import face_recognition
import cv2
import os

# Load reference images and encode faces
def load_reference_faces(reference_folder):
    known_encodings = []
    for filename in os.listdir(reference_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(reference_folder, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                print(f"[INFO] Loaded encoding for {filename}")
            else:
                print(f"[WARN] No face found in {filename}")
    return known_encodings

# Detect and highlight faces in video
def detect_and_annotate_video(video_path, known_encodings, output_path="output_with_matches.mp4", tolerance=0.6, resize_scale=0.25):
    video = cv2.VideoCapture(video_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    frame_number = 0
    found_frames = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_number += 1

        # Convert BGR to RGB and resize
        rgb_frame = frame[:, :, ::-1]
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=resize_scale, fy=resize_scale)

        try:
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        except Exception as e:
            print(f"[ERROR] Face processing failed at frame {frame_number}: {e}")
            out.write(frame)
            continue

        match_found = False

        for loc, face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance)
            if True in matches:
                match_found = True

                # Scale bounding box to original size
                top, right, bottom, left = [int(coord / resize_scale) for coord in loc]

                # Draw box and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "Match", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                found_frames.append((frame_number, timestamp))
                print(f"[MATCH] Found at frame {frame_number} (timestamp: {timestamp:.2f}s)")
                break

        out.write(frame)

    video.release()
    out.release()
    return found_frames

# Main
if __name__ == "__main__":
    reference_folder = "reference_images"
    video_path = "video2.mp4"

    known_faces = load_reference_faces(reference_folder)
    if not known_faces:
        print("No valid reference faces found.")
        exit()

    matches = detect_and_annotate_video(video_path, known_faces)

    if matches:
        print("\n[RESULT] Person detected at these timestamps:")
        for frame, timestamp in matches:
            print(f" - Frame {frame}, Time {timestamp:.2f}s")
        print("\n[INFO] Annotated video saved as 'output_with_matches.mp4'")
    else:
        print("\n[RESULT] Person not detected in the video.")
    