# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Streamlit client for Face Recognition pipeline.

Features:
- Real-time face detection and recognition via webcam
- Register new faces to the database
- View and manage registered faces
"""

import io
import logging
import time

import av
import cv2
import requests
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Keypoint colors (left eye, right eye, nose, left mouth, right mouth)
KP_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

# Identity colors for known vs unknown
KNOWN_COLOR = (0, 255, 0)  # Green for known faces
UNKNOWN_COLOR = (0, 0, 255)  # Red for unknown faces
TOO_SMALL_COLOR = (128, 128, 128)  # Gray for faces too small to recognize


class FaceRecognitionProcessor(VideoProcessorBase):
    """Video processor for real-time face recognition."""

    def __init__(self):
        self.api_url = None
        self.input_size = 640  # Customer requirement: 640x640
        self.detection_thresh = 0.5
        self.recognition_thresh = 0.5  # With correct preprocessing, 0.5 works well
        # FPS tracking
        self.last_time = time.time()
        self.fps = 0.0
        self.fps_alpha = 0.9  # Smoothing factor for FPS

    def set_api_url(self, url):
        self.api_url = url

    def set_params(self, input_size, detection_thresh, recognition_thresh):
        self.input_size = input_size
        self.detection_thresh = detection_thresh
        self.recognition_thresh = recognition_thresh

    def draw_results(self, image, faces, latency_ms=0, server_latency_ms=0, fps=0):
        """Draw face boxes, keypoints, confidence scores, and metrics overlay."""
        img = image.copy()
        height, width = img.shape[:2]

        scale = max(width, height) / 640
        thickness = max(1, int(2 * scale))
        kp_radius = max(2, int(3 * scale))
        font_scale = max(0.4, 0.5 * scale)

        for face in faces:
            # Convert normalized coordinates to pixels
            x1 = int(face["box"][0] * width)
            y1 = int(face["box"][1] * height)
            x2 = int(face["box"][2] * width)
            y2 = int(face["box"][3] * height)

            # Color coding: Green = known, Red = unknown
            is_known = face.get("is_known", False)
            color = KNOWN_COLOR if is_known else UNKNOWN_COLOR

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Only show label for recognized faces (green)
            if is_known:
                name = face.get("identity", "Unknown")
                similarity = face.get("similarity", 0.0)
                label = f"{name}: {similarity:.2f}"

                # Draw label background
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                cv2.rectangle(img, (x1, y1 - label_h - 8), (x1 + label_w + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

            # Draw keypoints
            for i, kp in enumerate(face["keypoints"]):
                kx = int(kp[0] * width)
                ky = int(kp[1] * height)
                cv2.circle(img, (kx, ky), kp_radius, KP_COLORS[i], -1)

        # Show inference time (real detection speed) - top left corner
        inference_text = f"Inference: {server_latency_ms:.0f}ms"
        cv2.rectangle(img, (5, 5), (160, 30), (0, 0, 0), -1)  # Black background
        cv2.putText(img, inference_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return img

    def recv(self, frame):
        """Process incoming video frame."""
        if self.api_url is None:
            return frame

        t0 = time.time()

        # Convert frame to PIL image
        pil_image = frame.to_image()

        # Save as JPEG with high quality for better detection
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=95)
        byte_im = buf.getvalue()

        url = f"{self.api_url}/recognize"
        server_latency_ms = 0
        faces = []

        try:
            response = requests.post(
                url,
                files={"file": byte_im},
                params={
                    "input_size": self.input_size,
                    "detection_thresh": self.detection_thresh,
                    "recognition_thresh": self.recognition_thresh,
                },
                timeout=30,  # First calls can be slow due to compilation
            )

            if response.status_code == 200:
                result = response.json()
                faces = result.get("faces", [])
                server_latency_ms = result.get("total_time_ms", 0)
            else:
                logging.error(f"Request failed: {response.status_code}")
                return frame

        except requests.exceptions.RequestException as e:
            logging.error(f"Request error: {e}")
            return frame

        t1 = time.time()

        # Calculate latency and FPS
        latency_ms = (t1 - t0) * 1000
        current_fps = 1.0 / (t1 - self.last_time) if (t1 - self.last_time) > 0 else 0
        self.fps = self.fps_alpha * self.fps + (1 - self.fps_alpha) * current_fps  # Smoothed FPS
        self.last_time = t1

        # Draw results on original frame with metrics overlay
        bgr_image = frame.to_ndarray(format="bgr24")
        result_image = self.draw_results(
            bgr_image, faces, latency_ms=latency_ms, server_latency_ms=server_latency_ms, fps=self.fps
        )

        logging.info(
            f"Server: {server_latency_ms:.1f}ms | Total: {latency_ms:.1f}ms | FPS: {self.fps:.1f} | Faces: {len(faces)}"
        )

        return av.VideoFrame.from_ndarray(result_image, format="bgr24")


def main():
    st.set_page_config(
        page_title="Face Recognition - Tenstorrent",
        page_icon="👤",
        layout="wide",
    )

    st.title("👤 Face Recognition")
    st.markdown("**YuNet (Detection) + SFace (Recognition) on Tenstorrent AI Hardware**")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    api_url = st.sidebar.text_input(
        "API Server URL", value="http://localhost:8000", help="URL of the Face Recognition FastAPI server"
    )

    input_size = st.sidebar.selectbox(
        "Detection Input Size", options=[640, 320], index=0, help="YuNet input resolution (640x640 recommended)"
    )

    detection_thresh = st.sidebar.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Face detection confidence threshold",
    )

    recognition_thresh = st.sidebar.slider(
        "Recognition Threshold",
        min_value=0.3,
        max_value=0.9,
        value=0.55,  # 0.55 helps avoid false positives with family members
        step=0.05,
        help="Face recognition similarity threshold (higher = stricter). Recommend 0.5-0.6",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Keypoint Legend")
    st.sidebar.markdown("- 🔵 Left Eye")
    st.sidebar.markdown("- 🟢 Right Eye")
    st.sidebar.markdown("- 🔴 Nose")
    st.sidebar.markdown("- 🟡 Left Mouth Corner")
    st.sidebar.markdown("- 🟣 Right Mouth Corner")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Color Legend")
    st.sidebar.markdown("- 🟩 **Green**: Registered face")
    st.sidebar.markdown("- 🟥 **Red**: Unknown face")

    # Custom CSS for fixed video size
    st.markdown(
        """
    <style>
    /* Fix video container size */
    .stVideo video,
    iframe[title="streamlit_webrtc.component"],
    div[data-testid="stWebRtc"] video {
        max-width: 640px !important;
        max-height: 480px !important;
        width: 640px !important;
        height: auto !important;
    }
    div[data-testid="stWebRtc"] {
        max-width: 660px !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Main content - tabs
    tab1, tab2, tab3 = st.tabs(["📹 Live Recognition", "📝 Register Faces", "🎬 Test Video"])

    with tab1:
        st.subheader("Live Camera Feed")

        # Show registered faces count
        try:
            resp = requests.get(f"{api_url}/registered", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                st.info(
                    f"📋 Registered faces: **{data['count']}** ({', '.join(data['registered_faces']) if data['registered_faces'] else 'None'})"
                )
        except:
            st.warning("⚠️ Cannot connect to server. Make sure the server is running.")

        # WebRTC streamer
        def processor_factory():
            processor = FaceRecognitionProcessor()
            processor.set_api_url(api_url)
            processor.set_params(input_size, detection_thresh, recognition_thresh)
            return processor

        webrtc_streamer(
            key="face-recognition",
            video_processor_factory=processor_factory,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                },
                "audio": False,
            },
            async_processing=True,
            # Use VP8 codec - more resilient to packet loss than H264
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
            },
            video_html_attrs={
                "style": {"width": "640px", "max-width": "100%"},
                "muted": True,
                "autoPlay": True,
            },
        )

    with tab2:
        st.subheader("Register New Face")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload a face image", type=["jpg", "jpeg", "png"], help="Upload a clear photo with one face visible"
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            name = st.text_input("Name", placeholder="Enter person's name")

            if st.button("Register Face", type="primary", disabled=uploaded_file is None or not name):
                with st.spinner("Registering face..."):
                    try:
                        # Reset file pointer
                        uploaded_file.seek(0)
                        files = {"file": uploaded_file.getvalue()}
                        data = {"name": name}

                        resp = requests.post(
                            f"{api_url}/register",
                            files=files,
                            data=data,
                            timeout=60,  # First call can be slow due to model compilation
                        )

                        if resp.status_code == 200:
                            result = resp.json()
                            st.success(f"✅ {result['message']}")
                            st.balloons()
                        else:
                            try:
                                error = resp.json().get("error", "Unknown error")
                            except:
                                error = f"Server error (status {resp.status_code}): {resp.text[:200]}"
                            st.error(f"❌ {error}")

                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        st.markdown("---")
        st.subheader("Registered Faces")

        try:
            resp = requests.get(f"{api_url}/registered", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                if data["registered_faces"]:
                    cols = st.columns(4)
                    for i, name in enumerate(data["registered_faces"]):
                        with cols[i % 4]:
                            st.markdown(f"**{name}**")
                            if st.button(f"Delete", key=f"del_{name}"):
                                del_resp = requests.delete(f"{api_url}/registered/{name}", timeout=2)
                                if del_resp.status_code == 200:
                                    st.success(f"Deleted {name}")
                                    st.rerun()
                else:
                    st.info("No faces registered yet. Upload an image above to register.")
        except:
            st.warning("Cannot connect to server.")

    with tab3:
        st.subheader("🎬 Test Video Demo")
        st.markdown(
            """
        Run a pre-recorded test video through the face recognition pipeline.
        This is useful for demos and benchmarking without a live camera.
        """
        )

        # Check for available test videos
        from pathlib import Path

        test_video_dir = Path(__file__).parent.parent / "server" / "test_videos"

        if test_video_dir.exists():
            video_files = list(test_video_dir.glob("*.mp4")) + list(test_video_dir.glob("*.avi"))

            if video_files:
                selected_video = st.selectbox(
                    "Select test video", options=[v.name for v in video_files], help="Choose a test video to process"
                )

                video_path = test_video_dir / selected_video

                if st.button("▶️ Run Live Detection", type="primary"):
                    st.markdown("**Live Detection:**")

                    # Live video display with detection overlay
                    video_display = st.empty()
                    status_text = st.empty()

                    cap = cv2.VideoCapture(str(video_path))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_fps = cap.get(cv2.CAP_PROP_FPS) or 1

                    results = []
                    frame_idx = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        height, width = frame.shape[:2]

                        # Send to API for recognition
                        _, img_encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

                        try:
                            resp = requests.post(
                                f"{api_url}/recognize",
                                files={"file": img_encoded.tobytes()},
                                params={
                                    "input_size": input_size,
                                    "detection_thresh": detection_thresh,
                                    "recognition_thresh": recognition_thresh,
                                },
                                timeout=30,
                            )

                            if resp.status_code == 200:
                                data = resp.json()

                                # Keypoint colors (left eye, right eye, nose, left mouth, right mouth)
                                kp_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

                                # Draw detection results on frame
                                for face in data.get("faces", []):
                                    # Get box coordinates (normalized 0-1)
                                    box = face.get("box", [0, 0, 1, 1])
                                    x1 = int(box[0] * width)
                                    y1 = int(box[1] * height)
                                    x2 = int(box[2] * width)
                                    y2 = int(box[3] * height)

                                    identity = face.get("identity", "Unknown")
                                    similarity = face.get("similarity", 0)
                                    is_known = face.get("is_known", False)

                                    if is_known:
                                        # Green for recognized - show name and score
                                        color = (0, 255, 0)
                                        label = f"{identity}: {similarity:.2f}"
                                        cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(label) * 12, y1), color, -1)
                                        cv2.putText(
                                            frame,
                                            label,
                                            (x1 + 2, y1 - 7),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6,
                                            (255, 255, 255),
                                            2,
                                        )
                                    else:
                                        # Red for detected but not recognized - just box, no label
                                        color = (0, 0, 255)

                                    # Draw bounding box for ALL detected faces
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                                    # Draw keypoints
                                    keypoints = face.get("keypoints", [])
                                    for i, kp in enumerate(keypoints):
                                        if len(kp) >= 2:
                                            kp_x = int(kp[0] * width)
                                            kp_y = int(kp[1] * height)
                                            cv2.circle(frame, (kp_x, kp_y), 3, kp_colors[i % len(kp_colors)], -1)

                                    # Store result (only for recognized faces in summary)
                                    if is_known:
                                        results.append(
                                            {
                                                "frame": frame_idx,
                                                "name": identity,
                                                "score": similarity,
                                            }
                                        )

                        except Exception as e:
                            logging.error(f"Error: {e}")

                        # Convert BGR to RGB for display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_display.image(frame_rgb, channels="RGB", use_container_width=True)

                        status_text.text(f"Frame {frame_idx+1}/{total_frames}")
                        frame_idx += 1

                        # Small delay for visual effect
                        import time

                        time.sleep(0.3)

                    cap.release()
                    status_text.text("✅ Complete!")

                    # Show summary of recognized faces
                    if results:
                        st.markdown("### Recognized Faces")
                        from collections import Counter

                        name_counts = Counter([r["name"] for r in results])

                        for name, count in name_counts.most_common():
                            scores = [r["score"] for r in results if r["name"] == name]
                            avg_score = sum(scores) / len(scores) if scores else 0
                            st.success(f"✅ **{name}**: {count}x (avg score: {avg_score:.2f})")
                    else:
                        st.info("No recognized faces in video")
            else:
                st.info("No test videos found. Create one using:")
                st.code("python -m models.experimental.sface.tests.benchmark.create_demo_video")
        else:
            st.info("Test video directory not found. Create test videos using:")
            st.code("python -m models.experimental.sface.tests.benchmark.create_demo_video")

        # Upload custom video option
        st.markdown("---")
        st.markdown("### Or upload your own video:")
        uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

        if uploaded_video:
            # Save temporarily and process
            temp_path = Path("/tmp/uploaded_test_video.mp4")
            with open(temp_path, "wb") as f:
                f.write(uploaded_video.read())
            st.video(str(temp_path))
            st.info(
                "Click 'Run Recognition' above after selecting the uploaded video from the dropdown (after refresh)."
            )


if __name__ == "__main__":
    main()
