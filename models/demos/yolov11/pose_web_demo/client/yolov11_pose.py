# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import argparse
import io
import logging

import av
import cv2
import numpy as np
import requests
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# Configure the logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()]
)


class VideoProcessor(VideoProcessorBase):
    def __init__(self, server_url="http://localhost:8000"):
        self.frame_count = 0
        self.server_url = server_url

    def load_class_names(self, namesfile):
        class_names = []
        with open(namesfile, "r") as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.rstrip()
                class_names.append(line)
        return class_names

    def plot_pose_cv2(self, bgr_img, detections, savename=None):
        """Plot pose keypoints and skeleton on image"""
        img = np.copy(bgr_img)
        height, width = img.shape[:2]

        # COCO pose keypoints (17 points)
        keypoints_names = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]

        # COCO pose skeleton connections
        skeleton = [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [5, 11],
            [6, 12],
            [5, 6],
            [5, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [1, 2],
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
        ]

        # Keypoint colors (COCO format)
        keypoint_colors = [
            (255, 0, 0),  # nose - red
            (0, 255, 0),  # eyes - green
            (0, 255, 0),
            (0, 0, 255),  # ears - blue
            (0, 0, 255),
            (255, 255, 0),  # shoulders - cyan
            (255, 255, 0),
            (255, 0, 255),  # elbows - magenta
            (255, 0, 255),
            (0, 255, 255),  # wrists - yellow
            (0, 255, 255),
            (128, 0, 128),  # hips - purple
            (128, 0, 128),
            (0, 128, 128),  # knees - teal
            (0, 128, 128),
            (128, 128, 0),  # ankles - olive
            (128, 128, 0),
        ]

        # Skeleton colors
        skeleton_colors = [
            (255, 255, 255),  # white for main connections
            (255, 255, 255),
            (255, 255, 255),
            (255, 255, 255),
            (255, 255, 255),
            (255, 255, 255),
            (255, 255, 255),
            (255, 255, 255),
            (255, 255, 255),
            (255, 255, 255),
            (255, 255, 255),
            (255, 255, 255),
            (255, 255, 255),
            (255, 255, 255),
            (255, 255, 255),
            (255, 255, 255),
            (255, 255, 255),
            (255, 255, 255),
            (255, 255, 255),
        ]

        person_count = 0
        for detection in detections:
            person_count += 1

            # Extract data from detection
            # Format: [x1,y1,x2,y2,conf,class,kpt1_x,kpt1_y,kpt1_v,...]
            box = detection[:4]  # normalized coordinates
            conf = detection[4]
            class_id = detection[5]

            # Convert normalized coordinates back to image coordinates
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add confidence text
            cv2.putText(img, ".2f", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Extract keypoints (51 values = 17 keypoints * 3)
            keypoints_data = detection[6:]
            keypoints = []
            for i in range(17):  # 17 keypoints
                base_idx = i * 3
                if base_idx + 2 < len(keypoints_data):
                    kx = keypoints_data[base_idx]
                    ky = keypoints_data[base_idx + 1]
                    kv = keypoints_data[base_idx + 2]

                    # Convert normalized to image coordinates
                    kx_img = int(kx * width)
                    ky_img = int(ky * height)
                    keypoints.append((kx_img, ky_img, kv))

            # Draw keypoints
            for i, (kx, ky, kv) in enumerate(keypoints):
                if kv > 0.3:  # Only draw visible keypoints
                    color = keypoint_colors[i]
                    cv2.circle(img, (kx, ky), 4, color, -1)
                    cv2.circle(img, (kx, ky), 2, (255, 255, 255), -1)  # white border

            # Draw skeleton connections
            for i, (start_idx, end_idx) in enumerate(skeleton):
                if (
                    start_idx < len(keypoints)
                    and end_idx < len(keypoints)
                    and keypoints[start_idx][2] > 0.3
                    and keypoints[end_idx][2] > 0.3
                ):
                    start_point = (keypoints[start_idx][0], keypoints[start_idx][1])
                    end_point = (keypoints[end_idx][0], keypoints[end_idx][1])
                    color = skeleton_colors[i]
                    cv2.line(img, start_point, end_point, color, 2)

            # Add person label
            cv2.putText(img, f"Person {person_count}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if savename is not None:
            cv2.imwrite(savename, img)

        return img

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Process every 30 frames to reduce load
        if self.frame_count % 30 == 0:
            try:
                # Convert to RGB for processing
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Convert to bytes
                success, encoded_img = cv2.imencode(".jpg", rgb_img)
                if not success:
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

                img_bytes = encoded_img.tobytes()

                # Send to server
                files = {"file": ("image.jpg", io.BytesIO(img_bytes), "image/jpeg")}
                server_endpoint = f"{self.server_url}/pose_estimation_v2"
                response = requests.post(server_endpoint, files=files, timeout=10)

                if response.status_code == 200:
                    detections = response.json()
                    if detections:
                        img = self.plot_pose_cv2(img, detections)
                else:
                    logging.error(f"Server error: {response.status_code}")

            except Exception as e:
                logging.error(f"Error processing frame: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Pose Estimation Web Demo")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the pose estimation server (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    st.title("YOLOv11 Pose Estimation Demo")
    st.markdown("Real-time pose estimation using YOLOv11 on Tenstorrent hardware")

    # Sidebar configuration
    st.sidebar.title("Configuration")
    st.sidebar.markdown("Adjust detection parameters:")

    # Create video processor factory with server URL
    def create_video_processor():
        return VideoProcessor(server_url=args.server_url)

    # Check if running on HTTPS (required for camera access)
    import streamlit.components.v1 as components

    # Show HTTPS requirement warning
    https_warning = components.html(
        """
    <div id="https-warning" style="display: none;">
        <div style="background: #ff9800; color: white; padding: 10px; border-radius: 5px; margin: 10px 0;">
            ⚠️ Camera access requires HTTPS. Please use one of these options:
        </div>
    </div>
    <script>
    if (window.location.protocol !== 'https:' &&
        window.location.hostname !== 'localhost' &&
        window.location.hostname !== '127.0.0.1' &&
        !window.location.hostname.startsWith('192.168.') &&
        !window.location.hostname.startsWith('10.') &&
        !window.location.hostname.startsWith('172.')) {
        document.getElementById('https-warning').style.display = 'block';
    }
    if (navigator.mediaDevices === undefined) {
        document.getElementById('https-warning').innerHTML =
            '<div style="background: #f44336; color: white; padding: 10px; border-radius: 5px; margin: 10px 0;">' +
            '❌ navigator.mediaDevices is undefined. Camera access not supported in this browser/environment.' +
            '</div>';
        document.getElementById('https-warning').style.display = 'block';
    }
    </script>
    """
    )

    st.markdown("---")

    # Start webcam stream with error handling
    try:
        ctx = webrtc_streamer(
            key="pose-estimation",
            video_processor_factory=create_video_processor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": {"deviceId": args.device}, "audio": False},
        )
    except Exception as e:
        st.error(f"WebRTC Error: {e}")
        st.error("This might be due to browser compatibility issues.")
        st.info("Try refreshing the page or using a different browser (Chrome recommended)")
        st.info("Make sure to allow camera permissions when prompted.")
        return

    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("1. Allow camera access when prompted")
    st.markdown("2. Position yourself in frame for pose detection")
    st.markdown("3. Pose keypoints and skeleton will be displayed in real-time")
    st.markdown("4. Each detected person will show bounding box and pose skeleton")

    if ctx.video_processor:
        st.markdown(f"**Status:** Processing frames at ~{30} FPS")


if __name__ == "__main__":
    main()
