# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import io
import logging
import time

import av
import cv2
import numpy as np
import orjson
import requests
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# Configure the logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()]
)


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0

    def load_class_names(self, namesfile):
        class_names = []
        with open(namesfile, "r") as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.rstrip()
                class_names.append(line)
        return class_names

    def get_consistent_color(self, index):
        """Generate consistent colors for segmentation masks"""
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 0),  # Dark Red
            (0, 128, 0),  # Dark Green
            (0, 0, 128),  # Dark Blue
            (128, 128, 0),  # Olive
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
            (255, 128, 0),  # Orange
            (128, 255, 0),  # Lime
            (0, 255, 128),  # Spring Green
            (128, 0, 255),  # Violet
            (255, 0, 128),  # Rose
            (0, 128, 255),  # Sky Blue
            (255, 128, 128),  # Light Red
            (128, 255, 128),  # Light Green
        ]
        return colors[index % len(colors)]

    def plot_boxes_and_masks_cv2(self, bgr_img, boxes, masks, savename=None, class_names=None):
        """Plot bounding boxes and segmentation masks on the image"""
        img = np.copy(bgr_img)
        width = img.shape[1]
        height = img.shape[0]

        # Create overlay for masks
        overlay = img.copy()

        # Process masks first (so they appear behind boxes)
        if masks and len(masks) > 0:
            for i, mask_data in enumerate(masks):
                try:
                    # Convert mask to numpy array
                    mask = np.array(mask_data)

                    # Validate mask dimensions
                    if mask.size == 0:
                        print(f"Warning: Empty mask at index {i}, skipping...")
                        continue

                    # Ensure mask is 2D
                    if mask.ndim > 2:
                        mask = mask.squeeze()

                    # Check if mask has valid dimensions after squeezing
                    if mask.ndim != 2:
                        print(f"Warning: Invalid mask dimensions at index {i}: {mask.shape}, skipping...")
                        continue

                    # Resize mask to image dimensions if needed
                    if mask.shape != (height, width):
                        try:
                            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
                        except cv2.error as e:
                            print(f"Error resizing mask at index {i}: {e}")
                            print(f"Mask shape: {mask.shape}, Target shape: ({height}, {width})")
                            continue

                    # Threshold mask to binary
                    mask_binary = (mask > 0.5).astype(np.uint8)

                    # Get color for this mask
                    color_tuple = self.get_consistent_color(i)
                    # Convert color tuple to numpy array for broadcasting
                    color = np.array(color_tuple, dtype=np.uint8)

                    # Create colored mask
                    colored_mask = np.zeros_like(img, dtype=np.uint8)

                    # Apply color to mask for each channel
                    for c in range(3):
                        colored_mask[:, :, c] = (mask_binary * color[c]).astype(np.uint8)

                    # Apply mask overlay
                    mask_bool = mask_binary.astype(bool)
                    overlay[mask_bool] = (0.6 * overlay[mask_bool] + 0.4 * colored_mask[mask_bool]).astype(np.uint8)

                except Exception as e:
                    print(f"Error processing mask at index {i}: {e}")
                    continue

        # Apply overlay to image
        img = overlay

        if savename:
            print("save plot results to %s" % savename)
            cv2.imwrite(savename, img)
        return img

    def recv(self, frame):
        t0 = time.time()

        # Convert frame to PIL image and resize
        pil_image = frame.to_image()
        pil_image = pil_image.resize((640, 640))  # Resize to target dimensions
        t1 = time.time()

        # Save image as JPEG in-memory with optimized settings
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=85, optimize=True)
        byte_im = buf.getvalue()
        file = {"file": byte_im}

        # Parse API URL once at the class level for efficiency
        if not hasattr(self, "api_url"):
            parser = argparse.ArgumentParser(description="YOLOv9c segmentation script")
            parser.add_argument("--api-url", type=str, required=True, help="URL for the segmentation API")
            args = parser.parse_args()
            self.api_url = args.api_url

        url = f"{self.api_url}/segmentation"

        try:
            # Use a persistent session for multiple requests
            with requests.Session() as session:
                # Post request with a timeout
                response = session.post(url, files=file, timeout=5)

                # Check if response is successful
                if response.status_code == 200:
                    # Parse JSON response
                    output = orjson.loads(response.content)
                else:
                    print(f"Request failed with status code {response.status_code}")
                    return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

        t3 = time.time()
        # Convert frame to ndarray and perform post-processing
        bgr_image = frame.to_ndarray(format="bgr24")

        # Load class names
        namesfile = "coco.names"
        class_names = self.load_class_names(namesfile)

        # Extract boxes and masks from response
        boxes = output.get("boxes", [])
        masks = output.get("masks", [])

        # Plot boxes and masks
        image_final = self.plot_boxes_and_masks_cv2(bgr_image, boxes, masks, None, class_names)

        t4 = time.time()
        logging.info(
            f" IMG-IN | WH | Post | Total time: {(t1-t0):.3f} | {(t3-t1):.3f} | {(t4-t3):.3f} || {(t4-t0):.3f} "
        )

        return av.VideoFrame.from_ndarray(image_final, format="bgr24")


st.sidebar.title("YOLOv9c Segmentation Demo")

# Try to load logo images, but don't fail if they don't exist
try:
    st.sidebar.image("TT.png", use_column_width=True)
except:
    st.sidebar.write("Tenstorrent")

try:
    st.sidebar.image("GS.png", use_column_width=True)
except:
    st.sidebar.write("GS")

st.title("YOLOv9c Segmentation Demo")
st.write("This demo shows real-time object detection with segmentation masks overlay.")

webrtc_streamer(
    key="segmentation_example",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"min": 320, "ideal": 400, "max": 960},
            "height": {"min": 320, "ideal": 400, "max": 960},
            "frameRate": {"min": 1, "ideal": 50, "max": 60},
        }
    },
)

st.markdown(
    """
### Features:
- **Real-time segmentation**: Process video stream with live segmentation masks
- **Instance segmentation**: Separate masks for each detected object
- **Color-coded masks**: Each instance gets a unique color for easy identification
- **Mask overlays**: Transparent mask visualization with contour outlines
- **Pure segmentation**: Focus on pixel-level segmentation without bounding boxes

### Performance:
- **Inference time**: ~0.04-0.05 seconds per frame
- **FPS**: 18-23 frames per second
- **Hardware**: Tenstorrent Wormhole B0
"""
)
