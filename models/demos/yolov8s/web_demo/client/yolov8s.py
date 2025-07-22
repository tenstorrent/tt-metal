# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import io
import logging
import math
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

    def plot_boxes_cv2(self, bgr_img, boxes, savename=None, class_names=None, color=None):
        img = np.copy(bgr_img)
        colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

        def get_color(c, x, max_val):
            ratio = float(x) / max_val * 5
            i = int(math.floor(ratio))
            j = int(math.ceil(ratio))
            ratio = ratio - i
            r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
            return int(r * 255)

        width = img.shape[1]
        height = img.shape[0]
        for i in range(len(boxes)):
            box = boxes[i]
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            bbox_thick = int(0.6 * (height + width) / 600)
            if color:
                rgb = color
            else:
                rgb = (255, 0, 0)
            if len(box) >= 7 and class_names:
                cls_conf = box[5]
                cls_id = int(box[6])
                print("%s: %f" % (class_names[cls_id], cls_conf))
                classes = len(class_names)
                offset = cls_id * 123457 % classes
                red = get_color(2, offset, classes)
                green = get_color(1, offset, classes)
                blue = get_color(0, offset, classes)
                if color is None:
                    rgb = (red, green, blue)
                msg = str(class_names[cls_id]) + " " + str(round(cls_conf, 3))
                t_size = cv2.getTextSize(msg, 0, 0.7, thickness=bbox_thick // 2)[0]
                c1, c2 = (x1, y1), (x2, y2)
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(img, (x1, y1), (int(np.float32(c3[0])), int(np.float32(c3[1]))), rgb, -1)
                img = cv2.putText(
                    img,
                    msg,
                    (c1[0], int(np.float32(c1[1] - 2))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    bbox_thick // 2,
                    lineType=cv2.LINE_AA,
                )

            img = cv2.rectangle(img, (x1, y1), (int(x2), int(y2)), rgb, bbox_thick)
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
            parser = argparse.ArgumentParser(description="YOLOv8 script")
            parser.add_argument("--api-url", type=str, required=True, help="URL for the object detection API")
            args = parser.parse_args()
            self.api_url = args.api_url

        url = f"{self.api_url}/objdetection_v2"

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
                    # return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

        t3 = time.time()
        # Convert frame to ndarray and perform post-processing
        bgr_image = frame.to_ndarray(format="bgr24")
        conf_thresh = 0.6
        nms_thresh = 0.5

        # Load class names and plot bounding boxes
        namesfile = "../../../../experimental/yolo_common/yolo_web_demo/coco.names"
        class_names = self.load_class_names(namesfile)
        image_final = self.plot_boxes_cv2(bgr_image, output, None, class_names)

        t4 = time.time()
        logging.info(
            f" IMG-IN | WH | Post | Total time: {(t1-t0):.3f} | {(t3-t1):.3f} | {(t4-t3):.3f} || {(t4-t0):.3f} "
        )

        return av.VideoFrame.from_ndarray(image_final, format="bgr24")


st.title("YOLOv8s Detection Demo")

webrtc_streamer(
    key="example",
    video_transformer_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"min": 640, "ideal": 400, "max": 960},
            "height": {"min": 640, "ideal": 400, "max": 960},
            "frameRate": {"min": 1, "ideal": 50, "max": 60},
        }
    },
)
