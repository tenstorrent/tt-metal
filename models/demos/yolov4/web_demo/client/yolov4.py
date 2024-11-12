# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import io
import math
import json
import random
import argparse
import cv2
import requests
import torch
import av
import streamlit as st
import numpy as np


from torch import nn
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0

    def post_processing(self, img, conf_thresh, nms_thresh, output):
        box_array = output[0]
        confs = output[1].float()

        t1 = time.time()

        if type(box_array).__name__ != "ndarray":
            box_array = box_array.cpu().detach().numpy()
            confs = confs.cpu().detach().numpy()

        num_classes = confs.shape[2]

        # [batch, num, 4]
        box_array = box_array[:, :, 0]

        # [batch, num, num_classes] --> [batch, num]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)

        t2 = time.time()

        bboxes_batch = []
        for i in range(box_array.shape[0]):
            argwhere = max_conf[i] > conf_thresh
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]

            bboxes = []
            # nms for each class
            for j in range(num_classes):
                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]

                keep = self.nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

                if keep.size > 0:
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]

                    for k in range(ll_box_array.shape[0]):
                        bboxes.append(
                            [
                                ll_box_array[k, 0],
                                ll_box_array[k, 1],
                                ll_box_array[k, 2],
                                ll_box_array[k, 3],
                                ll_max_conf[k],
                                ll_max_conf[k],
                                ll_max_id[k],
                            ]
                        )

            bboxes_batch.append(bboxes)

        t3 = time.time()

        print("-----------------------------------")
        print("       max and argmax : %f" % (t2 - t1))
        print("                  nms : %f" % (t3 - t2))
        print("Post processing total : %f" % (t3 - t1))
        print("-----------------------------------")

        return bboxes_batch

    def load_class_names(self, namesfile):
        class_names = []
        with open(namesfile, "r") as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.rstrip()
                class_names.append(line)
        return class_names

    def nms_cpu(self, boxes, confs, nms_thresh=0.5, min_mode=False):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = confs.argsort()[::-1]

        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]

            keep.append(idx_self)

            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            if min_mode:
                over = inter / np.minimum(areas[order[0]], areas[order[1:]])
            else:
                over = inter / (areas[order[0]] + areas[order[1:]] - inter)

            inds = np.where(over <= nms_thresh)[0]
            order = order[inds + 1]

        return np.array(keep)

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
                cls_id = box[6]
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
        pil_image = frame.to_image()
        # resize on the client side
        new_size = (320, 320)
        pil_image = pil_image.resize(new_size)
        t1 = time.time()
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        file = {"file": byte_im}
        # Argument Parser to grab namespace_id of server pod from user
        parser = argparse.ArgumentParser(description="YOLOv4 script")
        parser.add_argument("--api-url", type=str, help="URL for the object detection API", required=True)
        args = parser.parse_args()
        apiurl = args.api_url
        url = f"{apiurl}/objdetection_v2"
        r = requests.post(url, files=file)

        if r.status_code == 200:
            try:
                # Get the JSON response as a dictionary
                response_dict = r.json()
                output = [torch.tensor(tensor_data) for tensor_data in response_dict["output"]]
            except ValueError:
                st.error("Failed to parse JSON. The response is not in JSON format.")
        else:
            st.error(f"Request failed with status code {r.status_code}")

        t3 = time.time()
        bgr_image = frame.to_ndarray(format="bgr24")
        conf_thresh = 0.6
        nms_thresh = 0.5
        boxes = self.post_processing(bgr_image, conf_thresh, nms_thresh, output)
        namesfile = "coco.names"
        class_names = self.load_class_names(namesfile)

        # random_number = random.randint(1, 100)
        # save_name = "ttnn_prediction_demo" + str(random_number) + ".jpg"
        save_name = None

        image_final = self.plot_boxes_cv2(bgr_image, boxes[0], save_name, class_names)
        t4 = time.time()
        print()
        print(f" IMG-IN | WH | Post | Total time: ")
        print(f" {(t1-t0):.3f} | {(t3-t1):.3f} | {(t4-t3):.3f} || {(t4-t0):.3f} ")

        # return image_final
        return av.VideoFrame.from_ndarray(image_final, format="bgr24")


st.sidebar.image("TT.png", use_column_width=True)
st.sidebar.image("GS.png", use_column_width=True)

webrtc_streamer(
    key="example",
    video_transformer_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"min": 320, "ideal": 400, "max": 960},
            # "height": {"min": 180, "ideal": 225, "max": 450},
            "height": {"min": 320, "ideal": 400, "max": 960},
            "frameRate": {"min": 1, "ideal": 50, "max": 60},
        }
    },
    # async_processing=True  # Use asynchronous processing for long tasks
)
