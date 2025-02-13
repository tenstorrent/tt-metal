# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import cv2
import math
import time
import torch
import torchvision
import numpy as np
from pathlib import Path
from loguru import logger


def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)


IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}


class LoadImages:
    def __init__(self, path, batch=1, vid_stride=1):
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())
            if os.path.isfile(a):
                files.append(a)
            else:
                raise FileNotFoundError(f"{p} does not exist")

        images = []
        for f in files:
            suffix = f.split(".")[-1].lower()
            if suffix in IMG_FORMATS:
                images.append(f)
        ni = len(images)

        self.files = images
        self.nf = ni
        self.ni = ni
        self.mode = "image"
        self.vid_stride = vid_stride
        self.bs = batch
        if self.nf == 0:
            raise FileNotFoundError(f"No images or videos found in {p}")

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.count >= self.nf:
                if imgs:
                    return paths, imgs, info
                else:
                    raise StopIteration

            path = self.files[self.count]

            self.mode = "image"
            im0 = imread(path)
            if im0 is None:
                logger.warning(f"WARNING ⚠️ Image Read Error {path}")
            else:
                paths.append(path)
                imgs.append(im0)
                info.append(f"image {self.count + 1}/{self.nf} {path}: ")
            self.count += 1
            if self.count >= self.ni:
                break

        return paths, imgs, info

    def _new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video {path}")
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)

    def __len__(self):
        return math.ceil(self.nf / self.bs)


def postprocess(preds, img, orig_imgs, batch, names):
    args = {"conf": 0.5, "iou": 0.7, "agnostic_nms": False, "max_det": 300, "classes": None}

    preds = non_max_suppression(
        preds,
        args["conf"],
        args["iou"],
        agnostic=args["agnostic_nms"],
        max_det=args["max_det"],
        classes=args["classes"],
    )

    results = []
    for pred, orig_img, img_path in zip(preds, orig_imgs, batch[0]):
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results.append(Results(orig_img, path=img_path, names=names, boxes=pred))

    return results
