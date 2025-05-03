# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
import cv2
from loguru import logger
from datasets import load_dataset

from models.experimental.yolov5.reference.models.common import DetectMultiBackend
from models.experimental.yolov5.tt.yolov5_detection_model import (
    yolov5s_detection_model,
)
from models.experimental.yolov5.reference.utils.dataloaders import LoadImages
from models.experimental.yolov5.reference.utils.general import check_img_size
from models.experimental.yolov5.reference.utils.general import (
    non_max_suppression,
    scale_boxes,
    xyxy2xywh,
)
from models.experimental.yolov5.reference.utils.plots import (
    Annotator,
    colors,
)

from models.utility_functions import torch2tt_tensor


def download_images(path, imgsz):
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    if imgsz is not None:
        image = image.resize(imgsz)

    image.save(path / "input_image.jpg")


def test_detection_model(device):
    refence_model = DetectMultiBackend(
        ROOT / "yolov5s.pt",
        device=torch.device("cpu"),
        dnn=False,
        data=None,
        fp16=False,
    )

    refence_module = refence_model.model
    tt_module = yolov5s_detection_model(device)

    with torch.no_grad():
        tt_module.eval()
        refence_module.eval()

        # Load data
        stride = max(int(max(refence_module.stride)), 32)  # model stride
        imgsz = check_img_size((640, 640), s=stride)  # check image size

        download_images(Path(ROOT), None)
        dataset = LoadImages(ROOT, img_size=imgsz, stride=stride, auto=True)
        names = refence_module.names

        logger.info(f"Running inference for GS Yolov5")

        for path, im, im0s, _, s in dataset:
            im = torch.from_numpy(im)
            im = im.float()
            im /= 255  # 0 - 255 to 0.0 - 1.0

            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference- fused tt
            tt_im = torch2tt_tensor(im, device)
            pred = tt_module(tt_im)

            conf_thres, iou_thres = 0.25, 0.45
            classes = None  # filter by class
            agnostic_nms = False

            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=1000)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                logger.info(f"Save results in {f}")
                p, im0, frame = f, im0s.copy(), getattr(dataset, "frame", 0)

                p = Path(p)  # to Path
                save_path = str(p / "out_img.jpg")

                s += "%gx%g " % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                annotator = Annotator(im0, line_width=3, example=str(names))

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if True:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh)  # label format
                        # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if False else (f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))

                # Get results
                im0 = annotator.result()

                # Save results (image with detections)
                cv2.imwrite(save_path, im0)

    logger.info(f"Result image saved on {save_path}")
