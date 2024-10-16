# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import cv2
import time

from loguru import logger
from pathlib import Path
from numpy import random


from models.experimental.yolov7.reference.models.load_torch_model import (
    get_yolov7_fused_cpu_model,
)
from models.experimental.yolov7.reference.utils.datasets import LoadImages
from models.experimental.yolov7.reference.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    set_logging,
)
from models.experimental.yolov7.reference.utils.plots import plot_one_box
from models.experimental.yolov7.reference.utils.torch_utils import (
    time_synchronized,
)


file_path = f"{Path(__file__).parent}"


def test_cpu_demo(model_location_generator):
    torch.manual_seed(1234)
    logger.info(file_path)
    # Get data
    data_path = model_location_generator("data", model_subdir="Yolo")

    data_image_path = str(data_path / "images/horses.jpg")
    data_coco = str(data_path / "coco128.yaml")
    imgsz = 640
    save_img = True  # save inference images
    source = data_image_path
    view_img = False
    augment = False

    set_logging()

    # Load model
    model_path = model_location_generator("models", model_subdir="Yolo")
    weights = str(model_path / "yolov7.pt")

    # Load model
    model = get_yolov7_fused_cpu_model(model_location_generator)  # load FP32 model

    # Load data and setups
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    dataset = LoadImages(data_image_path, img_size=imgsz, stride=stride)
    names = model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    logger.info(f"Running inference for CPU Yolov7")
    t0 = time.time()
    with torch.no_grad():
        for (
            path,
            im,
            im0s,
            _,
        ) in dataset:
            im = torch.from_numpy(im)
            im = im.float()
            im /= 255
            if im.ndimension() == 3:
                im = im.unsqueeze(0)

            # Inference- fused
            t1 = time_synchronized()
            # Calculating gradients would cause a GPU memory leak
            pred = model(im, augment=augment)[0]
            t2 = time_synchronized()

            conf_thres, iou_thres = 0.25, 0.45
            classes = None  # filter by class
            agnostic_nms = False
            save_conf = True
            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=False)
            t3 = time_synchronized()

            # Process predictions
            for i, det in enumerate(pred):  # per image
                p, s, im0, frame = (
                    file_path,
                    "",
                    im0s.copy(),
                    getattr(dataset, "frame", 0),
                )
                p = Path(p)  # to Path
                save_path_input = str(p / "yolov7_cpu_input.jpg")
                save_path_output = str(p / "yolov7_cpu_output.jpg")

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img:  # Add bbox to image
                        label = f"{names[int(cls)]} {conf:.2f}"
                        plot_one_box(
                            xyxy,
                            im0,
                            label=label,
                            color=colors[int(cls)],
                            line_thickness=1,
                        )

                # Print time (inference + NMS)
                logger.info(f"{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS")

                # Save input image
                cv2.imwrite(save_path_input, im0s)
                # Save results (image with detections)
                cv2.imwrite(save_path_output, im0)

    logger.info(f"Input image saved as {save_path_input}")
    logger.info(f"Result image saved as {save_path_output}")
