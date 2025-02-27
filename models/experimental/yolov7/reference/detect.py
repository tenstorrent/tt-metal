# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import time
from pathlib import Path

import cv2
import torch
from numpy import random
from loguru import logger


from models.experimental.yolov7.reference.models.experimental import attempt_load
from models.experimental.yolov7.reference.utils.datasets import LoadImages
from models.experimental.yolov7.reference.utils.general import (
    check_img_size,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    set_logging,
    increment_path,
)
from models.experimental.yolov7.reference.utils.plots import plot_one_box
from models.experimental.yolov7.reference.utils.torch_utils import (
    select_device,
    load_classifier,
    time_synchronized,
)


def test_detect(model_location_generator):
    f = f"{Path(__file__).parent}"
    # Initialize
    set_logging()
    device = select_device("cpu")
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model_path = model_location_generator("models", model_subdir="Yolo")
    data_path = model_location_generator("data", model_subdir="Yolo")

    data_image_path = str(data_path / "images/horses.jpg")
    weights = str(model_path / "yolov7.pt")
    imgsz = 640
    save_img = True  # save inference images
    source = data_image_path
    view_img = False
    save_txt = False
    augment = False
    conf_thres, iou_thres = 0.25, 0.45
    classes = None  # filter by class
    agnostic_nms = False
    save_conf = True
    webcam = False

    model = attempt_load(weights, map_location=device)  # load FP32 model

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Directories
    save_dir = Path(increment_path(Path(f) / "exp"))  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Second-stage classifier
    # classify = False
    # if classify:
    #     modelc = load_classifier(name='resnet101', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != "cpu" and (
            old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]
        ):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=False)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], "%g: " % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or view_img:  # Add bbox to image
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

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                    logger.info(f" The image with the result is saved in: {save_path}")

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""

    logger.info(f"Done. ({time.time() - t0:.3f}s)")
