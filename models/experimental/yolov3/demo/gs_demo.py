# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import cv2

from pathlib import Path
from loguru import logger

from models.experimental.yolov3.reference.models.common import DetectMultiBackend
from models.experimental.yolov3.tt.yolov3_detection_model import (
    yolov3_fused_model,
)
from models.experimental.yolov3.reference.utils.dataloaders import LoadImages
from models.experimental.yolov3.reference.utils.general import (
    non_max_suppression,
    scale_boxes,
    xyxy2xywh,
    check_img_size,
)
from models.experimental.yolov3.reference.utils.plots import (
    Annotator,
    colors,
)
from models.utility_functions import torch2tt_tensor

f = f"{Path(__file__).parent}"


def test_gs_demo(model_location_generator, device):
    torch.manual_seed(1234)

    # Load yolo
    model_path = model_location_generator("models", model_subdir="Yolo")
    data_path = model_location_generator("data", model_subdir="Yolo")

    data_image_path = str(data_path / "images")
    data_coco = str(data_path / "coco128.yaml")
    model_config_path = str(data_path / "yolov3.yaml")
    weights_loc = str(model_path / "yolov3.pt")

    reference_model = DetectMultiBackend(weights_loc, device=torch.device("cpu"), dnn=False, data=data_coco, fp16=False)
    reference_model = reference_model.model

    tt_module = yolov3_fused_model(device, model_location_generator)

    with torch.no_grad():
        tt_module.eval()
        reference_model.eval()

        # Load data
        stride = max(int(max(reference_model.stride)), 32)  # model stride
        imgsz = check_img_size((640, 640), s=stride)  # check image size
        dataset = LoadImages(data_image_path, img_size=imgsz, stride=stride, auto=True)
        names = reference_model.names

        logger.info(f"Running inference for GS Yolov3")

        path, im, im0s, _, s = next(iter(dataset))
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
            save_path_input = str(p / "yolov3_gs_input.jpg")
            save_path_output = str(p / "yolov3_gs_output.jpg")

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
            # Save input image
            cv2.imwrite(save_path_input, im0s)
            # Save results (image with detections)
            cv2.imwrite(save_path_output, im0)

    logger.info(f"Input image saved as {save_path_input}")
    logger.info(f"Result image saved as {save_path_output}")
