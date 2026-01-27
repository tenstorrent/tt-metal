# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import logging
import time
from io import BytesIO

import numpy as np
import torch
import torchvision
from fastapi import FastAPI, File, UploadFile
from PIL import Image

import ttnn

# from models.experimental.yolo_common.yolo_web_demo.yolo_evaluation_utils import postprocess
from models.demos.utils.common_demo_utils import xywh2xyxy
from models.demos.yolov6l.common import YOLOV6L_L1_SMALL_SIZE
from models.demos.yolov6l.runner.performant_runner import YOLOv6lPerformantRunner

app = FastAPI(
    title="YOLOv6l object detection",
    description="Inference engine to detect objects in image.",
    version="0.0",
)


def non_max_suppression(
    prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300
):
    num_classes = prediction.shape[2] - 5  # number of classes
    pred_candidates = torch.logical_and(
        prediction[..., 4] > conf_thres, torch.max(prediction[..., 5:], axis=-1)[0] > conf_thres
    )  # candidates
    # Check the parameters.
    assert 0 <= conf_thres <= 1, f"conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided."
    assert 0 <= iou_thres <= 1, f"iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided."

    # Function settings.
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
    time_limit = 10.0  # quit the function when nms cost time exceed the limit time.
    multi_label &= num_classes > 1  # multiple labels per box

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence

        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # confidence multiply the objectness
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), 1)
        else:  # Only keep the class with highest scores.
            conf, class_idx = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores
        keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = x[keep_box_idx]

    return output


def Boxes(data):
    return {"xyxy": data[:, :4], "conf": data[:, -2], "cls": data[:, -1]}


def postprocess(preds, img, orig_imgs, conf=0.25, max_det=300):
    args = {"conf": conf, "iou": 0.7, "agnostic_nms": False, "max_det": max_det, "classes": None}

    preds = non_max_suppression(
        preds,
        args["conf"],
        max_det=args["max_det"],
    )

    results = []
    for pred, orig_img in zip(preds, orig_imgs):
        results.append({"orig_img": orig_img, "boxes": Boxes(pred)})

    return results


@app.get("/")
async def root():
    return {"message": "Hello World"}


# Configure the logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()]
)


@app.on_event("startup")
async def startup():
    global model
    device_id = 0
    device = ttnn.CreateDevice(
        device_id, l1_small_size=YOLOV6L_L1_SMALL_SIZE, trace_region_size=3211264, num_command_queues=2
    )
    device.enable_program_cache()
    model = YOLOv6lPerformantRunner(device, 1)


@app.on_event("shutdown")
async def shutdown():
    # model.release()
    model.release()


@app.post("/objdetection_v2")
async def objdetection_v2(file: UploadFile = File(...)):
    contents = await file.read()
    # Load and convert the image to RGB
    image = Image.open(BytesIO(contents)).convert("RGB")
    image1 = np.array(image)
    if type(image1) == np.ndarray and len(image1.shape) == 3:  # cv2 image
        image = torch.from_numpy(image1).float().div(255.0).unsqueeze(0)
    elif type(image1) == np.ndarray and len(image1.shape) == 4:
        image = torch.from_numpy(image1).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)

    image = torch.permute(image, (0, 3, 1, 2))
    t1 = time.time()
    response = model.run(image)
    response = ttnn.to_torch(response)
    results = postprocess(response, image, image1)[0]
    t2 = time.time()
    logging.info("The inference on the sever side took: %.3f seconds", t2 - t1)
    conf_thresh = 0.6
    nms_thresh = 0.5

    output = []
    for i in range(len(results["boxes"]["xyxy"])):
        output.append(
            torch.concat(
                (
                    results["boxes"]["xyxy"][i] / 640,
                    results["boxes"]["conf"][i].unsqueeze(0),
                    results["boxes"]["conf"][i].unsqueeze(0),
                    results["boxes"]["cls"][i].unsqueeze(0),
                ),
                dim=0,
            )
            .numpy()
            .tolist()
        )
    t3 = time.time()
    logging.info("The post-processing to get the boxes took: %.3f seconds", t3 - t2)

    return output
