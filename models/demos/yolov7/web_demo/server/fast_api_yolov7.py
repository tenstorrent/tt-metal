# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import logging
import os
import sys
import time
from io import BytesIO

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

import models.demos.yolov7.reference.yolov7_model as yolov7_model
import models.demos.yolov7.reference.yolov7_utils as yolov7_utils
import ttnn
from models.demos.yolov7.demo.demo_utils import load_coco_class_names
from models.demos.yolov7.runner.performant_runner import YOLOv7PerformantRunner

sys.modules["models.common"] = yolov7_utils
sys.modules["models.yolo"] = yolov7_model

app = FastAPI(
    title="YOLOv7 object detection",
    description="Inference engine to detect objects in image.",
    version="0.0",
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


# Configure the logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()]
)


def get_dispatch_core_config():
    # TODO: 11059 move dispatch_core_type to device_params when all tests are updated to not use WH_ARCH_YAML env flag
    dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
    if ("WH_ARCH_YAML" in os.environ) and os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
        dispatch_core_type = ttnn.device.DispatchCoreType.ETH
    dispatch_core_axis = ttnn.DispatchCoreAxis.ROW
    dispatch_core_config = ttnn.DispatchCoreConfig(dispatch_core_type, dispatch_core_axis)

    return dispatch_core_config


def postprocess_yolov7_custom(preds, img, orig_imgs):
    """Custom postprocessing for YOLOv7 to match YOLOv11 format"""
    # The model output is already processed and contains detections in format [x1, y1, x2, y2, conf, cls]
    # We just need to scale the coordinates to the original image size

    results = []
    for batch_idx in range(preds.shape[0]):
        if preds[batch_idx].numel() == 0:
            # No detections
            results.append(
                {"boxes": {"xyxy": torch.empty((0, 4)), "conf": torch.empty((0,)), "cls": torch.empty((0,))}}
            )
            continue

        # Extract detections for this batch
        detections = preds[batch_idx]  # Shape: [num_detections, 6] where 6 = [x1, y1, x2, y2, conf, cls]

        # Scale coordinates to original image size
        img_h, img_w = orig_imgs[batch_idx].shape[:2]
        scaled_boxes = detections[:, :4].clone()
        scaled_boxes[:, [0, 2]] *= img_w / 640  # scale x coordinates
        scaled_boxes[:, [1, 3]] *= img_h / 640  # scale y coordinates

        # Clip boxes to image boundaries
        scaled_boxes[:, 0] = torch.clamp(scaled_boxes[:, 0], 0, img_w)
        scaled_boxes[:, 1] = torch.clamp(scaled_boxes[:, 1], 0, img_h)
        scaled_boxes[:, 2] = torch.clamp(scaled_boxes[:, 2], 0, img_w)
        scaled_boxes[:, 3] = torch.clamp(scaled_boxes[:, 3], 0, img_h)

        results.append({"boxes": {"xyxy": scaled_boxes, "conf": detections[:, 4], "cls": detections[:, 5]}})

    return results


@app.on_event("startup")
async def startup():
    global model, class_names
    if ("WH_ARCH_YAML" in os.environ) and os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
        print("WH_ARCH_YAML:", os.environ.get("WH_ARCH_YAML"))
        device_id = 0
        device = ttnn.CreateDevice(
            device_id,
            dispatch_core_config=get_dispatch_core_config(),
            l1_small_size=24576,
            trace_region_size=6434816,
            num_command_queues=2,
        )
        device.enable_program_cache()
        model = YOLOv7PerformantRunner(device)
    else:
        device_id = 0
        device = ttnn.CreateDevice(device_id, l1_small_size=24576, trace_region_size=3211264, num_command_queues=2)
        device.enable_program_cache()
        model = YOLOv7PerformantRunner(device)
    model._capture_yolov7_trace_2cqs()

    # Load class names
    class_names = load_coco_class_names()


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
    t2 = time.time()

    # Use standard YOLOv7 postprocessing
    from models.demos.yolov7.demo.demo_utils import postprocess as postprocess_yolov7

    # Create batch info for postprocessing
    batch = [["input_image"], [image1], [image1.shape]]

    # Postprocess using standard YOLOv7 function
    results = postprocess_yolov7(response, image, [image1], batch, class_names, "input_image", image1, None)[0]
    logging.info("The inference on the sever side took: %.3f seconds", t2 - t1)

    output = []
    if results["boxes"]["xyxy"].numel() > 0 and results["boxes"]["xyxy"].shape[0] > 0:
        boxes = results["boxes"]["xyxy"]  # [num_detections, 4]
        confs = results["boxes"]["conf"]  # [num_detections]
        cls_ids = results["boxes"]["cls"]  # [num_detections]

        for i in range(len(boxes)):
            # Normalize coordinates to [0,1] range
            x1, y1, x2, y2 = boxes[i] / torch.tensor(
                [image1.shape[1], image1.shape[0], image1.shape[1], image1.shape[0]]
            )
            output.append(
                [x1.item(), y1.item(), x2.item(), y2.item(), confs[i].item(), confs[i].item(), cls_ids[i].item()]
            )
    else:
        # No detections
        pass
    t3 = time.time()
    return output
