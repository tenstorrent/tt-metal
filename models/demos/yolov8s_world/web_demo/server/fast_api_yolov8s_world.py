# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import logging
import os
import time
from io import BytesIO

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

import ttnn
from models.demos.yolov8s_world.runner.performant_runner import YOLOv8sWorldPerformantRunner
from models.experimental.yolo_common.yolo_web_demo.yolo_evaluation_utils import postprocess

app = FastAPI(
    title="YOLOv8s_world object detection",
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


@app.on_event("startup")
async def startup():
    global model
    if ("WH_ARCH_YAML" in os.environ) and os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
        print("WH_ARCH_YAML:", os.environ.get("WH_ARCH_YAML"))
        device_id = 0
        device = ttnn.CreateDevice(
            device_id,
            dispatch_core_config=get_dispatch_core_config(),
            l1_small_size=24576,
            trace_region_size=3211264,
            num_command_queues=2,
        )
        device.enable_program_cache()
        model = YOLOv8sWorldPerformantRunner(device, 1)
    else:
        device_id = 0
        device = ttnn.CreateDevice(device_id, l1_small_size=24576, trace_region_size=3211264, num_command_queues=2)
        device.enable_program_cache()
        model = YOLOv8sWorldPerformantRunner(device, 1)
    model._capture_yolov8s_world_trace_2cqs()


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
    response = ttnn.to_torch(response[0])
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
