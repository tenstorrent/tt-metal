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
from models.demos.yolov9c.runner.performant_runner import YOLOv9PerformantRunner
from models.demos.yolov9c.web_demo.server.demo_utils import postprocess

app = FastAPI(
    title="YOLOv9c segmentation",
    description="Inference engine to detect objects and generate segmentation masks in image.",
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
        model = YOLOv9PerformantRunner(device, 1, model_task="segment")
    else:
        device_id = 0
        device = ttnn.CreateDevice(device_id, l1_small_size=24576, trace_region_size=3211264, num_command_queues=2)
        device.enable_program_cache()
        model = YOLOv9PerformantRunner(device, 1, model_task="segment")
    model._capture_yolov9_trace_2cqs()


@app.on_event("shutdown")
async def shutdown():
    model.release()


def process_segmentation_output(output, image_shape):
    """Process segmentation output to extract masks and bounding boxes"""
    # Extract detection outputs
    detect1_out, detect2_out, detect3_out = [ttnn.to_torch(tensor, dtype=torch.float32) for tensor in output[1][0]]
    mask = ttnn.to_torch(output[1][1], dtype=torch.float32)
    proto = ttnn.to_torch(output[1][2], dtype=torch.float32)
    proto = proto.reshape((1, 160, 160, 32)).permute((0, 3, 1, 2))

    # Combine detection outputs
    detection_output = [[detect1_out, detect2_out, detect3_out], mask, proto]

    # Create batch info for postprocessing
    batch = [["input_image"], [np.zeros(image_shape)], [image_shape]]

    # Postprocess to get results
    results = postprocess([ttnn.to_torch(output[0]), detection_output], None, [np.zeros(image_shape)], batch)

    if len(results) > 0 and results[0] is not None:
        result = results[0]
        masks = []
        if result.masks is not None and len(result.masks) > 0:
            for i in range(len(result.masks)):
                mask_data = result.masks[i].data.cpu().numpy()
                masks.append(mask_data.tolist())

        return {
            "masks": masks,
        }

    return {
        "masks": [],
    }


@app.post("/segmentation")
async def segmentation(file: UploadFile = File(...)):
    contents = await file.read()
    # Load and convert the image to RGB
    image = Image.open(BytesIO(contents)).convert("RGB")
    image1 = np.array(image)
    if type(image1) == np.ndarray and len(image1.shape) == 3:  # cv2 image
        image = torch.from_numpy(image1).float().div(255.0).unsqueeze(0)
    elif type(image1) == np.ndarray and len(image1.shape) == 4:
        image = torch.from_numpy(image1).float().div(255.0)
    else:
        print("unknown image type")
        return {"error": "Invalid image format"}

    t1 = time.time()
    response = model.run(image)
    t2 = time.time()
    logging.info("The inference on the server side took: %.3f seconds", t2 - t1)

    # Process segmentation output
    result = process_segmentation_output(response, image1.shape)
    t3 = time.time()
    logging.info("The post-processing took: %.3f seconds", t3 - t2)

    return result
