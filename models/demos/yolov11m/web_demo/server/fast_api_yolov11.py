# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import logging
import time
from io import BytesIO

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

import ttnn
from models.demos.yolov11m.runner.performant_runner import YOLOv11PerformantRunner
from models.experimental.yolo_common.yolo_web_demo.yolo_evaluation_utils import postprocess

app = FastAPI(
    title="YOLOv11 object detection",
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


@app.on_event("startup")
async def startup():
    global model
    device_id = 0
    device = ttnn.CreateDevice(device_id, l1_small_size=24576, trace_region_size=3211264, num_command_queues=2)
    device.enable_program_cache()
    model = YOLOv11PerformantRunner(device)
    model._capture_yolov11_trace_2cqs()


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
    results = postprocess(response, image, image1)[0]
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
    return output
