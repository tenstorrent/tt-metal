# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import json
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
from models.experimental.yolov4.tests.yolov4_perfomant_webdemo import Yolov4Trace2CQ
import ttnn

import cv2
import numpy as np
import torch
import time

app = FastAPI(
    title="YOLOv4 object detection",
    description="Inference engine to detect objects in image.",
    version="0.0",
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.on_event("startup")
async def startup():
    device_id = 0
    device = ttnn.CreateDevice(device_id, l1_small_size=24576, trace_region_size=1617920, num_command_queues=2)
    ttnn.enable_program_cache(device)
    global model
    model = Yolov4Trace2CQ()
    model.initialize_yolov4_trace_2cqs_inference(device)


@app.on_event("shutdown")
async def shutdown():
    model.release_yolov4_trace_2cqs_inference()


def process_request(output):
    # Convert all tensors to lists for JSON serialization
    output_serializable = {"output": [tensor.tolist() for tensor in output]}
    return output_serializable


@app.post("/objdetection_v2")
async def objdetection_v2(file: UploadFile = File(...)):
    contents = await file.read()

    # Load and convert the image to RGB
    image = Image.open(BytesIO(contents)).convert("RGB")
    image = np.array(image)
    if type(image) == np.ndarray and len(image.shape) == 3:  # cv2 image
        image = torch.from_numpy(image).float().div(255.0).unsqueeze(0)
    elif type(image) == np.ndarray and len(image.shape) == 4:
        image = torch.from_numpy(image).float().div(255.0)
    else:
        print("unknow image type")
        exit(-1)
    t1 = time.time()
    response = model.run_traced_inference(image)
    t2 = time.time()
    print("the inference on the sever side took: ", t2 - t1)

    # Convert response tensors to JSON-serializable format
    output = process_request(response)
    return output
