import json
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
from models.demos.yolov4.tests.yolov4_perfomant import Yolov4Trace2CQ
import ttnn

import cv2
import numpy as np
import torch

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


# @app.post("/objdetection_v2")
# async def objdetection_v2(file: UploadFile = File(...)):
#    contents = await file.read()
#    response = model.run_traced_inference(Image.open(BytesIO(contents)))
#    return json.dumps(response, indent=4)
#
#
#


def process_request(output):
    # Convert all tensors to lists for JSON serialization
    # output_serializable = {'output': [tensor.tolist() for tensor in output['output']]}
    output_serializable = {"output": [tensor.tolist() for tensor in output]}
    return output_serializable


@app.post("/objdetection_v2")
async def objdetection_v2(file: UploadFile = File(...)):
    contents = await file.read()

    # Load and convert the image to RGB
    image = Image.open(BytesIO(contents)).convert("RGB")
    image = np.array(image)
    image = torch.from_numpy(image)
    # Perform object detection
    # response = model.do_detect(image)
    response = model.run_traced_inference(image)

    print("response in fastapi is:", response)

    # Convert response tensors to JSON-serializable format
    output = process_request(response)
    return output
