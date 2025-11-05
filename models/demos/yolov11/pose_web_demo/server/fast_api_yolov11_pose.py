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
from models.demos.utils.common_demo_utils import postprocess_pose
from models.demos.yolov11.runner.performant_runner_pose import YOLOv11PosePerformantRunner

app = FastAPI(
    title="YOLOv11 pose estimation",
    description="Inference engine to detect human poses in image.",
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
    model = YOLOv11PosePerformantRunner(device)
    # Disable trace capture for web demo (causes hangs during startup)
    # model._capture_yolov11_trace_2cqs() # Disabled to prevent startup hangs


@app.on_event("shutdown")
async def shutdown():
    # model.release()
    model.release()


@app.post("/pose_estimation_v2")
async def pose_estimation_v2(file: UploadFile = File(...)):
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

    # Postprocess for pose estimation
    results = postprocess_pose(response, image, [image1], [file.filename], ["person"])[0]
    logging.info("The inference on the sever side took: %.3f seconds", t2 - t1)

    conf_thresh = 0.6
    nms_thresh = 0.5

    output = []
    # Process pose keypoints
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        for i, (box, conf) in enumerate(zip(boxes, confs)):
            if conf > conf_thresh:
                # Normalize box coordinates to [0,1]
                normalized_box = box / 640.0

                # Add keypoints if available
                keypoints_data = []
                if hasattr(results, "keypoints") and results.keypoints is not None:
                    # Get keypoints for this detection
                    kpts = results.keypoints[i].xy.cpu().numpy()  # [17, 2]
                    kpts_conf = results.keypoints[i].conf.cpu().numpy()  # [17]

                    # Normalize keypoints to [0,1] relative to image
                    kpts_normalized = kpts / 640.0

                    # Flatten keypoints: [x1,y1,v1,x2,y2,v2,...]
                    for j in range(len(kpts_normalized)):
                        keypoints_data.extend(
                            [
                                kpts_normalized[j][0],  # x
                                kpts_normalized[j][1],  # y
                                kpts_conf[j],  # visibility/confidence
                            ]
                        )
                else:
                    # No keypoints available, fill with zeros
                    keypoints_data = [0.0] * (17 * 3)  # 17 keypoints * 3 values each

                # Output format: [x1,y1,x2,y2,conf,class,keypoints...]
                detection = list(normalized_box) + [conf, 0.0] + keypoints_data
                output.append(detection)

    t3 = time.time()
    logging.info(f"Processed {len(output)} pose detections")
    return output
