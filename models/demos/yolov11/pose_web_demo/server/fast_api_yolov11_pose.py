# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import logging
from io import BytesIO

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

# Import TTNN here so it's available for the model
import ttnn

app = FastAPI(
    title="YOLOv11 pose estimation",
    description="Inference engine to detect human poses in image.",
    version="0.0",
)

# Configure the logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()]
)

# Global model variable
model = None


@app.get("/")
async def root():
    return {"message": "Hello World", "status": "Server running", "model_loaded": model is not None}


@app.on_event("startup")
async def startup():
    global model
    try:
        print("=== STEP 1: TTNN IMPORT ===")
        import ttnn

        print("✓ TTNN imported successfully")

        print("=== STEP 2: DEVICE CREATION ===")
        device_id = 0
        # Use the correct L1 size like the tests (24576 instead of 8192)
        from models.demos.yolov11.common import YOLOV11_L1_SMALL_SIZE

        device = ttnn.CreateDevice(
            device_id, l1_small_size=YOLOV11_L1_SMALL_SIZE, trace_region_size=6434816, num_command_queues=2
        )
        print("✓ Device created successfully")

        print("=== STEP 3: PROGRAM CACHE ===")
        device.enable_program_cache()
        print("✓ Program cache enabled")

        print("=== STEP 4: MODEL LOADING ===")
        from models.demos.yolov11.runner.performant_runner_pose import YOLOv11PosePerformantRunner

        model = YOLOv11PosePerformantRunner(device)
        print("✓ YOLOv11 pose model loaded successfully")

        print("=== STEP 5: TRACE CAPTURE ===")
        model._capture_yolov11_pose_trace_2cqs()
        print("✓ Trace capture completed successfully")

        # Store device for later use
        global device_global
        device_global = device

        print("=== TTNN FULL SETUP COMPLETE ===")
        print("Ready for optimized pose detection!")

    except Exception as e:
        print(f"=== TTNN SETUP FAILED ===")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        model = None


@app.on_event("shutdown")
async def shutdown():
    global device_global
    if "device_global" in globals():
        try:
            device_global.disable_and_close()
            print("✓ Device closed")
        except:
            pass


@app.post("/pose_estimation_v2")
async def pose_estimation_v2(file: UploadFile = File(...)):
    if model is None:
        return {"error": "TTNN model not loaded - check server startup logs"}

    try:
        print(f"DEBUG: Received pose request for file: {file.filename}")
        # Read and process the uploaded image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_array = np.array(image)

        # Convert to torch tensor
        if len(image_array.shape) == 3:  # HWC format
            image_tensor = torch.from_numpy(image_array).float().div(255.0).unsqueeze(0)
        else:
            return {"error": "Invalid image format"}

        # Permute to CHW format
        image_tensor = torch.permute(image_tensor, (0, 3, 1, 2))

        # Run inference
        response = model.run(image_tensor)
        response = ttnn.to_torch(response)

        # Postprocess for pose estimation
        from models.demos.utils.common_demo_utils import postprocess_pose

        results = postprocess_pose(response, image_tensor, [image_array], [file.filename], ["person"])[0]

        # Format results
        output = []
        conf_thresh = 0.6

        if results.boxes is not None and len(results.boxes) > 0:
            # boxes shape: [num_detections, 6] - (x1,y1,x2,y2,conf,class)
            boxes_tensor = results.boxes.cpu().numpy()

            for i, box in enumerate(boxes_tensor):
                x1, y1, x2, y2, conf, class_id = box
                if conf > conf_thresh:
                    # Normalize box coordinates to [0,1]
                    normalized_box = [x1 / 640.0, y1 / 640.0, x2 / 640.0, y2 / 640.0]

                    # Process keypoints
                    keypoints_data = []
                    if hasattr(results, "keypoints") and results.keypoints is not None and len(results.keypoints) > i:
                        # keypoints shape: [num_detections, 51] - (17 points × 3 values: x,y,conf)
                        kpt_data = results.keypoints[i].cpu().numpy()  # [51]

                        # Reshape to [17, 3] - (x, y, confidence) for each keypoint
                        kpt_reshaped = kpt_data.reshape(17, 3)

                        # Normalize x,y coordinates to [0,1] (confidence stays as-is)
                        kpt_reshaped[:, 0] = kpt_reshaped[:, 0] / 640.0  # x coordinates
                        kpt_reshaped[:, 1] = kpt_reshaped[:, 1] / 640.0  # y coordinates

                        # Flatten to list: [x,y,confidence] for each of 17 points
                        keypoints_data = kpt_reshaped.flatten().tolist()
                    else:
                        # No keypoints available, fill with zeros
                        keypoints_data = [0.0] * (17 * 3)  # 17 keypoints * 3 values each

                    # Format: [x1,y1,x2,y2,conf,class,keypoints...]
                    detection = normalized_box + [float(conf), float(class_id)] + keypoints_data
                    output.append(detection)

        print(f"DEBUG: Returning {len(output)} detections")
        if len(output) > 0:
            print(f"DEBUG: First detection has {len(output[0])} values")
        return output

    except Exception as e:
        logging.error(f"Error in pose estimation: {e}")
        import traceback

        traceback.print_exc()
        return {"error": f"Pose estimation failed: {str(e)}"}
