#!/usr/bin/env python3

import sys

sys.path.insert(0, "/home/ubuntu/pose/tt-metal")

import ttnn
import torch
import cv2
import numpy as np
from PIL import Image

print("=== Testing TTNN Pose Model Loading ===")

try:
    print("1. Creating device with 2 command queues...")
    device = ttnn.CreateDevice(0, l1_small_size=32768, num_command_queues=2)
    print("✓ Device created")

    print("2. Loading YOLOv11PosePerformantRunner...")
    from models.demos.yolov11.runner.performant_runner_pose import YOLOv11PosePerformantRunner

    model = YOLOv11PosePerformantRunner(device)
    print("✓ Model loaded")

    print("3. Testing with sample image...")
    # Load test image
    image_path = "/home/ubuntu/pose/tt-metal/models/demos/yolov11/pose_web_demo/test_image.jpg"
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)
    print(f"Image shape: {image_array.shape}")

    # Preprocess
    image_tensor = torch.from_numpy(image_array).float().div(255.0).unsqueeze(0)
    image_tensor = torch.permute(image_tensor, (0, 3, 1, 2))
    print(f"Tensor shape: {image_tensor.shape}")

    print("4. Running inference...")
    response = model.run(image_tensor)
    print("✓ Inference completed")
    print(f"Output shape: {response.shape}")

    print("5. Converting to torch...")
    response = ttnn.to_torch(response)
    print(f"Torch output shape: {response.shape}")

    print("6. Running postprocess...")
    from models.demos.utils.common_demo_utils import postprocess_pose

    results = postprocess_pose(response, image_array.shape)
    print("✓ Postprocessing completed")
    print(f"Results: {results}")

    print("🎉 SUCCESS: TTNN pose model works!")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback

    traceback.print_exc()

finally:
    try:
        ttnn.close_device(device)
        print("✓ Device closed")
    except:
        pass
