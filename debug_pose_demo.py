#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, "/home/ubuntu/pose/tt-metal")

import torch
import ttnn
from models.demos.utils.common_demo_utils import LoadImages, preprocess
from models.demos.yolov11.demo.demo_pose_estimation import process_images
from models.demos.yolov11.reference.yolov11_pose_correct import YoloV11Pose
from models.demos.yolov11.tt.model_preprocessing_pose import create_yolov11_pose_model_parameters
from models.demos.yolov11.tt.ttnn_yolov11_pose_model import TtnnYoloV11Pose
from loguru import logger


def debug_model_inference():
    """Debug the model inference pipeline"""

    logger.info("Testing model inference...")

    input_loc = "models/demos/yolov11/demo/images"
    batch_size = 1
    res = (640, 640)

    try:
        # Load and preprocess image
        logger.info("Loading and preprocessing image...")
        dataset = LoadImages(path=os.path.abspath(input_loc), batch=batch_size)
        im_tensor, orig_images, paths_images = process_images(dataset, res, batch_size)
        logger.info(f"Image preprocessed: {im_tensor.shape}")

        # Test PyTorch model first
        logger.info("Testing PyTorch model...")
        torch_model = YoloV11Pose()
        weights_path = "models/demos/yolov11/reference/yolov11_pose_pretrained_correct.pth"
        if os.path.exists(weights_path):
            torch_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        else:
            logger.warning(f"Pretrained weights not found at {weights_path}")
        torch_model.eval()

        with torch.no_grad():
            torch_output = torch_model(im_tensor)
        logger.info(f"PyTorch output shape: {torch_output.shape}")
        logger.info(f"PyTorch output range: [{torch_output.min():.3f}, {torch_output.max():.3f}]")

        # Test TTNN model
        logger.info("Testing TTNN model...")
        device = ttnn.open_device(device_id=0)

        try:
            # Create TTNN model
            parameters = create_yolov11_pose_model_parameters(torch_model, im_tensor, device=device)
            ttnn_model = TtnnYoloV11Pose(device, parameters)

            # Convert input to TTNN format and move to device
            tt_input = ttnn.from_torch(im_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            tt_input = ttnn.to_device(tt_input, device)

            # Run inference
            logger.info("Running TTNN inference...")
            tt_output = ttnn_model(tt_input)

            # Convert back to torch
            output_tensor = ttnn.to_torch(tt_output)
            logger.info(f"TTNN output shape: {output_tensor.shape}")
            logger.info(f"TTNN output range: [{output_tensor.min():.3f}, {output_tensor.max():.3f}]")

            # Compare outputs
            logger.info("Comparing PyTorch vs TTNN outputs...")
            diff = torch.abs(torch_output - output_tensor)
            logger.info(f"Max difference: {diff.max():.6f}")
            logger.info(f"Mean difference: {diff.mean():.6f}")

        finally:
            ttnn.close_device(device)

    except Exception as e:
        logger.error(f"Error during model inference: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_model_inference()
