# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import base64
import gzip
import logging
import os
import time
from io import BytesIO

import numpy as np
import torch
from fastapi import FastAPI, File, Query, UploadFile
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


def compress_mask_binary(mask_data, threshold=0.5):
    """Convert mask to binary format and compress"""
    # Ensure mask_data is a numpy array
    mask_data = np.array(mask_data)

    # Ensure mask is 2D by squeezing extra dimensions
    if mask_data.ndim > 2:
        mask_data = mask_data.squeeze()
    elif mask_data.ndim == 1:
        # If 1D, try to reshape to square (this is a fallback)
        size = int(np.sqrt(mask_data.size))
        if size * size == mask_data.size:
            mask_data = mask_data.reshape(size, size)
        else:
            raise ValueError(f"Cannot reshape 1D mask of size {mask_data.size} to 2D")

    # Convert to binary mask
    binary_mask = (mask_data > threshold).astype(np.uint8)

    # Pack binary data (8 pixels per byte)
    height, width = binary_mask.shape
    packed_data = np.packbits(binary_mask.flatten())

    # Compress with gzip
    compressed_data = gzip.compress(packed_data.tobytes())

    # Encode as base64 for JSON transmission
    encoded_data = base64.b64encode(compressed_data).decode("utf-8")

    return {"data": encoded_data, "shape": [height, width], "format": "binary_compressed"}


def compress_mask_rle(mask_data, threshold=0.5):
    """Compress mask using Run-Length Encoding"""
    # Ensure mask_data is a numpy array
    mask_data = np.array(mask_data)

    # Ensure mask is 2D by squeezing extra dimensions
    if mask_data.ndim > 2:
        mask_data = mask_data.squeeze()
    elif mask_data.ndim == 1:
        # If 1D, try to reshape to square (this is a fallback)
        size = int(np.sqrt(mask_data.size))
        if size * size == mask_data.size:
            mask_data = mask_data.reshape(size, size)
        else:
            raise ValueError(f"Cannot reshape 1D mask of size {mask_data.size} to 2D")

    # Convert to binary mask
    binary_mask = (mask_data > threshold).astype(np.uint8)

    # Flatten the mask
    flat_mask = binary_mask.flatten()

    # Run-length encoding
    rle = []
    count = 1
    current_val = flat_mask[0]

    for val in flat_mask[1:]:
        if val == current_val:
            count += 1
        else:
            rle.append(count)
            count = 1
            current_val = val

    rle.append(count)  # Add the last run

    return {"data": rle, "shape": list(binary_mask.shape), "format": "rle"}


def process_segmentation_output(output, image_shape, compression="binary"):
    """Process segmentation output to extract masks and bounding boxes with compression options"""
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

                # Debug: Log mask information
                logging.info(f"Mask {i}: type={type(mask_data)}, shape={mask_data.shape}, dtype={mask_data.dtype}")

                # Apply compression based on method
                if compression == "binary":
                    compressed_mask = compress_mask_binary(mask_data)
                elif compression == "rle":
                    compressed_mask = compress_mask_rle(mask_data)
                elif compression == "none":
                    # Original format (for backward compatibility)
                    compressed_mask = {"data": mask_data.tolist(), "format": "raw"}
                else:
                    # Default to binary compression
                    compressed_mask = compress_mask_binary(mask_data)

                masks.append(compressed_mask)

        return {"masks": masks, "compression": compression, "downsample_factor": 1}  # Fixed at 1 for best performance

    return {"masks": [], "compression": compression, "downsample_factor": 1}  # Fixed at 1 for best performance


@app.post("/segmentation")
async def segmentation(
    file: UploadFile = File(...),
    compression: str = Query("binary", description="Compression method: 'binary', 'rle', or 'none'"),
):
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

    image = torch.permute(image, (0, 3, 1, 2))
    t1 = time.time()
    response = model.run(image)
    t2 = time.time()
    logging.info("The inference on the server side took: %.3f seconds", t2 - t1)

    # Process segmentation output with compression (no downsampling)
    result = process_segmentation_output(response, image1.shape, compression)
    t3 = time.time()
    logging.info("The post-processing took: %.3f seconds", t3 - t2)

    # Log compression statistics
    if result["masks"]:
        original_size = sum(len(str(mask.get("data", []))) for mask in result["masks"])
        logging.info(f"Compression: {compression}, Masks: {len(result['masks'])}")

    return result
