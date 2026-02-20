# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""FastAPI server for YuNet face detection inference on Tenstorrent hardware."""

import logging
import time
from io import BytesIO

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

import ttnn
from models.experimental.yunet.common import (
    YUNET_L1_SMALL_SIZE,
    STRIDES,
    DEFAULT_NMS_IOU_THRESHOLD,
    load_torch_model,
    get_default_weights_path,
)
from models.experimental.yunet.tt.ttnn_yunet import create_yunet_model

app = FastAPI(
    title="YuNet Face Detection",
    description="Inference engine to detect faces in images using Tenstorrent hardware.",
    version="1.0",
)

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Global variables
model = None
device = None
INPUT_SIZE = 640  # Default input size


@app.get("/")
async def root():
    return {"message": "YuNet Face Detection API - Powered by Tenstorrent"}


@app.on_event("startup")
async def startup():
    global model, device
    logging.info("Initializing Tenstorrent device...")
    device = ttnn.open_device(device_id=0, l1_small_size=YUNET_L1_SMALL_SIZE)

    logging.info("Loading YuNet model...")
    weights_path = get_default_weights_path()
    torch_model = load_torch_model(weights_path)
    torch_model = torch_model.to(torch.bfloat16)
    model = create_yunet_model(device, torch_model)

    logging.info("YuNet model loaded successfully!")


@app.on_event("shutdown")
async def shutdown():
    global device
    if device is not None:
        ttnn.close_device(device)
        logging.info("Device closed.")


def decode_detections(cls_outs, box_outs, obj_outs, kpt_outs, orig_w, orig_h, input_size, threshold):
    """Decode raw model outputs to detections."""
    detections = []

    for scale_idx in range(3):
        cls_out = ttnn.to_torch(cls_outs[scale_idx]).float().permute(0, 3, 1, 2)
        box_out = ttnn.to_torch(box_outs[scale_idx]).float().permute(0, 3, 1, 2)
        obj_out = ttnn.to_torch(obj_outs[scale_idx]).float().permute(0, 3, 1, 2)
        kpt_out = ttnn.to_torch(kpt_outs[scale_idx]).float().permute(0, 3, 1, 2)

        stride = STRIDES[scale_idx]
        score = cls_out.sigmoid() * obj_out.sigmoid()

        high_conf = score > threshold
        if high_conf.any():
            indices = torch.where(high_conf)
            for i in range(len(indices[0])):
                b, c, h, w = indices[0][i], indices[1][i], indices[2][i], indices[3][i]
                conf = score[b, c, h, w].item()
                anchor_x, anchor_y = w.item() * stride, h.item() * stride

                dx, dy = box_out[b, 0, h, w].item(), box_out[b, 1, h, w].item()
                dw, dh = box_out[b, 2, h, w].item(), box_out[b, 3, h, w].item()

                cx, cy = dx * stride + anchor_x, dy * stride + anchor_y
                bw, bh = np.exp(dw) * stride, np.exp(dh) * stride

                # Normalize coordinates to 0-1 range
                x1 = (cx - bw / 2) / input_size
                y1 = (cy - bh / 2) / input_size
                x2 = (cx + bw / 2) / input_size
                y2 = (cy + bh / 2) / input_size

                keypoints = []
                for k in range(5):
                    kpt_dx = kpt_out[b, k * 2, h, w].item()
                    kpt_dy = kpt_out[b, k * 2 + 1, h, w].item()
                    kx = (kpt_dx * stride + anchor_x) / input_size
                    ky = (kpt_dy * stride + anchor_y) / input_size
                    keypoints.append([kx, ky])

                detections.append({"box": [x1, y1, x2, y2], "conf": conf, "keypoints": keypoints})

    # NMS
    detections = sorted(detections, key=lambda x: x["conf"], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        remaining = []
        for det in detections:
            x1 = max(best["box"][0], det["box"][0])
            y1 = max(best["box"][1], det["box"][1])
            x2 = min(best["box"][2], det["box"][2])
            y2 = min(best["box"][3], det["box"][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (best["box"][2] - best["box"][0]) * (best["box"][3] - best["box"][1])
            area2 = (det["box"][2] - det["box"][0]) * (det["box"][3] - det["box"][1])
            if inter / max(area1 + area2 - inter, 1e-6) < DEFAULT_NMS_IOU_THRESHOLD:
                remaining.append(det)
        detections = remaining

    return keep


@app.post("/facedetection")
async def facedetection(file: UploadFile = File(...), input_size: int = 640, conf_thresh: float = 0.35):
    """Run face detection on uploaded image."""
    global model, device

    contents = await file.read()

    # Load and convert the image to RGB
    image = Image.open(BytesIO(contents)).convert("RGB")
    orig_w, orig_h = image.size

    # Resize to model input size
    image_resized = image.resize((input_size, input_size))
    image_np = np.array(image_resized)

    # Convert to tensor (NHWC, bfloat16)
    tensor = torch.from_numpy(image_np).float()
    tensor_nhwc = tensor.unsqueeze(0).to(torch.bfloat16)

    # Run inference
    t1 = time.time()
    tt_input = ttnn.from_torch(tensor_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    cls_out, box_out, obj_out, kpt_out = model(tt_input)
    ttnn.synchronize_device(device)
    t2 = time.time()

    # Decode detections (normalized coordinates 0-1)
    detections = decode_detections(
        cls_out, box_out, obj_out, kpt_out, orig_w, orig_h, input_size, threshold=conf_thresh
    )

    inference_time_ms = (t2 - t1) * 1000
    logging.info(f"Inference took: {inference_time_ms:.1f} ms, detected {len(detections)} faces")

    return {"detections": detections, "inference_time_ms": inference_time_ms, "num_faces": len(detections)}
