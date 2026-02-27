# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Faster-RCNN object detection demo using TTNN APIs.

Runs Faster-RCNN with ResNet-50-FPN backbone on Tenstorrent hardware.
The backbone, FPN, and RPN convolutions run on TT device while
post-processing (NMS, ROI Align) runs on CPU.

Usage:
    pytest models/demos/vision/detection/faster_rcnn/demo/demo.py::test_faster_rcnn_demo_sample -sv
"""

import os
import time

import cv2
import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image
from torchvision import transforms

import ttnn
from models.common.utility_functions import profiler
from models.demos.vision.detection.faster_rcnn.common import (
    COCO_INSTANCE_CATEGORY_NAMES,
    FASTER_RCNN_BATCH_SIZE,
    FASTER_RCNN_INPUT_HEIGHT,
    FASTER_RCNN_INPUT_WIDTH,
    FASTER_RCNN_L1_SMALL_SIZE,
    load_torch_faster_rcnn,
)
from models.demos.vision.detection.faster_rcnn.tt.model_preprocessing import (
    create_faster_rcnn_model_parameters,
)
from models.demos.vision.detection.faster_rcnn.tt.ttnn_faster_rcnn import TtFasterRCNN


def load_and_preprocess_image(image_path, target_size=(320, 320)):
    """Load an image and preprocess it for Faster-RCNN inference."""
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ]
    )
    tensor = transform(img).unsqueeze(0)
    return tensor, img


def draw_detections(image, detections, class_names, score_threshold=0.5):
    """Draw bounding boxes and labels on an image."""
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    boxes = detections["boxes"].cpu().numpy()
    labels = detections["labels"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()

    count = 0
    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue
        count += 1
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[label] if label < len(class_names) else f"class_{label}"
        label_text = f"{class_name}: {score:.2f}"

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_bgr, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    logger.info(f"Drew {count} detections with score >= {score_threshold}")
    return img_bgr


def run_faster_rcnn_inference(
    device,
    batch_size=1,
    input_height=320,
    input_width=320,
    input_path=None,
    score_threshold=0.5,
    save_output=True,
):
    """Run Faster-RCNN inference on sample images."""
    profiler.clear()

    logger.info("Loading pretrained Faster-RCNN model...")
    profiler.start("load_model")
    torch_model = load_torch_faster_rcnn(pretrained=True)
    profiler.end("load_model")

    logger.info("Preprocessing model weights for TTNN...")
    profiler.start("preprocess_weights")
    parameters = create_faster_rcnn_model_parameters(torch_model, device=device)
    profiler.end("preprocess_weights")

    logger.info("Creating TTNN Faster-RCNN model...")
    profiler.start("create_model")
    ttnn_model = TtFasterRCNN(
        parameters,
        device,
        torch_model,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
    )
    profiler.end("create_model")

    if input_path is None:
        input_path = "models/demos/vision/detection/faster_rcnn/demo/images/"

    if os.path.isdir(input_path):
        image_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
    elif os.path.isfile(input_path):
        image_files = [input_path]
    else:
        logger.info(f"No images found at {input_path}, using random input tensor")
        image_files = []

    if not image_files:
        logger.info("Running with random input tensor...")
        torch_input = torch.randn(batch_size, 3, input_height, input_width)

        profiler.start("warmup_inference")
        _ = ttnn_model(torch_input)
        ttnn.synchronize_device(device)
        profiler.end("warmup_inference")

        profiler.start("inference")
        detections = ttnn_model(torch_input)
        ttnn.synchronize_device(device)
        profiler.end("inference")

        for i, det in enumerate(detections):
            num_dets = len(det["boxes"])
            logger.info(f"Image {i}: {num_dets} detections")
            for j in range(min(num_dets, 10)):
                box = det["boxes"][j].tolist()
                label = det["labels"][j].item()
                score = det["scores"][j].item()
                class_name = (
                    COCO_INSTANCE_CATEGORY_NAMES[label]
                    if label < len(COCO_INSTANCE_CATEGORY_NAMES)
                    else f"class_{label}"
                )
                logger.info(f"  Detection {j}: {class_name} (score={score:.3f}), box={[f'{b:.1f}' for b in box]}")

        inference_time = profiler.get("inference")
        fps = batch_size / inference_time
        logger.info(f"Inference time: {inference_time * 1000:.1f} ms, FPS: {fps:.1f}")
        return detections

    all_detections = []
    for img_path in image_files:
        logger.info(f"Processing: {img_path}")
        torch_input, original_image = load_and_preprocess_image(img_path, (input_height, input_width))

        profiler.start(f"inference_{os.path.basename(img_path)}")
        detections = ttnn_model(torch_input)
        ttnn.synchronize_device(device)
        profiler.end(f"inference_{os.path.basename(img_path)}")

        det = detections[0]
        num_dets = len(det["boxes"])
        logger.info(f"  Found {num_dets} detections")

        for j in range(min(num_dets, 10)):
            label = det["labels"][j].item()
            score = det["scores"][j].item()
            class_name = (
                COCO_INSTANCE_CATEGORY_NAMES[label] if label < len(COCO_INSTANCE_CATEGORY_NAMES) else f"class_{label}"
            )
            if score >= score_threshold:
                logger.info(f"  {class_name}: {score:.3f}")

        if save_output:
            output_img = draw_detections(original_image, det, COCO_INSTANCE_CATEGORY_NAMES, score_threshold)
            output_path = img_path.replace(".", "_ttnn_output.")
            cv2.imwrite(output_path, output_img)
            logger.info(f"  Saved output to: {output_path}")

        all_detections.append(detections)

    return all_detections


def run_faster_rcnn_perf_test(device, batch_size=1, input_height=320, input_width=320, num_iterations=10):
    """Run Faster-RCNN performance benchmark."""
    torch_model = load_torch_faster_rcnn(pretrained=True)
    parameters = create_faster_rcnn_model_parameters(torch_model, device=device)

    ttnn_model = TtFasterRCNN(
        parameters,
        device,
        torch_model,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
    )

    torch_input = torch.randn(batch_size, 3, input_height, input_width)

    logger.info("Warming up (2 iterations)...")
    for _ in range(2):
        _ = ttnn_model(torch_input)
        ttnn.synchronize_device(device)

    logger.info(f"Running {num_iterations} iterations for benchmark...")
    start_time = time.time()
    for _ in range(num_iterations):
        _ = ttnn_model(torch_input)
        ttnn.synchronize_device(device)
    total_time = time.time() - start_time

    avg_time = total_time / num_iterations
    fps = batch_size / avg_time

    logger.info(f"Average inference time: {avg_time * 1000:.1f} ms")
    logger.info(f"Throughput: {fps:.1f} FPS")
    logger.info(f"Total time for {num_iterations} iterations: {total_time:.2f} s")

    return {"avg_inference_time_ms": avg_time * 1000, "fps": fps}


@pytest.mark.parametrize("device_params", [{"l1_small_size": FASTER_RCNN_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch_size", [FASTER_RCNN_BATCH_SIZE])
def test_faster_rcnn_demo_sample(device, batch_size):
    """Demo test: run Faster-RCNN on sample images or random input."""
    run_faster_rcnn_inference(
        device,
        batch_size=batch_size,
        input_height=FASTER_RCNN_INPUT_HEIGHT,
        input_width=FASTER_RCNN_INPUT_WIDTH,
        save_output=False,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": FASTER_RCNN_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch_size", [FASTER_RCNN_BATCH_SIZE])
@pytest.mark.parametrize("num_iterations", [10])
def test_faster_rcnn_perf(device, batch_size, num_iterations):
    """Performance test: benchmark Faster-RCNN throughput."""
    results = run_faster_rcnn_perf_test(
        device,
        batch_size=batch_size,
        input_height=FASTER_RCNN_INPUT_HEIGHT,
        input_width=FASTER_RCNN_INPUT_WIDTH,
        num_iterations=num_iterations,
    )
    assert results["fps"] >= 1.0, f"FPS too low: {results['fps']:.1f}"
