# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime

import cv2
import pytest
import torch
from loguru import logger
from ultralytics import YOLO

import ttnn
from models.demos.yolov8x.demo.demo_utils import LoadImages, load_coco_class_names, postprocess, preprocess
from models.demos.yolov8x.runner.performant_runner import YOLOv8xPerformantRunner
from models.utility_functions import disable_persistent_kernel_cache


def save_yolo_predictions(result, save_dir, image_path, model_name):
    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    box_color = (0, 255, 0) if model_name == "torch_model" else (255, 0, 0)
    label_color = (0, 255, 0) if model_name == "torch_model" else (255, 255, 0)

    for box, score, cls in zip(result["boxes"]["xyxy"], result["boxes"]["conf"], result["boxes"]["cls"]):
        x1, y1, x2, y2 = map(int, box)
        label = f"{result['names'][int(cls)]} {score.item():.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(model_dir, f"prediction_{timestamp}.jpg")
    cv2.imwrite(output_path, image)

    logger.info(f"Predictions saved to {output_path}")


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "source, model_type",
    [
        ("models/demos/yolov8x/demo/images/bus.jpg", "torch_model"),
        ("models/demos/yolov8x/demo/images/bus.jpg", "tt_model"),
    ],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo(device, source, model_type, res):
    disable_persistent_kernel_cache()

    if model_type == "torch_model":
        model = YOLO("yolov8x.pt").model.eval()
    else:
        runner = YOLOv8xPerformantRunner(device, device_batch_size=1)

    dataset = LoadImages(path=source)
    class_names = load_coco_class_names()
    save_dir = "models/demos/yolov8x/demo/runs"

    for batch in dataset:
        _, im0s, _ = batch
        im = preprocess(im0s, res=res)

        if model_type == "torch_model":
            preds = model(im)
        else:
            preds = runner.run(im.clone())
            preds = ttnn.to_torch(preds, dtype=torch.float32)

        results = postprocess(preds, im, im0s, batch, class_names)[0]
        save_yolo_predictions(results, save_dir, source, model_type)

    if model_type == "tt_model":
        runner.release()

    logger.info(f"Inference completed for {model_type}")
