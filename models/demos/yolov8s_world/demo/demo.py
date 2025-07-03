# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime

import cv2
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.yolov8s_world.demo.demo_utils import LoadImages, load_coco_class_names, postprocess, preprocess
from models.demos.yolov8s_world.reference import yolov8s_world
from models.demos.yolov8s_world.runner.performant_runner import YOLOv8sWorldPerformantRunner
from models.demos.yolov8s_world.tt.ttnn_yolov8s_world_utils import attempt_load
from models.utility_functions import disable_persistent_kernel_cache


def save_yolo_predictions_by_model(result, save_dir, image_path, model_name):
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if model_name == "torch_model":
        bounding_box_color, label_color = (0, 255, 0), (0, 255, 0)
    else:
        bounding_box_color, label_color = (255, 0, 0), (255, 255, 0)

    boxes = result["boxes"]["xyxy"]
    scores = result["boxes"]["conf"]
    classes = result["boxes"]["cls"]
    names = result["names"]

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[int(cls)]} {score.item():.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), bounding_box_color, 3)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"prediction_{timestamp}.jpg"
    output_path = os.path.join(model_save_dir, output_name)

    cv2.imwrite(output_path, image)

    logger.info(f"Predictions saved to {output_path}")


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "source",
    [
        ("models/demos/yolov8s_world/demo/images/bus.jpg"),
        # ("models/demos/yolov8s_world/demo/images/elephants.jpg"), # Uncomment to run the demo with another image for the second example.
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        "tt_model",
        # "torch_model", # Uncomment to run the demo with torch model.
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [True],
    ids=[
        "pretrained_weight_true",
    ],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo(device, source, model_type, res, use_pretrained_weight):
    disable_persistent_kernel_cache()

    if model_type == "torch_model":
        if use_pretrained_weight:
            model = attempt_load("yolov8s-world.pt", map_location="cpu")
        else:
            model = yolov8s_world.YOLOWorld()

        logger.info("Inferencing using Torch Model")
    else:
        yolov8s_world_trace_2cq = YOLOv8sWorldPerformantRunner(
            device,
            1,  # batch_size
            ttnn.bfloat16,  # act_dtype
            ttnn.bfloat8_b,  # weight_dtype
            resolution=(640, 640),
            model_location_generator=None,
        )
        yolov8s_world_trace_2cq._capture_yolov8s_world_trace_2cqs()
        logger.info("Inferencing using ttnn Model")

    save_dir = "models/demos/yolov8s_world/demo/runs"

    dataset = LoadImages(path=source)

    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)

    names = load_coco_class_names()

    for batch in dataset:
        paths, im0s, s = batch

        im = preprocess(im0s, res=res)

        if model_type == "torch_model":
            preds = model(im)
        else:
            preds = yolov8s_world_trace_2cq.run(im)
            preds[0] = ttnn.to_torch(preds[0], dtype=torch.float32)

        results = postprocess(preds, im, im0s, batch, names)[0]

        save_yolo_predictions_by_model(results, save_dir, source, model_type)
    yolov8s_world_trace_2cq.release()
    logger.info("Inference done")
