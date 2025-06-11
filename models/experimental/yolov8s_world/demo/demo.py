# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import torch
import pytest
import cv2
from loguru import logger
from datetime import datetime

import ttnn
from models.experimental.yolov8s_world.reference import yolov8s_world
from models.experimental.yolov8s_world.tt.ttnn_yolov8s_world import TtYOLOWorld
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.yolov8s_world.tt.ttnn_yolov8s_world_utils import (
    create_custom_preprocessor,
    attempt_load,
    move_to_device,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.yolov8s_world.demo.demo_utils import LoadImages, preprocess, postprocess, load_coco_class_names


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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "source",
    [
        ("models/experimental/yolov8s_world/demo/images/bus.jpg"),
        # ("models/experimental/yolov8s_world/demo/images/elephants.jpg"), # Uncomment to run the demo with another image for the second example.
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
        if use_pretrained_weight:
            weights_torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
            torch_model = yolov8s_world.YOLOWorld(model_torch=weights_torch_model)

            state_dict = weights_torch_model.state_dict()
            ds_state_dict = {k: v for k, v in state_dict.items()}
            new_state_dict = {}
            for (name1, parameter1), (name2, parameter2) in zip(
                torch_model.state_dict().items(), ds_state_dict.items()
            ):
                new_state_dict[name1] = parameter2

            torch_model.load_state_dict(new_state_dict)
            torch_model = torch_model.model
        else:
            torch_model = yolov8s_world.YOLOWorld()
            state_dict = torch_model.state_dict()
            torch_model = torch_model.model
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device)
        )

        for i in [12, 15, 19, 22]:
            parameters["model"][i]["attn"]["gl"]["weight"] = ttnn.to_device(
                parameters["model"][i]["attn"]["gl"]["weight"], device=device
            )
            parameters["model"][i]["attn"]["gl"]["bias"] = ttnn.to_device(
                parameters["model"][i]["attn"]["gl"]["bias"], device=device
            )
            parameters["model"][i]["attn"]["bias"] = ttnn.to_device(
                parameters["model"][i]["attn"]["bias"], device=device
            )

        parameters["model"][16] = move_to_device(parameters["model"][16], device)

        parameters["model"][23]["cv4"] = move_to_device(parameters["model"][23]["cv4"], device)
        model = TtYOLOWorld(device=device, parameters=parameters)
        logger.info("Inferencing using ttnn Model")

    save_dir = "models/experimental/yolov8s_world/demo/runs"

    dataset = LoadImages(path=source)

    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)

    names = load_coco_class_names()

    for batch in dataset:
        paths, im0s, s = batch

        im = preprocess(im0s, res=res)

        ttnn_im = im.permute((0, 2, 3, 1))
        ttnn_im = ttnn.from_torch(ttnn_im, dtype=ttnn.bfloat16)

        if model_type == "torch_model":
            preds = model(im)
        else:
            preds = model(x=ttnn_im)
            preds[0] = ttnn.to_torch(preds[0], dtype=torch.float32)

        results = postprocess(preds, im, im0s, batch, names)[0]

        save_yolo_predictions_by_model(results, save_dir, source, model_type)

    logger.info("Inference done")
