# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import cv2
import sys
import ttnn
import yaml
import torch
import pytest
import requests
import torch.nn as nn
from loguru import logger
from datetime import datetime
from functools import partial
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.functional_yolov9c.tt import ttnn_yolov9c
from models.experimental.functional_yolov9c.reference import yolov9c
from models.experimental.functional_yolov9c.tt.model_preprocessing import (
    create_yolov9c_input_tensors,
    create_yolov9c_model_parameters,
    create_yolov9c_model_parameters_detect,
)
from models.experimental.functional_yolov9c.demo.demo_utils import LoadImages, preprocess, postprocess


try:
    sys.modules["ultralytics"] = yolov9c
    sys.modules["ultralytics.nn.tasks"] = yolov9c
    sys.modules["ultralytics.nn.modules.conv"] = yolov9c
    sys.modules["ultralytics.nn.modules.block"] = yolov9c
    sys.modules["ultralytics.nn.modules.head"] = yolov9c

except KeyError:
    logger.error("models.experimental.functional_yolov9c.reference.yolov9c not found.")


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


def load_coco_class_names():
    url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    path = f"models/demos/yolov4/demo/coco.names"
    response = requests.get(url)
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.text.strip().split("\n")
    except requests.RequestException:
        pass
    if os.path.exists(path):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]

    raise Exception("Failed to fetch COCO class names from both online and local sources.")


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        w = "models/experimental/functional_yolov9c/reference/yolov9c.pt"
        ckpt = torch.load(w, map_location=map_location)
        model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().eval())
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None

    if len(model) == 1:
        return model[-1]
    else:
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model


def save_yolo_predictions_by_model(result, save_dir, image_path, model_name):
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if model_name == "torch_model":
        bounding_box_color, label_color = (0, 255, 0), (0, 255, 0)
    else:
        bounding_box_color, label_color = (255, 0, 0), (255, 0, 0)

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
        # "models/sample_data/huggingface_cat_image.jpg",
        "models/experimental/functional_yolov9c/demo/images/dog.jpg",
        # "models/experimental/functional_yolov9c/demo/images/cycle_girl.jpg",
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        "tt_model",
        # "torch_model",
    ],
)
def test_demo(device, source, model_type):
    disable_persistent_kernel_cache()

    if model_type == "torch_model":
        state_dict = None
        w = f"/models/experimental/functional_yolov9c/reference/yolov9c.pt"
        if os.path.exists(w):
            model = attempt_load(w, map_location="cpu")
            state_dict = model.state_dict()
        model = yolov9c.YoloV9()
        state_dict = torch_model.state_dict() if state_dict is None else state_dict

        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(model.state_dict().items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2
        model.load_state_dict(new_state_dict)
        model.eval()
        logger.info("Inferencing [Torch] Model")
    else:
        torch_input, ttnn_input = create_yolov9c_input_tensors(device)
        state_dict = None
        w = f"/models/experimental/functional_yolov9c/reference/yolov9c.pt"
        if os.path.exists(w):
            model = attempt_load(w, map_location="cpu")
            state_dict = model.state_dict()

        torch_model = yolov9c.YoloV9()
        state_dict = torch_model.state_dict() if state_dict is None else state_dict

        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2

        torch_model.load_state_dict(new_state_dict)
        torch_model.eval()
        parameters = create_yolov9c_model_parameters(torch_model, torch_input, device=device)
        model = ttnn_yolov9c.YoloV9(device, parameters)
        logger.info("Inferencing [TTNN] Model")

    save_dir = "models/experimental/functional_yolov9c/demo/runs"
    dataset = LoadImages(path=source)
    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)
    names = load_coco_class_names()

    for batch in dataset:
        paths, im0s, s = batch
        im = preprocess(im0s)
        if model_type == "torch_model":
            preds = model(im)
        else:
            img = torch.permute(im, (0, 2, 3, 1))
            img = img.reshape(
                1,
                1,
                img.shape[0] * img.shape[1] * img.shape[2],
                img.shape[3],
            )
            ttnn_im = ttnn.from_torch(img, dtype=ttnn.bfloat16)
            preds = model(ttnn_im)
            preds = ttnn.to_torch(preds, dtype=torch.float32)

        results = postprocess(preds, im, im0s, batch, names)[0]
        save_yolo_predictions_by_model(results, save_dir, source, model_type)

    logger.info("Inference done")
