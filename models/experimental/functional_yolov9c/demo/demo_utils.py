# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
import requests
from ultralytics import YOLO

from models.experimental.functional_yolov9c.reference.yolov9c import YoloV9


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


def load_torch_model(use_weights_from_ultralytics=True, module=None):
    state_dict = None
    if use_weights_from_ultralytics:
        model = YOLO("yolov9c.pt")
        model.load_state_dict(model.state_dict(), strict=False)

    model = YoloV9()
    new_state_dict = {name: param for name, param in model.state_dict().items() if isinstance(param, torch.FloatTensor)}

    model.load_state_dict(new_state_dict)
    model.eval()

    return model
