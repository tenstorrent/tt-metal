# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import cv2
import math
import time
import torch
import requests
import torchvision
import numpy as np
import torch.nn as nn
from pathlib import Path
from loguru import logger
from datetime import datetime
from functools import partial


def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


def attempt_download(file, repo="ultralytics/assets"):
    tests = Path(__file__).parent.parent / "reference"
    file_path = tests / Path(str(file).strip().replace("'", "").lower())

    if not file_path.exists():
        name = "yolov9c.pt"  # file_path.name
        msg = f"{file_path} missing, try downloading from https://github.com/{repo}/releases/"

        try:
            url = f"https://github.com/{repo}/releases/download/v8.3.0/{name}"

            logger.info(f"Downloading {url} to {file_path}...")
            torch.hub.download_url_to_file(url, file_path)

            # Validate the file
            assert file_path.exists() and file_path.stat().st_size > 1e6, f"Download failed for {name}"

        except Exception as e:
            logger.warning(f"Error downloading from GitHub: {e}. Trying secondary source...")

            url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
            logger.info(f"Downloading {url} to {file_path}...")
            os.system(f"curl -L {url} -o {file_path}")

            if not file_path.exists() or file_path.stat().st_size < 1e6:
                file_path.unlink(missing_ok=True)
                logger.error(f"ERROR: Download failure for {msg}")
            else:
                logger.info(f"Download succeeded from secondary source!")
    return file_path


# Function to load weights into the model
def attempt_load(weights, map_location=None):
    model = Ensemble()

    # Iterate through the weights and load them
    for w in weights if isinstance(weights, list) else [weights]:
        weight_path = attempt_download(w)
        ckpt = torch.load(weight_path, map_location=map_location)
        model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().eval())
    for m in model.modules():
        if isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
            m.inplace = True
        elif isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None

    if len(model) == 1:
        return model[-1]
    else:
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model


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
