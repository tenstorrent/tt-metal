# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv9 model implementation

Apdapted from : https://github.com/WongKinYiu/yolov9
License : https://github.com/WongKinYiu/yolov9/blob/main/LICENSE.md
"""

import torch
import torch.nn as nn
from pathlib import Path
import os
import importlib
import types
import sys
import numpy as np
import cv2
import math

from tests.ttnn.unit_tests.operations.silu.yolo import Detect, Model

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


class Ensemble(nn.ModuleList):
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 1)
        return y, None


def attempt_load(weights, device=None, inplace=True, fuse=True):

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:

        # Weights were saved with 'models.yolo' paths from original repo structure.
        # Create fake module mappings in sys.modules to redirect to our unified yolo.py during torch.load().
        src_dir = str(Path(__file__).resolve().parent)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        unified = importlib.import_module("yolo")

        # Create parent packages
        models_pkg = sys.modules.get("models")
        if models_pkg is None:
            models_pkg = types.ModuleType("models")
            sys.modules["models"] = models_pkg

        utils_pkg = sys.modules.get("utils")
        if utils_pkg is None:
            utils_pkg = types.ModuleType("utils")
            sys.modules["utils"] = utils_pkg

        # Map expected submodules to the unified implementation
        for modname in ("models.common", "models.yolo", "models.experimental"):
            sys.modules[modname] = unified
        for modname in (
            "utils.torch_utils",
            "utils.activations",
            "utils.autoanchor",
            "utils.general",
            "utils.loss",
        ):
            sys.modules[modname] = unified

        ckpt = torch.load(w, map_location="cpu")
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()

        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))

        model.append(
            ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval()
        )

    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None

    if len(model) == 1:
        return model[-1]

    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[
        torch.argmax(torch.tensor([m.stride.max() for m in model])).int()
    ].stride  # max stride
    assert all(
        model[0].nc == m.nc for m in model
    ), f"Models have different class counts: {[m.nc for m in model]}"
    return model


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):

    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, ratio, (dw, dh)


def check_img_size(img_size, s=32):
    new_size = math.ceil(img_size / s) * s
    if new_size != img_size:
        print(
            f"WARNING: image size {img_size} must be multiple of max stride {s}, updating to {new_size}"
        )
    return new_size
