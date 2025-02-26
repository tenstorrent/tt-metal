# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import torch
import pytest
from pathlib import Path
import torch.nn as nn
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.functional_yolov8x.reference import yolov8x_utils
from models.experimental.functional_yolov8x.reference import yolov8x

try:
    sys.modules["ultralytics"] = yolov8x_utils
    sys.modules["ultralytics.nn.tasks"] = yolov8x_utils
    sys.modules["ultralytics.nn.modules.conv"] = yolov8x_utils
    sys.modules["ultralytics.nn.modules.block"] = yolov8x_utils
    sys.modules["ultralytics.nn.modules.head"] = yolov8x_utils

except KeyError:
    logger.info("models.experimental.functional_yolov8x.reference.yolov8x_utils not found.")


# For testing reference
def decode_bboxes(distance, anchor_points, xywh=True, dim=1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset

        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)


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
        name = "yolov8x.pt"
        msg = f"{file_path} missing, try downloading from https://github.com/{repo}/releases/"
        try:
            url = f"https://github.com/{repo}/releases/download/v8.3.0/{name}"
            logger.info(f"Downloading {url} to {file_path}...")
            torch.hub.download_url_to_file(url, file_path)

            assert file_path.exists() and file_path.stat().st_size > 1e6, f"Download failed for {name}"
        except Exception as e:
            logger.info(f"Error downloading from GitHub: {e}. Trying secondary source...")

            url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
            logger.info(f"Downloading {url} to {file_path}...")
            os.system(f"curl -L {url} -o {file_path}")

            if not file_path.exists() or file_path.stat().st_size < 1e6:
                file_path.unlink(missing_ok=True)
                logger.info(f"ERROR: Download failure for {msg}")
            else:
                logger.info(f"Download succeeded from secondary source!")
    return file_path


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        weight_path = attempt_download(w)
        logger.info("Loading weights from:", weight_path)
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


def run_submodule(x, submodule):
    y = []
    for m in submodule:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x)
    return x


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor",
    [torch.rand((1, 3, 640, 640))],
    ids=["input_tensor1"],
)
def test_yolov8x_reference(device, input_tensor):
    disable_persistent_kernel_cache()

    torch_model = attempt_load("yolov8x.pt", map_location="cpu")
    state_dict = torch_model.state_dict()
    torch_model_reference = yolov8x.DetectionModel()

    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in torch_model_reference.state_dict():
            filtered_state_dict[k] = v

    torch_model_reference.load_state_dict(filtered_state_dict, strict=False)
    torch_model_reference.eval()

    with torch.inference_mode():
        torch_model_output = torch_model(input_tensor)[0]
        torch_model_reference_output = torch_model_reference(input_tensor)[0]

    passing, pcc = assert_with_pcc(torch_model_output, torch_model_reference_output, 1)
    logger.info(f"Passing: {passing}, PCC: {pcc}")
