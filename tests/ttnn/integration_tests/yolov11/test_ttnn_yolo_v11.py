# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import sys
import torch.nn as nn
import torch
import ttnn
import logging
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.yolov11.tt.model_preprocessing import (
    create_yolov11_input_tensors,
    create_yolov11_model_parameters,
)
from models.utility_functions import skip_for_grayskull
from models.experimental.yolov11.reference import yolov11
from models.experimental.yolov11.tt import ttnn_yolov11

logger = logging.getLogger(__name__)

try:
    sys.modules["ultralytics"] = yolov11
    sys.modules["ultralytics.nn"] = yolov11
    sys.modules["ultralytics.nn.tasks"] = yolov11
    sys.modules["ultralytics.nn.modules.conv"] = yolov11
    sys.modules["ultralytics.nn.modules.block"] = yolov11
    sys.modules["ultralytics.nn.modules.head"] = yolov11
except KeyError:
    logger.error("models.experimental.yolov11.reference.yolov11 not found.")


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
    from pathlib import Path
    import os

    tests = Path(__file__).parent.parent / "yolov11"
    file_path = tests / Path(str(file).strip().replace("'", "").lower())

    if not file_path.exists():
        name = "yolo11n.pt"
        msg = f"{file_path} missing, try downloading from https://github.com/{repo}/releases/"

        try:
            url = f"https://github.com/{repo}/releases/download/v8.3.0/{name}"

            logger.info(f"Downloading {url} to {file_path}...")
            torch.hub.download_url_to_file(url, file_path)

            assert file_path.exists() and file_path.stat().st_size > 1e6, f"Download failed for {name}"

        except Exception as e:
            logger.error(f"Error downloading from GitHub: {e}. Trying secondary source...")

            url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
            logger.info(f"Downloading {url} to {file_path}...")
            os.system(f"curl -L {url} -o {file_path}")

            if not file_path.exists() or file_path.stat().st_size < 1e6:
                file_path.unlink(missing_ok=True)
                logger.error(f"ERROR: Download failure for {msg}")
            else:
                logger.info(f"Download succeeded from secondary source!")
    return file_path


def attempt_load(weights, map_location=None):
    model = Ensemble()

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


@skip_for_grayskull()
@pytest.mark.parametrize(
    "resolution",
    [
        ([1, 3, 224, 224]),
        ([1, 3, 640, 640]),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov11(device, use_program_cache, reset_seeds, resolution):
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device, batch=resolution[0], input_channels=resolution[1], input_height=resolution[2], input_width=resolution[3]
    )

    torch_model = yolov11.YoloV11()
    torch_model.eval()

    torch_output = torch_model(torch_input)
    parameters = create_yolov11_model_parameters(torch_model, torch_input, device=device)
    ttnn_model = ttnn_yolov11.YoloV11(device, parameters)
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)
