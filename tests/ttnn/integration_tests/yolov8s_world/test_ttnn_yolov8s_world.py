# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import ttnn
import torch
import pytest
from pathlib import Path
import torch.nn as nn
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import disable_persistent_kernel_cache, skip_for_grayskull
from models.experimental.functional_yolov8s_world.reference import yolov8s_world_utils
from models.experimental.functional_yolov8s_world.tt.ttnn_yolov8s_world import ttnn_Conv, ttnn_C2f, ttnn_SPPF
from models.experimental.functional_yolov8s_world.tt.ttnn_yolov8s_world_utils import (
    # ttnn_decode_bboxes,
    create_custom_preprocessor,
)
from models.experimental.functional_yolov8s_world.reference.yolov8s_world import YOLOWorld

from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)

try:
    sys.modules["ultralytics"] = yolov8s_world_utils
    sys.modules["ultralytics.nn.tasks"] = yolov8s_world_utils
    sys.modules["ultralytics.nn.modules.conv"] = yolov8s_world_utils
    sys.modules["ultralytics.nn.modules.block"] = yolov8s_world_utils
    sys.modules["ultralytics.nn.modules.head"] = yolov8s_world_utils

except KeyError:
    print("models.experimental.functional_yolov8s_world.reference.yolov8s_world_utils not found.")


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
    tests = Path(__file__).parent.parent / "yolov8s_world"
    file_path = tests / Path(str(file).strip().replace("'", "").lower())

    if not file_path.exists():
        name = "yolov8s-world.pt"
        msg = f"{file_path} missing, try downloading from https://github.com/{repo}/releases/"
        try:
            url = f"https://github.com/{repo}/releases/download/v8.3.0/{name}"
            print(f"Downloading {url} to {file_path}...")
            torch.hub.download_url_to_file(url, file_path)

            assert file_path.exists() and file_path.stat().st_size > 1e6, f"Download failed for {name}"
        except Exception as e:
            print(f"Error downloading from GitHub: {e}. Trying secondary source...")

            url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
            print(f"Downloading {url} to {file_path}...")
            os.system(f"curl -L {url} -o {file_path}")

            if not file_path.exists() or file_path.stat().st_size < 1e6:
                file_path.unlink(missing_ok=True)
                print(f"ERROR: Download failure for {msg}")
            else:
                print(f"Download succeeded from secondary source!")
    return file_path


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        weight_path = attempt_download(w)
        print("Loading weights from:", weight_path)
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [(torch.rand((1, 3, 640, 640)))], ids=["input_tensor1"])
@skip_for_grayskull()
def test_Conv(device, input_tensor):
    disable_persistent_kernel_cache()

    torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
    torch_model.eval()

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))

    state_dict = torch_model.state_dict()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device)
    )

    with torch.inference_mode():
        conv_0 = ttnn_Conv(
            device,
            parameters["model"][0],
            input_params=[3, 2, 1, 32, 3],
            change_shard=False,
            deallocate_activation=True,
            act_block_h=True,
            is_dfl=True,
        )
        conv_0, out_h, out_w = conv_0(ttnn_input)
        ttnn_model_output = ttnn.to_torch(conv_0)
        ttnn_model_output = ttnn_model_output.reshape((1, out_h, out_w, ttnn_model_output.shape[-1]))
        ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))

    submodule = torch_model.get_submodule("model.0")

    with torch.inference_mode():
        torch_model_output = submodule(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [(torch.rand((1, 64, 160, 160)))], ids=["input_tensor1"])
@skip_for_grayskull()
def test_C2f(device, input_tensor, reset_seeds):
    disable_persistent_kernel_cache()

    torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
    torch_model.eval()

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))

    ttnn_input = ttnn.from_device(ttnn_input)

    state_dict = torch_model.state_dict()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device)
    )

    c2f_configs = {
        "model.2": {"input_params": ((1, 1, 0, 64, 64), (1, 1, 0, 64, 96), (3, 1, 1, 32, 32))},
    }

    with torch.inference_mode():
        c2f_2 = ttnn_C2f(
            device,
            parameters["model"][2],
            n=1,
            shortcut=True,
            change_shard=False,
            input_params=c2f_configs["model.2"]["input_params"],
        )
        c2f_2, out_h, out_w = c2f_2(ttnn_input)
        ttnn_model_output = ttnn.to_torch(c2f_2)
        ttnn_model_output = ttnn_model_output.reshape((1, out_h, out_w, ttnn_model_output.shape[-1]))
        ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))

    submodule = torch_model.get_submodule("model.2")

    with torch.inference_mode():
        torch_model_output = submodule(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [(torch.rand((1, 512, 20, 20)))], ids=["input_tensor1"])
@skip_for_grayskull()
def test_SPPF(device, input_tensor, reset_seeds):
    disable_persistent_kernel_cache()

    torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
    torch_model.eval()

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))

    ttnn_input = ttnn.from_device(ttnn_input)

    state_dict = torch_model.state_dict()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device)
    )

    sppf_configs = {"input_params": ((1, 1, 0, 256, 512), (1, 1, 0, 512, 1024))}

    with torch.inference_mode():
        sppf = ttnn_SPPF(device, parameters["model"][9], input_params=sppf_configs["input_params"], batch_size=1)
        ttnn_model_output, out_h, out_w = sppf(ttnn_input)
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)
        ttnn_model_output = ttnn_model_output.reshape((1, out_h, out_w, ttnn_model_output.shape[-1]))
        ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))

    submodule = torch_model.get_submodule("model.9")

    with torch.inference_mode():
        torch_model_output = submodule(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")
