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
from models.utility_functions import run_for_wormhole_b0
from models.experimental.functional_yolov8s_world.reference import yolov8s_world_utils
from models.experimental.functional_yolov8s_world.tt.ttnn_yolov8s_world import (
    ttnn_Conv,
    ttnn_C2f,
    ttnn_SPPF,
    ttnn_MaxSigmoidAttnBlock,
    ttnn_C2fAttn,
    ttnn_ContrastiveHead,
    ttnn_WorldModel,
    ttnn_WorldDetect,
    ttnn_YOLOWorld,
    ttnn_ImagePoolingAttn,
)
from models.experimental.functional_yolov8s_world.tt.ttnn_yolov8s_world_utils import (
    # ttnn_decode_bboxes,
    create_custom_preprocessor,
)
from ttnn.model_preprocessing import preprocess_model_parameters, ParameterDict, ParameterList

from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)

# Enable when using Real weights
# try:
#     sys.modules["ultralytics"] = yolov8s_world_utils
#     sys.modules["ultralytics.nn.tasks"] = yolov8s_world_utils
#     sys.modules["ultralytics.nn.modules.conv"] = yolov8s_world_utils
#     sys.modules["ultralytics.nn.modules.block"] = yolov8s_world_utils
#     sys.modules["ultralytics.nn.modules.head"] = yolov8s_world_utils

# except KeyError:
#     print("models.experimental.functional_yolov8s_world.reference.yolov8s_world_utils not found.")


def move_to_device(object, device):
    if isinstance(object, ParameterDict):
        for name, value in list(object.items()):
            if name in ["projections"]:
                continue
            object[name] = move_to_device(value, device)
        return object
    elif isinstance(object, ParameterList):
        for index, element in enumerate(object):
            object[index] = move_to_device(element, device)
        return object
    elif isinstance(object, ttnn.Tensor):
        return ttnn.to_device(object, device)
    else:
        return object


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
    tests = Path(__file__).parent.parent
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


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [(torch.rand((1, 3, 640, 640)))], ids=["input_tensor1"])
@run_for_wormhole_b0()
def test_Conv(device, input_tensor, use_pretrained_weight, reset_seeds):
    if use_pretrained_weight:
        torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
    else:
        torch_model = yolov8s_world_utils.YOLOWorld()
        torch_model = torch_model.model

    torch_model.eval()

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))

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


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [(torch.rand((1, 64, 160, 160)))], ids=["input_tensor1"])
@run_for_wormhole_b0()
def test_C2f(device, input_tensor, use_pretrained_weight, reset_seeds):
    if use_pretrained_weight:
        torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
    else:
        torch_model = yolov8s_world_utils.YOLOWorld()
        torch_model = torch_model.model
    torch_model.eval()

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))

    ttnn_input = ttnn.from_device(ttnn_input)

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


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_tensor", [(torch.rand((1, 512, 20, 20)))], ids=["input_tensor1"])
@run_for_wormhole_b0()
def test_SPPF(device, input_tensor, use_pretrained_weight, reset_seeds):
    if use_pretrained_weight:
        torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
    else:
        torch_model = yolov8s_world_utils.YOLOWorld()
        torch_model = torch_model.model
    torch_model.eval()

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))

    ttnn_input = ttnn.from_device(ttnn_input)

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


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@run_for_wormhole_b0()
def test_MaxSigmoidAttnBlock(device, use_pretrained_weight, reset_seeds):
    x = torch.randn(1, 128, 40, 40)
    guide = torch.randn(1, 80, 512)

    if use_pretrained_weight:
        torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
    else:
        torch_model = yolov8s_world_utils.YOLOWorld()
        torch_model = torch_model.model
    torch_model.eval()

    ttnn_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_guide = ttnn.from_torch(guide, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_x = ttnn.permute(ttnn_x, (0, 2, 3, 1))

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device)
    )
    parameters["model"][12]["attn"]["gl"]["weight"] = ttnn.to_device(
        parameters["model"][12]["attn"]["gl"]["weight"], device=device
    )
    parameters["model"][12]["attn"]["gl"]["bias"] = ttnn.to_device(
        parameters["model"][12]["attn"]["gl"]["bias"], device=device
    )
    parameters["model"][12]["attn"]["bias"] = ttnn.to_device(parameters["model"][12]["attn"]["bias"], device=device)

    maxsigmoisattnblockconfigs_configs = {"input_params": ((3, 1, 1, 128, 128))}

    with torch.inference_mode():
        multisigmoidattn = ttnn_MaxSigmoidAttnBlock(
            device,
            parameters["model"][12]["attn"],
            input_params=maxsigmoisattnblockconfigs_configs["input_params"],
            c1=128,
            c2=128,
            nh=4,
            ec=128,
            gc=512,
        )
        ttnn_model_output, out_h, out_w = multisigmoidattn(ttnn_x, ttnn_guide)
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)
        ttnn_model_output = ttnn_model_output.reshape((1, out_h, out_w, ttnn_model_output.shape[-1]))
        ttnn_model_output = ttnn_model_output.permute(0, 3, 1, 2)

    submodule = torch_model.get_submodule("model.12.attn")

    with torch.inference_mode():
        torch_model_output = submodule(x, guide)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@run_for_wormhole_b0()
def test_C2fAttn(device, use_pretrained_weight, reset_seeds):
    x = torch.randn(1, 768, 40, 40)
    guide = torch.randn(1, 80, 512)

    if use_pretrained_weight:
        torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
    else:
        torch_model = yolov8s_world_utils.YOLOWorld()
        torch_model = torch_model.model
    torch_model.eval()

    ttnn_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_guide = ttnn.from_torch(guide, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_x = ttnn.permute(ttnn_x, (0, 2, 3, 1))

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device)
    )
    parameters["model"][12]["attn"]["gl"]["weight"] = ttnn.to_device(
        parameters["model"][12]["attn"]["gl"]["weight"], device=device
    )
    parameters["model"][12]["attn"]["gl"]["bias"] = ttnn.to_device(
        parameters["model"][12]["attn"]["gl"]["bias"], device=device
    )
    parameters["model"][12]["attn"]["bias"] = ttnn.to_device(parameters["model"][12]["attn"]["bias"], device=device)

    c2fAttn_configs = {"input_params": ((1, 1, 0, 256, 768), (1, 1, 0, 256, 512), (3, 1, 1, 128, 128))}

    with torch.inference_mode():
        c2fAttn = ttnn_C2fAttn(
            device,
            parameters["model"][12],
            input_params=c2fAttn_configs["input_params"],
            c1=768,
            c2=256,
            n=1,
            nh=4,
            ec=128,
            gc=512,
            shortcut=False,
            g=1,
            e=0.5,
        )
        ttnn_model_output, out_h, out_w = c2fAttn(ttnn_x, ttnn_guide)
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)
        ttnn_model_output = ttnn_model_output.reshape((1, out_h, out_w, ttnn_model_output.shape[-1]))
        ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))

    submodule = torch_model.get_submodule("model.12")

    with torch.inference_mode():
        torch_model_output = submodule(x, guide)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@run_for_wormhole_b0()
def test_ImagePoolingAttn(device, use_pretrained_weight, reset_seeds):
    x = [torch.randn(1, 128, 80, 80), torch.randn(1, 256, 40, 40), torch.randn(1, 512, 20, 20)]
    text = torch.randn(1, 80, 512)

    if use_pretrained_weight:
        torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
    else:
        torch_model = yolov8s_world_utils.YOLOWorld()
        torch_model = torch_model.model
    torch_model.eval()

    ttnn_x_0 = ttnn.from_torch(x[0], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_text = ttnn.from_torch(text, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_x_0 = ttnn.permute(ttnn_x_0, (0, 2, 3, 1))

    ttnn_x_1 = ttnn.from_torch(x[1], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_x_1 = ttnn.permute(ttnn_x_1, (0, 2, 3, 1))

    ttnn_x_2 = ttnn.from_torch(x[2], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_x_2 = ttnn.permute(ttnn_x_2, (0, 2, 3, 1))

    ttnn_x = [ttnn_x_0, ttnn_x_1, ttnn_x_2]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device)
    )

    parameters["model"][16] = move_to_device(parameters["model"][16], device)

    ImagePoolingAttn_configs = {"input_params": ((1, 1, 0, 256, 128), (1, 1, 0, 256, 256), (1, 1, 0, 256, 512))}

    with torch.inference_mode():
        ImagePoolingAttn = ttnn_ImagePoolingAttn(
            device,
            parameters["model"][16],
            input_params=ImagePoolingAttn_configs["input_params"],
            ec=256,
            ch=[128, 256, 512],
            ct=512,
            nh=8,
            k=3,
            scale=False,
        )
        ttnn_model_output = ImagePoolingAttn(ttnn_x, ttnn_text)
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    submodule = torch_model.get_submodule("model.16")

    with torch.inference_mode():
        torch_model_output = submodule(x, text)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@run_for_wormhole_b0()
def test_ContrastiveHead(device, use_pretrained_weight, reset_seeds):
    x = torch.randn(1, 512, 80, 80)
    w = torch.randn(1, 80, 512)

    if use_pretrained_weight:
        torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
    else:
        torch_model = yolov8s_world_utils.YOLOWorld()
        torch_model = torch_model.model
    torch_model.eval()

    ttnn_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_w = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_x = ttnn.permute(ttnn_x, (0, 2, 3, 1))

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device)
    )
    parameters["model"][23]["cv4"] = move_to_device(parameters["model"][23]["cv4"], device)

    with torch.inference_mode():
        c2fAttn = ttnn_ContrastiveHead(
            device,
            parameters["model"][23]["cv4"][0],
        )
        ttnn_model_output = c2fAttn(ttnn_x, ttnn_w)
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)
        # ttnn_model_output = ttnn_model_output.reshape((1, out_h, out_w, ttnn_model_output.shape[-1]))
        ttnn_model_output = ttnn_model_output.permute((0, 3, 1, 2))

    submodule = torch_model.get_submodule("model.23.cv4.0")

    with torch.inference_mode():
        torch_model_output = submodule(x, w)

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@run_for_wormhole_b0()
def test_WorldDetect(device, use_pretrained_weight, reset_seeds):
    x = [torch.randn(1, 128, 80, 80), torch.randn(1, 256, 40, 40), torch.randn(1, 512, 20, 20)]
    text = torch.randn(1, 80, 512)

    if use_pretrained_weight:
        torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
    else:
        torch_model = yolov8s_world_utils.YOLOWorld()
        torch_model = torch_model.model
    torch_model.eval()

    ttnn_x_0 = ttnn.from_torch(x[0], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_text = ttnn.from_torch(text, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_x_0 = ttnn.permute(ttnn_x_0, (0, 2, 3, 1))

    ttnn_x_1 = ttnn.from_torch(x[1], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_x_1 = ttnn.permute(ttnn_x_1, (0, 2, 3, 1))

    ttnn_x_2 = ttnn.from_torch(x[2], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_x_2 = ttnn.permute(ttnn_x_2, (0, 2, 3, 1))

    ttnn_x = [ttnn_x_0, ttnn_x_1, ttnn_x_2]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device)
    )

    parameters["model"][23]["cv4"] = move_to_device(parameters["model"][23]["cv4"], device)
    world_detect_configs = {
        "cv2_params": [
            {
                "input_params": [
                    (3, 1, 1, 64, 128),
                    (3, 1, 1, 64, 64),
                    (1, 1, 0, 64, 64),
                ]
            },
            {
                "input_params": [
                    (3, 1, 1, 64, 256),
                    (3, 1, 1, 64, 64),
                    (1, 1, 0, 64, 64),
                ]
            },
            {
                "input_params": [
                    (3, 1, 1, 64, 512),
                    (3, 1, 1, 64, 64),
                    (1, 1, 0, 64, 64),
                ]
            },
        ],
        "cv3_params": [
            {
                "input_params": [
                    (3, 1, 1, 128, 128),
                    (3, 1, 1, 128, 128),
                    (1, 1, 0, 512, 128),
                ]
            },
            {
                "input_params": [
                    (3, 1, 1, 128, 256),
                    (3, 1, 1, 128, 128),
                    (1, 1, 0, 512, 128),
                ]
            },
            {
                "input_params": [
                    (3, 1, 1, 128, 512),
                    (3, 1, 1, 128, 128),
                    (1, 1, 0, 512, 128),
                ]
            },
        ],
        "dfl_params": {
            "input_params": (1, 1, 0, 1, 16),
        },
    }

    with torch.inference_mode():
        worldDetect = ttnn_WorldDetect(
            device,
            parameters["model"][23],
            input_params=world_detect_configs,
            nc=80,
            embed=512,
            with_bn=False,
            ch=[128, 256, 512],
        )
        ttnn_model_output_y, ttnn_model_output_x = worldDetect(ttnn_x, ttnn_text)
        ttnn_model_output_y = ttnn.to_torch(ttnn_model_output_y)
        for index, i in enumerate(ttnn_model_output_x):
            ttnn_model_output_x[index] = ttnn.to_torch(ttnn_model_output_x[index])
            ttnn_model_output_x[index] = ttnn_model_output_x[index].permute(0, 3, 1, 2)

    submodule = torch_model.get_submodule("model.23")

    with torch.inference_mode():
        torch_model_output = submodule(x, text)

    passing, pcc_1 = assert_with_pcc(ttnn_model_output_y, torch_model_output[0], 0.99)
    passing, pcc_2 = assert_with_pcc(ttnn_model_output_x[0], torch_model_output[1][0], 0.99)
    passing, pcc_3 = assert_with_pcc(ttnn_model_output_x[1], torch_model_output[1][1], 0.99)
    passing, pcc_4 = assert_with_pcc(ttnn_model_output_x[2], torch_model_output[1][2], 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc_1}")
    logger.info(f"Passing: {passing}, PCC: {pcc_2}")
    logger.info(f"Passing: {passing}, PCC: {pcc_3}")
    logger.info(f"Passing: {passing}, PCC: {pcc_4}")


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@run_for_wormhole_b0()
def test_WorldModel(device, use_pretrained_weight, reset_seeds):
    x = torch.randn(1, 3, 640, 640)

    if use_pretrained_weight:
        torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
    else:
        torch_model = yolov8s_world_utils.YOLOWorld()
        torch_model = torch_model.model
    torch_model.eval()

    ttnn_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_x = ttnn.permute(ttnn_x, (0, 2, 3, 1))

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
        parameters["model"][i]["attn"]["bias"] = ttnn.to_device(parameters["model"][i]["attn"]["bias"], device=device)

    parameters["model"][16] = move_to_device(parameters["model"][16], device)

    parameters["model"][23]["cv4"] = move_to_device(parameters["model"][23]["cv4"], device)

    with torch.inference_mode():
        world_model = ttnn_WorldModel(
            device,
            parameters,
        )
        ttnn_model_output_y, ttnn_model_output_x = world_model(ttnn_x)
        ttnn_model_output_y = ttnn.to_torch(ttnn_model_output_y)
        for index, i in enumerate(ttnn_model_output_x):
            ttnn_model_output_x[index] = ttnn.to_torch(ttnn_model_output_x[index])
            ttnn_model_output_x[index] = ttnn_model_output_x[index].permute(0, 3, 1, 2)

    with torch.inference_mode():
        torch_model_output = torch_model(x)

    passing, pcc_1 = assert_with_pcc(ttnn_model_output_y, torch_model_output[0], 0.99)
    passing, pcc_2 = assert_with_pcc(
        ttnn_model_output_x[0], torch_model_output[1][0], 0.98
    )  # 0.9818297046520124 for real weights
    passing, pcc_3 = assert_with_pcc(
        ttnn_model_output_x[1], torch_model_output[1][1], 0.97
    )  # 0.9730835624429178 for real weights
    passing, pcc_4 = assert_with_pcc(ttnn_model_output_x[2], torch_model_output[1][2], 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc_1}")
    logger.info(f"Passing: {passing}, PCC: {pcc_2}")
    logger.info(f"Passing: {passing}, PCC: {pcc_3}")
    logger.info(f"Passing: {passing}, PCC: {pcc_4}")


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@run_for_wormhole_b0()
def test_YoloModel(device, use_pretrained_weight, reset_seeds):
    x = torch.randn(1, 3, 640, 640)

    if use_pretrained_weight:
        torch_model = attempt_load("yolov8s-world.pt", map_location="cpu")
    else:
        torch_model = yolov8s_world_utils.YOLOWorld()
        torch_model = torch_model.model
    torch_model.eval()

    ttnn_x = x.permute(0, 2, 3, 1)
    ttnn_x = ttnn.from_torch(
        ttnn_x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

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
        parameters["model"][i]["attn"]["bias"] = ttnn.to_device(parameters["model"][i]["attn"]["bias"], device=device)

    parameters["model"][16] = move_to_device(parameters["model"][16], device)

    parameters["model"][23]["cv4"] = move_to_device(parameters["model"][23]["cv4"], device)

    with torch.inference_mode():
        world_model = ttnn_YOLOWorld(
            device,
            parameters,
        )
        ttnn_model_output_y, ttnn_model_output_x = world_model(ttnn_x)
        ttnn_model_output_y = ttnn.to_torch(ttnn_model_output_y)
        for index, i in enumerate(ttnn_model_output_x):
            ttnn_model_output_x[index] = ttnn.to_torch(ttnn_model_output_x[index])
            ttnn_model_output_x[index] = ttnn_model_output_x[index].permute(0, 3, 1, 2)

    with torch.inference_mode():
        torch_model_output = torch_model(x)

    passing, pcc_1 = assert_with_pcc(ttnn_model_output_y, torch_model_output[0], 0.99)
    passing, pcc_2 = assert_with_pcc(
        ttnn_model_output_x[0], torch_model_output[1][0], 0.98
    )  # 0.9818297046520124 for real weights
    passing, pcc_3 = assert_with_pcc(
        ttnn_model_output_x[1], torch_model_output[1][1], 0.97
    )  # 0.9730835624429178 for real weights
    passing, pcc_4 = assert_with_pcc(ttnn_model_output_x[2], torch_model_output[1][2], 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc_1}")
    logger.info(f"Passing: {passing}, PCC: {pcc_2}")
    logger.info(f"Passing: {passing}, PCC: {pcc_3}")
    logger.info(f"Passing: {passing}, PCC: {pcc_4}")
