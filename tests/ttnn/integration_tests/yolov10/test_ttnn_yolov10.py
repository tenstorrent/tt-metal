# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import sys
import ttnn
import torch
import pickle
import pytest
import requests
from pathlib import Path
import torch.nn as nn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_yolov10.reference import yolov10_utils
from models.experimental.functional_yolov10.reference.yolov10 import YOLOv10
from models.experimental.functional_yolov10.tt.ttnn_yolov10x import (
    ttnn_BottleNeck,
    ttnn_SCDown,
    ttnn_SPPF,
    ttnn_CIB,
    ttnn_Attention,
    ttnn_PSA,
    ttnn_DFL,
    ttnn_C2f,
    ttnn_C2fCIB,
    ttnn_V10Detect,
    ttnn_Yolov10,
)
from models.experimental.functional_yolov10.tt.model_preprocessing import (
    create_yolov10x_input_tensors,
    create_yolov10x_model_parameters,
    create_yolov10_model_parameters_detect,
)

try:
    sys.modules["ultralytics"] = yolov10_utils
    sys.modules["ultralytics.nn.tasks"] = yolov10_utils
    sys.modules["ultralytics.nn.modules.conv"] = yolov10_utils
    sys.modules["ultralytics.nn.modules.block"] = yolov10_utils
    sys.modules["ultralytics.nn.modules.head"] = yolov10_utils
except KeyError:
    print("models.experimental.functional_yolov10x.reference.YoloV10x not found.")


def download_yolov10x_weights():
    url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt"
    weights_path = "tests/ttnn/integration_tests/yolov10/yolov10x.pt"

    response = requests.get(url)
    if response.status_code == 200:
        with open(weights_path, "wb") as f:
            f.write(response.content)
        print(f"YOLOv10x weights downloaded successfully and saved to {weights_path}")
    else:
        print("Failed to download YOLOv10x weights. Status code:", response.status_code)


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


def attempt_download(file, repo="ultralytics/assets/"):
    tests = Path(__file__).parent.parent / "tests"
    file = tests / Path(str(file).strip().replace("'", "").lower())
    file = Path(str(file).strip().replace("'", "").lower())
    if not file.exists():
        assets = [
            "yolov10x.pt",
        ]
        name = file.name
        if name in assets:
            msg = f"{file} missing, try downloading from https://github.com/{repo}/releases/"
            redundant = False
            try:
                url = f"https://github.com/{repo}/releases/download/v8.3.0/{name}"
                torch.hub.download_url_to_file(url, file)
                assert file.exists() and file.stat().st_size > 1e6
            except Exception as e:
                assert redundant, "No secondary mirror"
                url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
                os.system(f"curl -L {url} -o {file}")
            finally:
                if not file.exists() or file.stat().st_size < 1e6:
                    file.unlink(missing_ok=True)
                return


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        w = Path(__file__).parent.parent / "yolov10" / "yolov10x.pt"
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


@pytest.mark.parametrize(
    "index, fwd_input_shape , shortcut",
    [
        (
            2,
            (1, 80, 160, 160),
            True,
        ),
        (
            4,
            (1, 160, 80, 80),
            True,
        ),
        (
            16,
            (1, 160, 80, 80),
            False,
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov10x_bottleneck(device, use_program_cache, reset_seeds, index, fwd_input_shape, shortcut):
    torch_input, ttnn_input = create_yolov10x_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
    download_yolov10x_weights()
    torch_model = attempt_load("models/experimental/functional_yolov10/reference/", map_location="cpu")
    torch_model.eval()
    torch_model = torch_model.model[index].m[0]

    state_dict = torch_model.state_dict()
    reference_model = YOLOv10()
    reference_model = reference_model.model[index].m[0]

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    torch_model_output = reference_model(torch_input)[0]

    parameters = create_yolov10x_model_parameters(reference_model, torch_input, device=device)

    ttnn_module = ttnn_BottleNeck(shortcut=shortcut, device=device, parameters=parameters.conv_args, conv_pt=parameters)

    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_model_output.shape)

    assert_with_pcc(torch_model_output, ttnn_output, 0.999)


@pytest.mark.parametrize(
    "index, fwd_input_shape",
    [
        (5, (1, 320, 80, 80)),
        (7, (1, 640, 40, 40)),
        (20, (1, 640, 40, 40)),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov10x_SCDown(device, use_program_cache, reset_seeds, index, fwd_input_shape):
    torch_input, ttnn_input = create_yolov10x_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
    download_yolov10x_weights()
    torch_model = attempt_load("models/experimental/functional_yolov10/reference/", map_location="cpu")
    torch_model.eval()
    torch_model = torch_model.model[index]

    state_dict = torch_model.state_dict()
    reference_model = YOLOv10()
    reference_model.eval()
    reference_model = reference_model.model[index]

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    torch_model_output = reference_model(torch_input)[0]

    parameters = create_yolov10x_model_parameters(reference_model, torch_input, device=device)

    ttnn_module = ttnn_SCDown(
        device=device,
        parameters=parameters.conv_args,
        conv_pt=parameters,
        torch_conv=True,
    )
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output).squeeze(0)

    assert_with_pcc(torch_model_output, ttnn_output, 0.999)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov10x_SPPF(device, use_program_cache, reset_seeds):
    fwd_input_shape = [1, 640, 20, 20]
    torch_input, ttnn_input = create_yolov10x_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
    download_yolov10x_weights()
    torch_model = attempt_load("models/experimental/functional_yolov10/reference/", map_location="cpu")
    torch_model.eval()
    torch_model = torch_model.model[9]

    state_dict = torch_model.state_dict()
    reference_model = YOLOv10()
    reference_model.eval()
    reference_model = reference_model.model[9]

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    torch_model_output = reference_model(torch_input)[0]

    parameters = create_yolov10x_model_parameters(reference_model, torch_input, device=device)

    ttnn_module = ttnn_SPPF(
        device=device,
        parameters=parameters.conv_args,
        conv_pt=parameters,
    )
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_model_output.shape)

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)  # 0.9980494743661721


@pytest.mark.parametrize(
    "index, fwd_input_shape",
    [
        (6, (1, 320, 40, 40)),
        (8, (1, 320, 40, 40)),
        (13, (1, 320, 20, 20)),
        (19, (1, 320, 40, 40)),
        (22, (1, 320, 40, 40)),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov10x_CIB(device, use_program_cache, reset_seeds, index, fwd_input_shape):
    torch_input, ttnn_input = create_yolov10x_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
    download_yolov10x_weights()
    torch_model = attempt_load("models/experimental/functional_yolov10/reference/", map_location="cpu")
    torch_model.eval()
    torch_model = torch_model.model[index].m[0]

    state_dict = torch_model.state_dict()
    reference_model = YOLOv10()
    reference_model.eval()
    reference_model = reference_model.model[index].m[0]

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    torch_model_output = reference_model(torch_input)[0]

    parameters = create_yolov10x_model_parameters(reference_model, torch_input, device=device)

    ttnn_module = ttnn_CIB(
        device=device,
        parameters=parameters.conv_args,
        conv_pt=parameters,
        torch_conv=True,
    )
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_model_output.shape)

    assert_with_pcc(torch_model_output, ttnn_output, 0.999)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov10x_Attention(device, use_program_cache, reset_seeds):
    fwd_input_shape = [1, 320, 20, 20]
    torch_input, ttnn_input = create_yolov10x_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
        input_dtype=ttnn.bfloat16,
    )

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
    download_yolov10x_weights()
    torch_model = attempt_load("tests/ttnn/integration_tests/yolov10/yolov10x.pt", map_location="cpu")
    torch_model.eval()
    torch_model = torch_model.model[10].attn

    state_dict = torch_model.state_dict()
    reference_model = YOLOv10()
    reference_model.eval()
    reference_model = reference_model.model[10].attn

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    torch_model_output = reference_model(torch_input)[0]

    parameters = create_yolov10x_model_parameters(reference_model, torch_input, device=device)

    ttnn_module = ttnn_Attention(
        dim=320,
        num_heads=5,
        attn_ratio=0.5,
        device=device,
        parameters=parameters.conv_args,
        conv_pt=parameters,
    )
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_model_output.shape)

    assert_with_pcc(torch_model_output, ttnn_output, 0.999)  # 0.99948606391443


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov10x_PSA(device, use_program_cache, reset_seeds):
    fwd_input_shape = [1, 640, 20, 20]
    torch_input, ttnn_input = create_yolov10x_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
        input_dtype=ttnn.bfloat16,
    )

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
    download_yolov10x_weights()
    torch_model = attempt_load("models/experimental/functional_yolov10/reference/", map_location="cpu")
    torch_model.eval()
    torch_model = torch_model.model[10]

    state_dict = torch_model.state_dict()
    reference_model = YOLOv10()
    reference_model.eval()
    reference_model = reference_model.model[10]

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    torch_model_output = reference_model(torch_input)[0]

    parameters = create_yolov10x_model_parameters(reference_model, torch_input, device=device)

    ttnn_module = ttnn_PSA(
        device=device,
        parameters=parameters.conv_args,
        conv_pt=parameters,
    )
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = ttnn_output.permute(0, 3, 1, 2).squeeze(0)
    ttnn_output = ttnn_output.reshape(torch_model_output.shape)

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov10x_DFL(device, use_program_cache, reset_seeds):
    torch_input = torch.randn(1, 64, 8400)
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    download_yolov10x_weights()
    torch_model = attempt_load("models/experimental/functional_yolov10/reference/", map_location="cpu")
    torch_model.eval()
    torch_model = torch_model.model[23].dfl

    state_dict = torch_model.state_dict()
    reference_model = YOLOv10()
    reference_model.eval()
    reference_model = reference_model.model[23].dfl

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    torch_model_output = reference_model(torch_input)[0]

    parameters = create_yolov10x_model_parameters(reference_model, torch_input, device=device)

    ttnn_module = ttnn_DFL(
        device=device,
        parameters=parameters.conv_args,
        conv_pt=parameters,
    )
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output).squeeze(0)

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "index, fwd_input_shape, num_layers, shortcut",
    [
        (2, (1, 160, 160, 160), 3, True),
        (4, (1, 320, 80, 80), 6, True),
        (16, (1, 960, 80, 80), 3, False),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov10x_C2f(device, use_program_cache, reset_seeds, index, fwd_input_shape, num_layers, shortcut):
    torch_input, ttnn_input = create_yolov10x_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    # download_yolov10x_weights()
    # torch_model = attempt_load("models/experimental/functional_yolov10/reference/", map_location="cpu")
    torch_model = attempt_load("tests/ttnn/integration_tests/yolov10/yolov10x.pt", map_location="cpu")
    torch_model.eval()
    torch_model = torch_model.model[index]

    state_dict = torch_model.state_dict()
    reference_model = YOLOv10()
    reference_model.eval()
    reference_model = reference_model.model[index]

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    torch_model_output = reference_model(torch_input)[0]
    parameters = create_yolov10x_model_parameters(torch_model, torch_input, device=device)
    ttnn_module = ttnn_C2f(
        shortcut=shortcut,
        n=num_layers,
        device=device,
        parameters=parameters.conv_args,
        conv_pt=parameters,
    )
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_model_output.shape)

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "index, fwd_input_shape, num_layers, shortcut",
    [
        (23, ([1, 320, 80, 80], [1, 640, 40, 40], [1, 640, 20, 20]), 3, True),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 83000}], indirect=True)
def test_yolov10x_v10Detect(device, reset_seeds, index, fwd_input_shape, num_layers, shortcut):
    torch_input_1, ttnn_input_1 = create_yolov10x_input_tensors(
        device,
        batch_size=fwd_input_shape[0][0],
        input_channels=fwd_input_shape[0][1],
        input_height=fwd_input_shape[0][2],
        input_width=fwd_input_shape[0][3],
    )
    torch_input_2, ttnn_input_2 = create_yolov10x_input_tensors(
        device,
        batch_size=fwd_input_shape[1][0],
        input_channels=fwd_input_shape[1][1],
        input_height=fwd_input_shape[1][2],
        input_width=fwd_input_shape[1][3],
    )
    torch_input_3, ttnn_input_3 = create_yolov10x_input_tensors(
        device,
        batch_size=fwd_input_shape[2][0],
        input_channels=fwd_input_shape[2][1],
        input_height=fwd_input_shape[2][2],
        input_width=fwd_input_shape[2][3],
    )

    torch_input = [torch_input_1, torch_input_2, torch_input_3]
    download_yolov10x_weights()
    torch_model = attempt_load("models/experimental/functional_yolov10/reference/", map_location="cpu")
    torch_model.eval()
    torch_model = torch_model.model[23]

    state_dict = torch_model.state_dict()
    reference_model = YOLOv10()
    reference_model.eval()
    reference_model = reference_model.model[23]

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()
    parameters = create_yolov10_model_parameters_detect(
        reference_model, torch_input[0], torch_input[1], torch_input[2], device=device
    )

    torch_model_output = reference_model(torch_input)[0]
    print("torch_model_output", torch_model_output.shape)
    ttnn_input = [ttnn.from_torch(torch_input_1), ttnn.from_torch(torch_input_2), ttnn.from_torch(torch_input_3)]

    ttnn_module = ttnn_V10Detect(
        device=device,
        parameters=parameters.model_args,
        conv_pt=parameters,
        torch_conv=True,
    )
    ttnn_output = ttnn_module(ttnn_input_1, ttnn_input_2, ttnn_input_3)
    ttnn_output = ttnn.to_torch(ttnn_output)[0]

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)  # PCC = 0.9986721809938076


# PCC 1= 0.9684003291054135
# PCC 2= 0.9784148638644237
# PCC 3= 0.9839501449608378


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        # False,
        True
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov10x_trial(use_pretrained_weight, device, reset_seeds):
    torch_input, ttnn_input = create_yolov10x_input_tensors(device)
    state_dict = None

    if use_pretrained_weight:
        torch_model = attempt_load("tests/ttnn/integration_tests/yolov10/yolov10x.pt", map_location="cpu")
        state_dict = torch_model.state_dict()

    torch_model = YOLOv10()
    state_dict = torch_model.state_dict() if state_dict is None else state_dict
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    torch_output = torch_model(torch_input)
    parameters = create_yolov10x_model_parameters(torch_model, torch_input, device)

    torch_model_output = torch_model(torch_input)[0]
    ttnn_module = ttnn_Yolov10(
        device=device,
        parameters=parameters,
        conv_pt=parameters,
        torch_conv=True,
    )
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)[0]

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)
