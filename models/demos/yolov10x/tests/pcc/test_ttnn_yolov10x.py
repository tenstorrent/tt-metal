# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.yolov10x.common import YOLOV10_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov10x.reference.yolov10x import YOLOv10
from models.demos.yolov10x.tt.attention import TtnnAttention
from models.demos.yolov10x.tt.bottleneck import TtnnBottleNeck
from models.demos.yolov10x.tt.c2f import TtnnC2f
from models.demos.yolov10x.tt.cib import TtnnCIB
from models.demos.yolov10x.tt.model_preprocessing import (
    create_yolov10_model_parameters_detect,
    create_yolov10x_input_tensors,
    create_yolov10x_input_tensors_submodules,
    create_yolov10x_model_parameters,
)
from models.demos.yolov10x.tt.psa import TtnnPSA
from models.demos.yolov10x.tt.scdown import TtnnSCDown
from models.demos.yolov10x.tt.sppf import TtnnSPPF
from models.demos.yolov10x.tt.v10detect import TtnnV10Detect
from models.demos.yolov10x.tt.yolov10x import TtnnYolov10
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "use_pretrained_weights",
    [True],
)
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
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV10_L1_SMALL_SIZE}], indirect=True)
def test_yolov10x_bottleneck(
    device, reset_seeds, index, fwd_input_shape, shortcut, use_pretrained_weights, model_location_generator
):
    torch_input, ttnn_input = create_yolov10x_input_tensors_submodules(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    state_dict = None
    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)

    torch_model = YOLOv10()
    torch_model = torch_model.model[index].m[0]
    state_dict = torch_model.state_dict()
    torch_model.eval()

    state_dict = torch_model.state_dict() if state_dict is None else state_dict
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_model_output = torch_model(torch_input)[0]

    parameters = create_yolov10x_model_parameters(torch_model, torch_input, device=device)

    ttnn_module = TtnnBottleNeck(shortcut=shortcut, device=device, parameters=parameters.conv_args, conv_pt=parameters)

    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_model_output.shape)

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "use_pretrained_weights",
    [True],
)
@pytest.mark.parametrize(
    "index, fwd_input_shape",
    [
        (5, (1, 320, 80, 80)),
        (7, (1, 640, 40, 40)),
        (20, (1, 640, 40, 40)),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV10_L1_SMALL_SIZE}], indirect=True)
def test_yolov10x_scdown(device, reset_seeds, index, fwd_input_shape, use_pretrained_weights, model_location_generator):
    torch_input, ttnn_input = create_yolov10x_input_tensors_submodules(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    state_dict = None
    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)

    torch_model = YOLOv10()
    torch_model = torch_model.model[index]
    state_dict = torch_model.state_dict()
    torch_model.eval()

    state_dict = torch_model.state_dict() if state_dict is None else state_dict
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_model_output = torch_model(torch_input)[0]

    parameters = create_yolov10x_model_parameters(torch_model, torch_input, device=device)

    ttnn_module = TtnnSCDown(
        device=device,
        parameters=parameters.conv_args,
        conv_pt=parameters,
    )
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output).squeeze(0)
    ttnn_output = ttnn_output.permute(0, 2, 1)
    ttnn_output = ttnn_output.reshape(torch_model_output.shape)

    assert_with_pcc(torch_model_output, ttnn_output, 0.999)


@pytest.mark.parametrize(
    "use_pretrained_weights",
    [True],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV10_L1_SMALL_SIZE}], indirect=True)
def test_yolov10x_sppf(device, reset_seeds, use_pretrained_weights, model_location_generator):
    fwd_input_shape = [1, 640, 20, 20]
    torch_input, ttnn_input = create_yolov10x_input_tensors_submodules(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    state_dict = None

    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)

    torch_model = YOLOv10()
    torch_model = torch_model.model[9]
    state_dict = torch_model.state_dict()
    torch_model.eval()

    state_dict = torch_model.state_dict() if state_dict is None else state_dict
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_model_output = torch_model(torch_input)[0]

    parameters = create_yolov10x_model_parameters(torch_model, torch_input, device=device)

    ttnn_module = TtnnSPPF(
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
    "use_pretrained_weights",
    [True],
)
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
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV10_L1_SMALL_SIZE}], indirect=True)
def test_yolov10x_cib(device, reset_seeds, index, fwd_input_shape, use_pretrained_weights, model_location_generator):
    torch_input, ttnn_input = create_yolov10x_input_tensors_submodules(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )

    state_dict = None
    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)

    torch_model = YOLOv10()
    torch_model = torch_model.model[index].m[0]
    state_dict = torch_model.state_dict()
    torch_model.eval()

    state_dict = torch_model.state_dict() if state_dict is None else state_dict
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_model_output = torch_model(torch_input)[0]

    parameters = create_yolov10x_model_parameters(torch_model, torch_input, device=device)

    ttnn_module = TtnnCIB(
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
    "use_pretrained_weights",
    [True],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV10_L1_SMALL_SIZE}], indirect=True)
def test_yolov10x_attention(device, reset_seeds, use_pretrained_weights, model_location_generator):
    fwd_input_shape = [1, 320, 20, 20]
    torch_input, ttnn_input = create_yolov10x_input_tensors_submodules(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
        input_dtype=ttnn.bfloat16,
    )
    state_dict = None
    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)

    torch_model = YOLOv10()
    torch_model = torch_model.model[10].attn
    state_dict = torch_model.state_dict()
    torch_model.eval()

    state_dict = torch_model.state_dict() if state_dict is None else state_dict
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_model_output = torch_model(torch_input)[0]

    parameters = create_yolov10x_model_parameters(torch_model, torch_input, device=device)

    ttnn_module = TtnnAttention(
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


@pytest.mark.parametrize(
    "use_pretrained_weights",
    [True],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV10_L1_SMALL_SIZE}], indirect=True)
def test_yolov10x_psa(device, reset_seeds, use_pretrained_weights, model_location_generator):
    fwd_input_shape = [1, 640, 20, 20]
    torch_input, ttnn_input = create_yolov10x_input_tensors_submodules(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
        input_dtype=ttnn.bfloat16,
    )
    state_dict = None

    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)

    torch_model = YOLOv10()
    torch_model = torch_model.model[10]
    state_dict = torch_model.state_dict()
    torch_model.eval()

    state_dict = torch_model.state_dict() if state_dict is None else state_dict
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_model_output = torch_model(torch_input)[0]

    parameters = create_yolov10x_model_parameters(torch_model, torch_input, device=device)

    ttnn_module = TtnnPSA(
        device=device,
        parameters=parameters.conv_args,
        conv_pt=parameters,
    )
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output).squeeze(0)
    ttnn_output = ttnn_output.permute(0, 2, 1)
    ttnn_output = ttnn_output.reshape(torch_model_output.shape)

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "use_pretrained_weights",
    [True],
)
@pytest.mark.parametrize(
    "index, fwd_input_shape, num_layers, shortcut, memory_config",
    [
        (2, (1, 160, 160, 160), 3, True, ttnn.DRAM_MEMORY_CONFIG),
        (4, (1, 320, 80, 80), 6, True, ttnn.L1_MEMORY_CONFIG),
        (16, (1, 960, 80, 80), 3, False, ttnn.L1_MEMORY_CONFIG),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV10_L1_SMALL_SIZE}], indirect=True)
def test_yolov10x_c2f(
    device,
    reset_seeds,
    index,
    fwd_input_shape,
    num_layers,
    shortcut,
    use_pretrained_weights,
    memory_config,
    model_location_generator,
):
    torch_input, ttnn_input = create_yolov10x_input_tensors_submodules(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    state_dict = None
    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)

    torch_model = YOLOv10()
    torch_model = torch_model.model[index]
    state_dict = torch_model.state_dict()
    torch_model.eval()

    state_dict = torch_model.state_dict() if state_dict is None else state_dict
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_model_output = torch_model(torch_input)[0]
    parameters = create_yolov10x_model_parameters(torch_model, torch_input, device=device)
    ttnn_module = TtnnC2f(
        shortcut=shortcut,
        n=num_layers,
        device=device,
        parameters=parameters.conv_args,
        conv_pt=parameters,
    )
    ttnn_output = ttnn_module(ttnn_input, memory_config=memory_config)
    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_model_output.shape)

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)


@pytest.mark.parametrize(
    "use_pretrained_weights",
    [True],
)
@pytest.mark.parametrize(
    "index, fwd_input_shape, num_layers, shortcut",
    [
        (23, ([1, 320, 80, 80], [1, 640, 40, 40], [1, 640, 20, 20]), 3, True),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV10_L1_SMALL_SIZE}], indirect=True)
def test_yolov10x_v10detect(
    device, reset_seeds, index, fwd_input_shape, num_layers, shortcut, use_pretrained_weights, model_location_generator
):
    torch_input_1, ttnn_input_1 = create_yolov10x_input_tensors_submodules(
        device,
        batch_size=fwd_input_shape[0][0],
        input_channels=fwd_input_shape[0][1],
        input_height=fwd_input_shape[0][2],
        input_width=fwd_input_shape[0][3],
    )
    torch_input_2, ttnn_input_2 = create_yolov10x_input_tensors_submodules(
        device,
        batch_size=fwd_input_shape[1][0],
        input_channels=fwd_input_shape[1][1],
        input_height=fwd_input_shape[1][2],
        input_width=fwd_input_shape[1][3],
    )
    torch_input_3, ttnn_input_3 = create_yolov10x_input_tensors_submodules(
        device,
        batch_size=fwd_input_shape[2][0],
        input_channels=fwd_input_shape[2][1],
        input_height=fwd_input_shape[2][2],
        input_width=fwd_input_shape[2][3],
    )

    state_dict = None
    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)

    torch_input = [torch_input_1, torch_input_2, torch_input_3]

    torch_model = YOLOv10()
    torch_model = torch_model.model[23]
    state_dict = torch_model.state_dict()
    torch_model.eval()

    state_dict = torch_model.state_dict() if state_dict is None else state_dict

    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    parameters = create_yolov10_model_parameters_detect(
        torch_model, torch_input[0], torch_input[1], torch_input[2], device=device
    )

    torch_model_output = torch_model(torch_input)[0]
    ttnn_input = [ttnn.from_torch(torch_input_1), ttnn.from_torch(torch_input_2), ttnn.from_torch(torch_input_3)]

    ttnn_module = TtnnV10Detect(
        device=device,
        parameters=parameters.model_args,
        conv_pt=parameters,
    )
    ttnn_output = ttnn_module(ttnn_input_1, ttnn_input_2, ttnn_input_3)
    ttnn_output = ttnn.to_torch(ttnn_output)[0]

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)  # PCC = 0.9986721809938076


@pytest.mark.parametrize(
    "use_pretrained_weights",
    [True],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV10_L1_SMALL_SIZE}], indirect=True)
def test_yolov10x(use_pretrained_weights, device, reset_seeds, model_location_generator):
    torch_input, ttnn_input = create_yolov10x_input_tensors(device)
    state_dict = None

    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)
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
    ttnn_module = TtnnYolov10(
        device=device,
        parameters=parameters,
        conv_pt=parameters,
    )
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)[0]

    assert_with_pcc(torch_model_output, ttnn_output, 0.999)
