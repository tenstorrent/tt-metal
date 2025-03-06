# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import ttnn
import torch
import pickle
import pytest
import torch.nn as nn
from models.utility_functions import run_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_yolov9c.tt.model_preprocessing import (
    create_yolov9c_input_tensors,
    create_yolov9c_model_parameters,
    create_yolov9c_model_parameters_detect,
)
from models.experimental.functional_yolov9c.tt import ttnn_yolov9c
from models.experimental.functional_yolov9c.reference import yolov9c
from models.experimental.functional_yolov9c.demo.demo_utils import attempt_load

try:
    sys.modules["ultralytics"] = yolov9c
    sys.modules["ultralytics.nn.tasks"] = yolov9c
    sys.modules["ultralytics.nn.modules.conv"] = yolov9c
    sys.modules["ultralytics.nn.modules.block"] = yolov9c
    sys.modules["ultralytics.nn.modules.head"] = yolov9c

except KeyError:
    logger.error("models.experimental.functional_yolov9c.reference.yolov9c not found.")


class DummyLoss:
    def __init__(self, *args, **kwargs):
        pass


def dummy_load_class(mod_name, name):
    if name == "v8DetectionLoss":
        return DummyLoss
    return torch.serialization.pickle.load_class(mod_name, name)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups, fwd_input_shape",
    [
        (64, 64, [3, 3], [1, 1], [1, 1], [1, 1], [1, 1], [1, 32, 160, 160]),
        (128, 128, [3, 3], [1, 1], [1, 1], [1, 1], [1, 1], [1, 64, 80, 80]),
        (256, 256, [3, 3], [1, 1], [1, 1], [1, 1], [1, 1], [1, 128, 40, 40]),
        (256, 256, [3, 3], [1, 1], [1, 1], [1, 1], [1, 1], [1, 128, 20, 20]),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov9c_repbottleneck(
    device,
    use_program_cache,
    reset_seeds,
    in_channel,
    out_channel,
    kernel,
    stride,
    padding,
    dilation,
    groups,
    fwd_input_shape,
):
    torch_module = yolov9c.RepBottleneck(in_channel, out_channel)
    torch_module.eval()
    torch_input, ttnn_input = create_yolov9c_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
    torch_output = torch_module(torch_input)
    parameters = create_yolov9c_model_parameters(torch_module, torch_input, device=device)
    ttnn_module = ttnn_yolov9c.RepBottleneck(device=device, parameter=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(x=ttnn_input, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups, is_bk_enabled, fwd_input_shape",
    [
        (128, 128, 3, 1, 0, 1, 1, True, [1, 64, 160, 160]),
        (256, 256, 3, 1, 0, 1, 1, True, [1, 128, 80, 80]),
        (512, 512, 3, 1, 0, 1, 1, True, [1, 256, 40, 40]),
        (512, 512, 3, 1, 0, 1, 1, True, [1, 256, 20, 20]),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov9c_repcsp(
    device,
    use_program_cache,
    reset_seeds,
    in_channel,
    out_channel,
    kernel,
    stride,
    padding,
    dilation,
    groups,
    is_bk_enabled,
    fwd_input_shape,
):
    torch_module = yolov9c.RepCSP(in_channel, out_channel)
    torch_module.eval()
    torch_input, ttnn_input = create_yolov9c_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    torch_output = torch_module(torch_input)
    parameters = create_yolov9c_model_parameters(torch_module, torch_input, device=device)
    ttnn_module = ttnn_yolov9c.RepCSP(device=device, parameter=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(x=ttnn_input, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "in_channel, out_channel, cv2_inc, cv2_outc, cv3_inc, cv3_outc, cv4_inc, cv4_out_c, fwd_input_shape",
    [
        (128, 128, 64, 64, 64, 64, 256, 256, [1, 128, 160, 160]),
        (256, 256, 128, 128, 128, 128, 512, 512, [1, 256, 80, 80]),
        (512, 512, 256, 256, 256, 256, 1024, 512, [1, 512, 40, 40]),
        (512, 512, 256, 256, 256, 256, 1024, 512, [1, 512, 20, 20]),
        (1024, 512, 256, 256, 256, 256, 1024, 512, [1, 1024, 40, 40]),
        (1024, 256, 128, 128, 128, 128, 512, 256, [1, 1024, 80, 80]),
        (768, 512, 256, 256, 256, 256, 1024, 512, [1, 768, 40, 40]),
        (1024, 512, 256, 256, 256, 256, 1024, 512, [1, 1024, 20, 20]),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov9c_repncspelan4(
    device,
    use_program_cache,
    reset_seeds,
    in_channel,
    out_channel,
    cv2_inc,
    cv2_outc,
    cv3_inc,
    cv3_outc,
    cv4_inc,
    cv4_out_c,
    fwd_input_shape,
):
    torch_module = yolov9c.RepNCSPELAN4(
        in_channel,
        out_channel,
        cv2_inc=cv2_inc,
        cv2_outc=cv2_outc,
        cv3_inc=cv3_inc,
        cv3_outc=cv3_outc,
        cv4_inc=cv4_inc,
        cv4_out_c=cv4_out_c,
    )
    torch_module.eval()
    torch_input, ttnn_input = create_yolov9c_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    torch_output = torch_module(torch_input)
    parameters = create_yolov9c_model_parameters(torch_module, torch_input, device=device)
    ttnn_module = ttnn_yolov9c.RepNCSPELAN4(device=device, parameter=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(x=ttnn_input, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "in_channel, out_channel, fwd_input_shape",
    [
        (128, 128, [1, 256, 160, 160]),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov9c_adown(
    device,
    use_program_cache,
    reset_seeds,
    in_channel,
    out_channel,
    fwd_input_shape,
):
    torch_module = yolov9c.ADown(in_channel, out_channel)
    torch_module.eval()
    torch_input, ttnn_input = create_yolov9c_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    torch_output = torch_module(torch_input)
    parameters = create_yolov9c_model_parameters(torch_module, torch_input, device=device)
    ttnn_module = ttnn_yolov9c.ADown(device=device, parameter=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(x=ttnn_input, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "in_channel, out_channel, fwd_input_shape",
    [
        (512, 256, [1, 512, 20, 20]),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov9c_sppelan(
    device,
    use_program_cache,
    reset_seeds,
    in_channel,
    out_channel,
    fwd_input_shape,
):
    torch_module = yolov9c.SPPELAN(in_channel, out_channel)
    torch_module.eval()
    torch_input, ttnn_input = create_yolov9c_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    torch_output = torch_module(torch_input)
    parameters = create_yolov9c_model_parameters(torch_module, torch_input, device=device)
    ttnn_module = ttnn_yolov9c.SPPELAN(device=device, parameter=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(x=ttnn_input, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "in_channel, fwd_input_shapes",
    [
        (
            [256, 64, 64, 512, 64, 64, 512, 64, 64, 256, 256, 256, 512, 256, 256, 512, 256, 256],
            [[1, 256, 80, 80], [1, 512, 40, 40], [1, 512, 20, 20]],
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov9c_detect(
    device,
    use_program_cache,
    reset_seeds,
    in_channel,
    fwd_input_shapes,
):
    torch_module = yolov9c.Detect(in_channel)
    torch_module.eval()
    (b1, c1, h1, w1), (b2, c2, h2, w2), (b3, c3, h3, w3) = fwd_input_shapes
    y1, ttnn_y1 = create_yolov9c_input_tensors(
        device, batch_size=b1, input_channels=c1, input_height=h1, input_width=w1
    )
    y2, ttnn_y2 = create_yolov9c_input_tensors(
        device, batch_size=b2, input_channels=c2, input_height=h2, input_width=w2
    )
    y3, ttnn_y3 = create_yolov9c_input_tensors(
        device, batch_size=b3, input_channels=c3, input_height=h3, input_width=w3
    )
    torch_output = torch_module(y1, y2, y3)
    parameters = create_yolov9c_model_parameters_detect(torch_module, y1, y2, y3, device=device)
    ttnn_module = ttnn_yolov9c.Detect(device=device, parameter=parameters.model, conv_pt=parameters)
    ttnn_output = ttnn_module(device=device, y1=ttnn_y1, y2=ttnn_y2, y3=ttnn_y3)
    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        False,
        # True
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov9c(use_pretrained_weight, device, use_program_cache, reset_seeds):
    torch_input, ttnn_input = create_yolov9c_input_tensors(device)
    state_dict = None

    if use_pretrained_weight:
        torch_model = attempt_load("yolov9c.pt", map_location="cpu")
        state_dict = torch_model.state_dict()

    torch_model = yolov9c.YoloV9()
    state_dict = torch_model.state_dict() if state_dict is None else state_dict
    ds_state_dict = {k: v for k, v in state_dict.items()}
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    torch_output = torch_model(torch_input)
    parameters = create_yolov9c_model_parameters(torch_model, torch_input, device)
    ttnn_model = ttnn_yolov9c.YoloV9(device, parameters)
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.98 if use_pretrained_weight else 0.99)
