# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import os
import pytest
import sys
import ttnn
import torch.nn as nn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.yolov11.tt.model_preprocessing import (
    create_yolov11_input_tensors,
    create_yolov11_model_parameters,
    create_yolov11_model_parameters_detect,
)
from models.experimental.yolov11.reference.yolov11 import (
    Attention as torch_attention,
    Bottleneck as torch_bottleneck,
    C2PSA as torch_c2psa_block,
    C3k as torch_c3k,
    C3k2 as torch_c3k2,
    PSABlock as torch_psa_block,
    SPPF as torch_sppf,
    Detect as torch_detect,
)
from models.experimental.yolov11.tt.ttnn_yolov11 import (
    Attention as ttnn_attention,
    Bottleneck as ttnn_bottleneck,
    C2PSA as ttnn_c2psa_block,
    C3K as ttnn_c3k,
    C3k2 as ttnn_c3k2,
    PSABlock as ttnn_psa_block,
    SPPF as ttnn_sppf,
    Detect as ttnn_detect,
)
from models.utility_functions import skip_for_grayskull

from models.experimental.yolov11.reference import yolov11
from models.experimental.yolov11.tt import ttnn_yolov11


@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups,fwd_input_shape",
    [
        ([128, 128, 128], [256, 128, 128], [1, 1, 3], [1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 128], [1, 128, 7, 7]),
    ],
)
@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolo_v11_attention(
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
    torch_module = torch_attention(in_channel, out_channel, kernel, stride, padding, dilation, groups)
    torch_module.eval()
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    torch_output = torch_module(torch_input)
    parameters = create_yolov11_model_parameters(torch_module, torch_input, device=device)
    ttnn_module = ttnn_attention(device=device, parameter=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(x=ttnn_input, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups,fwd_input_shape",
    [
        ([16, 8], [8, 16], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1], [1, 16, 56, 56]),  # 1
        ([32, 16], [16, 32], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1], [1, 32, 38, 28]),  # 2
        ([32, 32], [32, 32], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1], [1, 32, 14, 14]),  # 3
        ([64, 64], [64, 64], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1], [1, 64, 7, 7]),
        ([64, 32], [32, 64], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1], [1, 64, 14, 14]),
        ([32, 16], [16, 32], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1], [1, 32, 28, 28]),
        ([64, 32], [32, 64], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1], [1, 64, 14, 14]),
        ([64, 64], [64, 64], [3, 3], [1, 1], [1, 1], [1, 1], [1, 1], [1, 64, 7, 7]),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolo_v11_bottleneck(
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
    torch_module = torch_bottleneck(in_channel, out_channel, kernel, stride, padding, dilation, groups)
    torch_module.eval()
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    torch_output = torch_module(torch_input)
    parameters = create_yolov11_model_parameters(torch_module, torch_input, device=device)
    ttnn_module = ttnn_bottleneck(device=device, parameter=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(x=ttnn_input, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups,fwd_input_shape",
    [
        (
            [256, 256, 128, 128, 128, 128, 256],
            [256, 256, 256, 128, 128, 256, 128],
            [1, 1, 1, 1, 3, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 128, 1, 1],
            [1, 256, 7, 7],
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolo_v11_c2psa_block(
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
    torch_module = torch_c2psa_block(in_channel, out_channel, kernel, stride, padding, dilation, groups)
    torch_module.eval()
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    torch_output = torch_module(torch_input)
    parameters = create_yolov11_model_parameters(torch_module, torch_input, device=device)
    ttnn_module = ttnn_c2psa_block(device=device, parameter=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(x=ttnn_input, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups,fwd_input_shape",
    [
        (
            [64, 64, 64, 32, 32, 32, 32],
            [32, 32, 64, 32, 32, 32, 32],
            [1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 64, 14, 14],
        ),
        (
            [128, 128, 128, 64, 64, 64, 64],
            [64, 64, 128, 64, 64, 64, 64],
            [1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 128, 7, 7],
        ),
        (
            [64, 64, 64, 32, 32, 32, 32],
            [32, 32, 64, 32, 32, 32, 32],
            [1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 64, 40, 40],
        ),
        (
            [128, 128, 128, 64, 64, 64, 64],
            [64, 64, 128, 64, 64, 64, 64],
            [1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 128, 20, 20],
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolo_v11_c3k(
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
    torch_module = torch_c3k(in_channel, out_channel, kernel, stride, padding, dilation, groups)
    torch_module.eval()
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    torch_output = torch_module(torch_input)
    parameters = create_yolov11_model_parameters(torch_module, torch_input, device=device)
    ttnn_module = ttnn_c3k(device=device, parameter=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(x=ttnn_input, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups,is_bk_enabled,fwd_input_shape",
    [
        # 224
        (
            [32, 48, 16, 8],
            [32, 64, 8, 16],
            [1, 1, 3, 3],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            True,
            [1, 32, 56, 56],
        ),
        (
            [64, 96, 32, 16],
            [64, 128, 16, 32],
            [1, 1, 3, 3],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            True,
            [1, 64, 28, 28],
        ),
        (
            [128, 192, 64, 64, 64, 32, 32, 32, 32],
            [128, 128, 32, 32, 64, 32, 32, 32, 32],
            [1, 1, 1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            False,
            [1, 128, 14, 14],
        ),
        (
            [256, 384, 128, 128, 128, 64, 64, 64, 64],
            [256, 256, 64, 64, 128, 64, 64, 64, 64],
            [1, 1, 1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            False,
            [1, 256, 7, 7],
        ),
        (
            [384, 192, 64, 32],
            [128, 128, 32, 64],
            [1, 1, 3, 3],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            True,
            [1, 384, 14, 14],
        ),
        (
            [256, 96, 32, 16],
            [64, 64, 16, 32],
            [1, 1, 3, 3],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            True,
            [1, 256, 28, 28],
        ),
        (
            [192, 192, 64, 32],
            [128, 128, 32, 64],
            [1, 1, 3, 3],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            True,
            [1, 192, 14, 14],
        ),
        (
            [384, 384, 128, 128, 128, 64, 64, 64, 64],
            [256, 256, 64, 64, 128, 64, 64, 64, 64],
            [1, 1, 1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            False,
            [1, 384, 7, 7],
        ),
        # 640
        (
            [32, 48, 16, 8],
            [32, 64, 8, 16],
            [1, 1, 3, 3],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            True,
            [1, 32, 160, 160],
        ),
        (
            [64, 96, 32, 16],
            [64, 128, 16, 32],
            [1, 1, 3, 3],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            True,
            [1, 64, 80, 80],
        ),
        (
            [128, 192, 64, 64, 64, 32, 32, 32, 32],
            [128, 128, 32, 32, 64, 32, 32, 32, 32],
            [1, 1, 1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            False,
            [1, 128, 40, 40],
        ),
        (
            [256, 384, 128, 128, 128, 64, 64, 64, 64],
            [256, 256, 64, 64, 128, 64, 64, 64, 64],
            [1, 1, 1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            False,
            [1, 256, 20, 20],
        ),
        (
            [384, 192, 64, 32],
            [128, 128, 32, 64],
            [1, 1, 3, 3],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            True,
            [1, 384, 40, 40],
        ),
        (
            [256, 96, 32, 16],
            [64, 64, 16, 32],
            [1, 1, 3, 3],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            True,
            [1, 256, 80, 80],
        ),
        (
            [192, 192, 64, 32],
            [128, 128, 32, 64],
            [1, 1, 3, 3],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            True,
            [1, 192, 40, 40],
        ),
        (
            [384, 384, 128, 128, 128, 64, 64, 64, 64],
            [256, 256, 64, 64, 128, 64, 64, 64, 64],
            [1, 1, 1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            False,
            [1, 384, 20, 20],
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolo_v11_c3k2(
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
    torch_module = torch_c3k2(in_channel, out_channel, kernel, stride, padding, dilation, groups, is_bk_enabled)
    torch_module.eval()
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    ttnn_input = ttnn.to_device(ttnn_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)
    torch_output = torch_module(torch_input)
    parameters = create_yolov11_model_parameters(torch_module, torch_input, device=device)
    ttnn_module = ttnn_c3k2(
        device=device, parameter=parameters.conv_args, conv_pt=parameters, is_bk_enabled=is_bk_enabled
    )
    ttnn_output = ttnn_module(x=ttnn_input, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups,fwd_input_shape",
    [
        (
            [128, 128, 128, 128, 256],
            [256, 128, 128, 256, 128],
            [1, 1, 3, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 128, 1, 1],
            [1, 128, 7, 7],
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolo_v11_psa_block(
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
    torch_module = torch_psa_block(in_channel, out_channel, kernel, stride, padding, dilation, groups)
    torch_module.eval()
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    torch_output = torch_module(torch_input)
    parameters = create_yolov11_model_parameters(torch_module, torch_input, device=device)
    ttnn_module = ttnn_psa_block(device=device, parameter=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(x=ttnn_input, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups,fwd_input_shape",
    [
        ([256, 512], [128, 256], [1, 1], [1, 1], [0, 0], [1, 1], [1, 1], [1, 256, 20, 20]),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolo_v11_sppf(
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
    torch_module = torch_sppf(in_channel, out_channel, kernel, stride, padding, dilation, groups)
    torch_module.eval()
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    torch_output = torch_module(torch_input)
    parameters = create_yolov11_model_parameters(torch_module, torch_input, device=device)
    ttnn_module = ttnn_sppf(device=device, parameter=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(x=ttnn_input, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


try:
    sys.modules["ultralytics"] = yolov11
    sys.modules["ultralytics.nn.tasks"] = yolov11
    sys.modules["ultralytics.nn.modules.conv"] = yolov11
    sys.modules["ultralytics.nn.modules.block"] = yolov11
    sys.modules["ultralytics.nn.modules.head"] = yolov11

except KeyError:
    print("models.experimental.yolov11.reference.yolov11 not found.")


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
    tests = Path(__file__).parent.parent / "yolov11"
    file_path = tests / Path(str(file).strip().replace("'", "").lower())

    if not file_path.exists():
        name = "yolo11n.pt"  # file_path.name
        msg = f"{file_path} missing, try downloading from https://github.com/{repo}/releases/"

        try:
            url = f"https://github.com/{repo}/releases/download/v8.3.0/{name}"

            print(f"Downloading {url} to {file_path}...")
            torch.hub.download_url_to_file(url, file_path)

            # Validate the file
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


@skip_for_grayskull()
@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups,fwd_input_shape",
    [
        (
            [64, 64, 64, 128, 64, 64, 256, 64, 64, 64, 64, 80, 80, 80, 128, 128, 80, 80, 80, 256, 256, 80, 80, 80, 16],
            [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 80, 80, 80, 80, 128, 80, 80, 80, 80, 256, 80, 80, 80, 80, 1],
            [3, 3, 1, 3, 3, 1, 3, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 64, 1, 80, 1, 1, 128, 1, 80, 1, 1, 256, 1, 80, 1, 1, 1],
            [[1, 64, 28, 28], [1, 128, 14, 14], [1, 256, 7, 7]],
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolo_v11_detect(
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
    torch_module = torch_detect(in_channel, out_channel, kernel, stride, padding, dilation, groups)
    torch_module.eval()
    torch_input_1, ttnn_input_1 = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[0][0],
        input_channels=fwd_input_shape[0][1],
        input_height=fwd_input_shape[0][2],
        input_width=fwd_input_shape[0][3],
    )
    torch_input_2, ttnn_input_2 = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[1][0],
        input_channels=fwd_input_shape[1][1],
        input_height=fwd_input_shape[1][2],
        input_width=fwd_input_shape[1][3],
    )
    torch_input_3, ttnn_input_3 = create_yolov11_input_tensors(
        device,
        batch=fwd_input_shape[2][0],
        input_channels=fwd_input_shape[2][1],
        input_height=fwd_input_shape[2][2],
        input_width=fwd_input_shape[2][3],
    )
    ttnn_input_1 = ttnn.to_device(ttnn_input_1, device=device)
    ttnn_input_1 = ttnn.to_layout(ttnn_input_1, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn_input_2 = ttnn.to_device(ttnn_input_2, device=device)
    ttnn_input_2 = ttnn.to_layout(ttnn_input_2, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn_input_3 = ttnn.to_device(ttnn_input_3, device=device)
    ttnn_input_3 = ttnn.to_layout(ttnn_input_3, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    torch_output = torch_module(torch_input_1, torch_input_2, torch_input_3)
    parameters = create_yolov11_model_parameters_detect(
        torch_module, torch_input_1, torch_input_2, torch_input_3, device=device
    )
    ttnn_module = ttnn_detect(device=device, parameter=parameters.model, conv_pt=parameters)

    ttnn_output = ttnn_module(y1=ttnn_input_1, y2=ttnn_input_2, y3=ttnn_input_3, device=device)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "resolution",
    [
        ([1, 3, 224, 224]),
        ([1, 3, 640, 640]),
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [False, True],  # uncomment  to run the model for real weights
    ids=[
        "pretrained_weight_false",
        "pretrained_weight_true",  # uncomment to run the model for real weights
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov11(device, use_program_cache, reset_seeds, resolution, use_pretrained_weight):
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device, batch=resolution[0], input_channels=resolution[1], input_height=resolution[2], input_width=resolution[3]
    )
    if use_pretrained_weight:
        torch_model = attempt_load("yolov11n.pt", map_location="cpu")
        state_dict = torch_model.state_dict()
        torch_model = yolov11.YoloV11()
        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2
        torch_model.load_state_dict(new_state_dict)
    else:
        torch_model = yolov11.YoloV11()
    torch_model.eval()

    torch_output = torch_model(torch_input)
    parameters = create_yolov11_model_parameters(torch_model, torch_input, device=device)
    ttnn_model = ttnn_yolov11.YoloV11(device, parameters)
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)
