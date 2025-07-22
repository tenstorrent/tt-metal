# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.yolov11.tt.model_preprocessing import (
    create_yolov11_input_tensors,
    create_yolov11_model_parameters,
)
from models.demos.yolov11.tt.ttnn_yolov11_c3k2 import TtnnC3k2 as ttnn_c3k2
from models.demos.yolov11.reference.yolov11 import C3k2 as torch_c3k2


@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups,is_bk_enabled,fwd_input_shape",
    [
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
