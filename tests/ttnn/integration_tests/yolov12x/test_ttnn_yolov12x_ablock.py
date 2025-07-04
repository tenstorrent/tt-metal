# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.yolov12x.reference import yolov12x
from models.experimental.yolov12x.tt.ablock import TtnnABlock
from models.experimental.yolov12x.tt.model_preprocessing import (
    create_yolov12x_input_tensors,
    create_yolov12x_model_parameters,
)


@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups, dim, num_heads, mlp_ratio, area, fwd_input_shape",
    [
        (
            [384, 384, 384, 384, 460],
            [1152, 384, 384, 460, 384],
            [1, 1, 7, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 3, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 384, 1, 1],
            384,
            12,
            1.2,
            4,
            [1, 384, 40, 40],
        ),  # 1
        (
            [384, 384, 384, 384, 460],
            [1152, 384, 384, 460, 384],
            [1, 1, 7, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 3, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 384, 1, 1],
            384,
            12,
            1.2,
            1,
            [1, 384, 20, 20],
        ),  # 2
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov12x_ablock(
    device,
    reset_seeds,
    in_channel,
    out_channel,
    kernel,
    stride,
    padding,
    dilation,
    groups,
    dim,
    num_heads,
    mlp_ratio,
    area,
    fwd_input_shape,
):
    torch_module = yolov12x.ABlock(
        in_channel, out_channel, kernel, stride, padding, dilation, groups, dim, num_heads, mlp_ratio, area
    )
    torch_module.eval()
    torch_input, ttnn_input = create_yolov12x_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )
    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    torch_output = torch_module(torch_input)
    parameters = create_yolov12x_model_parameters(torch_module, torch_input, device=device)
    ttnn_module = TtnnABlock(device=device, parameter=parameters.conv_args, conv_pt=parameters)
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)
