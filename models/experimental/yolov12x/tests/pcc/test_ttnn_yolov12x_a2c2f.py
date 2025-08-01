# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.yolov12x.reference import yolov12x
from models.experimental.yolov12x.tt.a2c2f import TtnnA2C2f
from models.experimental.yolov12x.tt.model_preprocessing import (
    create_yolov12x_input_tensors,
    create_yolov12x_model_parameters,
)


@pytest.mark.parametrize(
    "in_channel, out_channel, kernel, stride, padding, dilation, groups, c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut, fwd_input_shape, use_1d_systolic_array, shard_layout, config_override",
    [
        (
            [768, 1920, 384, 384, 384, 384, 460, 384, 384, 384, 384, 460],
            [384, 768, 1152, 384, 384, 460, 384, 1152, 384, 384, 460, 384],
            [1, 1, 1, 1, 7, 1, 1, 1, 1, 7, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 384, 1, 1, 1, 1, 384, 1, 1],
            768,
            768,
            4,
            True,
            4,
            True,
            1.2,
            0.5,
            1,
            True,
            [1, 768, 40, 40],
            False,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            {"act_block_h": 32},
        ),  # 1
        (
            [1536, 1152, 384, 384, 384, 192, 192, 192, 192],
            [384, 768, 192, 192, 384, 192, 192, 192, 192],
            [1, 1, 1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            1536,
            768,
            2,
            False,
            -1,
            True,
            1.2,
            0.5,
            1,
            True,
            [1, 1536, 40, 40],
            True,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            None,
        ),  # 2
        (
            [1536, 576, 192, 192, 192, 96, 96, 96, 96],
            [192, 384, 96, 96, 192, 96, 96, 96, 96],
            [1, 1, 1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            1536,
            384,
            2,
            False,
            -1,
            True,
            1.2,
            0.5,
            1,
            True,
            [1, 1536, 80, 80],
            True,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            None,
        ),  # 3
        (
            [1152, 1152, 384, 384, 384, 192, 192, 192, 192],
            [384, 768, 192, 192, 384, 192, 192, 192, 192],
            [1, 1, 1, 1, 1, 3, 3, 3, 3],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            1152,
            768,
            2,
            False,
            -1,
            True,
            1.2,
            0.5,
            1,
            True,
            [1, 1152, 40, 40],
            True,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            None,
        ),  # 4
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov12x_a2c2f(
    device,
    reset_seeds,
    in_channel,
    out_channel,
    kernel,
    stride,
    padding,
    dilation,
    groups,
    c1,
    c2,
    n,
    a2,
    area,
    residual,
    mlp_ratio,
    e,
    g,
    shortcut,
    fwd_input_shape,
    use_1d_systolic_array,
    shard_layout,
    config_override,
):
    torch_module = yolov12x.A2C2f(
        in_channel,
        out_channel,
        kernel,
        stride,
        padding,
        dilation,
        groups,
        c1,
        c2,
        n,
        a2,
        area,
        residual,
        mlp_ratio,
        e,
        g,
        shortcut,
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
    ttnn_module = TtnnA2C2f(
        device=device,
        parameter=parameters.conv_args,
        conv_pt=parameters,
        c1=c1,
        c2=c2,
        n=n,
        a2=a2,
        area=area,
        residual=residual,
        mlp_ratio=mlp_ratio,
        e=e,
        g=g,
        shortcut=shortcut,
        use_1d_systolic_array=use_1d_systolic_array,
        shard_layout=shard_layout,
        config_override=config_override,
    )

    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 0.99)
