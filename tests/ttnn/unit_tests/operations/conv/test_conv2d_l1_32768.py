# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
from tests.ttnn.unit_tests.operations.conv.test_conv2d_common import run_conv, torch_tensor_map

pytestmark = pytest.mark.use_module_device({"l1_small_size": 32768})


@pytest.mark.parametrize(
    "input_layout, dtype",
    [[ttnn.TILE_LAYOUT, ttnn.bfloat8_b], [ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16]],
)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, weights_dtype, kernel, stride, padding, dilation, act_block_h_override,  math_fidelity",
    # fmt: off
    (
        (1, 400,  528,  192,   192,    ttnn.bfloat8_b, (1, 1), (1, 1), (0, 0), (1, 1),      0,  ttnn.MathFidelity.HiFi4 ),
        (2,  13,   31,  313,    71,    ttnn.bfloat8_b, (5, 5), (1, 1), (2, 2), (2, 2), 32 * 4,  ttnn.MathFidelity.LoFi  ),
        (2,  63,  129,  981,    39,    ttnn.bfloat8_b, (3, 3), (2, 2), (2, 2), (1, 1),      0,  ttnn.MathFidelity.LoFi  ),
        (2, 512,  512,  128,   128,    ttnn.bfloat8_b, (3, 3), (1, 1), (1, 1), (1, 1), 32 * 8,  ttnn.MathFidelity.LoFi  ),
        (2, 64,   64,   384,   64,     ttnn.bfloat8_b, (4, 4), (2, 2), (1, 1), (1, 1),      0,  ttnn.MathFidelity.LoFi  ),
        (1, 4,    32,   1024,  1024,   ttnn.bfloat8_b, (5, 5), (1, 1), (0, 0), (1, 1),     32,  ttnn.MathFidelity.LoFi  ),
        (1, 2904, 2904,   48,    48,   ttnn.bfloat8_b, (3, 3), (1, 1), (0, 0), (1, 1),     32,  ttnn.MathFidelity.HiFi4 ),
    )
    # fmt: on
)
@pytest.mark.parametrize(
    "has_bias, fp32_accum, packer_l1_acc",
    [[True, True, False]],
)
def test_conv_dram(
    device,
    torch_tensor_map,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    has_bias,
    weights_dtype,
    dtype,
    kernel,
    stride,
    padding,
    dilation,
    act_block_h_override,
    math_fidelity,
    fp32_accum,
    input_layout,
    packer_l1_acc,
):
    if device.core_grid.y == 7:
        pytest.skip("Tests have been configured for N150.")
    config = {
        "act_block_h": act_block_h_override,
    }

    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        kernel[0],
        kernel[1],
        stride[0],
        stride[1],
        padding,
        config,
        dilation_h=dilation[0],
        dilation_w=dilation[1],
        has_bias=has_bias,
        fp32_accum=fp32_accum,
        packer_l1_acc=packer_l1_acc,
        input_dtype=dtype,
        input_layout=input_layout,
        output_layout=input_layout,
        run_twice=True,
        fast_compare=True,
        use_dram_slicing=True,
    )
