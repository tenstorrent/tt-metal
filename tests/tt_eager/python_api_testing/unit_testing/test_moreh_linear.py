# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import tt_lib as ttl
from tt_lib.utils import _nearest_32
from models.utility_functions import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.common import skip_for_wormhole_b0

TILE_HEIGHT = TILE_WIDTH = 32


def shape_padded(shape):
    return [shape[0], shape[1], _nearest_32(shape[2]), _nearest_32(shape[3])]


def shape_1d_padded(shape):
    return [shape[0], shape[1], shape[2], _nearest_32(shape[3])]


@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "shapes",
    (
        ([1, 1, 1, 31], [1, 1, 30, 31], [1, 1, 1, 30], [1, 1, 1, 30]),
        ([1, 1, 1, 31], [1, 1, 30, 31], [1, 1, 1, 1], [1, 1, 1, 30]), # scalar bias
        ([1, 1, 1, 2047], [1, 1, 1023, 2047], [1, 1, 1, 1023], [1, 1, 1, 1023]),
        ([1, 1, 32, 64], [1, 1, 1024, 64], [1, 1, 1, 1024], [1, 1, 32, 1024]),
        ([1, 1, 32, 2047], [1, 1, 4095, 2047], [1, 1, 1, 4095], [1, 1, 32, 4095]),
        ([1, 1, 32, 2047], [1, 1, 4095, 2047], [1, 1, 1, 1], [1, 1, 32, 4095]), # scalar bias
    ),
)
@pytest.mark.parametrize("has_bias", [False, True])
def test_run_moreh_linear(shapes, has_bias, device):
    input_shape, weight_shape, bias_shape, out_shape = shapes

    dtype = ttl.tensor.DataType.BFLOAT16
    npu_layout = ttl.tensor.Layout.TILE
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR

    torch.manual_seed(0)
    input = torch.randn(input_shape, dtype=torch.bfloat16)
    weight = torch.randn(weight_shape, dtype=torch.bfloat16)

    tt_input = ttl.tensor.Tensor(torch.flatten(input).tolist(), input_shape, dtype, cpu_layout)
    tt_weight = ttl.tensor.Tensor(torch.flatten(weight).tolist(), weight_shape, dtype, cpu_layout)

    input_shape_padded = shape_padded(input_shape)
    weight_shape_padded = shape_padded(weight_shape)
    out_shape_padded = shape_padded(out_shape)

    if input_shape != input_shape_padded:
        tt_input = tt_input.pad_to_tile(0.0)
    if weight_shape != weight_shape_padded:
        tt_weight = tt_weight.pad_to_tile(float("nan"))

    tt_input = tt_input.to(npu_layout).to(device)
    tt_weight = tt_weight.to(npu_layout).to(device)

    if has_bias:
        bias_shape_1d_padded = shape_1d_padded(bias_shape)
        bias = torch.randn(bias_shape, dtype=torch.bfloat16)
        tt_bias = ttl.tensor.Tensor(torch.flatten(bias).tolist(), bias_shape, dtype, cpu_layout)
        if bias_shape != bias_shape_1d_padded:
            tt_bias = tt_bias.pad_to_tile(1.0)

        out = ttl.tensor.moreh_linear(tt_input, tt_weight, tt_bias)
    else:
        bias = None
        out = ttl.tensor.moreh_linear(tt_input, tt_weight)

    out = out.cpu().to(cpu_layout)
    if out_shape != out_shape_padded:
        out = out.unpad_from_tile(out_shape)
    out_pytorch = out.to_torch()

    ## reference
    golden_pytorch = torch.nn.functional.linear(input, weight[0][0], bias)

    ## test for equivalance
    passing_pcc, output_pcc = comp_pcc(golden_pytorch, out_pytorch, pcc=0.999)
    print(f"Passing PCC = {passing_pcc}")
    print(f"Output PCC = {output_pcc}")

    assert passing_pcc
