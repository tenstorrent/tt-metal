# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import tt_lib as ttl
from tt_lib.utils import _nearest_32
from models.utility_functions import comp_allclose_and_pcc
from tests.tt_eager.python_api_testing.sweep_tests.common import skip_for_wormhole_b0
from tests.tt_eager.python_api_testing.unit_testing.test_moreh_matmul import get_tensors
from loguru import logger


def shape_padded(shape):
    return [shape[0], shape[1], _nearest_32(shape[2]), _nearest_32(shape[3])]


def shape_1d_padded(shape):
    return [shape[0], shape[1], shape[2], _nearest_32(shape[3])]


# TODO: add this feature in get_tensors method
def get_bias_tensors(bias_shape, require_bias_grad, device):
    torch.manual_seed(2023)
    npu_dtype = ttl.tensor.DataType.BFLOAT16
    cpu_dtype = torch.bfloat16
    npu_layout = ttl.tensor.Layout.TILE
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR

    bias = torch.randint(-2, 3, bias_shape, dtype=cpu_dtype)
    tt_bias = (ttl.tensor.Tensor(
        bias.reshape(-1).tolist(), bias_shape, npu_dtype,
        cpu_layout).pad_to_tile(1).to(npu_layout).to(device))

    return tt_bias, bias, None


@skip_for_wormhole_b0
@pytest.mark.parametrize(
    "shapes",
    (
        ([1, 1, 1, 31], [1, 1, 30, 31], [1, 1, 1, 30], [1, 1, 1, 30]),
        ([1, 1, 1, 31], [1, 1, 30, 31], [1, 1, 1, 1], [1, 1, 1, 30
                                                       ]),  # scalar bias
        ([1, 1, 1, 2047], [1, 1, 1023, 2047], [1, 1, 1, 1023], [1, 1, 1, 1023
                                                                ]),
        ([1, 1, 32, 64], [1, 1, 1024, 64], [1, 1, 1, 1024], [1, 1, 32, 1024]),
        ([1, 1, 32, 1023], [1, 1, 2047, 1023], [1, 1, 1, 2047
                                                ], [1, 1, 32, 2047]),
        ([1, 1, 32, 1024], [1, 1, 2047, 1024
                            ], [1, 1, 1, 1], [1, 1, 32, 2047]),  # scalar bias
    ),
)
@pytest.mark.parametrize("has_bias", [False, True])
def test_run_moreh_linear(shapes, has_bias, device):
    input_shape, weight_shape, bias_shape, output_shape = shapes
    tt_input, tt_weight, _, _, _, torch_input, torch_weight, _ = get_tensors(
        input_shape, weight_shape, output_shape, False, False, False, device)

    if has_bias:
        tt_bias, torch_bias, _ = get_bias_tensors(bias_shape, False, device)
        tt_output = ttl.operations.primary.moreh_linear(
            tt_input, tt_weight, tt_bias)
    else:
        torch_bias = None
        tt_output = ttl.operations.primary.moreh_linear(tt_input, tt_weight)

    ## reference
    torch_output = torch.nn.functional.linear(torch_input, torch_weight[0][0],
                                              torch_bias)

    ## test for equivalance
    rtol = atol = 0.1
    cpu_layout = ttl.tensor.Layout.ROW_MAJOR
    ttcpu_output = tt_output.cpu().to(cpu_layout).unpad_from_tile(
        output_shape).to_torch()
    passing, output_pcc = comp_allclose_and_pcc(torch_output,
                                                ttcpu_output,
                                                pcc=0.999,
                                                rtol=rtol,
                                                atol=atol)
    logger.info(f"Passing = {passing}")
    logger.info(f"Output PCC = {output_pcc}")

    assert passing
