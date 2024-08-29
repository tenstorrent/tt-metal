# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import random

import pytest
import torch

import ttnn
from models.utility_functions import comp_allclose_and_pcc
from loguru import logger

from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import TILE_HEIGHT, TILE_WIDTH


def to_cpu(npu_tensor, shape, *, cpu_layout=ttnn.ROW_MAJOR_LAYOUT):
    if npu_tensor is None:
        return None
    cpu_tensor = npu_tensor.cpu().to(cpu_layout).unpad_from_tile(shape).to_torch()
    return cpu_tensor


def to_npu(
    cpu_tensor,
    device,
    *,
    npu_layout=ttnn.TILE_LAYOUT,
    npu_dtype=ttnn.bfloat16,
    padding_value=float("nan"),
):
    if cpu_tensor is None:
        return None
    npu_tensor = ttnn.Tensor(cpu_tensor, npu_dtype).pad_to_tile(padding_value).to(npu_layout).to(device)
    return npu_tensor


@pytest.mark.skip(reason="assertion fails during binary op input shape comparison because of different padding")
@pytest.mark.parametrize("num_iters_of_each_case", [2])
@pytest.mark.parametrize("range_of_padding", [(0, 21, 10)])  # [0, 10, 20]
@pytest.mark.parametrize("range_of_n", [(1, 4)])
@pytest.mark.parametrize("range_of_c", [(1, 4)])
@pytest.mark.parametrize("range_of_ht", [(1, 4)])
@pytest.mark.parametrize("range_of_wt", [(1, 4)])
@pytest.mark.parametrize("max_norm", [2.0, 1.0, -1.0])
@pytest.mark.parametrize("norm_type", [2.0, -0.8, 2.2])
@pytest.mark.parametrize("num_parameters", [32, 128])
def test_moreh_clip_grad_norm(
    num_iters_of_each_case,
    num_parameters,
    max_norm,
    norm_type,
    range_of_n,
    range_of_c,
    range_of_ht,
    range_of_wt,
    range_of_padding,
    device,
):
    torch.manual_seed(2023)
    random.seed(2023)

    cpu_dtype = torch.float32
    npu_dtype = ttnn.bfloat16

    cpu_inputs = []
    npu_inputs = []
    input_shapes = []

    for _ in range(num_iters_of_each_case):
        for _ in range(num_parameters):
            n = random.randint(*range_of_n)
            c = random.randint(*range_of_c)
            ht = random.randint(*range_of_ht)
            wt = random.randint(*range_of_wt)
            padding_h = random.randrange(*range_of_padding)
            padding_w = random.randrange(*range_of_padding)

            input_shape = (
                n,
                c,
                ht * TILE_HEIGHT - padding_h,
                wt * TILE_WIDTH - padding_w,
            )

            param = torch.nn.Parameter(torch.empty(input_shape, dtype=cpu_dtype))
            grad = torch.empty(input_shape, dtype=cpu_dtype).uniform_(0, 2.5)
            param.grad = grad

            cpu_inputs.append(param)
            npu_inputs.append(to_npu(grad.clone().bfloat16(), device, npu_dtype=npu_dtype))
            input_shapes.append(input_shape)

        cpu_total_norm = torch.nn.utils.clip_grad_norm_(cpu_inputs, max_norm, norm_type)
        npu_total_norm = ttnn.experimental.operations.primary.moreh_clip_grad_norm_(npu_inputs, max_norm, norm_type)

        expected_total_norm = cpu_total_norm
        actual_total_norm = to_cpu(npu_total_norm, [1, 1, 1, 1])

        rtol = atol = 0.1
        # Check total_norm
        pass_total_norm, out_total_norm = comp_allclose_and_pcc(
            actual_total_norm.double(), expected_total_norm.double(), rtol=rtol, atol=atol
        )
        logger.debug(f"total_norm's {out_total_norm}")
        assert pass_total_norm

        # Check inputs
        for i in range(num_parameters):
            expected_input_i = cpu_inputs[i].grad.double()
            actual_input_i = to_cpu(npu_inputs[i], input_shapes[i]).double()
            pass_input_i, out_input_i = comp_allclose_and_pcc(expected_input_i, actual_input_i, rtol=rtol, atol=atol)
            logger.debug(f"inputs[{i}]-shape[{input_shapes[i]}]'s {out_input_i}")
            assert pass_input_i


# @pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
# @pytest.mark.parametrize("error_if_nonfinite", [True, False])
# def test_moreh_clip_grad_norm_with_error_if_nonfinite(error_if_nonfinite, device):
#     torch.manual_seed(2023)

#     cpu_dtype = torch.bfloat16
#     npu_dtype = ttnn.bfloat16

#     input_shape = [4, 4, 4 * TILE_HEIGHT, 4 * TILE_WIDTH]
#     param = torch.nn.Parameter(torch.empty(input_shape, dtype=cpu_dtype))
#     grad = torch.randn(input_shape, dtype=cpu_dtype)
#     param.grad = grad

#     max_norm = 1.0
#     norm_type = float("nan")

#     expected_error_msg = (
#         f"The total norm of order {norm_type} for gradients from `parameters` is non-finite, so it cannot be clipped."
#     )

#     # Check vanilla torch behavior
#     try:
#         torch.nn.utils.clip_grad_norm_((param), max_norm, norm_type, error_if_nonfinite)
#         assert not error_if_nonfinite
#     except RuntimeError as actual_error_msg:
#         assert expected_error_msg in str(actual_error_msg)
#         assert error_if_nonfinite

#     # Check tt behavior
#     try:
#         ttnn.experimental.operations.primary.moreh_clip_grad_norm_(
#             [to_npu(param.grad.bfloat16(), device, npu_dtype=npu_dtype)], max_norm, norm_type, error_if_nonfinite
#         )
#         assert not error_if_nonfinite
#     except RuntimeError as actual_error_msg:
#         assert expected_error_msg in str(actual_error_msg)
#         assert error_if_nonfinite
