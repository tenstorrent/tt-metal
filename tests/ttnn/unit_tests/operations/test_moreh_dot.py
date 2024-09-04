# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import comp_allclose_and_pcc
from tests.tt_eager.python_api_testing.unit_testing.misc.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
)


def create_tt_tensor(tensor, layout, dtype):
    return ttnn.from_torch(tensor, layout=layout, dtype=dtype)


def get_tensors(
    input_shape, other_shape, output_shape, require_input_grad, require_other_grad, is_1d, device, use_randint=True
):
    npu_dtype = ttnn.bfloat16
    cpu_dtype = torch.bfloat16
    npu_layout = ttnn.TILE_LAYOUT
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT

    # create tensors for forward
    if use_randint:
        input = torch.randint(-2, 3, input_shape, dtype=cpu_dtype)
        other = torch.randint(-2, 3, other_shape, dtype=cpu_dtype)
        output = torch.randint(-2, 3, output_shape, dtype=cpu_dtype)
    else:
        input = torch.rand(input_shape, dtype=cpu_dtype)
        other = torch.rand(other_shape, dtype=cpu_dtype)
        output = torch.rand(output_shape, dtype=cpu_dtype)

    tt_input = create_tt_tensor(input, cpu_layout, npu_dtype).pad_to_tile(float(1)).to(npu_layout).to(device)
    tt_other = create_tt_tensor(other, cpu_layout, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_output = create_tt_tensor(output, cpu_layout, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    torch_input = input.reshape(-1) if is_1d else input
    torch_other = other.reshape(-1) if is_1d else other

    # tensors for backward
    output_grad = tt_output_grad = torch_output_grad = tt_input_grad = tt_other_grad = None
    if require_input_grad or require_other_grad:
        output_grad = torch.randint(-2, 3, output_shape, dtype=cpu_dtype)
        tt_output_grad = (
            create_tt_tensor(output_grad, cpu_layout, npu_dtype).pad_to_tile(float(-1)).to(npu_layout).to(device)
        )
        torch_output_grad = output_grad[0][0][0][0] if is_1d else output_grad

        if require_input_grad:
            input_grad = torch.full(input_shape, float("nan"), dtype=cpu_dtype)
            tt_input_grad = (
                create_tt_tensor(input_grad, cpu_layout, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
            )

        if require_other_grad:
            other_grad = torch.full(other_shape, float("nan"), dtype=cpu_dtype)
            tt_other_grad = (
                create_tt_tensor(other_grad, cpu_layout, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
            )

    return (
        tt_input,
        tt_other,
        tt_output,
        tt_output_grad,
        tt_input_grad,
        tt_other_grad,
        torch_input,
        torch_other,
        torch_output_grad,
    )


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 1, 1, 10],  # test not mutiple of 32 case
        [1, 1, 1, 32],  # test single tile
        [1, 1, 1, 352],  # test multiple tiles
        [1, 1, 1, 323],  # test multiple tiles, not a multiple of 32
    ),
)
def test_moreh_matmul_1d(input_shape, device):
    torch.manual_seed(3072)
    # get tensors
    output_shape = [1, 1, 1, 1]
    tt_input, tt_other, _, _, _, _, torch_input, torch_other, _ = get_tensors(
        input_shape, input_shape, output_shape, False, False, True, device
    )

    # tt matmul
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_out = ttnn.moreh_dot(tt_input, tt_other).cpu().to(cpu_layout).unpad_from_tile(output_shape).to_torch()

    # torch matmul
    torch_input = torch.reshape(torch_input, (torch_input.shape[-1],))
    torch_other = torch.reshape(torch_other, (torch_other.shape[-1],))
    torch_out = torch.matmul(torch_input, torch_other)

    # test for equivalance
    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_out, tt_out[0][0][0][0], pcc=0.999, rtol=rtol, atol=atol)
    logger.debug(f"Out passing={passing}")
    logger.debug(f"Output pcc={output_pcc}")

    assert passing
