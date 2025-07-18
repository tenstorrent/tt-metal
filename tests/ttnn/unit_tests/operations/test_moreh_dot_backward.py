# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    get_ttnn_torch_dtype,
)


def get_tensors(
    input_shape,
    other_shape,
    output_shape,
    require_input_grad,
    require_other_grad,
    is_1d,
    device,
    npu_dtype=ttnn.bfloat16,
    use_randint=True,
):
    cpu_dtype = get_ttnn_torch_dtype(npu_dtype)
    if cpu_dtype is None:
        # panic
        assert False

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

    tt_input = ttnn.from_torch(input, npu_dtype, layout=npu_layout, device=device)
    tt_other = ttnn.from_torch(other, npu_dtype, layout=npu_layout, device=device)
    tt_output = ttnn.from_torch(output, npu_dtype, layout=npu_layout, device=device)

    torch_input = input.reshape(-1) if is_1d else input
    torch_other = other.reshape(-1) if is_1d else other

    # tensors for backward
    output_grad = tt_output_grad = torch_output_grad = tt_input_grad = tt_other_grad = None
    if require_input_grad or require_other_grad:
        output_grad = torch.randint(-2, 3, output_shape, dtype=cpu_dtype)
        tt_output_grad = ttnn.from_torch(output_grad, npu_dtype, layout=npu_layout, device=device)
        torch_output_grad = output_grad[0][0][0][0] if is_1d else output_grad

        if require_input_grad:
            input_grad = torch.full(input_shape, float("nan"), dtype=cpu_dtype)
            tt_input_grad = ttnn.from_torch(input_grad, npu_dtype, layout=npu_layout, device=device)

        if require_other_grad:
            other_grad = torch.full(other_shape, float("nan"), dtype=cpu_dtype)
            tt_other_grad = ttnn.from_torch(other_grad, npu_dtype, layout=npu_layout, device=device)

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


def run_moreh_dot_backward(input_shape, requires_grad, device, dtype=ttnn.bfloat16, use_randint=True):
    torch.manual_seed(3072)
    require_input_grad, require_other_grad = requires_grad
    output_shape = [1, 1, 1, 1]
    # get tensors
    (
        tt_input,
        tt_other,
        _,
        tt_output_grad,
        tt_input_grad,
        tt_other_grad,
        torch_input,
        torch_other,
        torch_output_grad,
    ) = get_tensors(
        input_shape, input_shape, output_shape, require_input_grad, require_other_grad, True, device, dtype, use_randint
    )
    # torch matmul
    torch_out = torch.matmul(
        torch_input.requires_grad_(require_input_grad), torch_other.requires_grad_(require_other_grad)
    )
    torch_out.backward(torch_output_grad)

    # tt matmul backward
    ttnn.operations.moreh.dot_backward(
        tt_output_grad, tt_input, tt_other, input_grad=tt_input_grad, other_grad=tt_other_grad
    )

    # test for equivalance
    rtol = atol = 0.1
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    if require_input_grad:
        ttcpu_input_grad = ttnn.to_torch(tt_input_grad)

        passing, output_pcc = comp_allclose_and_pcc(
            torch_input.grad, ttcpu_input_grad.reshape(-1), pcc=0.999, rtol=rtol, atol=atol
        )
        logger.debug(f"input_grad passing={passing}")
        logger.debug(f"input_grad pcc={output_pcc}")
        assert passing

    if require_other_grad:
        ttcpu_other_grad = ttnn.to_torch(tt_other_grad)

        passing, output_pcc = comp_allclose_and_pcc(
            torch_other.grad, ttcpu_other_grad.reshape(-1), pcc=0.999, rtol=rtol, atol=atol
        )
        logger.debug(f"other_grad passing={passing}")
        logger.debug(f"other_grad pcc={output_pcc}")
        assert passing


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 1, 1, 10],  # test not mutiple of 32 case
        [1, 1, 1, 32],  # test single tile
        [1, 1, 1, 352],  # test multiple tiles
        [1, 1, 1, 323],  # test multiple tiles, not a multiple of 32
    ),
)
@pytest.mark.parametrize(
    "requires_grad",
    (
        [True, False],
        [False, True],
        [True, True],
    ),
)
@pytest.mark.parametrize("use_randint", (True, False))
@pytest.mark.parametrize("dtype", ([ttnn.bfloat16, ttnn.bfloat8_b]))
def test_moreh_dot_backward(input_shape, requires_grad, dtype, use_randint, device):
    run_moreh_dot_backward(input_shape, requires_grad, device, dtype, use_randint)


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 1, 1, 10],  # test not mutiple of 32 case
        [1, 1, 1, 32],  # test single tile
        [1, 1, 1, 352],  # test multiple tiles
        [1, 1, 1, 323],  # test multiple tiles, not a multiple of 32
    ),
)
@pytest.mark.parametrize(
    "requires_grad",
    (
        [True, False],
        [False, True],
        [True, True],
    ),
)
def test_moreh_dot_backward_callback(
    input_shape,
    requires_grad,
    device,
):
    num_program_in_cache = []
    for i in range(2):
        run_moreh_dot_backward(input_shape, requires_grad, device)
        num_program_in_cache.append(device.num_program_cache_entries())
        dummy = torch.randn([32, 32])
        tt_dummy = ttnn.from_torch(dummy, device=device)

    logger.info(f"num_program_in_cache={num_program_in_cache}")
    assert num_program_in_cache[0] > 0
    assert num_program_in_cache[0] == num_program_in_cache[1]
