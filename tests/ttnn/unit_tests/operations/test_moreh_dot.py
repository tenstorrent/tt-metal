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
    to_ttnn,
)


def create_tt_tensor(tensor, layout, dtype):
    return ttnn.from_torch(tensor, layout=layout, dtype=dtype)


def get_torch_dtype(dtype):
    if dtype == ttnn.int32:
        return torch.int32
    elif dtype == ttnn.float32:
        return torch.float32
    else:
        return torch.bfloat16


def get_tensors(
    input_shape,
    other_shape,
    output_shape,
    require_input_grad,
    require_other_grad,
    is_1d,
    device,
    npu_dtype,
    cpu_dtype,
    use_randint=True,
):
    npu_layout = ttnn.TILE_LAYOUT
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT

    # create tensors for forward
    if use_randint:
        input = torch.randint(-2, 3, input_shape, dtype=torch.bfloat16)
        other = torch.randint(-2, 3, other_shape, dtype=torch.bfloat16)
        output = torch.randint(-2, 3, output_shape, dtype=torch.bfloat16)
    else:
        input = torch.rand(input_shape, dtype=torch.bfloat16)
        other = torch.rand(other_shape, dtype=torch.bfloat16)
        output = torch.rand(output_shape, dtype=torch.bfloat16)

    # inputs must be of type bfloat16 or bfloat8
    tt_input = create_tt_tensor(input, cpu_layout, ttnn.bfloat16).pad_to_tile(float(1)).to(npu_layout).to(device)
    tt_other = create_tt_tensor(other, cpu_layout, ttnn.bfloat16).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_output = create_tt_tensor(output, cpu_layout, ttnn.bfloat16).pad_to_tile(float("nan")).to(npu_layout).to(device)

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
@pytest.mark.parametrize(
    "dtype",
    (
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.int32,
        None,
    ),
)
def test_moreh_matmul_1d(input_shape, dtype, device):
    torch.manual_seed(3072)
    # get tensors
    output_shape = [1, 1, 1, 1]
    if dtype is None:
        npu_dtype = torch.bfloat16
    npu_dtype = dtype
    cpu_dtype = get_torch_dtype(dtype)

    tt_input, tt_other, _, _, _, _, torch_input, torch_other, _ = get_tensors(
        input_shape, input_shape, output_shape, False, False, True, device, npu_dtype, cpu_dtype
    )

    # tt matmul
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_out = (
        ttnn.operations.moreh.dot(tt_input, tt_other, dtype=dtype)
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

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
    "dtype",
    (
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.int32,
        None,
    ),
)
def test_moreh_matmul_1d_callback(input_shape, dtype, device, use_program_cache):
    torch.manual_seed(3072)
    # get tensors
    output_shape = [1, 1, 1, 1]
    if dtype is None:
        npu_dtype = torch.bfloat16
    npu_dtype = dtype
    cpu_dtype = get_torch_dtype(dtype)

    # tt matmul
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    for i in range(2):
        tt_input, tt_other, _, _, _, _, torch_input, torch_other, _ = get_tensors(
            input_shape, input_shape, output_shape, False, False, True, device, npu_dtype, cpu_dtype
        )
        tt_out = (
            ttnn.operations.moreh.dot(tt_input, tt_other, dtype=dtype)
            .cpu()
            .to(cpu_layout)
            .unpad_from_tile(output_shape)
            .to_torch()
        )
        torch_dummy = torch.randn([32, 32])
        tt_dummy = to_ttnn(torch_dummy, device=device)
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries

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


@pytest.mark.parametrize(
    "input_shape",
    ([1, 1, 1, 10],),  # test not mutiple of 32 case
)
@pytest.mark.parametrize(
    "dtype",
    (
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.int32,
        None,
    ),
)
def test_moreh_dot_optional_output_tensor(input_shape, dtype, device):
    torch.manual_seed(3072)
    # get tensors
    output_shape = [1, 1, 1, 1]
    if dtype is None:
        npu_dtype = torch.bfloat16
    npu_dtype = dtype
    cpu_dtype = get_torch_dtype(dtype)

    tt_input, tt_other, optional_output_tensor, _, _, _, torch_input, torch_other, _ = get_tensors(
        input_shape, input_shape, output_shape, False, False, True, device, npu_dtype, cpu_dtype
    )
    # tt matmul
    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_out = (
        ttnn.operations.moreh.dot(tt_input, tt_other, output=optional_output_tensor, dtype=dtype)
        .cpu()
        .to(cpu_layout)
        .unpad_from_tile(output_shape)
        .to_torch()
    )

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
