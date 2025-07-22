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


def get_torch_dtype(dtype):
    if dtype == ttnn.int32:
        return torch.int32
    else:
        return torch.bfloat16


def run_moreh_dot_test(input_shape, ttnn_dtype, device, use_optional_output=False):
    # TODO @thanhnguyen-moreh: Support bfloat8_b in kernel
    if ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip(f"bfloat8_b is not supported in the kernel")
    torch_dtype = get_torch_dtype(ttnn_dtype)
    output_shape = [1, 1, 1, 1]

    if ttnn_dtype == ttnn.int32:
        torch_input = torch.randint(-2, 3, input_shape, dtype=torch_dtype)
        torch_other = torch.randint(-2, 3, input_shape, dtype=torch_dtype)
        torch_optional_output = torch.randint(-2, 3, output_shape, dtype=torch_dtype)

        tt_input = ttnn.from_torch(
            torch_input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )  # Input dtype must be of bfloat16 or bfloat8_b
        tt_other = ttnn.from_torch(
            torch_other, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )  # Input dtype must be of bfloat16 or bfloat8_b

    else:
        torch_input = torch.rand(input_shape, dtype=torch_dtype)
        torch_other = torch.rand(input_shape, dtype=torch_dtype)
        torch_optional_output = torch.rand(output_shape, dtype=torch_dtype)

        tt_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT)
        tt_other = ttnn.from_torch(torch_other, device=device, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT)

    optional_output = (
        ttnn.from_torch(torch_optional_output, device=device, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT)
        if use_optional_output
        else None
    )

    tt_out = ttnn.operations.moreh.dot(tt_input, tt_other, dtype=ttnn_dtype, output=optional_output)
    tt_out = ttnn.to_torch(tt_out).to(torch_dtype)

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
        ttnn.bfloat16,
        ttnn.int32,
        ttnn.bfloat8_b,
    ),
)
def test_moreh_dot(input_shape, dtype, device):
    torch.manual_seed(3072)
    run_moreh_dot_test(input_shape, dtype, device)


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
        ttnn.bfloat16,
        ttnn.int32,
        None,
    ),
)
def test_moreh_matmul_1d_callback(input_shape, dtype, device):
    torch.manual_seed(3072)

    for i in range(2):
        run_moreh_dot_test(input_shape, dtype, device)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = ttnn.from_torch(torch_dummy, device=device)
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries


@pytest.mark.parametrize(
    "input_shape",
    ([1, 1, 1, 10],),  # test not mutiple of 32 case
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
)
def test_moreh_dot_optional_output_tensor(input_shape, dtype, device):
    torch.manual_seed(3072)
    run_moreh_dot_test(input_shape, dtype, device, use_optional_output=True)
