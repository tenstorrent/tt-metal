# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn as nn
import ttnn
from models.utility_functions import comp_allclose
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc, divup


@pytest.mark.parametrize(
    "input_shape",
    [
        [10],
        [32, 32],  # single tile
        [5, 17, 31],  # multiple tiles
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [3, -1],
)
def test_full_int(device, input_shape, fill_value):
    torch_any = torch.randint(0, 100, (input_shape), dtype=torch.int32)
    torch_output_tensor = torch.full(input_shape, fill_value)

    any = ttnn.from_torch(torch_any, layout=ttnn.TILE_LAYOUT)
    any = ttnn.to_device(any, device)
    output_tensor = ttnn.full_2(input_shape, fill_value, any)
    assert ttnn.is_tensor_storage_on_device(output_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    passing, out = comp_allclose(torch_output_tensor, output_tensor, rtol=0.01, atol=0.01)
    logger.info(out)
    assert passing


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],  # single tile
        [5, 96, 64],  # multiple tiles
        [3, 91, 67, 77],  # not multiple of 32
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [0.15, -1.2],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float32,
    ],
)
def test_full_float(device, input_shape, fill_value, dtype):
    torch_any = torch.rand((input_shape), dtype=dtype)

    torch_output_tensor = torch.full(input_shape, fill_value)
    any = ttnn.from_torch(torch_any, layout=ttnn.TILE_LAYOUT)
    any = ttnn.to_device(any, device)
    output_tensor = ttnn.full_2(input_shape, fill_value, any)
    assert ttnn.is_tensor_storage_on_device(output_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    passing, out = comp_allclose(torch_output_tensor, output_tensor, rtol=0.01, atol=0.01)
    logger.info(out)
    assert passing


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],  # single tile
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [3],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,  # Currently only support tile layout
    ],
)
def test_full_callback(device, input_shape, fill_value, layout, use_program_cache):
    for i in range(2):
        torch_any = torch.randint(0, 100, (input_shape), dtype=torch.int32)
        torch_output_tensor = torch.full(input_shape, fill_value)

        any = ttnn.from_torch(torch_any, layout=layout)
        any = ttnn.to_device(any, device)
        output_tensor = ttnn.full_2(input_shape, fill_value, any)
        assert ttnn.is_tensor_storage_on_device(output_tensor)
        output_tensor = ttnn.from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries
        passing, out = comp_allclose(torch_output_tensor, output_tensor, rtol=0.01, atol=0.01)
        logger.info(out)
        assert passing
