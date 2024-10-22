# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import copy
import torch
import torch.nn as nn
import ttnn
from models.utility_functions import comp_allclose
from loguru import logger

from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],  # single tile
        [5, 96, 64],  # multiple tiles
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [3, -1],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,  # Currently only support tile layout
    ],
)
def test_full_like_int(device, input_shape, fill_value, layout):
    torch_input_tensor = torch.randint(0, 100, (input_shape), dtype=torch.int32)
    torch_output_tensor = torch.full_like(torch_input_tensor, fill_value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.moreh_full_like(input_tensor, fill_value)
    assert ttnn.is_tensor_storage_on_device(output_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output_tensor, output_tensor)


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
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,  # Currently only support tile layout
    ],
)
def test_full_like_float(device, input_shape, fill_value, dtype, layout):
    torch_input_tensor = torch.rand((input_shape), dtype=dtype)
    torch_output_tensor = torch.full_like(torch_input_tensor, fill_value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.moreh_full_like(input_tensor, fill_value)
    assert ttnn.is_tensor_storage_on_device(output_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output_tensor, output_tensor)


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
def test_full_like_callback(device, input_shape, fill_value, layout, use_program_cache):
    for i in range(2):
        torch_input_tensor = torch.randint(0, 100, (input_shape), dtype=torch.int32)
        torch_output_tensor = torch.full_like(torch_input_tensor, fill_value)

        input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
        input_tensor = ttnn.to_device(input_tensor, device)
        output_tensor = ttnn.moreh_full_like(input_tensor, fill_value)
        assert ttnn.is_tensor_storage_on_device(output_tensor)
        output_tensor = ttnn.from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries
        torch_dummy = torch.randn([32, 32])
        tt_dummy = ttnn.from_torch(torch_dummy, device=device)

    assert_equal(torch_output_tensor, output_tensor)
