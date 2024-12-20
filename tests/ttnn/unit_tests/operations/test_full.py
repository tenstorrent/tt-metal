# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn as nn
import ttnn
from models.utility_functions import comp_allclose
from loguru import logger

from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 3],  # single tile
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
    torch_output = torch.full(input_shape, fill_value, dtype=torch.int32)

    any = ttnn.from_torch(torch_any, device=device, layout=ttnn.TILE_LAYOUT)
    tt_output = ttnn.moreh_full(input_shape, fill_value, any)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 3],  # single tile
        [32, 32],  # single tile
        [5, 96, 64],  # multiple tiles
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        3.14,
        2.00781250,  # mantissa: 0000 0001, bf16 round down test
        2.00830080,  # mantissa: 0000 0001 0001, bf16 round up test
        2.02343750,  # mantissa: 0000 0011, bf16 round up test
        -3.9921875,  # test mantissa overflow. answer should be 4
    ],
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

    torch_output = torch.full(input_shape, fill_value, dtype=dtype)
    any = ttnn.from_torch(torch_any, device=device, layout=ttnn.TILE_LAYOUT)
    tt_output = ttnn.moreh_full(input_shape, fill_value, any)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


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
        torch_output = torch.full(input_shape, fill_value, dtype=torch.int32)

        any = ttnn.from_torch(torch_any, device=device, layout=ttnn.TILE_LAYOUT)
        tt_output = ttnn.moreh_full(input_shape, fill_value, any)
        assert ttnn.is_tensor_storage_on_device(tt_output)
        tt_output_cpu = ttnn.to_torch(tt_output)
        torch_dummy = torch.randn([32, 32])
        ttnn_dummy = ttnn.from_torch(torch_dummy, device=device)
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries

        assert torch.equal(torch_output, tt_output_cpu)
