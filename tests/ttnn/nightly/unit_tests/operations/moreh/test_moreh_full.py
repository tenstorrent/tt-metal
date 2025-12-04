# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn as nn
import ttnn
from models.common.utility_functions import comp_allclose
from loguru import logger

from tests.ttnn.utils_for_testing import assert_equal

torch_dtype_to_ttnn_dtype = {
    torch.int32: ttnn.int32,
    torch.bfloat16: ttnn.bfloat16,
    torch.float32: ttnn.float32,
}


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
    torch_output = torch.full(input_shape, fill_value, dtype=torch.int32)

    tt_output = ttnn.moreh_full(input_shape, fill_value, device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT)
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
    ids=["pi", "bf16_round_down", "bf16_round_up1", "bf16_round_up2", "bf16_mantissa_overflow"],
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
        ttnn.TILE_LAYOUT,
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
def test_full_float(device, input_shape, fill_value, dtype, layout):
    torch_output = torch.full(input_shape, fill_value, dtype=dtype)
    ttnn_dtype = torch_dtype_to_ttnn_dtype[dtype]
    tt_output = ttnn.moreh_full(input_shape, fill_value, device, dtype=ttnn_dtype, layout=layout)
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
    [3, 0],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,  # Currently only support tile layout
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
def test_full_callback(device, input_shape, fill_value, layout):
    for i in range(2):
        torch_output = torch.full(input_shape, fill_value, dtype=torch.int32)

        ttnn_dtype = torch_dtype_to_ttnn_dtype[torch.int32]
        tt_output = ttnn.moreh_full(input_shape, fill_value, device, dtype=ttnn_dtype, layout=layout)
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


@pytest.mark.parametrize(
    "input_shape",
    [
        [8, 1, 1, 7168],
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [
        0.0,
        1.0,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
    ],
)
def test_big_full(device, input_shape, fill_value, dtype, layout):
    torch_output = torch.full(input_shape, fill_value, dtype=dtype)
    ttnn_dtype = torch_dtype_to_ttnn_dtype[dtype]
    for i in range(10):
        tt_output = ttnn.moreh_full(input_shape, fill_value, device, dtype=ttnn_dtype, layout=layout)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)
