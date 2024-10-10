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
        [5, 96, 64],  # multiple tiles
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [3, -1],
)
def test_full_like_int(device, input_shape, fill_value):
    torch_input_tensor = torch.randint(0, 100, (input_shape), dtype=torch.int32)
    torch_output_tensor = torch.full_like(torch_input_tensor, fill_value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.full_like_2(input_tensor, fill_value)
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
def test_full_like_float(device, input_shape, fill_value, dtype):
    torch_input_tensor = torch.rand((input_shape), dtype=dtype)

    torch_output_tensor = torch.full_like(torch_input_tensor, fill_value)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.full_like_2(input_tensor, fill_value)
    assert ttnn.is_tensor_storage_on_device(output_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    passing, out = comp_allclose(torch_output_tensor, output_tensor, rtol=0.01, atol=0.01)
    logger.info(out)
    assert passing
