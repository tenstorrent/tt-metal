# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn as nn
import ttnn
from models.utility_functions import comp_allclose
from loguru import logger

from tests.ttnn.utils_for_testing import assert_equal, tt_dtype_to_torch_dtype


@pytest.mark.parametrize(
    "input_shape",
    [
        [3],  # single tile with rank 1
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

    tt_output = ttnn.full(input_shape, fill_value, layout=ttnn.TILE_LAYOUT, device=device)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    assert torch.equal(torch_output, tt_output_cpu)


@pytest.mark.parametrize(
    "input_shape",
    [
        [3],  # single tile with rank 1
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
    "tt_dtype",
    [
        ttnn.bfloat16,
        ttnn.float32,
    ],
)
def test_full_float(device, input_shape, fill_value, tt_dtype):
    torch_dtype = tt_dtype_to_torch_dtype[tt_dtype]
    torch_output = torch.full(input_shape, fill_value, dtype=torch_dtype)

    tt_output = ttnn.full(input_shape, fill_value, dtype=tt_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    assert ttnn.is_tensor_storage_on_device(tt_output)
    tt_output_cpu = ttnn.to_torch(tt_output)

    # TODO (issue #16579): Investigate why ttnn.full isn't exact match while ttnn.moreh_full is correct for ttnn.bfloat16
    if tt_dtype == ttnn.bfloat16:
        pytest.xfail("ttnn.full does not have exact match if dtype is ttnn.bfloat16")
    else:
        assert torch.equal(torch_output, tt_output_cpu)


# TODO (issue #16579): Add program cache test when ttnn.full is run on device
