# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import tt_dtype_to_torch_dtype

ALL_TYPES = [dtype for dtype, _ in ttnn.DataType.__entries.values() if dtype != ttnn.DataType.INVALID]
TEST_SHAPES = [(5, 5), (32, 32), (50, 50), (16, 16, 16), (16, 16, 16, 16), (16, 16, 16, 16, 16)]


@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype", ALL_TYPES)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("is_device", [False, True])
def test_to_list_all_types(device, is_device, shape, dtype, layout):
    if dtype == ttnn.bfloat4_b or dtype == ttnn.bfloat8_b and layout != ttnn.TILE_LAYOUT:
        pytest.skip("types `bfloat4_b` and `bfloat8_b` can only be used with tile layout")

    torch.manual_seed(0)
    torch_tensor = torch.randint(0, 100, shape, dtype=tt_dtype_to_torch_dtype[dtype])
    torch_list = torch_tensor.tolist()

    ttnn_tensor = ttnn.from_torch(
        torch_tensor, device=device if is_device == True else None, dtype=dtype, layout=layout
    )
    ttnn_list = ttnn_tensor.to_list()

    assert ttnn_list == torch_list
