# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn


@pytest.mark.parametrize(
    "size, dim",
    [
        ([], 0),
        ([0], 0),
        ([1], 0),
        ([10], 0),
        ([2, 3], 0),
        ([2, 3], 1),
        ([2, 3], -1),
        ([2, 3], -2),
        ([2, 3, 4], 0),
        ([2, 3, 4], 2),
        ([2, 3, 4], -3),
        ([0, 0, 0], 0),
        ([0, 0, 0], 1),
        ([1, 32, 64], 1),
        ([1, 1024, 32], 0),
        ([1, 1024, 32], 1),
        ([1, 1024, 32], 2),
        ([64, 1, 32], 1),
        ([64, 64, 1], 1),
        ([1, 32, 129], 1),
        ([33, 35, 37], 1),
    ],
)
@pytest.mark.parametrize("dtype", [None, ttnn.bfloat16, ttnn.float32])
def test_cumsum(size, dim, dtype, device):
    torch_input_tensor = torch.rand(size, dtype=torch.float32)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device)

    if dtype is not None:
        output_tensor = ttnn.experimental.cumsum(input_tensor, dim=dim, dtype=dtype)
    else:
        output_tensor = ttnn.experimental.cumsum(input_tensor, dim)

    assert output_tensor.shape == (size)

    if dtype is not None:
        assert output_tensor.dtype == dtype
    else:
        assert output_tensor.dtype == input_tensor.dtype


@pytest.mark.parametrize(
    "size, dim",
    [
        ([], 0),
        ([0], 0),
        ([1], 0),
        ([10], 0),
        ([2, 3], 0),
        ([2, 3], 1),
        ([2, 3], -1),
        ([2, 3], -2),
        ([2, 3, 4], 0),
        ([2, 3, 4], 2),
        ([2, 3, 4], -3),
        ([0, 0, 0], 0),
        ([0, 0, 0], 1),
        ([1, 32, 64], 1),
        ([1, 1024, 32], 0),
        ([1, 1024, 32], 1),
        ([1, 1024, 32], 2),
        ([64, 1, 32], 1),
        ([64, 64, 1], 1),
        ([1, 32, 129], 1),
        ([33, 35, 37], 1),
    ],
)
@pytest.mark.parametrize("dtype", [None, ttnn.bfloat16, ttnn.float32])
def test_cumsum_with_preallocated_output(size, dim, dtype, device):
    torch_input_tensor = torch.rand(size, dtype=torch.float32)
    torch_output_tensor = torch.rand(size, dtype=torch.float32)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device)

    if dtype is not None:
        preallocated_output_tensor = ttnn.from_torch(torch_output_tensor, device=device, dtype=dtype)
        output_tensor = ttnn.experimental.cumsum(input_tensor, dim=dim, dtype=dtype, output=preallocated_output_tensor)
    else:
        preallocated_output_tensor = ttnn.from_torch(torch_output_tensor, device=device)
        output_tensor = ttnn.experimental.cumsum(input_tensor, dim=dim, output=preallocated_output_tensor)

    assert output_tensor.shape == (size)

    if dtype is not None:
        assert output_tensor.dtype == dtype
