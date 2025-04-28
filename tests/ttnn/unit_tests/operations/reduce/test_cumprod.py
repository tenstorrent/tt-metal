# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.parametrize("dim", [-10, -1, 0, 3, 10])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        [0],
        [1],
        [3, 3],
        [16, 16],
        [32, 32, 32],
        [1, 0, 32, 32],
        [1000, 32, 32],
    ],
)
def test_cumprod_with_uint8(dim, shape, device):
    torch.manual_seed(22041997)

    input_tensor = torch.randint(0, 100, shape, dtype=torch.uint8)
    result_tensor_torch = torch.cumprod(input_tensor, 0)
    ttnn_tensor = ttnn.from_torch(input_tensor, ttnn.uint8, layout=ttnn.Layout.TILE, device=device)
    result_tensor = ttnn.experimental.cumprod(ttnn_tensor, dim)

    assert ttnn_tensor.shape == result_tensor.shape
    assert ttnn_tensor.dtype == result_tensor.dtype
    assert input_tensor.shape == ttnn_tensor.shape
    assert result_tensor_torch.shape == ttnn_tensor.shape
    assert result_tensor_torch.shape == result_tensor.shape


@pytest.mark.parametrize("dim", [-10, -1, 0, 3, 10])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        [0],
        [1],
        [3, 3],
        [16, 16],
        [32, 32, 32],
        [1, 0, 32, 32],
        [1000, 32, 32],
    ],
)
def test_cumprod_with_bfloat16(dim, shape, device):
    torch.manual_seed(22041997)

    input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    result_tensor_torch = torch.cumprod(input_tensor, 0)
    ttnn_tensor = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    result_tensor = ttnn.experimental.cumprod(ttnn_tensor, dim)

    assert ttnn_tensor.shape == result_tensor.shape
    assert ttnn_tensor.dtype == result_tensor.dtype
    assert input_tensor.shape == ttnn_tensor.shape
    assert result_tensor_torch.shape == ttnn_tensor.shape
    assert result_tensor_torch.shape == result_tensor.shape


@pytest.mark.parametrize("dim", [-10, -1, 0, 3, 10])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        [0],
        [1],
        [3, 3],
        [16, 16],
        [32, 32, 32],
        [1, 0, 32, 32],
        [1000, 32, 32],
    ],
)
def test_cumprod_with_bfloat16_and_preallocation(dim, shape, device):
    torch.manual_seed(22041997)

    input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    result_tensor_torch = torch.cumprod(input_tensor, 0)
    ttnn_tensor = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    preallocated_tensor = ttnn.zeros_like(ttnn_tensor)
    result_tensor = ttnn.experimental.cumprod(ttnn_tensor, dim, out=preallocated_tensor)

    assert ttnn_tensor.shape == result_tensor.shape
    assert ttnn_tensor.dtype == result_tensor.dtype
    assert preallocated_tensor.shape == result_tensor.shape
    assert preallocated_tensor.dtype == result_tensor.dtype
    assert input_tensor.shape == ttnn_tensor.shape
    assert result_tensor_torch.shape == ttnn_tensor.shape
    assert result_tensor_torch.shape == result_tensor.shape


@pytest.mark.parametrize("dim", [-10, -1, 0, 3, 10])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        [0],
        [1],
        [3, 3],
        [16, 16],
        [32, 32, 32],
        [1, 0, 32, 32],
        [1000, 32, 32],
    ],
)
def test_cumprod_with_uint8_and_preallocation(dim, shape, device):
    torch.manual_seed(22041997)

    input_tensor = torch.randint(0, 100, shape, dtype=torch.uint8)
    result_tensor_torch = torch.cumprod(input_tensor, 0)
    ttnn_tensor = ttnn.from_torch(input_tensor, ttnn.uint8, layout=ttnn.Layout.TILE, device=device)
    preallocated_tensor = ttnn.zeros_like(ttnn_tensor)
    result_tensor = ttnn.experimental.cumprod(ttnn_tensor, dim, out=preallocated_tensor)

    assert ttnn_tensor.shape == result_tensor.shape
    assert ttnn_tensor.dtype == result_tensor.dtype
    assert preallocated_tensor.shape == result_tensor.shape
    assert preallocated_tensor.dtype == result_tensor.dtype
    assert input_tensor.shape == ttnn_tensor.shape
    assert result_tensor_torch.shape == ttnn_tensor.shape
    assert result_tensor_torch.shape == result_tensor.shape
