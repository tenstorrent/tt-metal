# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        [10],
        [10000],
        [3, 3],
        [16, 16],
        [1, 0, 32, 32],
        [1, 1, 1, 1],
        [16, 8, 4, 16],
        [1000, 32, 32],
        [3, 4, 5, 4, 3],
        [3, 4, 5, 4, 1, 2, 1],
    ],
)
def test_cumprod_mix_params(dim, shape, device):
    torch.manual_seed(22041997)

    torch_input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    torch_result_tensor = torch.cumprod(torch_input_tensor, dim)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_result_tensor = ttnn.experimental.cumprod(ttnn_input_tensor, dim)

    # assert metadata
    assert ttnn_input_tensor.shape == ttnn_result_tensor.shape
    assert ttnn_input_tensor.dtype == ttnn_result_tensor.dtype
    assert torch_input_tensor.shape == ttnn_input_tensor.shape
    assert torch_result_tensor.shape == ttnn_input_tensor.shape
    assert torch_result_tensor.shape == ttnn_result_tensor.shape

    # assert values with pcc
    assert_with_pcc(ttnn.to_torch(ttnn_result_tensor), torch_result_tensor, -0.1)


@pytest.mark.parametrize("dim", [0, 2, -1, -3])
@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 1, 1],
        [16, 8, 4, 16],
        [1000, 32, 32],
        [1, 1, 1],
        [3, 4, 5, 4, 3],
        [3, 4, 5, 4, 1, 2, 1],
    ],
)
def test_cumprod_mix_params_min_3d(dim, shape, device):
    torch.manual_seed(22041997)

    torch_input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    torch_result_tensor = torch.cumprod(torch_input_tensor, dim)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_result_tensor = ttnn.experimental.cumprod(ttnn_input_tensor, dim)

    # assert metadata
    assert ttnn_input_tensor.shape == ttnn_result_tensor.shape
    assert ttnn_input_tensor.dtype == ttnn_result_tensor.dtype
    assert torch_input_tensor.shape == ttnn_input_tensor.shape
    assert torch_result_tensor.shape == ttnn_input_tensor.shape
    assert torch_result_tensor.shape == ttnn_result_tensor.shape

    # assert values with pcc
    assert_with_pcc(ttnn.to_torch(ttnn_result_tensor), torch_result_tensor, -0.1)


@pytest.mark.parametrize("dim", [0, 1, 3, -1, -4])
@pytest.mark.parametrize(
    "shape",
    [
        [1, 32, 1, 32],
        [32, 1, 64, 1],
        [3, 4, 5, 4, 3],
        [3, 4, 5, 4, 1, 2, 1],
    ],
)
def test_cumprod_mix_params_min_4d(dim, shape, device):
    torch.manual_seed(22041997)

    torch_input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    torch_result_tensor = torch.cumprod(torch_input_tensor, dim)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_result_tensor = ttnn.experimental.cumprod(ttnn_input_tensor, dim)

    # assert metadata
    assert ttnn_input_tensor.shape == ttnn_result_tensor.shape
    assert ttnn_input_tensor.dtype == ttnn_result_tensor.dtype
    assert torch_input_tensor.shape == ttnn_input_tensor.shape
    assert torch_result_tensor.shape == ttnn_input_tensor.shape
    assert torch_result_tensor.shape == ttnn_result_tensor.shape

    # assert values with pcc
    assert_with_pcc(ttnn.to_torch(ttnn_result_tensor), torch_result_tensor, -0.1)


@pytest.mark.parametrize("dim", [0, 1, 3, -1, -4])
@pytest.mark.parametrize(
    "shape",
    [
        [1, 32, 1, 32],
        [32, 1, 64, 1],
        [3, 4, 5, 4, 3],
        [3, 4, 5, 4, 1, 2, 1],
    ],
)
def test_cumprod_mix_params_min_4d_preallocated(dim, shape, device):
    torch.manual_seed(22041997)

    torch_input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    torch_preallocated_tensor = torch.zeros_like(torch_input_tensor)
    torch_result_tensor = torch.cumprod(torch_input_tensor, dim, out=torch_preallocated_tensor)
    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_preallocated_tensor = ttnn.zeros_like(ttnn_input_tensor)
    ttnn_result_tensor = ttnn.experimental.cumprod(ttnn_input_tensor, dim, out=ttnn_preallocated_tensor)

    # assert metadata
    assert ttnn_input_tensor.shape == ttnn_result_tensor.shape
    assert ttnn_preallocated_tensor.shape == ttnn_result_tensor.shape
    assert ttnn_preallocated_tensor.dtype == ttnn_result_tensor.dtype
    assert ttnn_input_tensor.dtype == ttnn_result_tensor.dtype
    assert torch_input_tensor.shape == ttnn_input_tensor.shape
    assert torch_result_tensor.shape == ttnn_input_tensor.shape
    assert torch_result_tensor.shape == ttnn_result_tensor.shape

    # assert values with pcc
    assert_with_pcc(ttnn.to_torch(ttnn_result_tensor), torch_result_tensor, -0.1)
    assert_with_pcc(ttnn.to_torch(ttnn_preallocated_tensor), torch_preallocated_tensor, -0.1)
