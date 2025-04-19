# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


# this does not work.
# @pytest.mark.parametrize("dim", [-10, -1, 0, 3, 10])
# @pytest.mark.parametrize(
#     "shape",
#     [
#         [],
#         [0],
#         [1],
#         [3, 3],
#         [16, 16],
#         [32, 32, 32],
#         [1, 0, 32, 32],
#         [1, 1, 1, 1],
#         [16, 8, 4, 16],
#         [1000, 32, 32],
#     ],
# )
# @pytest.mark.parametrize("dtype", [ttnn.DataType.UINT8, ttnn.DataType.UINT16, ttnn.DataType.UINT32])
# def test_cumprod_scaffold_with_uint(dim, shape, dype, device):
#     torch.manual_seed(22041997)

#     input_tensor = torch.randint(0, 100, shape, dtype=torch.uint8)
#     result_tensor_torch = torch.cumprod(input_tensor, 0)
#     ttnn_tensor = ttnn.from_torch(input_tensor, dtype=dtype, layout=ttnn.Layout.TILE, device=device)
#     result_tensor = ttnn.experimental.cumprod(ttnn_tensor, dim)

#     assert ttnn_tensor.shape == result_tensor.shape
#     assert ttnn_tensor.dtype == result_tensor.dtype
#     assert input_tensor.shape == ttnn_tensor.shape
#     assert result_tensor_torch.shape == ttnn_tensor.shape
#     assert result_tensor_torch.shape == result_tensor.shape

#     # the case with preallocation
#     input_tensor = torch.randint(0, 100, shape, dtype=torch.uint8)
#     result_tensor_torch = torch.cumprod(input_tensor, 0)
#     ttnn_tensor = ttnn.from_torch(input_tensor, ttnn.uint8, layout=ttnn.Layout.TILE, device=device)
#     preallocated_tensor = ttnn.zeros_like(ttnn_tensor)
#     result_tensor = ttnn.experimental.cumprod(ttnn_tensor, dim, out=preallocated_tensor)

#     assert ttnn_tensor.shape == result_tensor.shape
#     assert ttnn_tensor.dtype == result_tensor.dtype
#     assert preallocated_tensor.shape == result_tensor.shape
#     assert preallocated_tensor.dtype == result_tensor.dtype
#     assert input_tensor.shape == ttnn_tensor.shape
#     assert result_tensor_torch.shape == ttnn_tensor.shape
#     assert result_tensor_torch.shape == result_tensor.shape


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize(
    "shape",
    [
        # [],
        # [0],
        [1],
        [10],
        [3, 3],
        [16, 16],
        [32, 32, 32],
        [1, 0, 32, 32],
        [1, 1, 1, 1],
        [16, 8, 4, 16],
        [1000, 32, 32],
        [3, 4, 5, 4, 3],
    ],
)
def test_cumprod_scaffold_with_bfloat16(dim, shape, device):
    torch.manual_seed(22041997)

    input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    result_tensor_torch = torch.cumprod(input_tensor, 0)
    ttnn_tensor = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    result_tensor = ttnn.experimental.cumprod(ttnn_tensor, dim)
    ttnn_tensor_result_from_torch = ttnn.from_torch(
        result_tensor_torch, layout=ttnn.Layout.TILE, dtype=ttnn.DataType.BFLOAT16
    )

    assert ttnn_tensor.shape == result_tensor.shape
    assert ttnn_tensor.dtype == result_tensor.dtype
    assert input_tensor.shape == ttnn_tensor.shape
    assert result_tensor_torch.shape == ttnn_tensor.shape
    assert result_tensor_torch.shape == result_tensor.shape

    assert_with_pcc(result_tensor_torch, ttnn.to_torch(result_tensor), 0.3)

    # the case with preallocation
    # input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    # result_tensor_torch = torch.cumprod(input_tensor, 0)
    # ttnn_tensor = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    # preallocated_tensor = ttnn.zeros_like(ttnn_tensor)
    # result_tensor = ttnn.experimental.cumprod(ttnn_tensor, dim, out=preallocated_tensor)

    # assert ttnn_tensor.shape == result_tensor.shape
    # assert ttnn_tensor.dtype == result_tensor.dtype
    # assert preallocated_tensor.shape == result_tensor.shape
    # assert preallocated_tensor.dtype == result_tensor.dtype
    # assert input_tensor.shape == ttnn_tensor.shape
    # assert result_tensor_torch.shape == ttnn_tensor.shape
    # assert result_tensor_torch.shape == result_tensor.shape
