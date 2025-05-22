# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def select_torch_dtype(ttnn_dtype):
    if ttnn_dtype is ttnn.bfloat16:
        return torch.bfloat16
    if ttnn_dtype is ttnn.float32:
        return torch.float32
    if ttnn_dtype is ttnn.uint8:
        return torch.uint8
    if ttnn_dtype is ttnn.int32:
        return torch.int32


@pytest.mark.parametrize("input_shape", [[64, 32]])
@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize("index_and_src_shape", [[64, 32]])
@pytest.mark.parametrize("input_dtype", [ttnn.float32])
def test_scatter_floating_point(input_shape, dim, index_and_src_shape, input_dtype, device):
    torch.manual_seed(22041997)

    torch_dtype = select_torch_dtype(input_dtype)
    ##
    torch_input = torch.randn(input_shape, dtype=torch_dtype)
    ttnn_input = ttnn.from_torch(torch_input, dtype=input_dtype, layout=ttnn.Layout.TILE, device=device)

    torch_index = torch.randint(0, input_shape[dim], index_and_src_shape, dtype=torch.int64)
    ttnn_index = ttnn.from_torch(torch_index, dtype=ttnn.int32, layout=ttnn.Layout.TILE, device=device)

    torch_src = torch.randn(index_and_src_shape, dtype=torch_dtype)
    ttnn_src = ttnn.from_torch(torch_src, dtype=input_dtype, layout=ttnn.Layout.TILE, device=device)

    torch_result = torch.scatter(torch_input, dim, index=torch_index, src=torch_src)
    ttnn_result = ttnn.experimental.scatter_(ttnn_input, dim, ttnn_index, ttnn_src)

    torch_result_from_ttnn = ttnn.to_torch(ttnn_result)
    assert torch_result_from_ttnn.shape == torch_result.shape
    assert torch_result_from_ttnn.dtype == torch_result.dtype
    # assert_with_pcc(torch_result, torch_result_from_ttnn)

    torch_output_preallocated = torch.zeros(torch_input.shape)
    ttnn_output_preallocated = ttnn.zeros(
        ttnn_input.shape, dtype=ttnn_input.dtype, device=device, layout=ttnn.Layout.TILE
    )

    torch_preallocated_result = torch.scatter(
        torch_input, dim, index=torch_index, src=torch_src, out=torch_output_preallocated
    )
    ttnn_preallocated_result = ttnn.experimental.scatter_(
        ttnn_input, dim, index=ttnn_index, src=ttnn_src, out=ttnn_output_preallocated
    )

    torch_result_from_preallocated_ttnn = ttnn.to_torch(ttnn_preallocated_result)
    assert torch_result_from_preallocated_ttnn.shape == torch_preallocated_result.shape
    assert torch_result_from_preallocated_ttnn.dtype == torch_preallocated_result.dtype
    # assert_with_pcc(torch_preallocated_result, torch_result_from_preallocated_ttnn)


# @pytest.mark.parametrize("input_shape", [])
# @pytest.mark.parametrize("dim", [])
# @pytest.mark.parametrize("index_and_src_shape", [])
# @pytest.mark.parametrize("input_ttnn_dtype", [])
# def test_scatter_integer(input_shape, dim, index_and_src_shape, input_ttnn_dtype, device):
#     # input_torch = torch.
#     torch.manual_seed(22041997)
#     #
#     torch_dtype = select_torch_dtype(input_ttnn_dtype)
#     #
#     torch_input = torch.zeros(input_shape, dtype=torch_dtype)
#     #
#     pass
