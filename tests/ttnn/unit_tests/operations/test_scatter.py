# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
        return (
            torch.int64
        )  # !!! there is a strict requirement for the index tensor in Torch to be int64, and there is no int64 in ttnn


@pytest.mark.parametrize(
    "input_shape, dim, index_and_source_shape, input_dtype",
    [
        ([100], -1, [80], ttnn.float32),
        ([50, 200], -1, [50, 100], ttnn.bfloat16),
        ([2, 30, 200], -1, [2, 30, 200], ttnn.float32),
        ([1, 1, 20, 20, 200], -1, [1, 1, 20, 20, 20], ttnn.bfloat16),
        ([2, 2, 2, 2, 2, 2, 2, 2], -1, [2, 2, 2, 2, 2, 2, 2, 2], ttnn.float32),
        ([10, 1, 10, 1, 10], -1, [10, 1, 10, 1, 1], ttnn.bfloat16),
        ([10, 50, 10, 50, 10], -1, [10, 50, 10, 50, 10], ttnn.bfloat16),
        ([10, 50, 10, 50, 100], -1, [10, 50, 10, 50, 100], ttnn.bfloat16),
        ([700, 10, 30, 10], -1, [700, 10, 30, 10], ttnn.bfloat16),
        ([700, 10, 30, 100], -1, [700, 10, 30, 10], ttnn.bfloat16),
    ],
)
def test_scatter_normal(input_shape, dim, index_and_source_shape, input_dtype, device):
    torch.manual_seed(22052025)
    for _ in range(2):
        torch_dtype = select_torch_dtype(input_dtype)
        ##
        torch_input = torch.randn(input_shape, dtype=torch_dtype)
        ttnn_input = ttnn.from_torch(torch_input, dtype=input_dtype, layout=ttnn.Layout.TILE, device=device)

        torch_index = torch.randint(0, input_shape[dim], index_and_source_shape, dtype=torch.int64)
        ttnn_index = ttnn.from_torch(torch_index, dtype=ttnn.int32, layout=ttnn.Layout.TILE, device=device)

        torch_src = torch.randn(index_and_source_shape, dtype=torch_dtype)
        ttnn_src = ttnn.from_torch(torch_src, dtype=input_dtype, layout=ttnn.Layout.TILE, device=device)

        torch_result = torch.scatter(torch_input, dim, index=torch_index, src=torch_src)
        ttnn_result = ttnn.experimental.scatter_(ttnn_input, dim, ttnn_index, ttnn_src)

        torch_result_from_ttnn = ttnn.to_torch(ttnn_result)
        assert torch_result_from_ttnn.shape == torch_result.shape
        assert torch_result_from_ttnn.dtype == torch_result.dtype
        torch.testing.assert_close(torch_result_from_ttnn, torch_result)

        torch_output_preallocated = torch.zeros(torch_input.shape, dtype=torch_input.dtype)
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
        torch.testing.assert_close(ttnn.to_torch(ttnn_output_preallocated), torch_output_preallocated)
