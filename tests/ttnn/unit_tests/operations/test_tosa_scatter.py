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
    "N, K, W, C, input_dtype, index_dtype, input_layout",
    [
        (1, 1, 1, 1, ttnn.bfloat16, ttnn.uint32, ttnn.Layout.TILE),
        (20, 40, 40, 10, ttnn.bfloat16, ttnn.uint32, ttnn.Layout.ROW_MAJOR),
    ],
)
def test_tosa_scatter_normal(N, K, W, C, input_dtype, index_dtype, input_layout, device):
    torch.manual_seed(0)
    input_torch_dtype = select_torch_dtype(input_dtype)

    input_shape = [N, K, C]
    index_shape = [N, W]
    source_shape = [N, W, C]

    torch_input = torch.randn(input_shape, dtype=input_torch_dtype)
    ttnn_input = ttnn.from_torch(torch_input, dtype=input_dtype, layout=input_layout, device=device)
    torch_index = torch.randint(0, K, index_shape, dtype=torch.int64)
    ttnn_index = ttnn.from_torch(torch_index, dtype=index_dtype, layout=input_layout, device=device)
    torch_source = torch.randn(source_shape, dtype=input_torch_dtype)
    ttnn_source = ttnn.from_torch(torch_source, dtype=input_dtype, layout=input_layout, device=device)
    dim = 1

    # adapt torch_index (expand [N, W] into [N, W, C])
    torch_index = torch_index.unsqueeze(-1).expand([N, W, C])

    torch_output = torch.scatter(torch_input, dim=dim, index=torch_index, src=torch_source)
    for _ in range(2):
        ttnn_output = ttnn.experimental.tosa_scatter(ttnn_input, ttnn_index, ttnn_source)
        assert ttnn_output.shape == ttnn_input.shape
        assert ttnn_output.dtype == ttnn_input.dtype
        torch.testing.assert_close(ttnn.to_torch(ttnn_output), torch_output)
