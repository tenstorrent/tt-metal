# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_allclose


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
        (1, 1, 1, 1, ttnn.bfloat16, ttnn.int32, ttnn.Layout.TILE),
        (20, 40, 40, 10, ttnn.float32, ttnn.uint32, ttnn.Layout.ROW_MAJOR),
        (20, 10, 10, 10, ttnn.bfloat16, ttnn.uint32, ttnn.Layout.TILE),
        (20, 10, 10, 10, ttnn.float32, ttnn.uint32, ttnn.Layout.ROW_MAJOR),
        (20, 10, 10, 10, ttnn.bfloat16, ttnn.uint32, ttnn.Layout.ROW_MAJOR),
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
        ttnn_output = ttnn.tosa_scatter(ttnn_input, ttnn_index, ttnn_source)
        assert ttnn_output.shape == ttnn_input.shape
        assert ttnn_output.dtype == ttnn_input.dtype
        assert_allclose(ttnn.to_torch(ttnn_output), torch_output, rtol=1e-3)


# tosa_scatter builds the device op directly instead of going through ttnn::scatter, so it needs
# its own empty-operand guard: without one, a zero extent reached the program factory's
# logical_volume() / input_shape[-1] work split and SIGFPE'd the host process. See issue #56881.
@pytest.mark.parametrize(
    "N, K, W, C",
    [
        (2, 4, 3, 0),  # zero C - every operand is empty
        (2, 4, 0, 5),  # zero W - empty index and source, non-empty input
        (0, 4, 3, 5),  # zero N - every operand is empty
    ],
)
@pytest.mark.parametrize("input_layout", [ttnn.Layout.ROW_MAJOR, ttnn.Layout.TILE])
def test_tosa_scatter_zero_volume(N, K, W, C, input_layout, device):
    torch.manual_seed(0)

    torch_input = torch.randn([N, K, C], dtype=torch.bfloat16)
    torch_index = torch.randint(0, max(K, 1), [N, W], dtype=torch.int64)
    torch_source = torch.randn([N, W, C], dtype=torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=input_layout, device=device)
    ttnn_index = ttnn.from_torch(torch_index, dtype=ttnn.uint32, layout=input_layout, device=device)
    ttnn_source = ttnn.from_torch(torch_source, dtype=ttnn.bfloat16, layout=input_layout, device=device)

    torch_output = torch.scatter(
        torch_input, dim=1, index=torch_index.unsqueeze(-1).expand([N, W, C]), src=torch_source
    )

    ttnn_output = ttnn.tosa_scatter(ttnn_input, ttnn_index, ttnn_source)

    assert ttnn_output.shape == ttnn_input.shape
    assert ttnn_output.dtype == ttnn_input.dtype
    result = ttnn.to_torch(ttnn_output)
    if result.numel():
        assert_allclose(result, torch_output, rtol=1e-3)
