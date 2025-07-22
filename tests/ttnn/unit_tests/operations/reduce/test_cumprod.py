# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("dim", [0, 2, -1])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        [1],
        [2000],
        [1000, 32, 32],
        [5, 5, 5, 5, 1, 1, 1],
    ],
)
def test_cumprod_normal(dim, shape, device):
    torch.manual_seed(0)
    if dim < len(shape) and -len(shape) <= dim:
        for _ in range(2):
            torch_input_tensor = torch.randn(shape, dtype=torch.bfloat16)
            torch_result_tensor = torch.cumprod(torch_input_tensor, dim)
            ttnn_input_tensor = ttnn.from_torch(
                torch_input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device
            )
            ttnn_result_tensor = ttnn.cumprod(ttnn_input_tensor, dim)

            # assert metadata
            assert ttnn_input_tensor.shape == ttnn_result_tensor.shape
            assert ttnn_input_tensor.dtype == ttnn_result_tensor.dtype
            assert torch_input_tensor.shape == ttnn_input_tensor.shape
            assert torch_result_tensor.shape == ttnn_input_tensor.shape
            assert torch_result_tensor.shape == ttnn_result_tensor.shape

            # assert values with pcc
            assert_with_pcc(ttnn.to_torch(ttnn_result_tensor), torch_result_tensor, 0.99)


@pytest.mark.parametrize("dim", [0, 2, -1])
@pytest.mark.parametrize(
    "shape",
    [
        [],
        [1],
        [2000],
        [1000, 32, 32],
        [5, 5, 5, 5, 1, 1, 1],
    ],
)
def test_cumprod_preallocated(dim, shape, device):
    torch.manual_seed(0)
    if dim < len(shape) and -len(shape) <= dim:
        for _ in range(2):
            torch_input_tensor = torch.randn(shape, dtype=torch.bfloat16)
            torch_preallocated_tensor = torch.zeros_like(torch_input_tensor)
            torch_result_tensor = torch.cumprod(torch_input_tensor, dim, out=torch_preallocated_tensor)
            ttnn_input_tensor = ttnn.from_torch(
                torch_input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device
            )
            ttnn_preallocated_tensor = ttnn.zeros_like(ttnn_input_tensor)
            ttnn_result_tensor = ttnn.cumprod(ttnn_input_tensor, dim, out=ttnn_preallocated_tensor)

            # assert metadata
            assert ttnn_input_tensor.shape == ttnn_result_tensor.shape
            assert ttnn_preallocated_tensor.shape == ttnn_result_tensor.shape
            assert ttnn_preallocated_tensor.dtype == ttnn_result_tensor.dtype
            assert ttnn_input_tensor.dtype == ttnn_result_tensor.dtype
            assert torch_input_tensor.shape == ttnn_input_tensor.shape
            assert torch_result_tensor.shape == ttnn_input_tensor.shape
            assert torch_result_tensor.shape == ttnn_result_tensor.shape

            # assert values with pcc
            assert_with_pcc(ttnn.to_torch(ttnn_result_tensor), torch_result_tensor, 0.99)
            assert_with_pcc(ttnn.to_torch(ttnn_preallocated_tensor), torch_preallocated_tensor, 0.98)


@pytest.mark.parametrize(
    "dim, input_shape, output_shape, torch_dtype, input_dtype, output_dtype, memory_config, layout",
    [
        (
            -10,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
        ),  # input_rank vs dim
        (
            10,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
        ),  # input_rank vs dim
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
        ),  # input_shape vs output_shape
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 1],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
        ),  # input_shape vs output_shape
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.float32,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.TILE,
        ),  # input_dtype vs output_dtype
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.L1_MEMORY_CONFIG,
            ttnn.Layout.TILE,
        ),  # unsupported memory config
        (
            3,
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            torch.bfloat16,
            ttnn.bfloat16,
            ttnn.bfloat16,
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.Layout.ROW_MAJOR,
        ),  # unsupported layout
    ],
)
def test_cumprod_failing_cases(
    dim,
    input_shape,
    output_shape,
    torch_dtype,
    input_dtype,
    output_dtype,
    memory_config,
    layout,
    device,
):
    torch.manual_seed(0)
    torch_input_tensor = torch.randn(input_shape, dtype=torch_dtype)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=input_dtype, layout=layout, device=device, memory_config=memory_config
    )
    ttnn_preallocated_tensor = ttnn.zeros(output_shape, dtype=output_dtype)
    with pytest.raises(RuntimeError):
        ttnn.cumprod(ttnn_input_tensor, dim=dim, out=ttnn_preallocated_tensor)
