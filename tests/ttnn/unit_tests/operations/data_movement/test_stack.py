# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize(
    "input_shapes, dim, layout",
    [
        ([(1, 1, 253), (1, 1, 253), (1, 1, 253)], 2, ttnn.ROW_MAJOR_LAYOUT),
        ([(1, 253), (1, 253), (1, 253)], 1, ttnn.ROW_MAJOR_LAYOUT),
        ([(57, 83), (57, 83), (57, 83)], 0, ttnn.TILE_LAYOUT),
        ([(8732,), (8732,), (8732,)], 0, ttnn.ROW_MAJOR_LAYOUT),
        ([(123, 259), (123, 259), (123, 259)], -1, ttnn.TILE_LAYOUT),
        ([(1, 1, 253), (1, 1, 253), (1, 1, 253)], -3, ttnn.ROW_MAJOR_LAYOUT),
        ([(120, 1, 253, 3), (120, 1, 253, 3), (120, 1, 253, 3), (120, 1, 253, 3)], 3, ttnn.ROW_MAJOR_LAYOUT),
        ([(57, 83), (57, 83), (57, 83), (57, 83), (57, 83), (57, 83), (57, 83)], -1, ttnn.TILE_LAYOUT),
        ([(24, 50, 83), (24, 50, 83), (24, 50, 83), (24, 50, 83), (24, 50, 83), (24, 50, 83)], 2, ttnn.TILE_LAYOUT),
        # Row-major stacks on a new trailing axis take the leading-axis concat plus permute route.
        ([(13600,), (13600,), (13600,), (13600,)], 1, ttnn.ROW_MAJOR_LAYOUT),
        ([(1444,)] * 8, -1, ttnn.ROW_MAJOR_LAYOUT),
        ([(8732, 2), (8732, 2)], 2, ttnn.ROW_MAJOR_LAYOUT),
        ([(1, 23, 40, 64), (1, 23, 40, 64)], 4, ttnn.ROW_MAJOR_LAYOUT),
        ([(5, 7)], -1, ttnn.ROW_MAJOR_LAYOUT),
        # Tile-layout stacks on a new trailing axis take the same route for rank >= 2.
        ([(8732, 1), (8732, 1), (8732, 1), (8732, 1)], 2, ttnn.TILE_LAYOUT),
        ([(8732, 2), (8732, 2)], 2, ttnn.TILE_LAYOUT),
        ([(1, 23, 40, 64), (1, 23, 40, 64)], 4, ttnn.TILE_LAYOUT),
        ([(1, 5, 16, 16), (1, 5, 16, 16)], -1, ttnn.TILE_LAYOUT),
        # Rank-1 tile inputs keep the unsqueeze-then-concat path.
        ([(1444,)] * 8, -1, ttnn.TILE_LAYOUT),
    ],
)
def test_stack(device, input_shapes, dim, layout):
    torch_tensors = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]

    ttnn_tensors = [ttnn.from_torch(tensor) for tensor in torch_tensors]
    ttnn_tensors = [ttnn.to_device(tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG) for tensor in ttnn_tensors]
    ttnn_tensors = [ttnn.to_layout(tensor, layout) for tensor in ttnn_tensors]
    torch_result = torch.stack(torch_tensors, dim=dim)
    ttnn_result = ttnn.stack(ttnn_tensors, dim=dim)

    ttnn_result_torch = ttnn.to_torch(ttnn_result)
    assert_equal(torch_result, ttnn_result_torch)

    assert torch_result.shape == ttnn_result_torch.shape


@pytest.mark.parametrize(
    "input_shapes, dim, torch_dtype, ttnn_dtype",
    [
        # Trailing axis, 4-byte and float dtypes: concat and permute pick different kernels per element size.
        ([(1444,)] * 8, -1, torch.int32, ttnn.int32),
        ([(8732, 2), (8732, 2)], 2, torch.int32, ttnn.uint32),
        ([(1, 23, 40, 64), (1, 23, 40, 64)], 4, torch.float32, ttnn.float32),
        # A negative dim that is not the trailing axis stays on the unsqueeze-then-concat path.
        ([(2, 32, 64), (2, 32, 64)], -2, torch.bfloat16, ttnn.bfloat16),
        ([(13600,), (13600,), (13600,)], -2, torch.int32, ttnn.int32),
    ],
)
def test_stack_row_major_dtypes(device, input_shapes, dim, torch_dtype, ttnn_dtype):
    if torch_dtype.is_floating_point:
        torch_tensors = [torch.rand(shape, dtype=torch_dtype) for shape in input_shapes]
    else:
        torch_tensors = [torch.randint(0, 1000, shape, dtype=torch_dtype) for shape in input_shapes]

    ttnn_tensors = [
        ttnn.from_torch(
            tensor, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        for tensor in torch_tensors
    ]
    torch_result = torch.stack(torch_tensors, dim=dim)
    ttnn_result = ttnn.stack(ttnn_tensors, dim=dim)

    ttnn_result_torch = ttnn.to_torch(ttnn_result)
    assert torch_result.shape == ttnn_result_torch.shape
    assert_equal(torch_result, ttnn_result_torch.to(torch_dtype))
