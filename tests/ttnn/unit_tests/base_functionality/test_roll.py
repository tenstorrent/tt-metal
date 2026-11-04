# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal


@pytest.mark.parametrize(
    "input_tensor, shifts, dim, layout, dtype, memory_config",
    [
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 1, 0, ttnn.ROW_MAJOR_LAYOUT, torch.float32, ttnn.DRAM_MEMORY_CONFIG),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], -1, 0, ttnn.TILE_LAYOUT, torch.float32, ttnn.L1_MEMORY_CONFIG),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 0, 0, ttnn.ROW_MAJOR_LAYOUT, torch.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 4, 0, ttnn.TILE_LAYOUT, torch.float32, ttnn.L1_MEMORY_CONFIG),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 5, 0, ttnn.ROW_MAJOR_LAYOUT, torch.int32, ttnn.DRAM_MEMORY_CONFIG),
        ([[1, 1], [1, 1], [1, 1], [1, 1]], 1, 1, ttnn.TILE_LAYOUT, torch.bfloat16, ttnn.L1_MEMORY_CONFIG),
        (
            [
                [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
                [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29], [30, 31, 32, 33, 34], [35, 36, 37, 38, 39]],
                [[40, 41, 42, 43, 44], [45, 46, 47, 48, 49], [50, 51, 52, 53, 54], [55, 56, 57, 58, 59]],
                [[60, 61, 62, 63, 64], [65, 66, 67, 68, 69], [70, 71, 72, 73, 74], [75, 76, 77, 78, 79]],
                [[80, 81, 82, 83, 84], [85, 86, 87, 88, 89], [90, 91, 92, 93, 94], [95, 96, 97, 98, 99]],
                [
                    [100, 101, 102, 103, 104],
                    [105, 106, 107, 108, 109],
                    [110, 111, 112, 113, 114],
                    [115, 116, 117, 118, 119],
                ],
            ],
            [-2, -1],
            [0, 1],
            ttnn.TILE_LAYOUT,
            torch.float32,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
        (
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
            [1, 0],
            [0, 1],
            ttnn.ROW_MAJOR_LAYOUT,
            torch.bfloat16,
            ttnn.L1_MEMORY_CONFIG,
        ),
        (
            [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]],
            [1, 0, -2],
            [0, 1, 2],
            ttnn.TILE_LAYOUT,
            torch.float32,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
        (
            [[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]]],
            [0, -1, -2],
            [0, 1, 2],
            ttnn.ROW_MAJOR_LAYOUT,
            torch.bfloat16,
            ttnn.L1_MEMORY_CONFIG,
        ),
    ],
)
def test_roll(device, input_tensor, shifts, dim, layout, dtype, memory_config):
    tensor = torch.tensor(input_tensor, dtype=dtype)
    ttnn_tensor = ttnn.from_torch(tensor)
    ttnn_tensor = ttnn.to_layout(ttnn_tensor, layout)
    ttnn_tensor = ttnn.to_device(ttnn_tensor, device, memory_config)

    pytorch_out = torch.roll(tensor, shifts, dim)
    ttnn_out = ttnn.roll(ttnn_tensor, shifts, dim)
    ttnn_result_torch = ttnn.to_torch(ttnn_out)

    assert_equal(pytorch_out, ttnn_result_torch)


@pytest.mark.parametrize(
    "input_tensor, shifts",
    [
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 2),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 3),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], -2),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 4),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 5),
        ([[1, 1], [1, 1], [1, 1], [1, 1]], 2),
        (
            [
                [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
                [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29], [30, 31, 32, 33, 34], [35, 36, 37, 38, 39]],
                [[40, 41, 42, 43, 44], [45, 46, 47, 48, 49], [50, 51, 52, 53, 54], [55, 56, 57, 58, 59]],
                [[60, 61, 62, 63, 64], [65, 66, 67, 68, 69], [70, 71, 72, 73, 74], [75, 76, 77, 78, 79]],
                [[80, 81, 82, 83, 84], [85, 86, 87, 88, 89], [90, 91, 92, 93, 94], [95, 96, 97, 98, 99]],
                [
                    [100, 101, 102, 103, 104],
                    [105, 106, 107, 108, 109],
                    [110, 111, 112, 113, 114],
                    [115, 116, 117, 118, 119],
                ],
            ],
            -2,
        ),
        ([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], 4),
        ([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], 2),
        ([[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]]], -3),
    ],
)
def test_roll_without_dim(device, input_tensor, shifts):
    tensor = torch.tensor(input_tensor, dtype=torch.float32)
    ttnn_tensor = ttnn.from_torch(tensor)
    ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_tensor = ttnn.to_device(ttnn_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pytorch_out = torch.roll(tensor, shifts)
    ttnn_out = ttnn.roll(ttnn_tensor, shifts)
    ttnn_result_torch = ttnn.to_torch(ttnn_out)
    assert_equal(pytorch_out, ttnn_result_torch)


@pytest.mark.parametrize(
    "input_tensor, shifts, dim, layout, dtype",
    [
        (
            [
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [10, 11, 12],
                        [13, 14, 15],
                        [16, 17, 18],
                        [19, 20, 21],
                        [22, 23, 24],
                    ],
                    [
                        [25, 26, 27],
                        [28, 29, 30],
                        [31, 32, 33],
                        [34, 35, 36],
                        [37, 38, 39],
                        [40, 41, 42],
                        [43, 44, 45],
                        [46, 47, 48],
                    ],
                    [
                        [49, 50, 51],
                        [52, 53, 54],
                        [55, 56, 57],
                        [58, 59, 60],
                        [61, 62, 63],
                        [64, 65, 66],
                        [67, 68, 69],
                        [70, 71, 72],
                    ],
                    [
                        [73, 74, 75],
                        [76, 77, 78],
                        [79, 80, 81],
                        [82, 83, 84],
                        [85, 86, 87],
                        [88, 89, 90],
                        [91, 92, 93],
                        [94, 95, 96],
                    ],
                ]
            ],
            1,
            1,
            ttnn.TILE_LAYOUT,
            torch.bfloat16,
        ),
        (
            [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]],
            [1, 0],
            [2, 3],
            ttnn.TILE_LAYOUT,
            torch.float32,
        ),
    ],
)
def test_roll_tile_padding(device, input_tensor, shifts, dim, layout, dtype):
    tensor = torch.tensor(input_tensor, dtype=dtype)
    ttnn_tensor = ttnn.from_torch(tensor)
    ttnn_tensor = ttnn.to_layout(ttnn_tensor, layout)
    ttnn_tensor = ttnn.to_device(ttnn_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    pytorch_out = torch.roll(tensor, shifts, dim)
    ttnn_out = ttnn.roll(ttnn_tensor, shifts, dim)
    ttnn_result_torch = ttnn.to_torch(ttnn_out)

    assert_equal(pytorch_out, ttnn_result_torch)


@pytest.mark.parametrize(
    "shape, shifts, dims",
    [
        ((4, 4, 32, 32), [1, 0], [2, 3]),
        ((2, 2, 64, 64), [2, 1], [2, 3]),
        ((1, 1, 32, 32), [1, -1], [2, 3]),
    ],
)
def test_roll_with_program_cache(device, shape, shifts, dims):
    dtype = torch.float32
    torch_input_tensor = torch.arange(1, 1 + torch.tensor(shape).prod().item(), dtype=dtype).reshape(shape)
    torch_output_tensor = torch.roll(torch_input_tensor, shifts=shifts, dims=dims)

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    for i in range(2):
        ttnn_output_tensor = ttnn.roll(ttnn_input_tensor, shifts, dims)
        ttnn_output_torch = ttnn.to_torch(ttnn_output_tensor)

        assert_with_pcc(torch_output_tensor, ttnn_output_torch)
        assert torch.allclose(ttnn_output_torch, ttnn_output_torch)


@pytest.mark.parametrize(
    "first_shape, second_shape, shifts, dims",
    [
        ((1, 1, 32, 32), (1, 1, 128, 128), [1, 1], [2, 3]),
        ((2, 2, 32, 32), (2, 2, 96, 64), [2, -1], [2, 3]),
    ],
)
def test_roll_cache_hit_larger_shape(device, first_shape, second_shape, shifts, dims):
    # shift/dim/dtype/layout/memory_config match, so both shapes share one program-cache entry by design:
    # the hit path re-applies the new shape's transfer plan. Guards that reuse against a stale plan.
    for shape in (first_shape, second_shape):
        torch_input = torch.arange(1, 1 + torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
        expected = torch.roll(torch_input, shifts=shifts, dims=dims)
        ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
        actual = ttnn.to_torch(ttnn.roll(ttnn_input, shifts, dims))
        assert_with_pcc(expected, actual)


def _nd_sharded_l1_without_2d_spec(num_cores):
    # [1, 1, 32, 32] shards of a [1, 1, 64, 64] tensor form 2 shard columns, which cannot block-shard a grid
    # of 4 cores in a row, so the ND spec has no 2D shard spec.
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    nd_spec = ttnn.NdShardSpec(
        ttnn.Shape([1, 1, 32, 32]),
        cores,
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(ttnn.BufferType.L1, nd_spec)


@pytest.mark.parametrize("shift", [5, 32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_roll_nd_sharded_without_2d_shard_spec(device, shift, layout):
    memory_config = _nd_sharded_l1_without_2d_spec(4)
    torch_input = torch.arange(64 * 64, dtype=torch.float32).reshape(1, 1, 64, 64).bfloat16()
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=memory_config
    )
    assert ttnn_input.memory_config().shard_spec is None

    ttnn_output = ttnn.roll(ttnn_input, shift, -2)

    assert ttnn_output.memory_config() == memory_config
    assert_equal(torch.roll(torch_input, shift, -2), ttnn.to_torch(ttnn_output))
