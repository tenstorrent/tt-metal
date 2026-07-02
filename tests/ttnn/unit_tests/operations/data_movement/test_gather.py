# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import numpy as np
from tests.ttnn.utils_for_testing import assert_allclose

TILE_HEIGHT = 32


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([1, 4, 4, 2], [1, 4, 128, 2], 2),
        ([8, 8, 8, 8], [8, 8, 8, 8], -1),
        ([32, 64, 128], [32, 64, 128], -1),
        ([64, 128, 256], [64, 128, 128], -1),
        ([1, 2048, 1, 64], [1, 2048, 1, 32], -1),
        ([1, 1, 1, 1], [1, 1, 1, 1], -1),
        ([4, 4], [4, 4], 1),
        ([128, 64], [128, 32], 1),
        ([16, 16, 16], [16, 16, 16], 0),
        ([1, 1, 1, 1], [1, 1, 1, 1], 1),
        ([64, 128, 256], [64, 128, 128], 1),
        ([256, 2, 32], [160, 2, 32], 1),
        ([2, 256, 2, 32], [2, 128, 2, 32], 1),
        ([2, 32, 96], [2, 32, 32], 1),
        ([128, 128], [128, 64], 1),
        ([1, 2, 128, 1, 768], [1, 2, 8, 1, 768], 2),
        ([1, 2, 8, 1, 768], [1, 2, 8, 1, 128], -1),
        ([1, 2, 8, 2, 768], [1, 2, 8, 2, 128], -1),
        ([1, 1, 2, 8, 2, 768], [1, 1, 2, 8, 2, 128], -2),
    ],
)
def test_gather_general(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(
        0, input_shape[dim], index_shape, dtype=torch.int64
    )  # torch.int64 is required for torch.gather

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([8, 8, 8, 8], [8, 8, 8, 8], -1),
        ([32, 64, 128], [32, 64, 128], -1),
        ([64, 128, 256], [64, 128, 128], -1),
        ([1, 2048, 1, 64], [1, 2048, 1, 32], -1),
        ([1, 1, 1, 1], [1, 1, 1, 1], -1),
    ],
)
def test_gather_preallocated_output(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(0, input_shape[dim], index_shape, dtype=torch.int64)
    output = torch.zeros_like(index, dtype=torch_dtype)

    torch_gather = torch.gather(input, dim, index, out=output)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)
    ttnn_output = ttnn.from_torch(output, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    ttnn.gather(ttnn_input, dim, index=ttnn_index, out=ttnn_output)

    assert ttnn_output.shape == index.shape

    assert_allclose(torch_gather, ttnn.to_torch(ttnn_output))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([1, 1, 512, 64], [1, 1, 512, 32], -1),  # 16 cores
        ([1, 1, 2048, 64], [1, 1, 2048, 32], -1),  # 64 cores
        ([1, 1, 2240, 64], [1, 1, 2240, 32], -1),  # 70 cores
    ],
)
def test_gather_multicore_cases(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(
        0, input_shape[dim], index_shape, dtype=torch.int64
    )  # torch.int64 is required for torch.gather

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim, torch_input_datatype, ttnn_input_datatype, ttnn_index_datatype",
    [
        ([1, 1, 512, 64], [1, 1, 512, 32], -1, torch.float32, ttnn.float32, ttnn.uint16),
        ([128, 64], [128, 32], 1, torch.bfloat16, ttnn.bfloat16, ttnn.uint16),
        ([2, 32, 96], [2, 32, 32], -1, torch.float32, ttnn.float32, ttnn.uint32),
    ],
)
def test_gather_datatype_cases(
    input_shape, index_shape, dim, torch_input_datatype, ttnn_input_datatype, ttnn_index_datatype, device
):
    torch.manual_seed(0)

    input = torch.randn(input_shape, dtype=torch_input_datatype)
    index = torch.randint(
        0, input_shape[dim], index_shape, dtype=torch.int64
    )  # torch.int64 is required for torch.gather

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn_input_datatype, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn_index_datatype, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([32, 256 * TILE_HEIGHT], [32, 64 * TILE_HEIGHT], -1),
        ([1, 1, 32, 256 * TILE_HEIGHT], [1, 1, 32, 128 * TILE_HEIGHT], -1),
        ([1, 1, 32, 63 * TILE_HEIGHT], [1, 1, 32, 63 * TILE_HEIGHT], -1),
        ([1, 1, 32, 20 * TILE_HEIGHT], [1, 1, 32, 20 * TILE_HEIGHT], -1),
        ([1, 1, 32, 96 * TILE_HEIGHT], [1, 1, 32, 96 * TILE_HEIGHT], -1),
        ([1, 1, 32, 256 * TILE_HEIGHT], [1, 1, 32, 256 * TILE_HEIGHT], -1),
        ([1, 151936], [1, 151936], -1),
        ([1, 128256], [1, 128256], -1),
    ],
)
def test_gather_long_tensor(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    max_uint32 = np.iinfo(np.uint32).max
    max_idx_val = min(input_shape[dim], max_uint32)
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(0, max_idx_val, index_shape, dtype=torch.int64)  # torch.int64 is required for torch.gather

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint32, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim, runs",
    [
        ([64, 64], [64, 32], -1, 10),
        ([1, 1, 32, 2048 * TILE_HEIGHT], [1, 1, 32, 2048 * TILE_HEIGHT], -1, 2),
        ([32, 128], [32, 128], -1, 5),
    ],
)
def test_gather_cache_run(input_shape, index_shape, dim, runs, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16

    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(0, input_shape[dim], index_shape, dtype=torch.int64)

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    for _ in range(runs):
        ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)
        assert ttnn_gather.shape == index.shape
        assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([32, 64, 128], [32, 64, 128], -1),
        ([32, 8192], [32, 2048], -1),
    ],
)
def test_gather_sub_core_grids(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(
        0, input_shape[dim], index_shape, dtype=torch.int64
    )  # torch.int64 is required for torch.gather

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
        ]
    )

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index, sub_core_grids=sub_core_grids)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([32, 32, 64 * TILE_HEIGHT], [32, 32, 64 * TILE_HEIGHT], -1),
        ([64, 64, 128 * TILE_HEIGHT], [64, 64, 128 * TILE_HEIGHT], -1),
    ],
)
def test_gather_multirow(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    max_uint32 = np.iinfo(np.uint32).max
    max_idx_val = min(input_shape[dim], max_uint32)
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(0, max_idx_val, index_shape, dtype=torch.int64)

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint32, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


# --- ND / legacy sharding (TILE): memory_config patterns mirror test_untilize / test_tilize_with_val_padding ---

_SHARD_GRID_2X1 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
_SHARD_GRID_2X2 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})


def _align_tile(value):
    return (value + 31) // 32 * 32


def _legacy_output_memory_config(tensor_shape, grid, memory_layout, orientation):
    num_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[-1]
    num_cores = grid.num_cores()

    height_shard_shape = (_align_tile(tensor_height // num_cores), tensor_width)
    width_shard_shape = (tensor_height, _align_tile(tensor_width // num_cores))
    block_grid_size = grid.bounding_box().grid_size()
    if orientation == ttnn.ShardOrientation.ROW_MAJOR:
        block_sharded_shard_shape = (
            _align_tile(tensor_height // block_grid_size.y),
            _align_tile(tensor_width // block_grid_size.x),
        )
    else:
        block_sharded_shard_shape = (
            _align_tile(tensor_height // block_grid_size.x),
            _align_tile(tensor_width // block_grid_size.y),
        )

    layout_map = {
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED: {"shard_grid": grid, "shard_shape": height_shard_shape},
        ttnn.TensorMemoryLayout.WIDTH_SHARDED: {"shard_grid": grid, "shard_shape": width_shard_shape},
        ttnn.TensorMemoryLayout.BLOCK_SHARDED: {"shard_grid": grid, "shard_shape": block_sharded_shard_shape},
    }
    info = layout_map[memory_layout]
    shard_spec = ttnn.ShardSpec(info["shard_grid"], info["shard_shape"], orientation)
    return ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, shard_spec)


@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize(
    "tensor_shape, index_shape, input_shard_shape, output_shard_shape",
    [
        ([1, 1, 32, 128], [1, 1, 32, 64], [1, 1, 32, 64], None),
        ([1, 1, 32, 128], [1, 1, 32, 64], [1, 1, 32, 64], [1, 1, 32, 32]),
        ([1, 1, 48, 96], [1, 1, 37, 55], [1, 1, 37, 55], [1, 1, 37, 32]),
    ],
)
@pytest.mark.parametrize("shard_core_grid", [_SHARD_GRID_2X1])
@pytest.mark.parametrize(
    "input_orientation, output_orientation",
    [
        (ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.ROW_MAJOR),
        (ttnn.ShardOrientation.COL_MAJOR, ttnn.ShardOrientation.COL_MAJOR),
    ],
)
def test_gather_nd_sharded_to_nd_sharded(
    device,
    dim,
    tensor_shape,
    index_shape,
    input_shard_shape,
    output_shard_shape,
    shard_core_grid,
    input_orientation,
    output_orientation,
):
    """ND-sharded TILE input and ND-sharded TILE output (optionally different shard_shape)."""
    if output_shard_shape is None:
        output_shard_shape = input_shard_shape
    torch.manual_seed(42)
    input_torch = torch.randn(tensor_shape, dtype=torch.bfloat16)
    index_torch = torch.randint(0, tensor_shape[dim], index_shape, dtype=torch.int64)
    torch_out = torch.gather(input_torch, dim, index_torch)

    input_nd = ttnn.NdShardSpec(shard_shape=input_shard_shape, grid=shard_core_grid, orientation=input_orientation)
    index_shard_shape = [min(input_shard_shape[i], index_shape[i]) for i in range(len(tensor_shape))]
    index_nd = ttnn.NdShardSpec(shard_shape=index_shard_shape, grid=shard_core_grid, orientation=input_orientation)
    input_spec = ttnn.TensorSpec(
        shape=tensor_shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        nd_shard_spec=input_nd,
        buffer_type=ttnn.BufferType.L1,
    )
    index_spec = ttnn.TensorSpec(
        shape=index_shape,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        nd_shard_spec=index_nd,
        buffer_type=ttnn.BufferType.L1,
    )
    ttnn_input = ttnn.from_torch(input_torch, spec=input_spec, device=device)
    ttnn_index = ttnn.from_torch(index_torch.to(torch.uint16), spec=index_spec, device=device)

    output_nd = ttnn.NdShardSpec(shard_shape=output_shard_shape, grid=shard_core_grid, orientation=output_orientation)
    output_mem = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd)
    ttnn_out = ttnn.gather(ttnn_input, dim, index=ttnn_index, memory_config=output_mem)
    assert_allclose(torch_out, ttnn.to_torch(ttnn_out))


@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize(
    "tensor_shape, index_shape, input_shard_shape",
    [
        ([1, 1, 32, 128], [1, 1, 32, 64], [1, 1, 32, 64]),
        ([1, 1, 48, 96], [1, 1, 37, 55], [1, 1, 37, 55]),
    ],
)
@pytest.mark.parametrize("shard_core_grid", [_SHARD_GRID_2X1])
@pytest.mark.parametrize("input_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_gather_nd_sharded_to_interleaved(
    device, dim, tensor_shape, index_shape, input_shard_shape, shard_core_grid, input_orientation
):
    torch.manual_seed(42)
    input_torch = torch.randn(tensor_shape, dtype=torch.bfloat16)
    index_torch = torch.randint(0, tensor_shape[dim], index_shape, dtype=torch.int64)
    torch_out = torch.gather(input_torch, dim, index_torch)

    input_nd = ttnn.NdShardSpec(shard_shape=input_shard_shape, grid=shard_core_grid, orientation=input_orientation)
    input_spec = ttnn.TensorSpec(
        shape=tensor_shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        nd_shard_spec=input_nd,
        buffer_type=ttnn.BufferType.L1,
    )
    ttnn_input = ttnn.from_torch(input_torch, spec=input_spec, device=device)
    ttnn_index = ttnn.from_torch(
        index_torch.to(torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    output_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    ttnn_out = ttnn.gather(ttnn_input, dim, index=ttnn_index, memory_config=output_mem)
    assert_allclose(torch_out, ttnn.to_torch(ttnn_out))


@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize(
    "tensor_shape, index_shape, output_shard_shape",
    [
        ([1, 1, 32, 128], [1, 1, 32, 64], [1, 1, 32, 64]),
        ([1, 1, 48, 96], [1, 1, 37, 55], [1, 1, 37, 55]),
    ],
)
@pytest.mark.parametrize("shard_core_grid", [_SHARD_GRID_2X1])
@pytest.mark.parametrize("output_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_gather_interleaved_to_nd_sharded(
    device, dim, tensor_shape, index_shape, output_shard_shape, shard_core_grid, output_orientation
):
    torch.manual_seed(42)
    input_torch = torch.randn(tensor_shape, dtype=torch.bfloat16)
    index_torch = torch.randint(0, tensor_shape[dim], index_shape, dtype=torch.int64)
    torch_out = torch.gather(input_torch, dim, index_torch)

    ttnn_input = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    ttnn_index = ttnn.from_torch(
        index_torch.to(torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    output_nd = ttnn.NdShardSpec(shard_shape=output_shard_shape, grid=shard_core_grid, orientation=output_orientation)
    output_mem = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd)
    ttnn_out = ttnn.gather(ttnn_input, dim, index=ttnn_index, memory_config=output_mem)
    assert_allclose(torch_out, ttnn.to_torch(ttnn_out))


@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize(
    "tensor_shape, index_shape, input_shard_shape",
    [
        ([1, 1, 32, 128], [1, 1, 32, 64], [1, 1, 32, 64]),
        ([1, 1, 48, 96], [1, 1, 37, 55], [1, 1, 37, 55]),
    ],
)
@pytest.mark.parametrize("shard_core_grid", [_SHARD_GRID_2X1])
@pytest.mark.parametrize(
    "output_memory_layout",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
@pytest.mark.parametrize("output_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
def test_gather_nd_sharded_to_legacy_sharded(
    device,
    dim,
    tensor_shape,
    index_shape,
    input_shard_shape,
    shard_core_grid,
    output_memory_layout,
    output_orientation,
):
    torch.manual_seed(42)
    input_torch = torch.randn(tensor_shape, dtype=torch.bfloat16)
    index_torch = torch.randint(0, tensor_shape[dim], index_shape, dtype=torch.int64)
    torch_out = torch.gather(input_torch, dim, index_torch)

    input_nd = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    input_spec = ttnn.TensorSpec(
        shape=tensor_shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        nd_shard_spec=input_nd,
        buffer_type=ttnn.BufferType.L1,
    )
    ttnn_input = ttnn.from_torch(input_torch, spec=input_spec, device=device)
    ttnn_index = ttnn.from_torch(
        index_torch.to(torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    grid = _SHARD_GRID_2X2 if output_memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED else shard_core_grid
    output_mem = _legacy_output_memory_config(index_shape, grid, output_memory_layout, output_orientation)
    ttnn_out = ttnn.gather(ttnn_input, dim, index=ttnn_index, memory_config=output_mem)
    assert_allclose(torch_out, ttnn.to_torch(ttnn_out))


@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize(
    "tensor_shape, index_shape",
    [
        ([1, 1, 32, 128], [1, 1, 32, 64]),
        ([1, 1, 48, 96], [1, 1, 37, 55]),
    ],
)
@pytest.mark.parametrize("shard_core_grid", [_SHARD_GRID_2X1])
@pytest.mark.parametrize("input_orientation", [ttnn.ShardOrientation.ROW_MAJOR])
def test_gather_legacy_sharded_to_nd_sharded(
    device, dim, tensor_shape, index_shape, shard_core_grid, input_orientation
):
    torch.manual_seed(42)
    input_torch = torch.randn(tensor_shape, dtype=torch.bfloat16)
    index_torch = torch.randint(0, tensor_shape[dim], index_shape, dtype=torch.int64)
    torch_out = torch.gather(input_torch, dim, index_torch)

    num_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[-1]
    num_cores = shard_core_grid.num_cores()
    legacy_shard_shape = (_align_tile(tensor_height // num_cores), tensor_width)
    legacy_spec = ttnn.ShardSpec(shard_core_grid, legacy_shard_shape, input_orientation)
    input_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, legacy_spec)
    ttnn_input = ttnn.from_torch(
        input_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_mem
    )
    ttnn_index = ttnn.from_torch(
        index_torch.to(torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    output_nd = ttnn.NdShardSpec(
        shard_shape=[1, 1, 32, 64], grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    output_mem = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd)
    ttnn_out = ttnn.gather(ttnn_input, dim, index=ttnn_index, memory_config=output_mem)
    assert_allclose(torch_out, ttnn.to_torch(ttnn_out))


@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize(
    "tensor_shape, index_shape",
    [
        ([1, 1, 32, 128], [1, 1, 32, 64]),
        ([1, 1, 48, 96], [1, 1, 37, 55]),
    ],
)
def test_gather_interleaved_to_legacy_sharded(device, dim, tensor_shape, index_shape):
    torch.manual_seed(42)
    input_torch = torch.randn(tensor_shape, dtype=torch.bfloat16)
    index_torch = torch.randint(0, tensor_shape[dim], index_shape, dtype=torch.int64)
    torch_out = torch.gather(input_torch, dim, index_torch)

    ttnn_input = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    ttnn_index = ttnn.from_torch(
        index_torch.to(torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    output_mem = _legacy_output_memory_config(
        index_shape, _SHARD_GRID_2X1, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.ShardOrientation.ROW_MAJOR
    )
    ttnn_out = ttnn.gather(ttnn_input, dim, index=ttnn_index, memory_config=output_mem)
    assert_allclose(torch_out, ttnn.to_torch(ttnn_out))


@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize(
    "tensor_shape, index_shape",
    [
        ([1, 1, 32, 128], [1, 1, 32, 64]),
        ([1, 1, 48, 96], [1, 1, 37, 55]),
    ],
)
def test_gather_legacy_sharded_to_interleaved(device, dim, tensor_shape, index_shape):
    torch.manual_seed(42)
    input_torch = torch.randn(tensor_shape, dtype=torch.bfloat16)
    index_torch = torch.randint(0, tensor_shape[dim], index_shape, dtype=torch.int64)
    torch_out = torch.gather(input_torch, dim, index_torch)

    num_dims = len(tensor_shape)
    tensor_height = 1
    for i in range(num_dims - 1):
        tensor_height *= tensor_shape[i]
    tensor_width = tensor_shape[-1]
    legacy_shard_shape = (_align_tile(tensor_height // _SHARD_GRID_2X1.num_cores()), tensor_width)
    legacy_spec = ttnn.ShardSpec(_SHARD_GRID_2X1, legacy_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, legacy_spec)
    ttnn_input = ttnn.from_torch(
        input_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_mem
    )
    ttnn_index = ttnn.from_torch(
        index_torch.to(torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    ttnn_out = ttnn.gather(
        ttnn_input,
        dim,
        index=ttnn_index,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    assert_allclose(torch_out, ttnn.to_torch(ttnn_out))


@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize(
    "tensor_shape, index_shape",
    [
        ([1, 1, 32, 128], [1, 1, 32, 64]),
        ([1, 1, 48, 96], [1, 1, 37, 55]),
    ],
)
def test_gather_preallocated_nd_sharded_output(device, dim, tensor_shape, index_shape):
    torch.manual_seed(42)
    input_torch = torch.randn(tensor_shape, dtype=torch.bfloat16)
    index_torch = torch.randint(0, tensor_shape[dim], index_shape, dtype=torch.int64)
    torch_out = torch.empty(index_shape, dtype=torch.bfloat16)
    torch.gather(input_torch, dim, index_torch, out=torch_out)

    ttnn_input = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    ttnn_index = ttnn.from_torch(
        index_torch.to(torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    output_nd = ttnn.NdShardSpec(
        shard_shape=[1, 1, 32, 64], grid=_SHARD_GRID_2X1, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    out_spec = ttnn.TensorSpec(
        shape=index_shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        nd_shard_spec=output_nd,
        buffer_type=ttnn.BufferType.L1,
    )
    out_torch = torch.zeros(index_shape, dtype=torch.bfloat16)
    ttnn_out = ttnn.from_torch(out_torch, spec=out_spec, device=device)
    ttnn.gather(ttnn_input, dim, index=ttnn_index, out=ttnn_out)
    assert_allclose(torch_out, ttnn.to_torch(ttnn_out))


@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize(
    "tensor_shape, index_shape, input_shard_shape",
    [
        ([1, 1, 2, 32, 128], [1, 1, 2, 32, 64], [1, 1, 2, 32, 64]),
        ([1, 1, 2, 48, 96], [1, 1, 2, 37, 55], [1, 1, 2, 37, 55]),
    ],
)
def test_gather_nd_sharded_high_rank(device, dim, tensor_shape, index_shape, input_shard_shape):
    """Rank-5 tensors: gather on last dim with ND-sharded input and output."""
    torch.manual_seed(42)
    input_torch = torch.randn(tensor_shape, dtype=torch.bfloat16)
    index_torch = torch.randint(0, tensor_shape[dim], index_shape, dtype=torch.int64)
    torch_out = torch.gather(input_torch, dim, index_torch)

    shard_core_grid = _SHARD_GRID_2X1
    input_nd = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    input_spec = ttnn.TensorSpec(
        shape=tensor_shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        nd_shard_spec=input_nd,
        buffer_type=ttnn.BufferType.L1,
    )
    index_spec = ttnn.TensorSpec(
        shape=index_shape,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        nd_shard_spec=input_nd,
        buffer_type=ttnn.BufferType.L1,
    )
    ttnn_input = ttnn.from_torch(input_torch, spec=input_spec, device=device)
    ttnn_index = ttnn.from_torch(index_torch.to(torch.uint16), spec=index_spec, device=device)
    output_nd = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=shard_core_grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    output_mem = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=output_nd)
    ttnn_out = ttnn.gather(ttnn_input, dim, index=ttnn_index, memory_config=output_mem)
    assert_allclose(torch_out, ttnn.to_torch(ttnn_out))


@pytest.mark.parametrize("dim", [-1])
def test_gather_multicore_sharded_wt_threshold(device, dim):
    """Wt_input > 60 triggers GatherProgramFactorySingleRowMultiCore; TILE sharded buffers."""
    w_tiles = 65
    tensor_shape = [1, 1, 32, w_tiles * TILE_HEIGHT]
    index_shape = [1, 1, 32, 32 * TILE_HEIGHT]
    input_shard_shape = [1, 1, 32, 64]
    torch.manual_seed(42)
    input_torch = torch.randn(tensor_shape, dtype=torch.bfloat16)
    index_torch = torch.randint(0, tensor_shape[dim], index_shape, dtype=torch.int64)
    torch_out = torch.gather(input_torch, dim, index_torch)

    input_nd = ttnn.NdShardSpec(
        shard_shape=input_shard_shape, grid=_SHARD_GRID_2X1, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    input_spec = ttnn.TensorSpec(
        shape=tensor_shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        nd_shard_spec=input_nd,
        buffer_type=ttnn.BufferType.L1,
    )
    ttnn_input = ttnn.from_torch(input_torch, spec=input_spec, device=device)
    ttnn_index = ttnn.from_torch(
        index_torch.to(torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    ttnn_out = ttnn.gather(
        ttnn_input,
        dim,
        index=ttnn_index,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    )
    assert_allclose(torch_out, ttnn.to_torch(ttnn_out))
