# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_equal


def run_index_fill_test(shape, dim, indices, value, dtype, device):
    torch.manual_seed(2025)

    if dim > len(shape) - 1:
        pytest.skip("Given dim is higher than tensor rank")

    if dtype == torch.int32:
        torch_input = torch.randint(0, 100, shape, dtype=torch.int32)
    else:
        torch_input = torch.rand(shape, dtype=dtype)
    torch_index = torch.tensor(indices)
    torch_output = torch.index_fill(torch_input, dim, torch_index, value)

    tt_input = ttnn.from_torch(torch_input, device=device)
    tt_index = ttnn.from_torch(torch_index, device=device)
    tt_output = ttnn.index_fill(tt_input, dim, tt_index, value)
    tt_output = ttnn.to_torch(tt_output)

    assert_equal(tt_output, torch_output)


@pytest.mark.parametrize(
    "shape",
    [
        [32, 32],  # 2D, multiple of 32
        [12, 24],  # 2D, not multiple of 32
        [23, 41, 32],  # 3D, multiple of 32
        [9, 5, 38],  # 3D, not multiple of 32
        [3, 4, 5, 32],  # 4D, multiple of 32
        [41, 21, 33, 34],  # 4D, not multiple of 32
    ],
)
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("indices", [[0, 2]])
@pytest.mark.parametrize("value", [2.5, 1.72])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_index_fill_float(shape, dim, indices, value, dtype, device):
    run_index_fill_test(shape, dim, indices, value, dtype, device)


@pytest.mark.parametrize(
    "shape",
    [
        [32, 32],  # 2D, multiple of 32
        [12, 23],  # 2D, not multiple of 32
        [27, 12, 32],  # 3D, multiple of 32
        [61, 3, 6],  # 3D, not multiple of 32
        [6, 3, 7, 32],  # 4D, multiple of 32
        [13, 15, 22, 13],  # 4D, not multiple of 32
    ],
)
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("indices", [[0, 2]])
@pytest.mark.parametrize("value", [15, 12])
@pytest.mark.parametrize("dtype", [torch.int32])
def test_index_fill_int(shape, dim, indices, value, dtype, device):
    run_index_fill_test(shape, dim, indices, value, dtype, device)


@pytest.mark.parametrize(
    "shape, dim, indices",
    [
        ([1], 0, [0]),  # 1 element tensor
        ([1, 1], 0, [0]),  # 1 element tensor
        ([4, 32], 1, list(range(14))),  # large index tensor
    ],
)
@pytest.mark.parametrize("value", [15])
@pytest.mark.parametrize("dtype", [torch.int32])
def test_index_fill_cornercases(shape, dim, indices, value, dtype, device):
    run_index_fill_test(shape, dim, indices, value, dtype, device)


@pytest.mark.parametrize(
    "shape",
    [
        [32, 32],  # 2D, multiple of 32
        [12, 23],  # 2D, not multiple of 32
        [27, 12, 32],  # 3D, multiple of 32
        [61, 3, 6],  # 3D, not multiple of 32
        [4, 3, 7, 32],  # 4D, multiple of 32
        [13, 15, 22, 13],  # 4D, not multiple of 32
    ],
)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("indices", [[0, 2]])
@pytest.mark.parametrize("value", [2002])
def test_index_fill_callback(shape, dim, indices, value, device):
    for i in range(2):
        run_index_fill_test(shape, dim, indices, value, torch.int32, device)
        if i == 0:
            num_program_cache_entries = device.num_program_cache_entries()
            assert num_program_cache_entries > 0
        else:
            assert device.num_program_cache_entries() == num_program_cache_entries


# ---------------------------------------------------------------------------
# Sharding tests (HEIGHT / WIDTH / BLOCK)
# ---------------------------------------------------------------------------


def _make_index_tensor(indices: torch.Tensor, device) -> ttnn.Tensor:
    """Create a 1-D UINT32 device tensor from a 1-D int32 torch tensor."""
    assert indices.dim() == 1
    return ttnn.from_torch(indices, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


# Shapes chosen so that num_rows (product of all dims but last) is divisible by
# the 8-core grid used below, and the last dim fits in BFLOAT16 row-major pages.
# All core grids stay within the 8×8 Wormhole device compute grid.
@pytest.mark.parametrize(
    "shape, dim, num_indices, value",
    [
        # shape (2, 4, 8, 32): num_rows = 2*4*8 = 64 → 8 cores, 8 rows each
        ((2, 4, 8, 32), 0, 2, 1.5),  # fill along batch dim
        ((2, 4, 8, 32), 1, 3, -0.5),  # fill along channel dim
        ((2, 4, 8, 32), 2, 5, 0.0),  # fill along height dim
        ((2, 4, 8, 32), 3, 8, 2.0),  # fill along last dim
    ],
    ids=["dim0", "dim1", "dim2", "dim3-last"],
)
def test_index_fill_height_sharded(device, shape, dim, num_indices, value):
    """index_fill with HEIGHT_SHARDED input and output."""
    sharded_mem_cfg = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(y=1, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
    )

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_index = torch.randint(0, shape[dim], (num_indices,), dtype=torch.int32)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_mem_cfg,
    )
    tt_index = _make_index_tensor(torch_index, device)

    tt_out = ttnn.index_fill(tt_input, dim, tt_index, value, memory_config=sharded_mem_cfg)

    golden = torch.index_fill(torch_input, dim, torch_index.long(), value)
    assert_equal(ttnn.to_torch(tt_out), golden)


@pytest.mark.parametrize(
    "shape, dim, num_indices, value",
    [
        # shape (2, 4, 4, 64): last dim 64 split across 4 col-shards of 16 each
        ((2, 4, 4, 64), 0, 1, -1.0),  # dim != last
        ((2, 4, 4, 64), 1, 2, 3.0),  # dim != last
        ((2, 4, 4, 64), 3, 16, 0.5),  # dim == last: column-local fill
    ],
    ids=["dim0", "dim1", "dim3-last"],
)
def test_index_fill_width_sharded(device, shape, dim, num_indices, value):
    """index_fill with WIDTH_SHARDED input and output (same shard spec required)."""
    sharded_mem_cfg = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(y=1, x=4),
        strategy=ttnn.ShardStrategy.WIDTH,
    )

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_index = torch.randint(0, shape[dim], (num_indices,), dtype=torch.int32)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_mem_cfg,
    )
    tt_index = _make_index_tensor(torch_index, device)

    tt_out = ttnn.index_fill(tt_input, dim, tt_index, value, memory_config=sharded_mem_cfg)

    golden = torch.index_fill(torch_input, dim, torch_index.long(), value)
    assert_equal(ttnn.to_torch(tt_out), golden)


@pytest.mark.parametrize(
    "shape, dim, num_indices, value",
    [
        # shape (4, 4, 4, 64): 2×4 block grid → 2 row-shards, 4 col-shards
        ((4, 4, 4, 64), 0, 2, 1.0),  # dim != last
        ((4, 4, 4, 64), 1, 3, -2.0),  # dim != last
        ((4, 4, 4, 64), 3, 16, 0.25),  # dim == last: column-local fill
    ],
    ids=["dim0", "dim1", "dim3-last"],
)
def test_index_fill_block_sharded(device, shape, dim, num_indices, value):
    """index_fill with BLOCK_SHARDED input and output (same shard spec required)."""
    sharded_mem_cfg = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(y=2, x=4),
        strategy=ttnn.ShardStrategy.BLOCK,
    )

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_index = torch.randint(0, shape[dim], (num_indices,), dtype=torch.int32)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_mem_cfg,
    )
    tt_index = _make_index_tensor(torch_index, device)

    tt_out = ttnn.index_fill(tt_input, dim, tt_index, value, memory_config=sharded_mem_cfg)

    golden = torch.index_fill(torch_input, dim, torch_index.long(), value)
    assert_equal(ttnn.to_torch(tt_out), golden)


# ---------------------------------------------------------------------------
# Cross-layout sharding tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, dim, num_indices, value, out_strategy",
    [
        ((2, 4, 8, 32), 0, 2, 1.5, ttnn.ShardStrategy.WIDTH),
        ((2, 4, 8, 32), 3, 8, 2.0, ttnn.ShardStrategy.WIDTH),  # last dim
        ((4, 4, 4, 64), 1, 3, -1.0, ttnn.ShardStrategy.BLOCK),
        ((4, 4, 4, 64), 3, 16, 0.5, ttnn.ShardStrategy.BLOCK),  # last dim
    ],
    ids=["interleaved_to_width_dim0", "interleaved_to_width_dim3", "height_to_block_dim1", "height_to_block_dim3"],
)
def test_index_fill_cross_layout_to_col_sharded(device, shape, dim, num_indices, value, out_strategy):
    """INTERLEAVED/HEIGHT input with WIDTH/BLOCK output."""
    if out_strategy == ttnn.ShardStrategy.WIDTH:
        in_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        out_cfg = ttnn.create_sharded_memory_config(
            shape=shape, core_grid=ttnn.CoreGrid(y=1, x=4), strategy=out_strategy
        )
    else:
        in_cfg = ttnn.create_sharded_memory_config(
            shape=shape, core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.HEIGHT
        )
        out_cfg = ttnn.create_sharded_memory_config(
            shape=shape, core_grid=ttnn.CoreGrid(y=2, x=4), strategy=out_strategy
        )

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_index = torch.randint(0, shape[dim], (num_indices,), dtype=torch.int32)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=in_cfg,
    )
    tt_index = _make_index_tensor(torch_index, device)
    tt_out = ttnn.index_fill(tt_input, dim, tt_index, value, memory_config=out_cfg)

    golden = torch.index_fill(torch_input, dim, torch_index.long(), value)
    assert_equal(ttnn.to_torch(tt_out), golden)


@pytest.mark.parametrize(
    "shape, dim, num_indices, value, in_strategy",
    [
        ((2, 4, 4, 64), 0, 1, -1.0, ttnn.ShardStrategy.WIDTH),
        ((2, 4, 4, 64), 3, 16, 0.5, ttnn.ShardStrategy.WIDTH),  # last dim
        ((4, 4, 4, 64), 1, 3, 1.0, ttnn.ShardStrategy.BLOCK),
        ((4, 4, 4, 64), 3, 16, 2.0, ttnn.ShardStrategy.BLOCK),  # last dim
    ],
    ids=["width_to_interleaved_dim0", "width_to_interleaved_dim3", "block_to_height_dim1", "block_to_height_dim3"],
)
def test_index_fill_cross_layout_from_col_sharded(device, shape, dim, num_indices, value, in_strategy):
    """WIDTH/BLOCK input with INTERLEAVED/HEIGHT output."""
    if in_strategy == ttnn.ShardStrategy.WIDTH:
        in_cfg = ttnn.create_sharded_memory_config(shape=shape, core_grid=ttnn.CoreGrid(y=1, x=4), strategy=in_strategy)
        out_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    else:
        in_cfg = ttnn.create_sharded_memory_config(shape=shape, core_grid=ttnn.CoreGrid(y=2, x=4), strategy=in_strategy)
        out_cfg = ttnn.create_sharded_memory_config(
            shape=shape, core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.HEIGHT
        )

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_index = torch.randint(0, shape[dim], (num_indices,), dtype=torch.int32)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=in_cfg,
    )
    tt_index = _make_index_tensor(torch_index, device)
    tt_out = ttnn.index_fill(tt_input, dim, tt_index, value, memory_config=out_cfg)

    golden = torch.index_fill(torch_input, dim, torch_index.long(), value)
    assert_equal(ttnn.to_torch(tt_out), golden)
