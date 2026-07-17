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
        ((2, 4, 4, 64), 2, 2, 4.0),  # height dim
        ((2, 4, 4, 64), 3, 16, 0.5),  # dim == last: column-local fill
    ],
    ids=["dim0", "dim1", "dim2", "dim3-last"],
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
        ((4, 4, 4, 64), 2, 2, 6.0),  # height dim
        ((4, 4, 4, 64), 3, 16, 0.25),  # dim == last: column-local fill
    ],
    ids=["dim0", "dim1", "dim2", "dim3-last"],
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
    if out_strategy == ttnn.ShardStrategy.BLOCK:
        pytest.skip(
            "HEIGHT->BLOCK alignment mismatch not yet fixed (see https://github.com/tenstorrent/tt-metal/issues/49987)"
        )
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


@pytest.mark.parametrize(
    "shape, dim, num_indices, value, direction",
    [
        # shape (4, 4, 4, 64): num_rows=64, column shard width 16 (KW=4) on both sides.
        # WIDTH grid 1x4  -> shard (64, 16);  BLOCK grid 2x4 -> shard (32, 16).
        ((4, 4, 4, 64), 0, 2, 1.0, "width_to_block"),
        ((4, 4, 4, 64), 1, 3, -2.0, "width_to_block"),
        ((4, 4, 4, 64), 2, 2, 7.0, "width_to_block"),  # height dim
        ((4, 4, 4, 64), 3, 16, 0.5, "width_to_block"),  # last dim: column-local fill
        ((4, 4, 4, 64), 0, 2, 3.0, "block_to_width"),
        ((4, 4, 4, 64), 1, 3, -1.0, "block_to_width"),
        ((4, 4, 4, 64), 2, 2, 5.0, "block_to_width"),  # height dim
        ((4, 4, 4, 64), 3, 16, 0.25, "block_to_width"),  # last dim
    ],
    ids=[
        "w2b_dim0",
        "w2b_dim1",
        "w2b_dim2",
        "w2b_dim3-last",
        "b2w_dim0",
        "b2w_dim1",
        "b2w_dim2",
        "b2w_dim3-last",
    ],
)
def test_index_fill_width_block_conversion(device, shape, dim, num_indices, value, direction):
    """WIDTH <-> BLOCK conversion with matching column shard width."""
    width_cfg = ttnn.create_sharded_memory_config(
        shape=shape, core_grid=ttnn.CoreGrid(y=1, x=4), strategy=ttnn.ShardStrategy.WIDTH
    )
    block_cfg = ttnn.create_sharded_memory_config(
        shape=shape, core_grid=ttnn.CoreGrid(y=2, x=4), strategy=ttnn.ShardStrategy.BLOCK
    )
    in_cfg, out_cfg = (width_cfg, block_cfg) if direction == "width_to_block" else (block_cfg, width_cfg)

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


def test_index_fill_col_sharded_width_mismatch_raises(device, expect_error):
    """WIDTH -> BLOCK with mismatched column shard width must raise a validation error."""
    shape = (4, 4, 4, 64)
    # WIDTH: 4 col shards (width 16);  BLOCK: 2 col shards (width 32) -> mismatch.
    width_cfg = ttnn.create_sharded_memory_config(
        shape=shape, core_grid=ttnn.CoreGrid(y=1, x=4), strategy=ttnn.ShardStrategy.WIDTH
    )
    block_cfg = ttnn.create_sharded_memory_config(
        shape=shape, core_grid=ttnn.CoreGrid(y=4, x=2), strategy=ttnn.ShardStrategy.BLOCK
    )

    tt_input = ttnn.from_torch(
        torch.rand(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=width_cfg,
    )
    tt_index = _make_index_tensor(torch.tensor([0, 1, 2, 3], dtype=torch.int32), device)

    with expect_error(RuntimeError, "column shard width"):
        ttnn.index_fill(tt_input, 3, tt_index, 1.0, memory_config=block_cfg)


@pytest.mark.parametrize(
    "width_grid, block_grid",
    [
        # All keep matching column shard width; only KW and block row-shard count (KH) vary.
        ((1, 2), (2, 2)),  # KW=2 (col width 32), block KH=2
        ((1, 4), (4, 4)),  # KW=4 (col width 16), block KH=4 (shard_h differs from the 2x4 case)
        ((1, 8), (2, 8)),  # KW=8 (col width 8),  block KH=2
    ],
    ids=["KW2", "KW4-KH4", "KW8"],
)
@pytest.mark.parametrize("direction", ["width_to_block", "block_to_width"])
def test_index_fill_width_block_varied_geometry(device, width_grid, block_grid, direction):
    """WIDTH <-> BLOCK across varied column-shard counts and block row-shard heights."""
    shape = (4, 4, 4, 64)
    dim = 3  # exercise the column-local last-dim fill, the most geometry-sensitive path
    width_cfg = ttnn.create_sharded_memory_config(
        shape=shape, core_grid=ttnn.CoreGrid(y=width_grid[0], x=width_grid[1]), strategy=ttnn.ShardStrategy.WIDTH
    )
    block_cfg = ttnn.create_sharded_memory_config(
        shape=shape, core_grid=ttnn.CoreGrid(y=block_grid[0], x=block_grid[1]), strategy=ttnn.ShardStrategy.BLOCK
    )
    in_cfg, out_cfg = (width_cfg, block_cfg) if direction == "width_to_block" else (block_cfg, width_cfg)

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_index = torch.randint(0, shape[dim], (16,), dtype=torch.int32)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg
    )
    tt_index = _make_index_tensor(torch_index, device)
    tt_out = ttnn.index_fill(tt_input, dim, tt_index, 1.5, memory_config=out_cfg)

    golden = torch.index_fill(torch_input, dim, torch_index.long(), 1.5)
    assert_equal(ttnn.to_torch(tt_out), golden)


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype, value",
    [
        (torch.float32, ttnn.float32, 2.5),
        (torch.int32, ttnn.int32, 7),
    ],
    ids=["float32", "int32"],
)
@pytest.mark.parametrize("direction", ["width_to_block", "block_to_width"])
def test_index_fill_width_block_dtypes(device, torch_dtype, ttnn_dtype, value, direction):
    """WIDTH <-> BLOCK conversion for non-bfloat16 dtypes (4-byte element path)."""
    shape = (4, 4, 4, 64)
    dim = 3
    width_cfg = ttnn.create_sharded_memory_config(
        shape=shape, core_grid=ttnn.CoreGrid(y=1, x=4), strategy=ttnn.ShardStrategy.WIDTH
    )
    block_cfg = ttnn.create_sharded_memory_config(
        shape=shape, core_grid=ttnn.CoreGrid(y=2, x=4), strategy=ttnn.ShardStrategy.BLOCK
    )
    in_cfg, out_cfg = (width_cfg, block_cfg) if direction == "width_to_block" else (block_cfg, width_cfg)

    if torch_dtype == torch.int32:
        torch_input = torch.randint(0, 100, shape, dtype=torch.int32)
    else:
        torch_input = torch.rand(shape, dtype=torch_dtype)
    torch_index = torch.randint(0, shape[dim], (16,), dtype=torch.int32)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg
    )
    tt_index = _make_index_tensor(torch_index, device)
    tt_out = ttnn.index_fill(tt_input, dim, tt_index, value, memory_config=out_cfg)

    golden = torch.index_fill(torch_input, dim, torch_index.long(), value)
    assert_equal(ttnn.to_torch(tt_out), golden)


@pytest.mark.parametrize(
    "shape, dim, num_indices, value",
    [
        ((2, 4, 8, 32), 0, 2, 1.5),
        ((2, 4, 8, 32), 2, 5, -0.5),
        ((2, 4, 8, 32), 3, 8, 2.0),  # last dim
    ],
    ids=["dim0", "dim2", "dim3-last"],
)
@pytest.mark.parametrize("direction", ["interleaved_to_height", "height_to_interleaved"])
def test_index_fill_interleaved_height_conversion(device, shape, dim, num_indices, value, direction):
    """INTERLEAVED <-> HEIGHT_SHARDED conversion (full-row pages on both sides)."""
    interleaved_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    height_cfg = ttnn.create_sharded_memory_config(
        shape=shape, core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.HEIGHT
    )
    in_cfg, out_cfg = (
        (interleaved_cfg, height_cfg) if direction == "interleaved_to_height" else (height_cfg, interleaved_cfg)
    )

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_index = torch.randint(0, shape[dim], (num_indices,), dtype=torch.int32)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg
    )
    tt_index = _make_index_tensor(torch_index, device)
    tt_out = ttnn.index_fill(tt_input, dim, tt_index, value, memory_config=out_cfg)

    golden = torch.index_fill(torch_input, dim, torch_index.long(), value)
    assert_equal(ttnn.to_torch(tt_out), golden)
