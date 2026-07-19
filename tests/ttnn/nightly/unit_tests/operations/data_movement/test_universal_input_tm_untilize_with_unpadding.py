# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Universal input/output support tests for ttnn.untilize_with_unpadding.

Extends the coverage in test_untilize_with_unpadding.py with the memory-config
and layout combinations added as part of the universal I/O support effort (see
issue #49900):
  - interleaved input -> HEIGHT_SHARDED / WIDTH_SHARDED / BLOCK_SHARDED output
    (previously hard-rejected; WIDTH/BLOCK needed a writer-kernel fix - see
    writer_unary_stick_layout_split_rows_multicore.cpp, which now uses
    noc_async_write_sharded with a per-shard page-size override to correctly
    split a row's write across shard columns - an earlier naive host-side-only
    relaxation was hardware-tested and found to silently produce wrong data)
  - BLOCK_SHARDED input -> BLOCK_SHARDED output (previously forced to INTERLEAVED)
  - ROW_MAJOR input/output, interleaved and legacy 2D-sharded (HEIGHT/WIDTH/
    BLOCK_SHARDED), in any combination including cross-shard-type
    (previously hard-rejected entirely). Reuses ttnn::slice's RM reader/writer
    kernels directly (same kernel-reuse convention this op already uses for its
    TILE path) rather than hand-authoring new low-level kernels or composing
    with a separate op call. Those kernels address purely via TensorAccessor
    page-id addressing and noc_async_{read,write}_sharded (which splits a
    logical row across shards when needed), independent of any shard-type
    match between input and output.
  - WIDTH_SHARDED <-> BLOCK_SHARDED cross-shard-type sharded output for the TILE
    path (matching column shard width, unbatched only), via a dedicated writer
    (writer_unary_unpad_cross_sharded.cpp) that addresses the destination through
    TensorAccessor + noc_async_write_sharded instead of the same-shard-type
    writer's same-core L1-to-L1 CB copy.

Follows the test-file convention established by test_split_universal_io.py /
test_sort_universal_io.py: a module-scoped device fixture, inline shard-spec
helper builders, and dtype/rank/shape sweeps per newly-supported combination.

Known gaps not yet covered by source changes (tracked for follow-up, not tested
here as passing):
  - ND-sharded ROW_MAJOR input/output (only legacy 2D sharded is supported)
  - HEIGHT_SHARDED sharded TILE input -> a *different* shard-type sharded output
    (e.g. HEIGHT_SHARDED -> WIDTH_SHARDED) - only WIDTH<->BLOCK cross-type is
    supported so far (matching column shard width); HEIGHT_SHARDED's cross-type
    counterpart hasn't been attempted
  - Sharded output from interleaved input when the block-interleaved factory
    would otherwise be selected (very wide rows) - select_program_factory()
    forces the row-split factory instead, which is correct but forgoes that
    heuristic's performance benefit for those shapes
  - DRAM-sharded *input* for any legacy 2D shard type (HEIGHT/WIDTH/BLOCK) - the
    legacy sharded factory binds the input buffer directly as a circular buffer,
    which only works for L1; this is a pre-existing architectural gap, not
    introduced by this work
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

pytestmark = pytest.mark.use_module_device

TILE = 32
L1 = ttnn.BufferType.L1
DRAM = ttnn.BufferType.DRAM


def _make_height_sharded_cfg(num_shards, shard_h, shard_w, buffer=L1):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_shards - 1))})
    spec = ttnn.ShardSpec(grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer, spec)


def _make_block_sharded_cfg(grid_y, grid_x, shard_h, shard_w, buffer=L1):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    spec = ttnn.ShardSpec(grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, buffer, spec)


def _check(
    device,
    torch_tensor,
    output_end,
    input_memory_config,
    output_memory_config,
    dtype=ttnn.bfloat16,
    input_layout=ttnn.TILE_LAYOUT,
):
    tile_tensor = ttnn.from_torch(torch_tensor, dtype=dtype, layout=input_layout)
    tile_tensor = ttnn.to_device(tile_tensor, device, memory_config=input_memory_config)
    untilized = ttnn.untilize_with_unpadding(
        tile_tensor, output_tensor_end=output_end, memory_config=output_memory_config
    )
    result = ttnn.to_torch(untilized)
    slices = tuple(slice(0, output_end[i] + 1) for i in range(len(output_end)))
    expected = torch_tensor[slices]
    assert_equal(result, expected)
    return untilized


# ---------------------------------------------------------------------------
# Interleaved input -> HEIGHT_SHARDED output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("input_buffer_type", [L1, DRAM])
@pytest.mark.parametrize(
    "shape, output_end, num_shards",
    [
        # Tile-aligned truncation.
        ([1, 1, 4 * TILE, 2 * TILE], [0, 0, 4 * TILE - 1, 2 * TILE - 1], 4),
        # Non-tile-aligned truncation (real unpadding).
        ([1, 1, 8 * TILE, 2 * TILE], [0, 0, 100, 2 * TILE - 1], 4),
        # Single core.
        ([1, 1, 4 * TILE, TILE], [0, 0, 50, TILE - 1], 1),
        # Multi-batch input, still unbatched by output_end (rank-4, batch=1 in output slice).
        ([1, 2, 4 * TILE, TILE], [0, 1, 4 * TILE - 1, TILE - 1], 2),
    ],
)
def test_interleaved_input_height_sharded_output(device, dtype, input_buffer_type, shape, output_end, num_shards):
    torch.manual_seed(0)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_tensor = torch.rand(shape, dtype=torch_dtype)

    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, input_buffer_type)
    out_height = output_end[-2] + 1
    out_width = output_end[-1] + 1
    output_memory_config = _make_height_sharded_cfg(num_shards, TILE, out_width)

    untilized = _check(device, torch_tensor, output_end, input_memory_config, output_memory_config, dtype=dtype)
    assert untilized.memory_config().is_sharded(), "Output should be sharded"
    assert untilized.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED


@pytest.mark.parametrize(
    "shard_layout, grid, shard_shape",
    [
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (1, 0), (4 * TILE, 2 * TILE)),
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, (1, 1), (2 * TILE, 2 * TILE)),
    ],
    ids=["width_sharded", "block_sharded"],
)
@pytest.mark.parametrize(
    "shape, output_end",
    [
        ([1, 1, 4 * TILE, 4 * TILE], [0, 0, 4 * TILE - 1, 4 * TILE - 1]),  # exact pass-through
        ([1, 1, 4 * TILE, 3 * TILE], [0, 0, 100, 70]),  # non-tile-aligned unpad, stresses row-split
    ],
)
def test_interleaved_input_width_block_sharded_output(device, shard_layout, grid, shard_shape, shape, output_end):
    """Interleaved input -> WIDTH_SHARDED / BLOCK_SHARDED output. The row-split multi-core
    writer (writer_unary_stick_layout_split_rows_multicore.cpp) now uses
    noc_async_write_sharded with a per-shard page-size override to split each row's write
    across shard columns, which is the real fix for what an earlier naive host-side-only
    relaxation got wrong (see gap #2b in the source comments) - select_program_factory()
    forces this factory whenever output is sharded, bypassing the block-interleaved/
    wide-row heuristic path (not yet updated for sharded output)."""
    torch.manual_seed(0)
    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)
    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(*grid))
    spec = ttnn.ShardSpec(ttnn.CoreRangeSet({core_range}), shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_memory_config = ttnn.MemoryConfig(shard_layout, L1, spec)

    untilized = _check(
        device,
        torch_tensor,
        output_end,
        ttnn.DRAM_MEMORY_CONFIG,
        output_memory_config,
    )
    assert untilized.memory_config().is_sharded()
    assert untilized.memory_config().memory_layout == shard_layout


# ---------------------------------------------------------------------------
# BLOCK_SHARDED input -> BLOCK_SHARDED output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize(
    "shape, output_end, shard_shape, grid_size",
    [
        # Tile-aligned, full grid.
        ([1, 1, 128, 128], [0, 0, 127, 127], (64, 64), (2, 2)),
        # Non-tile-aligned unpadding on both dims (real regression target for gap #3).
        ([1, 1, 128, 128], [0, 0, 100, 100], (64, 64), (2, 2)),
        ([1, 1, 256, 256], [0, 0, 200, 200], (64, 64), (4, 4)),
    ],
)
def test_block_sharded_input_block_sharded_output(device, dtype, shape, output_end, shard_shape, grid_size):
    torch.manual_seed(42)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_tensor = torch.rand(shape, dtype=torch_dtype)

    input_memory_config = _make_block_sharded_cfg(grid_size[1], grid_size[0], shard_shape[0], shard_shape[1])
    # Output shard shape only needs to describe the grid - compute_output_specs() derives
    # the true per-shard height/width from the unpadded output shape.
    output_memory_config = _make_block_sharded_cfg(grid_size[1], grid_size[0], shard_shape[0], shard_shape[1])

    untilized = _check(device, torch_tensor, output_end, input_memory_config, output_memory_config, dtype=dtype)
    assert untilized.memory_config().is_sharded(), "Output should be sharded"
    assert untilized.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED


def test_block_sharded_input_block_sharded_output_col_major_orientation(device):
    torch.manual_seed(7)
    shape = [1, 1, 128, 128]
    output_end = [0, 0, 100, 100]
    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    input_spec = ttnn.ShardSpec(grid, [64, 64], ttnn.ShardOrientation.COL_MAJOR)
    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, L1, input_spec)
    output_spec = ttnn.ShardSpec(grid, [64, 64], ttnn.ShardOrientation.COL_MAJOR)
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, L1, output_spec)

    _check(device, torch_tensor, output_end, input_memory_config, output_memory_config)


def test_block_sharded_input_interleaved_output_still_works(device):
    """Regression: the pre-existing BLOCK_SHARDED -> INTERLEAVED path must be unaffected."""
    torch.manual_seed(1)
    shape = [1, 1, 128, 128]
    output_end = [0, 0, 100, 100]
    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)

    input_memory_config = _make_block_sharded_cfg(2, 2, 64, 64)
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, L1)

    untilized = _check(device, torch_tensor, output_end, input_memory_config, output_memory_config)
    assert not untilized.memory_config().is_sharded()


# ---------------------------------------------------------------------------
# Cross-shard-type sharded-to-sharded output (WIDTH_SHARDED <-> BLOCK_SHARDED)
# ---------------------------------------------------------------------------


def _make_width_sharded_cfg(num_shards, shard_h, shard_w, buffer=L1):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_shards - 1, 0))})
    spec = ttnn.ShardSpec(grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer, spec)


@pytest.mark.parametrize(
    "direction, output_end",
    [
        ("width_to_block", [0, 0, 4 * TILE - 1, 4 * TILE - 1]),  # exact pass-through
        ("width_to_block", [0, 0, 100, 100]),  # real unpad, stresses row + column trimming
        ("block_to_width", [0, 0, 4 * TILE - 1, 4 * TILE - 1]),
        ("block_to_width", [0, 0, 100, 100]),
    ],
    ids=["w2b_exact", "w2b_unpad", "b2w_exact", "b2w_unpad"],
)
def test_width_block_sharded_cross_type_output(device, direction, output_end):
    """WIDTH_SHARDED <-> BLOCK_SHARDED with a matching column shard width. The executing core
    (owner of the input's column shard) need not be the physically-owning core of the output
    shard, so the writer (writer_unary_unpad_cross_sharded.cpp) addresses the destination via
    TensorAccessor + noc_async_write_sharded rather than the same-core L1-to-L1 copy the
    same-shard-type path uses."""
    torch.manual_seed(11)
    shape = [1, 1, 4 * TILE, 4 * TILE]
    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)

    width_cfg = _make_width_sharded_cfg(4, output_end[2] + 1, TILE)
    block_cfg = _make_block_sharded_cfg(2, 4, 2 * TILE, TILE)
    input_memory_config, output_memory_config = (
        (_make_width_sharded_cfg(4, 4 * TILE, TILE), block_cfg)
        if direction == "width_to_block"
        else (block_cfg, width_cfg)
    )

    untilized = _check(device, torch_tensor, output_end, input_memory_config, output_memory_config)
    assert untilized.memory_config().is_sharded()
    expected_layout = (
        ttnn.TensorMemoryLayout.BLOCK_SHARDED
        if direction == "width_to_block"
        else ttnn.TensorMemoryLayout.WIDTH_SHARDED
    )
    assert untilized.memory_config().memory_layout == expected_layout


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize("direction", ["width_to_block", "block_to_width"])
def test_width_block_sharded_cross_type_output_dtype(device, dtype, direction):
    """WIDTH_SHARDED <-> BLOCK_SHARDED cross-type output across non-bfloat16 dtypes - the
    main test above only exercises bfloat16."""
    torch.manual_seed(14)
    shape = [1, 1, 4 * TILE, 4 * TILE]
    output_end = [0, 0, 4 * TILE - 1, 4 * TILE - 1]
    if dtype in (ttnn.int32, ttnn.uint32):
        torch_tensor = torch.randint(-1000, 1000, shape, dtype=torch.int32)
        if dtype == ttnn.uint32:
            torch_tensor = torch_tensor.abs()
    else:
        torch_tensor = torch.rand(shape, dtype=torch.float32)

    width_cfg = _make_width_sharded_cfg(4, 4 * TILE, TILE)
    block_cfg = _make_block_sharded_cfg(2, 4, 2 * TILE, TILE)
    input_memory_config, output_memory_config = (
        (width_cfg, block_cfg) if direction == "width_to_block" else (block_cfg, width_cfg)
    )

    untilized = _check(device, torch_tensor, output_end, input_memory_config, output_memory_config, dtype=dtype)
    assert untilized.memory_config().is_sharded()


def test_width_block_sharded_cross_type_program_cache_reuse(device):
    torch.manual_seed(15)
    shape = [1, 1, 4 * TILE, 4 * TILE]
    output_end = [0, 0, 100, 100]
    input_memory_config = _make_width_sharded_cfg(4, 4 * TILE, TILE)
    output_memory_config = _make_block_sharded_cfg(2, 4, 2 * TILE, TILE)

    for _ in range(2):
        torch_tensor = torch.rand(shape, dtype=torch.bfloat16)
        _check(device, torch_tensor, output_end, input_memory_config, output_memory_config)


# ---------------------------------------------------------------------------
# Program-cache sanity across the newly-supported paths
# ---------------------------------------------------------------------------


def test_program_cache_reuse_interleaved_to_height_sharded(device):
    torch.manual_seed(3)
    shape = [1, 1, 4 * TILE, 2 * TILE]
    output_end = [0, 0, 4 * TILE - 1, 2 * TILE - 1]
    input_memory_config = ttnn.DRAM_MEMORY_CONFIG
    output_memory_config = _make_height_sharded_cfg(4, TILE, 2 * TILE)

    for _ in range(2):
        torch_tensor = torch.rand(shape, dtype=torch.bfloat16)
        _check(device, torch_tensor, output_end, input_memory_config, output_memory_config)


def test_program_cache_reuse_block_sharded_to_block_sharded(device):
    torch.manual_seed(4)
    shape = [1, 1, 128, 128]
    output_end = [0, 0, 100, 100]
    input_memory_config = _make_block_sharded_cfg(2, 2, 64, 64)
    output_memory_config = _make_block_sharded_cfg(2, 2, 64, 64)

    for _ in range(2):
        torch_tensor = torch.rand(shape, dtype=torch.bfloat16)
        _check(device, torch_tensor, output_end, input_memory_config, output_memory_config)


# ---------------------------------------------------------------------------
# DRAM-sharded input/output
# ---------------------------------------------------------------------------


def _height_shard_dram_single_core(shard_h, shard_w):
    """DRAM has a much narrower core grid than the compute grid - a single DRAM core
    is the safe, portable config (same convention as test_sort_universal_io_extended.py)."""
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    spec = ttnn.ShardSpec(grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, spec)


def test_interleaved_dram_input_dram_sharded_output(device):
    """Gap #2 with a DRAM (not L1) sharded output buffer type."""
    torch.manual_seed(11)
    shape = [1, 1, 4 * TILE, 2 * TILE]
    output_end = [0, 0, 100, 2 * TILE - 1]
    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)

    input_memory_config = ttnn.DRAM_MEMORY_CONFIG
    output_memory_config = _height_shard_dram_single_core(4 * TILE, 2 * TILE)

    untilized = _check(device, torch_tensor, output_end, input_memory_config, output_memory_config)
    assert untilized.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert untilized.memory_config().is_sharded()


# ---------------------------------------------------------------------------
# int32 / uint32 dtype coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize("buffer_type", [L1, DRAM])
def test_interleaved_dtype_int_uint32(device, dtype, buffer_type):
    torch.manual_seed(21)
    shape = [1, 1, 3 * TILE, 2 * TILE]
    output_end = [0, 0, 70, 50]
    torch_dtype = torch.int32
    torch_tensor = torch.randint(-1000, 1000, shape, dtype=torch_dtype)
    if dtype == ttnn.uint32:
        torch_tensor = torch_tensor.abs()

    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)
    _check(device, torch_tensor, output_end, input_memory_config, output_memory_config, dtype=dtype)


@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.uint32])
def test_height_sharded_dtype_int_uint32(device, dtype):
    torch.manual_seed(22)
    shape = [1, 1, 4 * TILE, 2 * TILE]
    output_end = [0, 0, 100, 2 * TILE - 1]
    torch_tensor = torch.randint(-1000, 1000, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        torch_tensor = torch_tensor.abs()

    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, L1)
    output_memory_config = _make_height_sharded_cfg(4, TILE, 2 * TILE)
    _check(device, torch_tensor, output_end, input_memory_config, output_memory_config, dtype=dtype)


@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.uint32])
def test_block_sharded_dtype_int_uint32(device, dtype):
    torch.manual_seed(23)
    shape = [1, 1, 128, 128]
    output_end = [0, 0, 100, 100]
    torch_tensor = torch.randint(-1000, 1000, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        torch_tensor = torch_tensor.abs()

    input_memory_config = _make_block_sharded_cfg(2, 2, 64, 64)
    output_memory_config = _make_block_sharded_cfg(2, 2, 64, 64)
    _check(device, torch_tensor, output_end, input_memory_config, output_memory_config, dtype=dtype)


# ---------------------------------------------------------------------------
# ROW_MAJOR input (interleaved only - sharded RM input/output not yet supported)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize("buffer_type", [L1, DRAM])
@pytest.mark.parametrize(
    "shape, output_end",
    [
        ([1, 1, 64, 64], [0, 0, 50, 40]),  # 2D-ish, non-tile-aligned unpad both dims
        ([2, 3, 100, 200], [1, 1, 50, 120]),  # multi-dim unpadding across batch and channel
        ([1, 1, TILE, TILE], [0, 0, TILE - 1, TILE - 1]),  # exact pass-through, no unpadding
        ([237], [100]),  # rank 1
        ([1, 55, 43], [0, 40, 30]),  # rank 3, no tile alignment anywhere
    ],
)
def test_row_major_input_interleaved(device, dtype, buffer_type, shape, output_end):
    torch.manual_seed(0)
    if dtype in (ttnn.int32, ttnn.uint32):
        torch_tensor = torch.randint(-1000, 1000, shape, dtype=torch.int32)
        if dtype == ttnn.uint32:
            torch_tensor = torch_tensor.abs()
    else:
        torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
        torch_tensor = torch.rand(shape, dtype=torch_dtype)

    input_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type)

    untilized = _check(
        device,
        torch_tensor,
        output_end,
        input_memory_config,
        output_memory_config,
        dtype=dtype,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    assert untilized.layout == ttnn.ROW_MAJOR_LAYOUT
    assert not untilized.memory_config().is_sharded()


@pytest.mark.parametrize(
    "make_cfg, cfg_args",
    [
        (_make_height_sharded_cfg, (4, TILE, 2 * TILE)),
        (_make_width_sharded_cfg, (2, 4 * TILE, TILE)),
        (_make_block_sharded_cfg, (2, 2, 2 * TILE, TILE)),
    ],
    ids=["height_sharded", "width_sharded", "block_sharded"],
)
@pytest.mark.parametrize(
    "shape, output_end",
    [
        ([1, 1, 4 * TILE, 2 * TILE], [0, 0, 4 * TILE - 1, 2 * TILE - 1]),  # exact pass-through
        ([1, 1, 4 * TILE, 2 * TILE], [0, 0, 100, 50]),  # non-tile-aligned unpad on both dims
    ],
    ids=["exact", "unpad"],
)
def test_row_major_sharded_input_interleaved_output(device, make_cfg, cfg_args, shape, output_end):
    """Legacy 2D-sharded ROW_MAJOR input -> INTERLEAVED output. The reused slice
    reader/writer kernels split a logical row across shards via
    noc_async_{read,write}_sharded when needed (BLOCK/WIDTH_SHARDED), so this works
    even though the TILE-path writers cannot (see gap #2b in the source comments).
    Output is interleaved, so the unpad amount doesn't affect the input shard config's
    validity - only the RM kernels' row-splitting math is exercised differently."""
    torch.manual_seed(0)
    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)
    input_memory_config = make_cfg(*cfg_args)
    output_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, L1)

    untilized = _check(
        device,
        torch_tensor,
        output_end,
        input_memory_config,
        output_memory_config,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    assert not untilized.memory_config().is_sharded()


@pytest.mark.parametrize(
    "make_cfg, cfg_args, shape, output_end",
    [
        (_make_height_sharded_cfg, (4, TILE, 2 * TILE), [1, 1, 4 * TILE, 2 * TILE], [0, 0, 4 * TILE - 1, 2 * TILE - 1]),
        (_make_width_sharded_cfg, (2, 4 * TILE, TILE), [1, 1, 4 * TILE, 2 * TILE], [0, 0, 4 * TILE - 1, 2 * TILE - 1]),
        (
            _make_block_sharded_cfg,
            (2, 2, 2 * TILE, TILE),
            [1, 1, 4 * TILE, 2 * TILE],
            [0, 0, 4 * TILE - 1, 2 * TILE - 1],
        ),
        # Non-tile-aligned unpad (real unpadding, not exact pass-through): shape 128x64 -> 100x64,
        # chosen so it divides evenly across each layout's grid (4 / 2 / 2x2 cores).
        (_make_height_sharded_cfg, (4, 25, 64), [1, 1, 4 * TILE, 2 * TILE], [0, 0, 99, 63]),
        (_make_width_sharded_cfg, (2, 100, 32), [1, 1, 4 * TILE, 2 * TILE], [0, 0, 99, 63]),
        (_make_block_sharded_cfg, (2, 2, 50, 32), [1, 1, 4 * TILE, 2 * TILE], [0, 0, 99, 63]),
    ],
    ids=["height_exact", "width_exact", "block_exact", "height_unpad", "width_unpad", "block_unpad"],
)
def test_row_major_interleaved_input_sharded_output(device, make_cfg, cfg_args, shape, output_end):
    """Interleaved ROW_MAJOR input -> sharded output. The reused slice writer kernel
    addresses the destination via TensorAccessor (sharding-agnostic) and
    noc_async_write_sharded (splits a row across shards for B/W-sharded), so the
    output shard spec is used verbatim from the caller with no tile-alignment
    derivation needed - see compute_output_specs()."""
    torch.manual_seed(0)
    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)
    input_memory_config = ttnn.DRAM_MEMORY_CONFIG
    output_memory_config = make_cfg(*cfg_args)

    untilized = _check(
        device,
        torch_tensor,
        output_end,
        input_memory_config,
        output_memory_config,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    assert untilized.memory_config().is_sharded()


def test_row_major_cross_shard_type_input_to_output(device):
    """Sharded ROW_MAJOR input -> differently-sharded ROW_MAJOR output (e.g.
    HEIGHT_SHARDED -> WIDTH_SHARDED). Unlike the TILE path (see gap #2b), the RM
    kernels address purely by page id via TensorAccessor, independent of any
    shard-type match between input and output, so this just works."""
    torch.manual_seed(0)
    shape = [1, 1, 4 * TILE, 2 * TILE]
    output_end = [0, 0, 4 * TILE - 1, 2 * TILE - 1]
    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)
    input_memory_config = _make_height_sharded_cfg(4, TILE, 2 * TILE)
    output_memory_config = _make_width_sharded_cfg(2, 4 * TILE, TILE)

    untilized = _check(
        device,
        torch_tensor,
        output_end,
        input_memory_config,
        output_memory_config,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    assert untilized.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED


def test_row_major_program_cache_reuse(device):
    torch.manual_seed(5)
    shape = [1, 1, 4 * TILE, 2 * TILE]
    output_end = [0, 0, 100, 50]
    input_memory_config = ttnn.DRAM_MEMORY_CONFIG
    output_memory_config = ttnn.DRAM_MEMORY_CONFIG

    for _ in range(2):
        torch_tensor = torch.rand(shape, dtype=torch.bfloat16)
        _check(
            device,
            torch_tensor,
            output_end,
            input_memory_config,
            output_memory_config,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
        )


def test_row_major_sharded_output_program_cache_reuse(device):
    torch.manual_seed(6)
    shape = [1, 1, 4 * TILE, 2 * TILE]
    output_end = [0, 0, 4 * TILE - 1, 2 * TILE - 1]
    input_memory_config = ttnn.DRAM_MEMORY_CONFIG
    output_memory_config = _make_height_sharded_cfg(4, TILE, 2 * TILE)

    for _ in range(2):
        torch_tensor = torch.rand(shape, dtype=torch.bfloat16)
        _check(
            device,
            torch_tensor,
            output_end,
            input_memory_config,
            output_memory_config,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
