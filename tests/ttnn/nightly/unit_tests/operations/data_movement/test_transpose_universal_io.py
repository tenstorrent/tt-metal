# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Tests for transpose universal input/output support.

Each test covers one distinct code path — no duplicate shapes or redundant combos.
"""

import pytest
import torch

import ttnn

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


def run_transpose_test(
    shape,
    dim0,
    dim1,
    device,
    input_layout=ttnn.TILE_LAYOUT,
    input_mem_config=None,
    output_mem_config=None,
    dtype=ttnn.bfloat16,
):
    """Helper to run a single transpose test with the given configs."""
    torch.manual_seed(12345)
    x = torch.rand(shape).bfloat16().float()

    if input_mem_config is None:
        input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    ttnn_input = ttnn.from_torch(x, layout=input_layout, dtype=dtype, device=device, memory_config=input_mem_config)
    result = ttnn.transpose(ttnn_input, dim0, dim1, memory_config=output_mem_config)

    if output_mem_config is not None and output_mem_config.shard_spec is not None:
        actual = result.memory_config()
        assert (
            actual.memory_layout == output_mem_config.memory_layout
        ), f"Expected output memory layout {output_mem_config.memory_layout}, got {actual.memory_layout}"

    ref = x.transpose(dim0, dim1)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))

    passing, output = comp_equal(ref, got)
    logger.info(output)
    assert passing, f"Transpose mismatch for shape={shape}, dims=({dim0},{dim1})"


def _block_shard_config(shape, device):
    """Create a 2x2 block-sharded MemoryConfig for the given 4D shape."""
    compute_grid = device.compute_with_storage_grid_size()
    grid_x = min(2, compute_grid.x)
    grid_y = min(2, compute_grid.y)
    total_height = shape[0] * shape[1] * shape[2]
    shard_shape = (total_height // grid_y, shape[3] // grid_x)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)


def _height_shard_config(shape, device, num_cores=4, buffer_type=ttnn.BufferType.L1):
    """Create a height-sharded MemoryConfig for the given 4D shape."""
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, compute_grid.x * compute_grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
    total_height = shape[0] * shape[1] * shape[2]
    shard_shape = (total_height // num_cores, shape[3])
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type, shard_spec)


def _width_shard_config(shape, device, num_cores=4):
    """Create a width-sharded MemoryConfig for the given 4D shape."""
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, compute_grid.x * compute_grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
    total_height = shape[0] * shape[1] * shape[2]
    shard_shape = (total_height, shape[3] // num_cores)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
DRAM_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


# ---------------------------------------------------------------------------
# 1. Block-sharded input → interleaved output  (WH)
# ---------------------------------------------------------------------------
def test_transpose_block_sharded_to_interleaved_wh(device):
    shape = (1, 1, 64, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_block_shard_config(shape, device),
        output_mem_config=L1_INTERLEAVED,
    )


# ---------------------------------------------------------------------------
# 2. Block-sharded input → interleaved output  (CN, needs N,C > 1)
# ---------------------------------------------------------------------------
def test_transpose_block_sharded_to_interleaved_cn(device):
    shape = (2, 4, 32, 64)
    run_transpose_test(
        shape,
        0,
        1,
        device,
        input_mem_config=_block_shard_config(shape, device),
        output_mem_config=L1_INTERLEAVED,
    )


# ---------------------------------------------------------------------------
# 3. Width-sharded input → interleaved output  (WH)
# ---------------------------------------------------------------------------
def test_transpose_width_sharded_to_interleaved_wh(device):
    shape = (1, 1, 64, 128)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_width_shard_config(shape, device),
        output_mem_config=L1_INTERLEAVED,
    )


# ---------------------------------------------------------------------------
# 4. Interleaved input → height-sharded output  (WH)
# ---------------------------------------------------------------------------
def test_transpose_interleaved_to_height_sharded(device):
    shape = (1, 1, 64, 128)
    out_shape = (1, 1, 128, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=L1_INTERLEAVED,
        output_mem_config=_height_shard_config(out_shape, device),
    )


# ---------------------------------------------------------------------------
# 5. Interleaved input → block-sharded output  (WH)
# ---------------------------------------------------------------------------
def test_transpose_interleaved_to_block_sharded(device):
    shape = (1, 1, 64, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=L1_INTERLEAVED,
        output_mem_config=_block_shard_config(shape, device),
    )


# ---------------------------------------------------------------------------
# 6. Block-sharded input → height-sharded output  (WH)
# ---------------------------------------------------------------------------
def test_transpose_block_to_height_sharded(device):
    shape = (1, 1, 64, 128)
    out_shape = (1, 1, 128, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_block_shard_config(shape, device),
        output_mem_config=_height_shard_config(out_shape, device, num_cores=2),
    )


# ---------------------------------------------------------------------------
# 7. Width-sharded input → height-sharded output  (WH, cross shard-type)
# ---------------------------------------------------------------------------
def test_transpose_width_to_height_sharded(device):
    shape = (1, 1, 64, 128)
    out_shape = (1, 1, 128, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_width_shard_config(shape, device),
        output_mem_config=_height_shard_config(out_shape, device),
    )


# ---------------------------------------------------------------------------
# 8. HC transpose: height-sharded input → interleaved output
# ---------------------------------------------------------------------------
def test_transpose_hc_height_sharded(device):
    shape = (1, 4, 32, 64)
    run_transpose_test(
        shape,
        1,
        2,
        device,
        input_mem_config=_height_shard_config(shape, device),
        output_mem_config=L1_INTERLEAVED,
    )


# ---------------------------------------------------------------------------
# 9. CN transpose: height-sharded input → interleaved output
# ---------------------------------------------------------------------------
def test_transpose_cn_height_sharded(device):
    shape = (2, 4, 32, 64)
    run_transpose_test(
        shape,
        0,
        1,
        device,
        input_mem_config=_height_shard_config(shape, device, num_cores=8),
        output_mem_config=L1_INTERLEAVED,
    )


# ---------------------------------------------------------------------------
# 10. Default output derivation: block-sharded input, no explicit output
# ---------------------------------------------------------------------------
def test_transpose_block_sharded_default_output(device):
    shape = (1, 1, 64, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_block_shard_config(shape, device),
    )


# ---------------------------------------------------------------------------
# 11. Default output derivation: height-sharded input, no explicit output
# ---------------------------------------------------------------------------
def test_transpose_height_sharded_default_output(device):
    shape = (2, 4, 32, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_height_shard_config(shape, device),
    )


# ---------------------------------------------------------------------------
# 12. Sharded output without shard_spec (system derives it)
# ---------------------------------------------------------------------------
def test_transpose_sharded_output_no_shard_spec(device):
    shape = (1, 1, 128, 64)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_height_shard_config(shape, device),
        output_mem_config=output_mem_config,
    )


# ---------------------------------------------------------------------------
# 13. Native L1 height-sharded (verifies native path still selected)
# ---------------------------------------------------------------------------
def test_transpose_native_height_sharded_wh(device):
    shape = (1, 1, 32, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_height_shard_config(shape, device, num_cores=1),
    )


# ---------------------------------------------------------------------------
# 14. Interleaved baseline (sanity)
# ---------------------------------------------------------------------------
def test_transpose_interleaved_baseline(device):
    shape = (2, 3, 64, 96)
    run_transpose_test(shape, 2, 3, device)


# ---------------------------------------------------------------------------
# 15. ROW_MAJOR height-sharded input (exercises ROW_MAJOR branch in
# get_transpose_shard_specs / is_shard_tile_aligned)
# ---------------------------------------------------------------------------
def test_transpose_row_major_height_sharded_hc(device):
    shape = (1, 4, 32, 64)
    run_transpose_test(
        shape,
        1,
        2,
        device,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        input_mem_config=_height_shard_config(shape, device),
        output_mem_config=L1_INTERLEAVED,
    )


# ---------------------------------------------------------------------------
# 16. DRAM-sharded input falls back to interleaved
# (is_native_transpose_sharding rejects BufferType::DRAM)
# ---------------------------------------------------------------------------
def test_transpose_dram_sharded_fallback(device):
    shape = (1, 1, 128, 64)
    dram_sharded = _height_shard_config(shape, device, buffer_type=ttnn.BufferType.DRAM)

    torch.manual_seed(12345)
    x = torch.rand(shape).bfloat16().float()
    ttnn_input = ttnn.from_torch(
        x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=dram_sharded
    )
    result = ttnn.transpose(ttnn_input, 2, 3, memory_config=L1_INTERLEAVED)
    assert (
        result.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    ), f"Expected fallback to INTERLEAVED, got {result.memory_config().memory_layout}"

    ref = x.transpose(2, 3)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    passing, output = comp_equal(ref, got)
    logger.info(output)
    assert passing, f"Transpose mismatch for DRAM-sharded fallback, shape={shape}"


# ---------------------------------------------------------------------------
# 17. Block-sharded output without shard_spec — exercises the block-shard
# branch of generate_transpose_shard_spec (interleaved input, no spec)
# ---------------------------------------------------------------------------
def test_transpose_block_sharded_output_no_shard_spec(device):
    shape = (1, 1, 128, 128)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=L1_INTERLEAVED,
        output_mem_config=output_mem_config,
    )


# ---------------------------------------------------------------------------
# 18. ROW_MAJOR + BLOCK_SHARDED input → interleaved (WH)
# Composite fallback: sharded→interleaved hop before transpose. Neither the
# native sharded kernels nor prim::permute handle block-sharded RM pages that
# span multiple cores.
# ---------------------------------------------------------------------------
def test_transpose_row_major_block_sharded_to_interleaved_wh(device):
    shape = (1, 1, 64, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        input_mem_config=_block_shard_config(shape, device),
        output_mem_config=L1_INTERLEAVED,
    )


# ---------------------------------------------------------------------------
# 19. ROW_MAJOR + WIDTH_SHARDED input → interleaved (WH)
# Composite fallback. Width sharding on RM also splits pages across cores.
# ---------------------------------------------------------------------------
def test_transpose_row_major_width_sharded_to_interleaved_wh(device):
    shape = (1, 1, 32, 128)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        input_mem_config=_width_shard_config(shape, device),
        output_mem_config=L1_INTERLEAVED,
    )


# ---------------------------------------------------------------------------
# 20. ROW_MAJOR interleaved input → BLOCK_SHARDED output requested (WH)
# Composite fallback goes through interleaved output, then reshards to the
# requested block-sharded memory config.
# ---------------------------------------------------------------------------
def test_transpose_row_major_interleaved_to_block_sharded_wh(device):
    shape = (1, 1, 64, 64)
    out_shape = (1, 1, 64, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        input_mem_config=L1_INTERLEAVED,
        output_mem_config=_block_shard_config(out_shape, device),
    )


# ---------------------------------------------------------------------------
# 21. ROW_MAJOR interleaved input → WIDTH_SHARDED output requested (WH)
# Composite fallback — mirror of 20 for width sharding output.
# ---------------------------------------------------------------------------
def test_transpose_row_major_interleaved_to_width_sharded_wh(device):
    shape = (1, 1, 64, 128)
    out_shape = (1, 1, 128, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        input_mem_config=L1_INTERLEAVED,
        output_mem_config=_width_shard_config(out_shape, device),
    )


# ---------------------------------------------------------------------------
# 22. ROW_MAJOR + BLOCK_SHARDED input → BLOCK_SHARDED output (WH)
# Both conditions of the composite-fallback guard fire: input needs an
# interleaved hop in, and output needs a reshard out.
# ---------------------------------------------------------------------------
def test_transpose_row_major_block_sharded_to_block_sharded_wh(device):
    shape = (1, 1, 64, 64)
    out_shape = (1, 1, 64, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        input_mem_config=_block_shard_config(shape, device),
        output_mem_config=_block_shard_config(out_shape, device),
    )


# ---------------------------------------------------------------------------
# 23. ROW_MAJOR + HEIGHT_SHARDED input → interleaved (WH)
# NOT part of the composite fallback — HEIGHT_SHARDED RM keeps pages on a
# single core and is handled natively. Included here as a regression guard so
# the fallback predicate stays narrow.
# ---------------------------------------------------------------------------
def test_transpose_row_major_height_sharded_to_interleaved_wh(device):
    shape = (1, 1, 64, 32)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        input_mem_config=_height_shard_config(shape, device),
        output_mem_config=L1_INTERLEAVED,
    )


# ===========================================================================
# Irregular-shape coverage for interleaved inputs (TILE and ROW_MAJOR).
# Shapes include:
#   - last-two dims not multiples of TILE_HEIGHT / TILE_WIDTH (tile padding)
#   - non-power-of-two C for HC (transposed_padded_shape rounding)
# ===========================================================================
@pytest.mark.parametrize(
    "shape, dim0, dim1",
    [
        # WH, last-two dims irregular
        ((1, 1, 65, 97), 2, 3),
        ((2, 3, 71, 79), 2, 3),
        # HC, irregular C
        ((1, 13, 47, 64), 1, 2),
        ((1, 7, 33, 96), 1, 2),
        # CN
        ((3, 5, 32, 64), 0, 1),
    ],
)
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_transpose_irregular_shapes_interleaved(shape, dim0, dim1, input_layout, device):
    run_transpose_test(shape, dim0, dim1, device, input_layout=input_layout)


# ===========================================================================
# ROW_MAJOR interleaved input + sharded (BLOCK/WIDTH) output WITHOUT shard_spec.
# Exercises the RM branch of `transposed_padded_shape`
# (transpose_device_operation.cpp) and the composite-fallback spec-synthesis
# path in `transpose.cpp` that populates a shard_spec before the reshard.
# ===========================================================================
@pytest.mark.parametrize(
    "shape, dim0, dim1, memory_layout",
    [
        ((1, 1, 64, 128), 2, 3, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        ((1, 1, 64, 128), 2, 3, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        ((1, 4, 32, 64), 1, 2, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        ((1, 4, 32, 64), 1, 2, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    ],
)
def test_transpose_row_major_sharded_output_no_shard_spec(shape, dim0, dim1, memory_layout, device):
    output_mem_config = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1)
    run_transpose_test(
        shape,
        dim0,
        dim1,
        device,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        input_mem_config=L1_INTERLEAVED,
        output_mem_config=output_mem_config,
    )


# ===========================================================================
# TILE + sharded output WITHOUT shard_spec, irregular C for HC.
# Exercises `transposed_padded_shape` with `C_p = round_up(C, tile_height)`
# in transpose_device_operation.cpp.
# ===========================================================================
@pytest.mark.parametrize(
    "shape, dim0, dim1",
    [
        ((1, 13, 32, 64), 1, 2),
        ((1, 47, 32, 64), 1, 2),
        ((1, 7, 96, 64), 1, 2),
    ],
)
def test_transpose_tile_hc_irregular_c_sharded_output_no_shard_spec(shape, dim0, dim1, device):
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)
    run_transpose_test(
        shape,
        dim0,
        dim1,
        device,
        input_layout=ttnn.TILE_LAYOUT,
        input_mem_config=L1_INTERLEAVED,
        output_mem_config=output_mem_config,
    )


# ===========================================================================
# Sharded inputs with shapes that don't divide evenly.
#
# Two categories:
#
#  (a) Logical shape has dims < tile multiple but padded shape is tile-aligned
#      and divides evenly into a tile-aligned shard. The native shard path
#      still runs; the irregular logical dim must be round-tripped correctly.
#
#  (b) Uneven sharding: padded shape does NOT divide evenly into the shard
#      shape (last shard partially filled). `is_native_transpose_sharding`
#      returns false via `is_uneven`, so the op falls through to interleaved
#      factories with TensorAccessorArgs.
#
# RM inputs additionally exercise the composite fallback from transpose.cpp.
# ===========================================================================


def _explicit_block_shard_config(grid_y, grid_x, sh, sw):
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)


def _explicit_height_shard_config(device, ncores, sh, sw):
    spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, device.compute_with_storage_grid_size(), True),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


def _explicit_width_shard_config(device, ncores, sh, sw):
    spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, device.compute_with_storage_grid_size(), True),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)


# (a) TILE + sharded inputs with IRREGULAR logical shape but tile-aligned padded + shard.
@pytest.mark.parametrize(
    "shape, dim0, dim1, mc_factory",
    [
        pytest.param((1, 1, 65, 64), 2, 3, lambda d: _explicit_block_shard_config(3, 2, 32, 32), id="block_3x2_32x32"),
        pytest.param((1, 1, 64, 97), 2, 3, lambda d: _explicit_block_shard_config(2, 4, 32, 32), id="block_2x4_32x32"),
        pytest.param((1, 1, 65, 64), 2, 3, lambda d: _explicit_height_shard_config(d, 3, 32, 64), id="height_3_32x64"),
        pytest.param((1, 1, 32, 97), 2, 3, lambda d: _explicit_width_shard_config(d, 4, 32, 32), id="width_4_32x32"),
        pytest.param(
            (1, 13, 32, 64), 1, 2, lambda d: _explicit_block_shard_config(2, 2, 512, 32), id="block_2x2_hc_512x32"
        ),
    ],
)
def test_transpose_tile_sharded_irregular_shapes(shape, dim0, dim1, mc_factory, device):
    run_transpose_test(shape, dim0, dim1, device, input_layout=ttnn.TILE_LAYOUT, input_mem_config=mc_factory(device))


# (a) ROW_MAJOR + BLOCK/WIDTH sharded inputs with IRREGULAR W.
# These exercise the composite fallback (transpose.cpp is_block_or_width_sharded_mc guard)
# with shapes whose last dim is not a tile multiple.
@pytest.mark.parametrize(
    "shape, dim0, dim1, mc_factory",
    [
        pytest.param((1, 1, 64, 96), 2, 3, lambda d: _explicit_block_shard_config(2, 3, 32, 32), id="block_2x3_32x32"),
        pytest.param((1, 1, 64, 96), 2, 3, lambda d: _explicit_width_shard_config(d, 3, 64, 32), id="width_3_64x32"),
    ],
)
def test_transpose_row_major_sharded_irregular_shapes(shape, dim0, dim1, mc_factory, device):
    run_transpose_test(
        shape, dim0, dim1, device, input_layout=ttnn.ROW_MAJOR_LAYOUT, input_mem_config=mc_factory(device)
    )


# (b) TILE UNEVEN sharding — padded shape doesn't divide evenly into the shard
# shape. `is_native_transpose_sharding` returns false; op routes through the
# interleaved factories' TensorAccessor fallback.
@pytest.mark.parametrize(
    "shape, dim0, dim1, mc_factory",
    [
        pytest.param((1, 1, 96, 64), 2, 3, lambda d: _explicit_block_shard_config(2, 2, 64, 32), id="block_2x2_64x32"),
        pytest.param((1, 1, 64, 96), 2, 3, lambda d: _explicit_block_shard_config(2, 2, 32, 64), id="block_2x2_32x64"),
    ],
)
def test_transpose_tile_sharded_uneven(shape, dim0, dim1, mc_factory, device):
    run_transpose_test(shape, dim0, dim1, device, input_layout=ttnn.TILE_LAYOUT, input_mem_config=mc_factory(device))


# ===========================================================================
# Additional regression guards for transpose-specific code paths.
# Dtype, COL_MAJOR, identity, non-canonical dims, DRAM no-spec, etc. are not
# duplicated here — they are exercised by generic ttnn tests or by the
# existing universal I/O tests above.
# ===========================================================================


# CN + ROW_MAJOR + BLOCK/WIDTH sharded input.
# Exercises the composite fallback in transpose_impl for the CN dim (the guard
# is keyed on memory layout, not dim).
@pytest.mark.parametrize(
    "shape, mc_factory",
    [
        pytest.param((2, 4, 32, 64), lambda d: _explicit_block_shard_config(2, 2, 128, 32), id="block_2x2_128x32"),
        pytest.param((2, 4, 32, 64), lambda d: _explicit_width_shard_config(d, 2, 256, 32), id="width_2_256x32"),
    ],
)
def test_transpose_cn_row_major_block_width_sharded(shape, mc_factory, device):
    run_transpose_test(shape, 0, 1, device, input_layout=ttnn.ROW_MAJOR_LAYOUT, input_mem_config=mc_factory(device))


# N>1 and/or C>1 with sharded input for WH transpose.
# Exercises shard-geometry helpers that flatten over the leading dims.
@pytest.mark.parametrize(
    "shape, layout, mc_factory",
    [
        pytest.param(
            (2, 1, 64, 64),
            ttnn.TILE_LAYOUT,
            lambda d: _explicit_height_shard_config(d, 4, 32, 64),
            id="tile_height_4_32x64",
        ),
        pytest.param(
            (1, 2, 64, 64),
            ttnn.TILE_LAYOUT,
            lambda d: _explicit_block_shard_config(2, 2, 64, 32),
            id="tile_block_2x2_64x32",
        ),
        pytest.param(
            (2, 2, 32, 64),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d: _explicit_height_shard_config(d, 4, 32, 64),
            id="rm_height_4_32x64",
        ),
    ],
)
def test_transpose_multi_batch_channel_sharded_wh(shape, layout, mc_factory, device):
    run_transpose_test(shape, 2, 3, device, input_layout=layout, input_mem_config=mc_factory(device))


# ROW_MAJOR HEIGHT_SHARDED inputs whose shard element count (shard_h * shard_w) is NOT a
# multiple of the tile footprint (32 * 32 = 1024). Before the is_native_transpose_sharding
# fix, these silently hit the native RM height-sharded kernel and produced garbage or hit a
# CB allocation failure. After the fix, the predicate returns false for such shards and the
# device op selects the interleaved WH factory, which reads/writes the sharded buffer
# directly over NOC via TensorAccessorArgs (no physical reshard). Tile-aligned RM HS
# (e.g. (32,64)) remains on the fast specialized native path.
@pytest.mark.parametrize(
    "shape, shard_shape",
    [
        ((1, 1, 52, 64), (13, 64)),
        ((1, 1, 40, 40), (10, 40)),
    ],
    ids=["HS_nontile_13x64", "HS_nontile_10x40"],
)
def test_transpose_row_major_height_sharded_nontile_aligned_wh(shape, shard_shape, device):
    imc = _explicit_height_shard_config(device, 4, shard_shape[0], shard_shape[1])
    run_transpose_test(shape, 2, 3, device, input_layout=ttnn.ROW_MAJOR_LAYOUT, input_mem_config=imc)
