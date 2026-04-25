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

    # Whenever the caller specified an output MemoryConfig, the result must honor the requested
    # memory_layout. If the request was sharded we additionally require a concrete shard_spec on
    # the output so silent fallbacks to interleaved (or a dropped spec) surface as test failures.
    if output_mem_config is not None:
        actual = result.memory_config()
        assert (
            actual.memory_layout == output_mem_config.memory_layout
        ), f"Expected output memory layout {output_mem_config.memory_layout}, got {actual.memory_layout}"
        if output_mem_config.memory_layout in (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ):
            assert (
                actual.shard_spec is not None
            ), f"Sharded output requested but result has no shard_spec (silently fell back?)"

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


# 1. WH: block-sharded input → interleaved output.
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


# 2. CN: block-sharded input → interleaved output.
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


# 3. WH: width-sharded input → interleaved output.
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


# 4. WH: interleaved input → height-sharded output.
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


# 5. WH: interleaved input → block-sharded output.
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


# 6. WH: block-sharded input → height-sharded output.
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


# 7. WH: width-sharded input → height-sharded output (cross shard-type).
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


# 8. HC: height-sharded input → interleaved output.
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


# 9. CN: height-sharded input → interleaved output.
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


# 10. Block-sharded input with no explicit output config (default output derivation).
def test_transpose_block_sharded_default_output(device):
    shape = (1, 1, 64, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_block_shard_config(shape, device),
    )


# 11. Height-sharded input with no explicit output config (default output derivation).
def test_transpose_height_sharded_default_output(device):
    shape = (2, 4, 32, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_height_shard_config(shape, device),
    )


# 12. Height-sharded input → height-sharded output without shard_spec.
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


# 13. WH: native L1 height-sharded path regression guard.
def test_transpose_native_height_sharded_wh(device):
    shape = (1, 1, 32, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_height_shard_config(shape, device, num_cores=1),
    )


# 14. Interleaved baseline (sanity).
def test_transpose_interleaved_baseline(device):
    shape = (2, 3, 64, 96)
    run_transpose_test(shape, 2, 3, device)


# 15. HC transpose: ROW_MAJOR height-sharded input → interleaved output.
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


# 16. DRAM-sharded input falls back to L1 interleaved output.
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


# 17. Interleaved input → BLOCK_SHARDED output requested without shard_spec.
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


# 17b. Non-native (BLOCK) sharded input → user-requested sharded output without shard_spec.
@pytest.mark.parametrize(
    "requested_out_layout",
    [
        pytest.param(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, id="H_out"),
        pytest.param(ttnn.TensorMemoryLayout.WIDTH_SHARDED, id="W_out"),
        pytest.param(ttnn.TensorMemoryLayout.BLOCK_SHARDED, id="B_out"),
    ],
)
def test_transpose_non_native_sharded_input_to_sharded_nospec(requested_out_layout, device):
    shape = (1, 1, 64, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_block_shard_config(shape, device),
        output_mem_config=ttnn.MemoryConfig(requested_out_layout, ttnn.BufferType.L1),
    )


# 18. WH: ROW_MAJOR + BLOCK_SHARDED input → interleaved output (composite fallback).
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


# 19. WH: ROW_MAJOR + WIDTH_SHARDED input → interleaved output (composite fallback).
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


# 20. WH: ROW_MAJOR interleaved input → BLOCK_SHARDED output (composite fallback).
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


# 21. WH: ROW_MAJOR interleaved input → WIDTH_SHARDED output (composite fallback).
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


# 22. WH: ROW_MAJOR + BLOCK_SHARDED input → BLOCK_SHARDED output (composite fallback both ends).
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


# 23. WH: ROW_MAJOR + HEIGHT_SHARDED input → interleaved output (regression guard, not composite fallback).
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


# Interleaved input with irregular shapes (TILE and ROW_MAJOR) across WH/HC/CN.
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


# ROW_MAJOR interleaved input → BLOCK/WIDTH sharded output without shard_spec.
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


# HC TILE with irregular C, interleaved input → BLOCK_SHARDED output without shard_spec.
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


# Sharded inputs with shapes that don't divide evenly — covers (a) tile-aligned padded shape with
# irregular logical dim and (b) uneven sharding (padded shape doesn't divide into shard shape).


def _explicit_block_shard_config(device, grid_y, grid_x, sh, sw):
    """Block-shard config clamped to the device's compute grid. Skips if clamping would change
    the requested grid (the test was authored for a specific grid layout)."""
    compute_grid = device.compute_with_storage_grid_size()
    if grid_y > compute_grid.y or grid_x > compute_grid.x:
        pytest.skip(f"Device grid ({compute_grid.y}x{compute_grid.x}) too small for requested {grid_y}x{grid_x}")
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)


def _explicit_height_shard_config(device, ncores, sh, sw):
    """Height-shard config; skips if the device can't host `ncores`."""
    compute_grid = device.compute_with_storage_grid_size()
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has {compute_grid.x * compute_grid.y} cores, test needs {ncores}")
    spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


def _explicit_width_shard_config(device, ncores, sh, sw):
    """Width-shard config; skips if the device can't host `ncores`."""
    compute_grid = device.compute_with_storage_grid_size()
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has {compute_grid.x * compute_grid.y} cores, test needs {ncores}")
    spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)


# (a) TILE + sharded input with irregular logical shape (padded shape and shard are tile-aligned).
@pytest.mark.parametrize(
    "shape, dim0, dim1, mc_factory",
    [
        pytest.param(
            (1, 1, 65, 64), 2, 3, lambda d: _explicit_block_shard_config(d, 3, 2, 32, 32), id="block_3x2_32x32"
        ),
        pytest.param(
            (1, 1, 64, 97), 2, 3, lambda d: _explicit_block_shard_config(d, 2, 4, 32, 32), id="block_2x4_32x32"
        ),
        pytest.param((1, 1, 65, 64), 2, 3, lambda d: _explicit_height_shard_config(d, 3, 32, 64), id="height_3_32x64"),
        pytest.param((1, 1, 32, 97), 2, 3, lambda d: _explicit_width_shard_config(d, 4, 32, 32), id="width_4_32x32"),
        pytest.param(
            (1, 13, 32, 64), 1, 2, lambda d: _explicit_block_shard_config(d, 2, 2, 512, 32), id="block_2x2_hc_512x32"
        ),
    ],
)
def test_transpose_tile_sharded_irregular_shapes(shape, dim0, dim1, mc_factory, device):
    run_transpose_test(shape, dim0, dim1, device, input_layout=ttnn.TILE_LAYOUT, input_mem_config=mc_factory(device))


# (a) ROW_MAJOR + BLOCK/WIDTH sharded input with irregular W (composite fallback).
@pytest.mark.parametrize(
    "shape, dim0, dim1, mc_factory",
    [
        pytest.param(
            (1, 1, 64, 96), 2, 3, lambda d: _explicit_block_shard_config(d, 2, 3, 32, 32), id="block_2x3_32x32"
        ),
        pytest.param((1, 1, 64, 96), 2, 3, lambda d: _explicit_width_shard_config(d, 3, 64, 32), id="width_3_64x32"),
    ],
)
def test_transpose_row_major_sharded_irregular_shapes(shape, dim0, dim1, mc_factory, device):
    run_transpose_test(
        shape, dim0, dim1, device, input_layout=ttnn.ROW_MAJOR_LAYOUT, input_mem_config=mc_factory(device)
    )


# (b) TILE uneven sharding — padded shape doesn't divide evenly into the shard shape.
@pytest.mark.parametrize(
    "shape, dim0, dim1, mc_factory",
    [
        pytest.param(
            (1, 1, 96, 64), 2, 3, lambda d: _explicit_block_shard_config(d, 2, 2, 64, 32), id="block_2x2_64x32"
        ),
        pytest.param(
            (1, 1, 64, 96), 2, 3, lambda d: _explicit_block_shard_config(d, 2, 2, 32, 64), id="block_2x2_32x64"
        ),
    ],
)
def test_transpose_tile_sharded_uneven(shape, dim0, dim1, mc_factory, device):
    run_transpose_test(shape, dim0, dim1, device, input_layout=ttnn.TILE_LAYOUT, input_mem_config=mc_factory(device))


# Additional regression guards for transpose-specific code paths.


# CN: ROW_MAJOR + BLOCK/WIDTH sharded input (composite fallback for non-WH dim).
@pytest.mark.parametrize(
    "shape, mc_factory",
    [
        pytest.param((2, 4, 32, 64), lambda d: _explicit_block_shard_config(d, 2, 2, 128, 32), id="block_2x2_128x32"),
        pytest.param((2, 4, 32, 64), lambda d: _explicit_width_shard_config(d, 2, 256, 32), id="width_2_256x32"),
    ],
)
def test_transpose_cn_row_major_block_width_sharded(shape, mc_factory, device):
    run_transpose_test(shape, 0, 1, device, input_layout=ttnn.ROW_MAJOR_LAYOUT, input_mem_config=mc_factory(device))


# WH transpose with N>1 and/or C>1 sharded inputs.
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
            lambda d: _explicit_block_shard_config(d, 2, 2, 64, 32),
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


# WH: ROW_MAJOR HEIGHT_SHARDED input with non-tile-aligned shard (shard_h * shard_w not a tile multiple).
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


# CN TILE → HEIGHT_SHARDED output without shard_spec, from interleaved and height-sharded inputs.
@pytest.mark.parametrize(
    "input_mem_factory",
    [
        pytest.param(lambda d, shape: L1_INTERLEAVED, id="in_interleaved"),
        pytest.param(lambda d, shape: _height_shard_config(shape, d), id="in_height_sharded"),
    ],
)
def test_transpose_cn_sharded_output(input_mem_factory, device):
    shape = (2, 4, 32, 64)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    run_transpose_test(
        shape,
        0,
        1,
        device,
        input_layout=ttnn.TILE_LAYOUT,
        input_mem_config=input_mem_factory(device, shape),
        output_mem_config=output_mem_config,
    )


# DRAM interleaved input → DRAM interleaved output across WH/HC/CN.
@pytest.mark.parametrize(
    "dim0, dim1",
    [
        pytest.param(2, 3, id="WH"),
        pytest.param(1, 2, id="HC"),
        pytest.param(0, 1, id="CN"),
    ],
)
def test_transpose_dram_interleaved(dim0, dim1, device):
    run_transpose_test(
        (1, 4, 64, 128),
        dim0,
        dim1,
        device,
        input_layout=ttnn.TILE_LAYOUT,
        input_mem_config=DRAM_INTERLEAVED,
        output_mem_config=DRAM_INTERLEAVED,
    )


# WH on tile-aligned height-sharded inputs whose transposed width shrinks below a tile.
@pytest.mark.parametrize(
    "shape, ncores, shard_shape",
    [
        pytest.param((1, 1, 64, 32), 2, (32, 32), id="1x1_64x32_hs_2x_32x32"),
        pytest.param((1, 1, 128, 32), 4, (32, 32), id="1x1_128x32_hs_4x_32x32"),
        pytest.param((2, 1, 64, 32), 4, (32, 32), id="2x1_64x32_hs_4x_32x32"),
        pytest.param((2, 2, 64, 32), 8, (32, 32), id="2x2_64x32_hs_8x_32x32"),
    ],
)
def test_transpose_wh_shrink_sub_tile_sharded(shape, ncores, shard_shape, device):
    imc = _explicit_height_shard_config(device, ncores, shard_shape[0], shard_shape[1])
    run_transpose_test(shape, 2, 3, device, input_layout=ttnn.TILE_LAYOUT, input_mem_config=imc)


# HC TILE transpose with irregular logical C and a tile-aligned height-sharded input.
@pytest.mark.parametrize(
    "shape, ncores, shard_shape",
    [
        pytest.param((1, 13, 32, 64), 13, (32, 64), id="C13_H32_hs_13x_32x64"),
        pytest.param((1, 5, 32, 64), 5, (32, 64), id="C5_H32_hs_5x_32x64"),
    ],
)
def test_transpose_hc_tile_irregular_sharded_input(shape, ncores, shard_shape, device):
    imc = _explicit_height_shard_config(device, ncores, shard_shape[0], shard_shape[1])
    run_transpose_test(shape, 1, 2, device, input_layout=ttnn.TILE_LAYOUT, input_mem_config=imc)


# HC/CN: ROW_MAJOR BLOCK/WIDTH-sharded input → BLOCK/WIDTH-sharded output (composite fallback round-trip).
@pytest.mark.parametrize(
    "shape, dim0, dim1, input_factory, output_layout",
    [
        pytest.param(
            (1, 1, 64, 64), 1, 2, _block_shard_config, ttnn.TensorMemoryLayout.BLOCK_SHARDED, id="HC_block_to_block"
        ),
        pytest.param(
            (1, 1, 64, 64), 1, 2, _width_shard_config, ttnn.TensorMemoryLayout.WIDTH_SHARDED, id="HC_width_to_width"
        ),
        pytest.param(
            (2, 2, 64, 64), 0, 1, _block_shard_config, ttnn.TensorMemoryLayout.BLOCK_SHARDED, id="CN_block_to_block"
        ),
    ],
)
def test_transpose_rm_block_or_width_sharded_to_sharded(shape, dim0, dim1, input_factory, output_layout, device):
    input_mem_config = input_factory(shape, device)
    output_mem_config = ttnn.MemoryConfig(output_layout, ttnn.BufferType.L1)
    run_transpose_test(
        shape,
        dim0,
        dim1,
        device,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        input_mem_config=input_mem_config,
        output_mem_config=output_mem_config,
    )
