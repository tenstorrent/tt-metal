# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Tests for transpose universal input/output support.

Each test covers one distinct code path — no duplicate shapes or redundant combos.
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_ulp, assert_with_pcc

_TTNN_TO_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.bfloat8_b: torch.bfloat16,
}


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
    torch_dtype = _TTNN_TO_TORCH_DTYPE[dtype]
    x = torch.rand(shape, dtype=torch_dtype)

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

    # bf16 is bit-exact (ULP=0); bf8_b and f32 use PCC because composite paths can perturb
    # individual elements (block-quantization for bf8_b, bf16-precision intermediates for f32).
    if dtype == ttnn.bfloat16:
        assert_with_ulp(ref, got, ulp_threshold=0)
    else:
        assert_with_pcc(ref.float(), got.float(), 0.9999)


_TILE_HEIGHT = 32
_TILE_WIDTH = 32


def _round_up(x, mult):
    return ((x + mult - 1) // mult) * mult


def _padded_hw(shape, layout):
    """Return (total_height, width) using tile-padded dims for TILE_LAYOUT, logical for RM.
    Pads each dim independently to match ttnn's per-dim tile padding (e.g. (2,3,71,79) →
    total_h = 2*3*96 = 576, w = 96)."""
    if layout == ttnn.TILE_LAYOUT:
        return shape[0] * shape[1] * _round_up(shape[2], _TILE_HEIGHT), _round_up(shape[3], _TILE_WIDTH)
    return shape[0] * shape[1] * shape[2], shape[3]


def _tile_align(shard_shape, layout):
    """Round shard dims up to the nearest tile for TILE_LAYOUT; pass through for ROW_MAJOR."""
    h, w = shard_shape
    if layout == ttnn.TILE_LAYOUT:
        return (_round_up(h, _TILE_HEIGHT), _round_up(w, _TILE_WIDTH))
    return (h, w)


def _block_shard_config(shape, device, layout=ttnn.TILE_LAYOUT):
    """Create a 2x2 block-sharded MemoryConfig; shard rounded up to tile for TILE inputs."""
    compute_grid = device.compute_with_storage_grid_size()
    grid_x = min(2, compute_grid.x)
    grid_y = min(2, compute_grid.y)
    total_h, w = _padded_hw(shape, layout)
    shard_shape = _tile_align(((total_h + grid_y - 1) // grid_y, (w + grid_x - 1) // grid_x), layout)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)


def _height_shard_config(shape, device, num_cores=4, buffer_type=ttnn.BufferType.L1, layout=ttnn.TILE_LAYOUT):
    """Create a height-sharded MemoryConfig; shard rounded up to tile for TILE inputs."""
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, compute_grid.x * compute_grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
    total_h, w = _padded_hw(shape, layout)
    shard_shape = _tile_align(((total_h + num_cores - 1) // num_cores, w), layout)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type, shard_spec)


def _width_shard_config(shape, device, num_cores=4, layout=ttnn.TILE_LAYOUT):
    """Create a width-sharded MemoryConfig; shard rounded up to tile for TILE inputs."""
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, compute_grid.x * compute_grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
    total_h, w = _padded_hw(shape, layout)
    shard_shape = _tile_align((total_h, (w + num_cores - 1) // num_cores), layout)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
DRAM_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


def _interleaved(_d):
    return L1_INTERLEAVED


def _sharded_no_spec(memory_layout):
    return lambda _d: ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1)


# Universal-IO matrix: every TILE_LAYOUT input × output combination the device op must support.
# Each row is one distinct routing path through `transpose_impl` / `select_program_factory`.
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="f32"),
        pytest.param(ttnn.bfloat8_b, id="bf8b"),
    ],
)
@pytest.mark.parametrize(
    "shape, dim0, dim1, input_layout, input_factory, output_factory",
    [
        pytest.param(
            (1, 1, 64, 64),
            2,
            3,
            ttnn.TILE_LAYOUT,
            lambda d: _block_shard_config((1, 1, 64, 64), d),
            _interleaved,
            id="WH_block_to_interleaved",
        ),
        pytest.param(
            (2, 4, 32, 64),
            0,
            1,
            ttnn.TILE_LAYOUT,
            lambda d: _block_shard_config((2, 4, 32, 64), d),
            _interleaved,
            id="CN_block_to_interleaved",
        ),
        pytest.param(
            (1, 1, 64, 128),
            2,
            3,
            ttnn.TILE_LAYOUT,
            lambda d: _width_shard_config((1, 1, 64, 128), d),
            _interleaved,
            id="WH_width_to_interleaved",
        ),
        pytest.param(
            (1, 1, 64, 128),
            2,
            3,
            ttnn.TILE_LAYOUT,
            _interleaved,
            lambda d: _height_shard_config((1, 1, 128, 64), d),
            id="WH_interleaved_to_height",
        ),
        pytest.param(
            (1, 1, 64, 64),
            2,
            3,
            ttnn.TILE_LAYOUT,
            _interleaved,
            lambda d: _block_shard_config((1, 1, 64, 64), d),
            id="WH_interleaved_to_block",
        ),
        pytest.param(
            (1, 1, 64, 128),
            2,
            3,
            ttnn.TILE_LAYOUT,
            lambda d: _block_shard_config((1, 1, 64, 128), d),
            lambda d: _height_shard_config((1, 1, 128, 64), d, num_cores=2),
            id="WH_block_to_height_2cores",
        ),
        pytest.param(
            (1, 1, 64, 128),
            2,
            3,
            ttnn.TILE_LAYOUT,
            lambda d: _width_shard_config((1, 1, 64, 128), d),
            lambda d: _height_shard_config((1, 1, 128, 64), d),
            id="WH_width_to_height",
        ),
        pytest.param(
            (1, 4, 32, 64),
            1,
            2,
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((1, 4, 32, 64), d),
            _interleaved,
            id="HC_height_to_interleaved",
        ),
        pytest.param(
            (2, 4, 32, 64),
            0,
            1,
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((2, 4, 32, 64), d, num_cores=8),
            _interleaved,
            id="CN_height_8cores_to_interleaved",
        ),
        pytest.param(
            (1, 1, 64, 64),
            2,
            3,
            ttnn.TILE_LAYOUT,
            lambda d: _block_shard_config((1, 1, 64, 64), d),
            None,
            id="WH_block_default_output",
        ),
        pytest.param(
            (2, 4, 32, 64),
            2,
            3,
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((2, 4, 32, 64), d),
            None,
            id="WH_height_default_output",
        ),
        pytest.param(
            (1, 1, 128, 64),
            2,
            3,
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((1, 1, 128, 64), d),
            _sharded_no_spec(ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            id="WH_height_to_height_nospec",
        ),
        pytest.param(
            (1, 1, 32, 64),
            2,
            3,
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((1, 1, 32, 64), d, num_cores=1),
            None,
            id="WH_native_height_1core",
        ),
        pytest.param(
            (2, 3, 64, 96),
            2,
            3,
            ttnn.TILE_LAYOUT,
            _interleaved,
            None,
            id="WH_interleaved_baseline",
        ),
        pytest.param(
            (1, 4, 32, 64),
            1,
            2,
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d: _height_shard_config((1, 4, 32, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            _interleaved,
            id="HC_RM_height_to_interleaved",
        ),
        pytest.param(
            (1, 1, 128, 128),
            2,
            3,
            ttnn.TILE_LAYOUT,
            _interleaved,
            _sharded_no_spec(ttnn.TensorMemoryLayout.BLOCK_SHARDED),
            id="WH_interleaved_to_block_nospec",
        ),
    ],
)
def test_transpose_universal_io_tile(shape, dim0, dim1, input_layout, input_factory, output_factory, dtype, device):
    if dtype == ttnn.bfloat8_b and input_layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("bfloat8_b is only supported on TILE_LAYOUT inputs")
    # HC sharded kernels mis-account bfp8's per-block byte layout (each 32x32 tile is 1088B, not
    # 1024 * elem_size). The legacy `test_transpose_sharded` carries the same skip; tracked as an
    # op-side gap, not a regression of this PR.
    is_hc = (dim0, dim1) in {(1, 2), (2, 1)}
    if dtype == ttnn.bfloat8_b and is_hc and input_factory(device).is_sharded():
        pytest.skip("sharded bfloat8_b is not supported for HC transpose (physical-size mismatch)")
    run_transpose_test(
        shape,
        dim0,
        dim1,
        device,
        input_layout=input_layout,
        input_mem_config=input_factory(device),
        output_mem_config=output_factory(device) if output_factory is not None else None,
        dtype=dtype,
    )


# 16. DRAM-sharded input falls back to L1 interleaved output.
def test_transpose_dram_sharded_fallback(device):
    shape = (1, 1, 128, 64)
    dram_sharded = _height_shard_config(shape, device, buffer_type=ttnn.BufferType.DRAM)

    torch.manual_seed(12345)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=dram_sharded
    )
    result = ttnn.transpose(ttnn_input, 2, 3, memory_config=L1_INTERLEAVED)
    assert (
        result.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    ), f"Expected fallback to INTERLEAVED, got {result.memory_config().memory_layout}"

    ref = x.transpose(2, 3)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert_with_ulp(ref, got, ulp_threshold=1)


# Non-native (BLOCK) sharded input → user-requested sharded output without shard_spec.
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


# Universal-IO matrix for ROW_MAJOR composite-fallback paths.
# Every row exercises the composite (un-tilize → transpose → re-tilize) path that the device op
# falls back to when one or both endpoints are ROW_MAJOR + sharded.
# bfloat8_b is excluded from this matrix — it requires TILE layout.
_RM = ttnn.ROW_MAJOR_LAYOUT


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="f32"),
    ],
)
@pytest.mark.parametrize(
    "shape, dim0, dim1, input_factory, output_factory",
    [
        pytest.param(
            (1, 1, 64, 64),
            2,
            3,
            lambda d: _block_shard_config((1, 1, 64, 64), d, layout=_RM),
            _interleaved,
            id="WH_RM_block_to_interleaved",
        ),
        pytest.param(
            (1, 1, 32, 128),
            2,
            3,
            lambda d: _width_shard_config((1, 1, 32, 128), d, layout=_RM),
            _interleaved,
            id="WH_RM_width_to_interleaved",
        ),
        pytest.param(
            (1, 1, 64, 64),
            2,
            3,
            _interleaved,
            lambda d: _block_shard_config((1, 1, 64, 64), d, layout=_RM),
            id="WH_RM_interleaved_to_block",
        ),
        pytest.param(
            (1, 1, 64, 128),
            2,
            3,
            _interleaved,
            lambda d: _width_shard_config((1, 1, 128, 64), d, layout=_RM),
            id="WH_RM_interleaved_to_width",
        ),
        pytest.param(
            (1, 1, 64, 64),
            2,
            3,
            lambda d: _block_shard_config((1, 1, 64, 64), d, layout=_RM),
            lambda d: _block_shard_config((1, 1, 64, 64), d, layout=_RM),
            id="WH_RM_block_to_block",
        ),
        pytest.param(
            (1, 1, 64, 32),
            2,
            3,
            lambda d: _height_shard_config((1, 1, 64, 32), d, layout=_RM),
            _interleaved,
            id="WH_RM_height_to_interleaved",
        ),
    ],
)
def test_transpose_universal_io_row_major(shape, dim0, dim1, input_factory, output_factory, dtype, device):
    in_mc = input_factory(device)
    out_mc = output_factory(device)
    run_transpose_test(
        shape,
        dim0,
        dim1,
        device,
        input_layout=_RM,
        input_mem_config=in_mc,
        output_mem_config=out_mc,
        dtype=dtype,
    )


# Irregular logical shapes (last-two not tile multiples) across WH/HC/CN, both interleaved
# and sharded. Sharded uses the rounded-up shard helpers (tile-aligned shards on irregular
# logical shapes — per review).
_IRREGULAR_SHAPES = [
    ((1, 1, 65, 97), 2, 3),
    ((2, 3, 71, 79), 2, 3),
    ((1, 13, 47, 64), 1, 2),
    ((1, 7, 33, 96), 1, 2),
    ((3, 5, 32, 64), 0, 1),
]


@pytest.mark.parametrize("shape, dim0, dim1", _IRREGULAR_SHAPES)
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_transpose_irregular_shapes_interleaved(shape, dim0, dim1, input_layout, device):
    run_transpose_test(shape, dim0, dim1, device, input_layout=input_layout)


@pytest.mark.parametrize("shape, dim0, dim1", _IRREGULAR_SHAPES)
@pytest.mark.parametrize(
    "shard_factory",
    [
        pytest.param(_block_shard_config, id="block"),
        pytest.param(_height_shard_config, id="height"),
        pytest.param(_width_shard_config, id="width"),
    ],
)
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_transpose_irregular_shapes_sharded(shape, dim0, dim1, shard_factory, input_layout, device):
    run_transpose_test(
        shape,
        dim0,
        dim1,
        device,
        input_layout=input_layout,
        input_mem_config=shard_factory(shape, device, layout=input_layout),
    )


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
    input_mem_config = input_factory(shape, device, layout=ttnn.ROW_MAJOR_LAYOUT)
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
