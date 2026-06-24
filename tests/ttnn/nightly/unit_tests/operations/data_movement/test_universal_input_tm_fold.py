# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

_TTNN_TO_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
}

_TILE_H = 32
_TILE_W = 32

L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
DRAM_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


# ──────────────────────────────────────────────────────────────────────────────
# Reference + helpers
# ──────────────────────────────────────────────────────────────────────────────


def _fold_torch_nhwc(x_nhwc, stride_h, stride_w):
    """Reference fold: (N,H,W,C) → (N, H/sh, W/sw, C*sh*sw) (NHWC space-to-depth)."""
    n, h, w, c = x_nhwc.shape
    reshaped = x_nhwc.reshape(n, h // stride_h, stride_h, w // stride_w, stride_w, c)
    transposed = reshaped.permute(0, 1, 3, 2, 4, 5).contiguous()
    return transposed.reshape(n, h // stride_h, w // stride_w, c * stride_h * stride_w)


def _round_up(x, m):
    return ((x + m - 1) // m) * m


def _height_sharded_nhwc(shape, device, ncores, layout, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    """HEIGHT_SHARDED memory config for an (N,H,W,C) tensor: shard along total pixels (N*H*W)."""
    n, h, w, c = shape
    grid = device.compute_with_storage_grid_size()
    if ncores > grid.x * grid.y:
        pytest.skip(f"Device has {grid.x * grid.y} cores, test needs {ncores}")
    shard_grid = ttnn.num_cores_to_corerangeset(ncores, grid, True)
    total_pixels = n * h * w
    if total_pixels % ncores != 0:
        pytest.skip(f"total pixels {total_pixels} not divisible by ncores {ncores}")
    shard_h = total_pixels // ncores
    shard_w = c
    if layout == ttnn.TILE_LAYOUT:
        shard_h = _round_up(shard_h, _TILE_H)
        shard_w = _round_up(shard_w, _TILE_W)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, (shard_h, shard_w), orientation),
    )


def _width_sharded_nhwc(shape, device, ncores, layout, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    """WIDTH_SHARDED memory config: shard along C across cores."""
    n, h, w, c = shape
    grid = device.compute_with_storage_grid_size()
    if ncores > grid.x * grid.y:
        pytest.skip(f"Device has {grid.x * grid.y} cores, test needs {ncores}")
    if c % ncores != 0:
        pytest.skip(f"C={c} not divisible by ncores={ncores}")
    shard_grid = ttnn.num_cores_to_corerangeset(ncores, grid, True)
    total_pixels = n * h * w
    shard_h = total_pixels
    shard_w = c // ncores
    if layout == ttnn.TILE_LAYOUT:
        shard_h = _round_up(shard_h, _TILE_H)
        shard_w = _round_up(shard_w, _TILE_W)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, (shard_h, shard_w), orientation),
    )


def _block_sharded_nhwc(shape, device, grid_y, grid_x, layout, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    """BLOCK_SHARDED memory config: 2D grid over pixel-rows × C."""
    n, h, w, c = shape
    compute_grid = device.compute_with_storage_grid_size()
    if grid_y > compute_grid.y or grid_x > compute_grid.x:
        pytest.skip(f"Device grid ({compute_grid.y}x{compute_grid.x}) too small for {grid_y}x{grid_x}")
    total_pixels = n * h * w
    if total_pixels % grid_y != 0 or c % grid_x != 0:
        pytest.skip(f"total_pixels={total_pixels} % grid_y={grid_y} or C={c} % grid_x={grid_x} != 0")
    shard_h = total_pixels // grid_y
    shard_w = c // grid_x
    if layout == ttnn.TILE_LAYOUT:
        shard_h = _round_up(shard_h, _TILE_H)
        shard_w = _round_up(shard_w, _TILE_W)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
            (shard_h, shard_w),
            orientation,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Shared runner
# ──────────────────────────────────────────────────────────────────────────────


def _assert_output_layout_contract(result, input_mem_config):
    """Universal-IO contract: output TensorMemoryLayout + ShardOrientation match input's (no silent demotion)."""
    expected_layout = input_mem_config.memory_layout
    actual_layout = result.memory_config().memory_layout
    assert actual_layout == expected_layout, (
        f"Universal-IO contract violated: input layout={expected_layout}, "
        f"output layout={actual_layout} (sharded input must yield sharded output)."
    )
    sharded_layouts = {
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    }
    if expected_layout in sharded_layouts:
        out_spec = result.memory_config().shard_spec
        assert out_spec is not None, "Sharded output missing shard_spec"
        in_spec = input_mem_config.shard_spec
        if in_spec is not None:
            assert out_spec.orientation == in_spec.orientation, (
                f"ShardOrientation silently demoted: input={in_spec.orientation}, " f"output={out_spec.orientation}."
            )


def _run_fold(shape, stride_h, stride_w, layout, input_mem_config, dtype, device, pcc=0.9999):
    torch.manual_seed(12345)
    torch_dtype = _TTNN_TO_TORCH_DTYPE[dtype]
    x_nhwc = torch.rand(shape, dtype=torch_dtype)
    expected = _fold_torch_nhwc(x_nhwc, stride_h, stride_w)

    ttnn_in = ttnn.from_torch(x_nhwc, layout=layout, dtype=dtype, device=device, memory_config=input_mem_config)
    result = ttnn.fold(ttnn_in, stride_h=stride_h, stride_w=stride_w)

    _assert_output_layout_contract(result, input_mem_config)

    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))

    n, h, w, c = expected.shape
    assert (
        got.numel() == expected.numel()
    ), f"fold output numel mismatch: got {tuple(got.shape)} ({got.numel()}) vs expected {tuple(expected.shape)} ({expected.numel()})"
    got_4d = got.reshape(n, h, w, c)
    assert_with_pcc(expected.float(), got_4d.float(), pcc)


# ──────────────────────────────────────────────────────────────────────────────
# Group A: interleaved baselines (cross-product of buffer × layout × dtype × stride)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "mem_config",
    [
        pytest.param(DRAM_INTERLEAVED, id="dram"),
        pytest.param(L1_INTERLEAVED, id="l1"),
    ],
)
@pytest.mark.parametrize(
    "shape, stride",
    [
        # TILE pads H,W to multiples of 32, so only strides dividing 32 (2, 4) work; RM irregular strides covered separately.
        pytest.param((1, 8, 8, 32), (2, 2), id="s2x2_small"),
        pytest.param((1, 16, 16, 32), (4, 4), id="s4x4"),
    ],
)
def test_fold_interleaved_baselines(layout, mem_config, dtype, shape, stride, device):
    """DRAM/L1 × RM/TILE × bf16/f32 × {2x2, 4x4}; TILE must NOT pass through composite to_layout(RM)."""
    _run_fold(shape, stride[0], stride[1], layout, mem_config, dtype, device)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize(
    "mem_config",
    [
        pytest.param(DRAM_INTERLEAVED, id="dram"),
        pytest.param(L1_INTERLEAVED, id="l1"),
    ],
)
@pytest.mark.parametrize(
    "shape, stride",
    [
        pytest.param((1, 12, 12, 32), (3, 3), id="s3x3"),
        pytest.param((1, 6, 8, 32), (3, 2), id="s3x2_asym"),
        pytest.param((1, 8, 6, 32), (2, 3), id="s2x3_asym"),
    ],
)
def test_fold_interleaved_baselines_rm_only(mem_config, dtype, shape, stride, device):
    """Irregular strides on RM inputs (TILE padding would inflate H,W past stride divisibility)."""
    _run_fold(shape, stride[0], stride[1], ttnn.ROW_MAJOR_LAYOUT, mem_config, dtype, device)


# ──────────────────────────────────────────────────────────────────────────────
# Group B: HEIGHT_SHARDED — RM fast path + TILE native + orientations
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, ncores, stride",
    [
        pytest.param((1, 8, 8, 32), 2, (2, 2), id="small_ncores2_s2x2"),
        pytest.param((1, 8, 8, 32), 4, (2, 2), id="small_ncores4_s2x2"),
        pytest.param((1, 16, 16, 64), 8, (2, 2), id="medium_s2x2"),
        pytest.param((1, 16, 16, 32), 4, (4, 4), id="s4x4"),
        pytest.param((1, 12, 8, 32), 2, (3, 2), id="s3x2_asym"),
    ],
)
def test_fold_height_sharded_rm_fast_path(shape, ncores, stride, device):
    """HEIGHT_SHARDED + ROW_MAJOR → zero-NOC MultiCore fast path."""
    mc = _height_sharded_nhwc(shape, device, ncores=ncores, layout=ttnn.ROW_MAJOR_LAYOUT)
    _run_fold(shape, stride[0], stride[1], ttnn.ROW_MAJOR_LAYOUT, mc, ttnn.bfloat16, device)


@pytest.mark.parametrize(
    "shape, ncores, stride",
    [
        pytest.param((1, 32, 32, 32), 8, (2, 2), id="small_s2x2"),
        pytest.param((1, 64, 64, 32), 8, (2, 2), id="larger_s2x2"),
        pytest.param((1, 32, 32, 32), 8, (4, 4), id="s4x4"),
        pytest.param((1, 64, 32, 32), 8, (4, 2), id="s4x2_asym"),
    ],
)
def test_fold_height_sharded_tile_native(shape, ncores, stride, device):
    """HEIGHT_SHARDED + TILE — native via MultiCoreDRAMFold's tiled branch (no composite untilize)."""
    mc = _height_sharded_nhwc(shape, device, ncores=ncores, layout=ttnn.TILE_LAYOUT)
    _run_fold(shape, stride[0], stride[1], ttnn.TILE_LAYOUT, mc, ttnn.bfloat16, device)


@pytest.mark.parametrize(
    "shape, ncores, layout",
    [
        pytest.param((1, 8, 8, 32), 2, ttnn.ROW_MAJOR_LAYOUT, id="rm"),
        pytest.param((1, 32, 32, 32), 8, ttnn.TILE_LAYOUT, id="tile"),
    ],
)
def test_fold_height_sharded_col_major_orientation(shape, ncores, layout, device):
    """COL_MAJOR shard orientation must be preserved end-to-end (no silent demotion)."""
    mc = _height_sharded_nhwc(shape, device, ncores=ncores, layout=layout, orientation=ttnn.ShardOrientation.COL_MAJOR)
    _run_fold(shape, 2, 2, layout, mc, ttnn.bfloat16, device)


# ──────────────────────────────────────────────────────────────────────────────
# Group C: WIDTH_SHARDED (any layout) — universal-IO output contract
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, ncores, stride, layout",
    [
        pytest.param((1, 8, 8, 64), 2, (2, 2), ttnn.ROW_MAJOR_LAYOUT, id="rm_small_s2x2"),
        pytest.param((1, 8, 8, 128), 4, (2, 2), ttnn.ROW_MAJOR_LAYOUT, id="rm_ncores4_s2x2"),
        pytest.param((1, 16, 16, 64), 4, (4, 4), ttnn.ROW_MAJOR_LAYOUT, id="rm_s4x4"),
        pytest.param((1, 12, 8, 64), 2, (3, 2), ttnn.ROW_MAJOR_LAYOUT, id="rm_s3x2_asym"),
        pytest.param((1, 32, 32, 64), 2, (2, 2), ttnn.TILE_LAYOUT, id="tile_small_s2x2"),
        pytest.param((1, 32, 32, 128), 4, (2, 2), ttnn.TILE_LAYOUT, id="tile_ncores4_s2x2"),
        pytest.param((1, 32, 32, 64), 2, (4, 4), ttnn.TILE_LAYOUT, id="tile_s4x4"),
    ],
)
def test_fold_width_sharded(shape, ncores, stride, layout, device):
    """WIDTH_SHARDED → W-sharded native via noc_async_write_sharded splitting the per-pixel stick."""
    mc = _width_sharded_nhwc(shape, device, ncores=ncores, layout=layout)
    _run_fold(shape, stride[0], stride[1], layout, mc, ttnn.bfloat16, device)


@pytest.mark.parametrize(
    "shape, ncores, layout",
    [
        pytest.param((1, 8, 8, 64), 2, ttnn.ROW_MAJOR_LAYOUT, id="rm"),
        pytest.param((1, 32, 32, 64), 2, ttnn.TILE_LAYOUT, id="tile"),
    ],
)
def test_fold_width_sharded_col_major(shape, ncores, layout, device):
    """W-sharded with COL_MAJOR orientation must be preserved end-to-end."""
    mc = _width_sharded_nhwc(shape, device, ncores=ncores, layout=layout, orientation=ttnn.ShardOrientation.COL_MAJOR)
    _run_fold(shape, 2, 2, layout, mc, ttnn.bfloat16, device)


# ──────────────────────────────────────────────────────────────────────────────
# Group D: BLOCK_SHARDED (any layout) — universal-IO output contract
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, grid, stride, layout",
    [
        pytest.param((1, 8, 8, 64), (2, 2), (2, 2), ttnn.ROW_MAJOR_LAYOUT, id="rm_2x2_s2x2"),
        pytest.param((1, 16, 16, 64), (2, 2), (4, 4), ttnn.ROW_MAJOR_LAYOUT, id="rm_2x2_s4x4"),
        pytest.param((1, 12, 8, 64), (2, 2), (3, 2), ttnn.ROW_MAJOR_LAYOUT, id="rm_2x2_s3x2_asym"),
        pytest.param((1, 32, 32, 64), (2, 2), (2, 2), ttnn.TILE_LAYOUT, id="tile_2x2_s2x2"),
        pytest.param((1, 32, 32, 64), (2, 2), (4, 4), ttnn.TILE_LAYOUT, id="tile_2x2_s4x4"),
    ],
)
def test_fold_block_sharded(shape, grid, stride, layout, device):
    """BLOCK_SHARDED → B-sharded native via noc_async_write_sharded (splits per-pixel stick across grid_x cores)."""
    mc = _block_sharded_nhwc(shape, device, grid_y=grid[0], grid_x=grid[1], layout=layout)
    _run_fold(shape, stride[0], stride[1], layout, mc, ttnn.bfloat16, device)


@pytest.mark.parametrize(
    "shape, grid, layout",
    [
        pytest.param((1, 8, 8, 64), (2, 2), ttnn.ROW_MAJOR_LAYOUT, id="rm"),
        pytest.param((1, 32, 32, 64), (2, 2), ttnn.TILE_LAYOUT, id="tile"),
    ],
)
def test_fold_block_sharded_col_major(shape, grid, layout, device):
    """B-sharded with COL_MAJOR orientation."""
    mc = _block_sharded_nhwc(
        shape, device, grid_y=grid[0], grid_x=grid[1], layout=layout, orientation=ttnn.ShardOrientation.COL_MAJOR
    )
    _run_fold(shape, 2, 2, layout, mc, ttnn.bfloat16, device)


# ──────────────────────────────────────────────────────────────────────────────
# Group E: float32 — representative path each
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, layout, mc_factory",
    [
        pytest.param(
            (1, 8, 8, 32),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d, s: _height_sharded_nhwc(s, d, ncores=2, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="rm_height_sharded",
        ),
        pytest.param(
            (1, 32, 32, 32),
            ttnn.TILE_LAYOUT,
            lambda d, s: _height_sharded_nhwc(s, d, ncores=8, layout=ttnn.TILE_LAYOUT),
            id="tile_height_sharded",
        ),
        pytest.param(
            (1, 32, 32, 64),
            ttnn.TILE_LAYOUT,
            lambda d, s: _width_sharded_nhwc(s, d, ncores=2, layout=ttnn.TILE_LAYOUT),
            id="tile_width_sharded",
        ),
        pytest.param(
            (1, 32, 32, 64),
            ttnn.TILE_LAYOUT,
            lambda d, s: _block_sharded_nhwc(s, d, grid_y=2, grid_x=2, layout=ttnn.TILE_LAYOUT),
            id="tile_block_sharded",
        ),
    ],
)
def test_fold_f32(shape, layout, mc_factory, device):
    """float32 over the major routing paths."""
    _run_fold(shape, 2, 2, layout, mc_factory(device, shape), ttnn.float32, device)


# ──────────────────────────────────────────────────────────────────────────────
# Group F: multi-batch (N > 1)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, stride, layout, mc_factory",
    [
        pytest.param(
            (2, 8, 8, 32),
            (2, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d, s: DRAM_INTERLEAVED,
            id="N2_dram_rm",
        ),
        pytest.param(
            (2, 8, 8, 32),
            (2, 2),
            ttnn.TILE_LAYOUT,
            lambda d, s: DRAM_INTERLEAVED,
            id="N2_dram_tile",
        ),
        pytest.param(
            (2, 8, 8, 32),
            (2, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d, s: L1_INTERLEAVED,
            id="N2_l1_rm",
        ),
        pytest.param(
            (2, 8, 8, 32),
            (2, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d, s: _height_sharded_nhwc(s, d, ncores=4, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="N2_height_sharded_rm",
        ),
        pytest.param(
            (2, 32, 32, 32),
            (2, 2),
            ttnn.TILE_LAYOUT,
            lambda d, s: _height_sharded_nhwc(s, d, ncores=8, layout=ttnn.TILE_LAYOUT),
            id="N2_height_sharded_tile",
        ),
        pytest.param(
            (2, 8, 8, 64),
            (2, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d, s: _width_sharded_nhwc(s, d, ncores=2, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="N2_width_sharded_rm",
        ),
        pytest.param(
            (2, 8, 8, 64),
            (2, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d, s: _block_sharded_nhwc(s, d, grid_y=2, grid_x=2, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="N2_block_sharded_rm",
        ),
        pytest.param(
            (4, 8, 8, 32),
            (2, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d, s: _height_sharded_nhwc(s, d, ncores=4, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="N4_height_sharded_rm",
        ),
    ],
)
def test_fold_multi_batch(shape, stride, layout, mc_factory, device):
    """N > 1 across all major routing paths."""
    _run_fold(shape, stride[0], stride[1], layout, mc_factory(device, shape), ttnn.bfloat16, device)


# ──────────────────────────────────────────────────────────────────────────────
# Group G: asymmetric strides
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, stride, mc_factory, layout",
    [
        pytest.param(
            (1, 6, 8, 32),
            (3, 2),
            lambda d, s: _height_sharded_nhwc(s, d, ncores=2, layout=ttnn.ROW_MAJOR_LAYOUT),
            ttnn.ROW_MAJOR_LAYOUT,
            id="s3x2_height_rm",
        ),
        pytest.param(
            (1, 6, 8, 64),
            (3, 2),
            lambda d, s: _width_sharded_nhwc(s, d, ncores=2, layout=ttnn.ROW_MAJOR_LAYOUT),
            ttnn.ROW_MAJOR_LAYOUT,
            id="s3x2_width_rm",
        ),
        pytest.param(
            (1, 6, 8, 64),
            (3, 2),
            lambda d, s: _block_sharded_nhwc(s, d, grid_y=2, grid_x=2, layout=ttnn.ROW_MAJOR_LAYOUT),
            ttnn.ROW_MAJOR_LAYOUT,
            id="s3x2_block_rm",
        ),
        pytest.param(
            (1, 16, 8, 32),
            (4, 2),
            lambda d, s: DRAM_INTERLEAVED,
            ttnn.ROW_MAJOR_LAYOUT,
            id="s4x2_rm_dram",
        ),
    ],
)
def test_fold_asymmetric_strides(shape, stride, mc_factory, layout, device):
    """stride_h ≠ stride_w over sharded paths (interleaved RM asym covered by Group A rm_only)."""
    _run_fold(shape, stride[0], stride[1], layout, mc_factory(device, shape), ttnn.bfloat16, device)


# ──────────────────────────────────────────────────────────────────────────────
# Group H: larger / real-world-like shapes
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, stride, mc_factory, layout",
    [
        # YOLO/SSD-style fold input shapes
        pytest.param((1, 64, 64, 32), (2, 2), lambda d, s: DRAM_INTERLEAVED, ttnn.TILE_LAYOUT, id="64x64x32_tile_dram"),
        pytest.param((1, 64, 64, 32), (2, 2), lambda d, s: L1_INTERLEAVED, ttnn.TILE_LAYOUT, id="64x64x32_tile_l1"),
        pytest.param(
            (1, 32, 32, 64),
            (2, 2),
            lambda d, s: _height_sharded_nhwc(s, d, ncores=8, layout=ttnn.TILE_LAYOUT),
            ttnn.TILE_LAYOUT,
            id="32x32x64_height_tile",
        ),
        pytest.param(
            (1, 32, 32, 64),
            (2, 2),
            lambda d, s: _width_sharded_nhwc(s, d, ncores=4, layout=ttnn.TILE_LAYOUT),
            ttnn.TILE_LAYOUT,
            id="32x32x64_width_tile",
        ),
    ],
)
def test_fold_larger_shapes(shape, stride, mc_factory, layout, device):
    """Real-world-ish shapes across the routing paths."""
    _run_fold(shape, stride[0], stride[1], layout, mc_factory(device, shape), ttnn.bfloat16, device)


# ──────────────────────────────────────────────────────────────────────────────
# Group I: irregular/larger ncores configurations
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, ncores, layout",
    [
        pytest.param((1, 16, 8, 32), 8, ttnn.ROW_MAJOR_LAYOUT, id="rm_ncores8"),
        pytest.param((1, 8, 8, 32), 16, ttnn.ROW_MAJOR_LAYOUT, id="rm_ncores16"),
        pytest.param((1, 64, 64, 32), 16, ttnn.TILE_LAYOUT, id="tile_ncores16"),
    ],
)
def test_fold_height_sharded_ncores_scan(shape, ncores, layout, device):
    """Scan ncores values; HEIGHT_SHARDED routing should preserve sharding for all of them."""
    mc = _height_sharded_nhwc(shape, device, ncores=ncores, layout=layout)
    _run_fold(shape, 2, 2, layout, mc, ttnn.bfloat16, device)


# ──────────────────────────────────────────────────────────────────────────────
# Group J: explicit override_memory_config — user-requested output honored end-to-end.
# ──────────────────────────────────────────────────────────────────────────────


def _run_fold_with_override(shape, stride, layout, input_mc, override_mc, device, dtype=ttnn.bfloat16, pcc=0.9999):
    """Run fold with an explicit override_memory_config and assert output matches the request."""
    torch.manual_seed(12345)
    torch_dtype = _TTNN_TO_TORCH_DTYPE[dtype]
    x_nhwc = torch.rand(shape, dtype=torch_dtype)
    expected = _fold_torch_nhwc(x_nhwc, stride[0], stride[1])

    ttnn_in = ttnn.from_torch(x_nhwc, layout=layout, dtype=dtype, device=device, memory_config=input_mc)
    result = ttnn.fold(ttnn_in, stride_h=stride[0], stride_w=stride[1], override_memory_config=override_mc)

    assert (
        result.memory_config().memory_layout == override_mc.memory_layout
    ), f"override violated: requested {override_mc.memory_layout}, got {result.memory_config().memory_layout}"
    assert (
        result.memory_config().buffer_type == override_mc.buffer_type
    ), f"override buffer_type violated: requested {override_mc.buffer_type}, got {result.memory_config().buffer_type}"
    # Orientation contract: explicit override spec wins; otherwise inherit input's orientation when sharded.
    out_spec = result.memory_config().shard_spec
    if out_spec is not None:
        expected_orientation = (
            override_mc.shard_spec.orientation
            if override_mc.shard_spec is not None
            else (
                input_mc.shard_spec.orientation if input_mc.shard_spec is not None else ttnn.ShardOrientation.ROW_MAJOR
            )
        )
        assert (
            out_spec.orientation == expected_orientation
        ), f"override orientation violated: expected {expected_orientation}, got {out_spec.orientation}"

    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    n, h, w, c = expected.shape
    assert got.numel() == expected.numel(), f"numel mismatch: {tuple(got.shape)} vs {tuple(expected.shape)}"
    assert_with_pcc(expected.float(), got.reshape(n, h, w, c).float(), pcc)


@pytest.mark.parametrize(
    "shape, stride, layout, input_mc_factory, override_mc, id_suffix",
    [
        # Sharded input → interleaved output override.
        pytest.param(
            (1, 8, 8, 32),
            (2, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d, s: _height_sharded_nhwc(s, d, ncores=2, layout=ttnn.ROW_MAJOR_LAYOUT),
            DRAM_INTERLEAVED,
            "height_rm_to_dram",
            id="height_rm_to_dram",
        ),
        pytest.param(
            (1, 8, 8, 32),
            (2, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d, s: _height_sharded_nhwc(s, d, ncores=2, layout=ttnn.ROW_MAJOR_LAYOUT),
            L1_INTERLEAVED,
            "height_rm_to_l1",
            id="height_rm_to_l1",
        ),
        pytest.param(
            (1, 8, 8, 64),
            (2, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d, s: _width_sharded_nhwc(s, d, ncores=2, layout=ttnn.ROW_MAJOR_LAYOUT),
            DRAM_INTERLEAVED,
            "width_rm_to_dram",
            id="width_rm_to_dram",
        ),
        pytest.param(
            (1, 8, 8, 64),
            (2, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d, s: _block_sharded_nhwc(s, d, grid_y=2, grid_x=2, layout=ttnn.ROW_MAJOR_LAYOUT),
            L1_INTERLEAVED,
            "block_rm_to_l1",
            id="block_rm_to_l1",
        ),
        pytest.param(
            (1, 32, 32, 32),
            (2, 2),
            ttnn.TILE_LAYOUT,
            lambda d, s: _height_sharded_nhwc(s, d, ncores=8, layout=ttnn.TILE_LAYOUT),
            DRAM_INTERLEAVED,
            "height_tile_to_dram",
            id="height_tile_to_dram",
        ),
        # Interleaved input → user-requested sharded output (no shard_spec → synthesised inline).
        pytest.param(
            (1, 8, 8, 32),
            (2, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d, s: DRAM_INTERLEAVED,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1),
            "dram_to_height_no_spec",
            id="dram_to_height_no_spec",
        ),
        pytest.param(
            (1, 8, 8, 64),
            (2, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d, s: L1_INTERLEAVED,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1),
            "l1_to_width_no_spec",
            id="l1_to_width_no_spec",
        ),
        pytest.param(
            (1, 8, 8, 64),
            (2, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d, s: L1_INTERLEAVED,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1),
            "l1_to_block_no_spec",
            id="l1_to_block_no_spec",
        ),
        # TILE input → W/B sharded override (exercises the inverse-rescale path in compute_output_specs).
        # Shapes chosen so synthesised shard_w is divisible by stride_h*stride_w (no FATAL).
        pytest.param(
            (1, 32, 32, 128),
            (2, 2),
            ttnn.TILE_LAYOUT,
            lambda d, s: L1_INTERLEAVED,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1),
            "tile_l1_to_width_no_spec",
            id="tile_l1_to_width_no_spec",
        ),
        pytest.param(
            (1, 32, 32, 128),
            (2, 2),
            ttnn.TILE_LAYOUT,
            lambda d, s: L1_INTERLEAVED,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1),
            "tile_l1_to_block_no_spec",
            id="tile_l1_to_block_no_spec",
        ),
        pytest.param(
            (1, 32, 32, 128),
            (2, 2),
            ttnn.TILE_LAYOUT,
            lambda d, s: _height_sharded_nhwc(s, d, ncores=8, layout=ttnn.TILE_LAYOUT),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1),
            "tile_height_to_width_no_spec",
            id="tile_height_to_width_no_spec",
        ),
        pytest.param(
            (1, 32, 32, 128),
            (2, 2),
            ttnn.TILE_LAYOUT,
            lambda d, s: _height_sharded_nhwc(s, d, ncores=8, layout=ttnn.TILE_LAYOUT),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1),
            "tile_height_to_block_no_spec",
            id="tile_height_to_block_no_spec",
        ),
    ],
)
def test_fold_override_memory_config(shape, stride, layout, input_mc_factory, override_mc, id_suffix, device):
    """User-requested override_memory_config is honored end-to-end (layout + buffer_type)."""
    _run_fold_with_override(shape, stride, layout, input_mc_factory(device, shape), override_mc, device)
