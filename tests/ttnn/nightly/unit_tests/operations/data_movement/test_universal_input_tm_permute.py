# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Tests for permute universal input/output support.

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

_TILE_HEIGHT = 32
_TILE_WIDTH = 32


def _round_up(x, mult):
    return ((x + mult - 1) // mult) * mult


def _padded_hw(shape, layout):
    """Return (total_height, width) with tile padding for TILE, logical for RM.
    Pads H and W independently to match ttnn's per-dim tile padding."""
    if layout == ttnn.TILE_LAYOUT:
        total_h = _round_up(shape[-2], _TILE_HEIGHT)
        for d in shape[:-2]:
            total_h *= d
        return total_h, _round_up(shape[-1], _TILE_WIDTH)
    h = 1
    for d in shape[:-1]:
        h *= d
    return h, shape[-1]


def _tile_align(shard_shape, layout):
    h, w = shard_shape
    if layout == ttnn.TILE_LAYOUT:
        return (_round_up(h, _TILE_HEIGHT), _round_up(w, _TILE_WIDTH))
    return (h, w)


def _height_shard_config(shape, device, num_cores=4, layout=ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1):
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, compute_grid.x * compute_grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
    total_h, w = _padded_hw(shape, layout)
    shard_shape = _tile_align(((total_h + num_cores - 1) // num_cores, w), layout)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type, shard_spec)


def _width_shard_config(shape, device, num_cores=4, layout=ttnn.TILE_LAYOUT):
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, compute_grid.x * compute_grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
    total_h, w = _padded_hw(shape, layout)
    shard_shape = _tile_align((total_h, (w + num_cores - 1) // num_cores), layout)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


def _block_shard_config(shape, device, layout=ttnn.TILE_LAYOUT):
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


def _explicit_height_shard_config(device, ncores, sh, sw):
    compute_grid = device.compute_with_storage_grid_size()
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has {compute_grid.x * compute_grid.y} cores, test needs {ncores}")
    spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


def _explicit_block_shard_config(device, grid_y, grid_x, sh, sw):
    compute_grid = device.compute_with_storage_grid_size()
    if grid_y > compute_grid.y or grid_x > compute_grid.x:
        pytest.skip(f"Device grid ({compute_grid.y}x{compute_grid.x}) too small for {grid_y}x{grid_x}")
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)


def _explicit_width_shard_config(device, ncores, sh, sw):
    compute_grid = device.compute_with_storage_grid_size()
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has {compute_grid.x * compute_grid.y} cores, test needs {ncores}")
    spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)


L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
DRAM_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


def _sharded_no_spec(memory_layout):
    return lambda _d: ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1)


def run_permute_test(
    shape,
    dims,
    device,
    input_layout=ttnn.TILE_LAYOUT,
    input_mem_config=None,
    output_mem_config=None,
    dtype=ttnn.bfloat16,
):
    torch.manual_seed(12345)
    torch_dtype = _TTNN_TO_TORCH_DTYPE[dtype]
    x = torch.rand(shape, dtype=torch_dtype)

    if input_mem_config is None:
        input_mem_config = L1_INTERLEAVED

    ttnn_input = ttnn.from_torch(x, layout=input_layout, dtype=dtype, device=device, memory_config=input_mem_config)
    result = ttnn.permute(ttnn_input, dims, memory_config=output_mem_config)

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
            assert actual.shard_spec is not None, "Sharded output requested but result has no shard_spec"

    ref = x.permute(dims)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))

    if dtype == ttnn.bfloat16:
        assert_with_ulp(ref, got, ulp_threshold=0)
    else:
        assert_with_pcc(ref.float(), got.float(), 0.9999)


# ──────────────────────────────────────────────────────────────
# 1. Universal-IO matrix: TILE input × output combos (bf16 only)
# ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, dims, input_layout, input_factory, output_factory",
    [
        # Decomposable (N=0): WH — height/block/width sharded inputs
        pytest.param(
            (2, 4, 32, 64),
            (0, 1, 3, 2),
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((2, 4, 32, 64), d),
            lambda _d: L1_INTERLEAVED,
            id="WH_height_to_interleaved",
        ),
        pytest.param(
            (1, 1, 64, 64),
            (0, 1, 3, 2),
            ttnn.TILE_LAYOUT,
            lambda d: _block_shard_config((1, 1, 64, 64), d),
            lambda _d: L1_INTERLEAVED,
            id="WH_block_to_interleaved",
        ),
        pytest.param(
            (1, 1, 64, 128),
            (0, 1, 3, 2),
            ttnn.TILE_LAYOUT,
            lambda d: _width_shard_config((1, 1, 64, 128), d),
            lambda _d: L1_INTERLEAVED,
            id="WH_width_to_interleaved",
        ),
        # Decomposable (N=0): WH — interleaved to sharded outputs
        pytest.param(
            (1, 1, 64, 128),
            (0, 1, 3, 2),
            ttnn.TILE_LAYOUT,
            lambda _d: L1_INTERLEAVED,
            lambda d: _height_shard_config((1, 1, 128, 64), d),
            id="WH_interleaved_to_height",
        ),
        pytest.param(
            (1, 1, 64, 64),
            (0, 1, 3, 2),
            ttnn.TILE_LAYOUT,
            lambda _d: L1_INTERLEAVED,
            lambda d: _block_shard_config((1, 1, 64, 64), d),
            id="WH_interleaved_to_block",
        ),
        pytest.param(
            (1, 1, 64, 128),
            (0, 1, 3, 2),
            ttnn.TILE_LAYOUT,
            lambda _d: L1_INTERLEAVED,
            lambda d: _width_shard_config((1, 1, 128, 64), d),
            id="WH_interleaved_to_width",
        ),
        # Default output (preserve input shard layout)
        pytest.param(
            (2, 4, 32, 64),
            (0, 1, 3, 2),
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((2, 4, 32, 64), d),
            None,
            id="WH_height_default_output",
        ),
        # Sharded-to-sharded without shard_spec
        pytest.param(
            (1, 1, 128, 64),
            (0, 1, 3, 2),
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((1, 1, 128, 64), d),
            _sharded_no_spec(ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
            id="WH_height_to_height_nospec",
        ),
        # Decomposable (N=0): HC
        pytest.param(
            (1, 4, 32, 64),
            (0, 2, 1, 3),
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((1, 4, 32, 64), d),
            lambda _d: L1_INTERLEAVED,
            id="HC_height_to_interleaved",
        ),
        # Decomposable (N=0): composite chain (one representative)
        pytest.param(
            (2, 4, 32, 64),
            (0, 2, 3, 1),
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((2, 4, 32, 64), d),
            None,
            id="0231_height_default_output",
        ),
        # CN (batch-channel swap)
        pytest.param(
            (2, 4, 32, 64),
            (1, 0, 2, 3),
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((2, 4, 32, 64), d, num_cores=8),
            lambda _d: L1_INTERLEAVED,
            id="CN_height_to_interleaved",
        ),
        pytest.param(
            (2, 4, 32, 64),
            (1, 0, 2, 3),
            ttnn.TILE_LAYOUT,
            lambda d: _block_shard_config((2, 4, 32, 64), d),
            lambda _d: L1_INTERLEAVED,
            id="CN_block_to_interleaved",
        ),
        # Non-decomposable (N moves) — prim::permute direct sharded read
        pytest.param(
            (2, 4, 32, 64),
            (2, 0, 1, 3),
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((2, 4, 32, 64), d),
            lambda _d: L1_INTERLEAVED,
            id="2013_height_to_interleaved",
        ),
        pytest.param(
            (2, 4, 32, 64),
            (3, 2, 1, 0),
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((2, 4, 32, 64), d),
            None,
            id="3210_height_default_output",
        ),
        pytest.param(
            (2, 4, 32, 64),
            (1, 2, 0, 3),
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((2, 4, 32, 64), d),
            lambda _d: L1_INTERLEAVED,
            id="1203_height_to_interleaved",
        ),
        # Transpose-delegated (one representative: CW)
        pytest.param(
            (2, 4, 32, 64),
            (0, 3, 2, 1),
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((2, 4, 32, 64), d),
            None,
            id="CW_height_default_output",
        ),
        # Block-sharded non-decomposable
        pytest.param(
            (2, 2, 64, 64),
            (2, 0, 1, 3),
            ttnn.TILE_LAYOUT,
            lambda d: _block_shard_config((2, 2, 64, 64), d),
            None,
            id="2013_block_default_output",
        ),
        # Width-sharded non-decomposable
        pytest.param(
            (2, 4, 32, 128),
            (2, 0, 1, 3),
            ttnn.TILE_LAYOUT,
            lambda d: _width_shard_config((2, 4, 32, 128), d),
            None,
            id="2013_width_default_output",
        ),
        # Interleaved baselines
        pytest.param(
            (2, 3, 64, 96),
            (0, 1, 3, 2),
            ttnn.TILE_LAYOUT,
            lambda _d: L1_INTERLEAVED,
            None,
            id="WH_interleaved_baseline",
        ),
        pytest.param(
            (2, 4, 32, 64),
            (2, 0, 1, 3),
            ttnn.TILE_LAYOUT,
            lambda _d: L1_INTERLEAVED,
            None,
            id="2013_interleaved_baseline",
        ),
        # ROW_MAJOR paths
        pytest.param(
            (2, 4, 32, 64),
            (0, 1, 3, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d: _height_shard_config((2, 4, 32, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda _d: L1_INTERLEAVED,
            id="WH_RM_height_to_interleaved",
        ),
        pytest.param(
            (2, 4, 32, 64),
            (2, 0, 1, 3),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda _d: L1_INTERLEAVED,
            None,
            id="2013_RM_interleaved_baseline",
        ),
    ],
)
def test_permute_universal_io(shape, dims, input_layout, input_factory, output_factory, device):
    run_permute_test(
        shape,
        dims,
        device,
        input_layout=input_layout,
        input_mem_config=input_factory(device),
        output_mem_config=output_factory(device) if output_factory is not None else None,
    )


# ──────────────────────────────────────────────────────────────
# 1b. Float32 spot-checks (dtype-sensitive paths only)
# ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, dims, input_layout, input_factory, output_factory",
    [
        pytest.param(
            (2, 4, 32, 64),
            (0, 1, 3, 2),
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((2, 4, 32, 64), d),
            lambda _d: L1_INTERLEAVED,
            id="WH_height_to_interleaved_f32",
        ),
        pytest.param(
            (2, 4, 32, 64),
            (0, 1, 3, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d: _height_shard_config((2, 4, 32, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda _d: L1_INTERLEAVED,
            id="WH_RM_height_to_interleaved_f32",
        ),
        pytest.param(
            (2, 4, 32, 64),
            (2, 0, 1, 3),
            ttnn.TILE_LAYOUT,
            lambda _d: L1_INTERLEAVED,
            None,
            id="2013_interleaved_f32",
        ),
    ],
)
def test_permute_universal_io_f32(shape, dims, input_layout, input_factory, output_factory, device):
    run_permute_test(
        shape,
        dims,
        device,
        input_layout=input_layout,
        input_mem_config=input_factory(device),
        output_mem_config=output_factory(device) if output_factory is not None else None,
        dtype=ttnn.float32,
    )


# ──────────────────────────────────────────────────────────────
# 2. Sharded input → sharded output without shard_spec
# ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "requested_out_layout",
    [
        pytest.param(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, id="H_out"),
        pytest.param(ttnn.TensorMemoryLayout.WIDTH_SHARDED, id="W_out"),
        pytest.param(ttnn.TensorMemoryLayout.BLOCK_SHARDED, id="B_out"),
    ],
)
def test_permute_sharded_input_to_sharded_nospec_wh(requested_out_layout, device):
    shape = (1, 1, 64, 64)
    run_permute_test(
        shape,
        (0, 1, 3, 2),
        device,
        input_mem_config=_block_shard_config(shape, device),
        output_mem_config=ttnn.MemoryConfig(requested_out_layout, ttnn.BufferType.L1),
    )


@pytest.mark.parametrize(
    "requested_out_layout",
    [
        pytest.param(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, id="H_out"),
        pytest.param(ttnn.TensorMemoryLayout.BLOCK_SHARDED, id="B_out"),
    ],
)
def test_permute_sharded_input_to_sharded_nospec_nondecomp(requested_out_layout, device):
    """Non-decomposable with same-layout input/output for spec generation."""
    shape = (2, 4, 32, 64)
    run_permute_test(
        shape,
        (2, 0, 1, 3),
        device,
        input_mem_config=_height_shard_config(shape, device),
        output_mem_config=ttnn.MemoryConfig(requested_out_layout, ttnn.BufferType.L1),
    )


# ──────────────────────────────────────────────────────────────
# 3. Cross-memory-config
# ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "dims",
    [
        pytest.param((0, 1, 3, 2), id="WH"),
        pytest.param((0, 2, 1, 3), id="HC"),
        pytest.param((1, 0, 2, 3), id="CN"),
    ],
)
def test_permute_dram_interleaved(dims, device):
    run_permute_test(
        (1, 4, 64, 128),
        dims,
        device,
        input_mem_config=DRAM_INTERLEAVED,
        output_mem_config=DRAM_INTERLEAVED,
    )


def test_permute_height_sharded_to_l1_interleaved(device):
    shape = (2, 4, 32, 64)
    run_permute_test(
        shape,
        (2, 0, 1, 3),
        device,
        input_mem_config=_height_shard_config(shape, device),
        output_mem_config=L1_INTERLEAVED,
    )


def test_permute_cross_shard_height_to_width(device):
    shape = (1, 1, 64, 128)
    run_permute_test(
        shape,
        (0, 1, 3, 2),
        device,
        input_mem_config=_height_shard_config(shape, device),
        output_mem_config=_width_shard_config((1, 1, 128, 64), device),
    )


def test_permute_cross_shard_block_to_height(device):
    shape = (1, 1, 64, 64)
    run_permute_test(
        shape,
        (0, 1, 3, 2),
        device,
        input_mem_config=_block_shard_config(shape, device),
        output_mem_config=_height_shard_config((1, 1, 64, 64), device),
    )


# ──────────────────────────────────────────────────────────────
# 4. Irregular shapes (not tile multiples)
# ──────────────────────────────────────────────────────────────

_IRREGULAR_SHAPES = [
    ((1, 1, 65, 97), (0, 1, 3, 2)),
    ((2, 3, 71, 79), (0, 1, 3, 2)),
    ((1, 13, 47, 64), (0, 2, 1, 3)),
    ((3, 5, 32, 64), (1, 0, 2, 3)),
]


@pytest.mark.parametrize("shape, dims", _IRREGULAR_SHAPES)
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_permute_irregular_shapes_interleaved(shape, dims, input_layout, device):
    run_permute_test(shape, dims, device, input_layout=input_layout)


# Rotate shard type across shapes to cover all 3 without full cross-product.
@pytest.mark.parametrize(
    "shape, dims, shard_factory",
    [
        pytest.param((1, 1, 65, 97), (0, 1, 3, 2), _block_shard_config, id="65x97_block"),
        pytest.param((2, 3, 71, 79), (0, 1, 3, 2), _height_shard_config, id="71x79_height"),
        pytest.param((1, 13, 47, 64), (0, 2, 1, 3), _width_shard_config, id="47x64_width"),
        pytest.param((3, 5, 32, 64), (1, 0, 2, 3), _block_shard_config, id="32x64_block"),
    ],
)
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_permute_irregular_shapes_sharded(shape, dims, shard_factory, input_layout, device):
    run_permute_test(
        shape,
        dims,
        device,
        input_layout=input_layout,
        input_mem_config=shard_factory(shape, device, layout=input_layout),
    )


# ──────────────────────────────────────────────────────────────
# 5. Explicit shard configs (tile-aligned shards on irregular shapes)
# ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, dims, mc_factory",
    [
        pytest.param(
            (1, 1, 65, 64),
            (0, 1, 3, 2),
            lambda d: _explicit_block_shard_config(d, 3, 2, 32, 32),
            id="WH_block_3x2_32x32",
        ),
        pytest.param(
            (1, 1, 64, 97),
            (0, 1, 3, 2),
            lambda d: _explicit_block_shard_config(d, 2, 4, 32, 32),
            id="WH_block_2x4_32x32",
        ),
        pytest.param(
            (1, 1, 65, 64),
            (0, 1, 3, 2),
            lambda d: _explicit_height_shard_config(d, 3, 32, 64),
            id="WH_height_3_32x64",
        ),
        pytest.param(
            (1, 1, 32, 97),
            (0, 1, 3, 2),
            lambda d: _explicit_width_shard_config(d, 4, 32, 32),
            id="WH_width_4_32x32",
        ),
        pytest.param(
            (1, 13, 32, 64),
            (0, 2, 1, 3),
            lambda d: _explicit_block_shard_config(d, 2, 2, 512, 32),
            id="HC_block_2x2_512x32",
        ),
    ],
)
def test_permute_tile_sharded_irregular(shape, dims, mc_factory, device):
    run_permute_test(shape, dims, device, input_mem_config=mc_factory(device))


# ──────────────────────────────────────────────────────────────
# 6. Uneven sharding (padded shape doesn't divide evenly into shard)
# ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, dims, mc_factory",
    [
        pytest.param(
            (1, 1, 96, 64),
            (0, 1, 3, 2),
            lambda d: _explicit_block_shard_config(d, 2, 2, 64, 32),
            id="block_2x2_64x32",
        ),
        pytest.param(
            (1, 1, 64, 96),
            (0, 1, 3, 2),
            lambda d: _explicit_block_shard_config(d, 2, 2, 32, 64),
            id="block_2x2_32x64",
        ),
    ],
)
def test_permute_tile_sharded_uneven(shape, dims, mc_factory, device):
    run_permute_test(shape, dims, device, input_mem_config=mc_factory(device))


# ──────────────────────────────────────────────────────────────
# 7. ROW_MAJOR sharded input combos
# ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, dims, input_factory, output_factory",
    [
        pytest.param(
            (1, 1, 64, 64),
            (0, 1, 3, 2),
            lambda d: _block_shard_config((1, 1, 64, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda _d: L1_INTERLEAVED,
            id="WH_RM_block_to_interleaved",
        ),
        pytest.param(
            (1, 1, 32, 128),
            (0, 1, 3, 2),
            lambda d: _width_shard_config((1, 1, 32, 128), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda _d: L1_INTERLEAVED,
            id="WH_RM_width_to_interleaved",
        ),
        pytest.param(
            (1, 1, 64, 64),
            (0, 1, 3, 2),
            lambda _d: L1_INTERLEAVED,
            lambda d: _block_shard_config((1, 1, 64, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="WH_RM_interleaved_to_block",
        ),
        pytest.param(
            (1, 1, 32, 128),
            (0, 1, 3, 2),
            lambda _d: L1_INTERLEAVED,
            lambda d: _width_shard_config((1, 1, 128, 32), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="WH_RM_interleaved_to_width",
        ),
        pytest.param(
            # Irregular output: interleaved input + BLOCK-sharded output, exercises noc_async_write_sharded with a tile-padded spec.
            (1, 1, 48, 65),
            (0, 1, 3, 2),
            lambda _d: L1_INTERLEAVED,
            lambda d: _block_shard_config((1, 1, 65, 48), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="WH_RM_interleaved_irregular_to_block",
        ),
        pytest.param(
            (1, 1, 64, 64),
            (0, 1, 3, 2),
            lambda d: _block_shard_config((1, 1, 64, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda d: _block_shard_config((1, 1, 64, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="WH_RM_block_to_block",
        ),
        pytest.param(
            (2, 4, 32, 64),
            (2, 0, 1, 3),
            lambda d: _block_shard_config((2, 4, 32, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda _d: L1_INTERLEAVED,
            id="2013_RM_block_to_interleaved",
        ),
        pytest.param(
            (2, 4, 32, 128),
            (2, 0, 1, 3),
            lambda d: _width_shard_config((2, 4, 32, 128), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda _d: L1_INTERLEAVED,
            id="2013_RM_width_to_interleaved",
        ),
        pytest.param(
            (2, 4, 32, 64),
            (2, 0, 1, 3),
            lambda d: _height_shard_config((2, 4, 32, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda _d: L1_INTERLEAVED,
            id="2013_RM_height_to_interleaved",
        ),
    ],
)
def test_permute_row_major_sharded(shape, dims, input_factory, output_factory, device):
    run_permute_test(
        shape,
        dims,
        device,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        input_mem_config=input_factory(device),
        output_mem_config=output_factory(device),
    )


@pytest.mark.parametrize(
    "shape, dims, memory_layout",
    [
        ((1, 1, 64, 128), (0, 1, 3, 2), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        ((1, 1, 64, 128), (0, 1, 3, 2), ttnn.TensorMemoryLayout.BLOCK_SHARDED),
        ((1, 1, 64, 128), (0, 1, 3, 2), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    ],
)
def test_permute_rm_sharded_output_no_shard_spec(shape, dims, memory_layout, device):
    run_permute_test(
        shape,
        dims,
        device,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        input_mem_config=L1_INTERLEAVED,
        output_mem_config=ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1),
    )


# ──────────────────────────────────────────────────────────────
# 8. Rank > 4
# ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, dims, input_layout, input_factory",
    [
        pytest.param(
            (2, 2, 2, 32, 64),
            (0, 2, 1, 4, 3),
            ttnn.TILE_LAYOUT,
            lambda _d: L1_INTERLEAVED,
            id="5d_tile_interleaved",
        ),
        pytest.param(
            (2, 2, 2, 32, 64),
            (0, 2, 1, 3, 4),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda _d: L1_INTERLEAVED,
            id="5d_rm_interleaved",
        ),
        pytest.param(
            (2, 2, 2, 32, 64),
            (0, 2, 1, 4, 3),
            ttnn.TILE_LAYOUT,
            lambda d: _height_shard_config((2, 2, 2, 32, 64), d, num_cores=8),
            id="5d_tile_height_sharded",
        ),
        pytest.param(
            (2, 2, 2, 32, 64),
            (0, 2, 1, 3, 4),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d: _height_shard_config((2, 2, 2, 32, 64), d, num_cores=8, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="5d_rm_height_sharded",
        ),
        pytest.param(
            (2, 2, 2, 32, 64),
            (0, 2, 1, 4, 3),
            ttnn.TILE_LAYOUT,
            lambda d: _width_shard_config((2, 2, 2, 32, 64), d, num_cores=4),
            id="5d_tile_width_sharded",
        ),
        pytest.param(
            (2, 2, 2, 32, 64),
            (0, 2, 1, 3, 4),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d: _block_shard_config((2, 2, 2, 32, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="5d_rm_block_sharded",
        ),
    ],
)
def test_permute_rank_gt4(shape, dims, input_layout, input_factory, device):
    run_permute_test(
        shape,
        dims,
        device,
        input_layout=input_layout,
        input_mem_config=input_factory(device),
    )


# ──────────────────────────────────────────────────────────────
# 9. Multi-batch/channel sharded inputs
# ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, dims, layout, mc_factory",
    [
        pytest.param(
            (2, 1, 64, 64),
            (0, 1, 3, 2),
            ttnn.TILE_LAYOUT,
            lambda d: _explicit_height_shard_config(d, 4, 32, 64),
            id="N2_tile_height_WH",
        ),
        pytest.param(
            (1, 2, 64, 64),
            (0, 1, 3, 2),
            ttnn.TILE_LAYOUT,
            lambda d: _explicit_block_shard_config(d, 2, 2, 64, 32),
            id="C2_tile_block_WH",
        ),
        pytest.param(
            (2, 2, 32, 64),
            (0, 1, 3, 2),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d: _explicit_height_shard_config(d, 4, 32, 64),
            id="N2C2_rm_height_WH",
        ),
    ],
)
def test_permute_multi_batch_channel_sharded(shape, dims, layout, mc_factory, device):
    run_permute_test(shape, dims, device, input_layout=layout, input_mem_config=mc_factory(device))


# ──────────────────────────────────────────────────────────────
# 10. DRAM-sharded fallback
# ──────────────────────────────────────────────────────────────


def test_permute_dram_sharded_fallback(device):
    shape = (1, 1, 128, 64)
    dram_sharded = _height_shard_config(shape, device, buffer_type=ttnn.BufferType.DRAM)

    torch.manual_seed(12345)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=dram_sharded
    )
    result = ttnn.permute(ttnn_input, (0, 1, 3, 2), memory_config=L1_INTERLEAVED)
    assert (
        result.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    ), f"Expected INTERLEAVED, got {result.memory_config().memory_layout}"

    ref = x.permute(0, 1, 3, 2)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert_with_ulp(ref, got, ulp_threshold=0)
