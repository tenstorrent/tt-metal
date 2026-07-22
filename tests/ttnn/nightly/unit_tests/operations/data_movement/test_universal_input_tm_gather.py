# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Universal input/output tests for ttnn.gather — one case per routing path."""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc

_TTNN_TO_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
}

_TILE_H = 32
_TILE_W = 32

L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
DRAM_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


def _round_up(x, m):
    return ((x + m - 1) // m) * m


def _layout_hw_for_shard(shape, layout):
    """Return (total_height, width). TILE pads last-two dims to a tile each; RM uses logical dims."""
    if layout == ttnn.TILE_LAYOUT:
        total_h = 1
        for d in shape[:-2]:
            total_h *= d
        total_h *= _round_up(shape[-2], _TILE_H)
        return total_h, _round_up(shape[-1], _TILE_W)
    total_h = 1
    for d in shape[:-1]:
        total_h *= d
    return total_h, shape[-1]


def _tile_align(shard_shape, layout):
    h, w = shard_shape
    if layout == ttnn.TILE_LAYOUT:
        return (_round_up(h, _TILE_H), _round_up(w, _TILE_W))
    return (h, w)


def _interleaved_l1(_d):
    return L1_INTERLEAVED


def _interleaved_dram(_d):
    return DRAM_INTERLEAVED


def _height_sharded(
    shape,
    device,
    num_cores=4,
    layout=ttnn.TILE_LAYOUT,
    buffer_type=ttnn.BufferType.L1,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
):
    grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, grid.x * grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    total_h, w = _layout_hw_for_shard(shape, layout)
    shard_shape = _tile_align(((total_h + num_cores - 1) // num_cores, w), layout)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type,
        ttnn.ShardSpec(shard_grid, shard_shape, orientation),
    )


def _width_sharded(shape, device, num_cores=4, layout=ttnn.TILE_LAYOUT, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, grid.x * grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    total_h, w = _layout_hw_for_shard(shape, layout)
    shard_shape = _tile_align((total_h, (w + num_cores - 1) // num_cores), layout)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, shard_shape, orientation),
    )


def _block_sharded(shape, device, layout=ttnn.TILE_LAYOUT, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    grid = device.compute_with_storage_grid_size()
    gx = min(2, grid.x)
    gy = min(2, grid.y)
    total_h, w = _layout_hw_for_shard(shape, layout)
    shard_shape = _tile_align(((total_h + gy - 1) // gy, (w + gx - 1) // gx), layout)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))}),
            shard_shape,
            orientation,
        ),
    )


# Explicit shard helpers — caller-controlled grid + shard shape (irregular, uneven, sub-tile).


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


def _explicit_block_shard_config(device, grid_y, grid_x, sh, sw):
    compute_grid = device.compute_with_storage_grid_size()
    if grid_y > compute_grid.y or grid_x > compute_grid.x:
        pytest.skip(f"Device grid ({compute_grid.y}x{compute_grid.x}) too small for requested {grid_y}x{grid_x}")
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)


# ────────────────────────────────────────────────────────────────────────────────
# Shared runner — mirrors slice / permute conventions.
# ────────────────────────────────────────────────────────────────────────────────


def _run_gather(
    input_shape,
    index_shape,
    dim,
    layout,
    input_mem_config,
    index_mem_config,
    output_mem_config,
    dtype,
    device,
    expect_bit_exact=True,
    expected_shard_orientation=None,
):
    """Run gather; validate memory_layout + shard_spec match the request, then compare values."""
    torch.manual_seed(12345)
    torch_dtype = _TTNN_TO_TORCH_DTYPE[dtype]
    x = torch.randn(input_shape, dtype=torch_dtype)
    idx = torch.randint(0, input_shape[dim], index_shape, dtype=torch.int64)

    ttnn_in = ttnn.from_torch(x, layout=layout, dtype=dtype, device=device, memory_config=input_mem_config)
    ttnn_idx = ttnn.from_torch(idx, layout=layout, dtype=ttnn.uint32, device=device, memory_config=index_mem_config)

    result = ttnn.gather(ttnn_in, dim, index=ttnn_idx, memory_config=output_mem_config)

    if output_mem_config is not None:
        actual = result.memory_config()
        assert (
            actual.memory_layout == output_mem_config.memory_layout
        ), f"Expected memory_layout {output_mem_config.memory_layout}, got {actual.memory_layout}"
        if output_mem_config.memory_layout in (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ):
            assert actual.shard_spec is not None, "Sharded output requested but result has no shard_spec"
            if expected_shard_orientation is not None:
                actual_orientation = actual.shard_spec.orientation
                assert (
                    actual_orientation == expected_shard_orientation
                ), f"Expected shard orientation {expected_shard_orientation}, got {actual_orientation}"

    ref = torch.gather(x, dim, idx)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))

    # Gather is pure data-movement for bf16/f32 native paths; composite arm still routes through
    # tilize which shuffles (not quantizes) non-block dtypes, so bit-exact holds for all paths here.
    if expect_bit_exact:
        try:
            assert_equal(ref, got)
        except AssertionError as e:
            raise AssertionError(f"dtype={dtype}: {e}") from e
    else:
        assert_with_pcc(ref.float(), got.float(), 0.9999)


# ────────────────────────────────────────────────────────────────────────────────
# Group A: TILE — interleaved & sharded in/out, bf16 + f32 (bf8b unsupported)
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "input_shape, index_shape, dim, input_factory, index_factory, output_factory",
    [
        pytest.param(
            (1, 1, 128, 128),
            (1, 1, 128, 64),
            -1,
            _interleaved_l1,
            _interleaved_l1,
            _interleaved_l1,
            id="L1_interleaved_baseline",
        ),
        pytest.param(
            (1, 1, 128, 128),
            (1, 1, 128, 64),
            -1,
            _interleaved_dram,
            _interleaved_dram,
            _interleaved_dram,
            id="DRAM_interleaved",
        ),
        # TILE sharded in/index → interleaved out (HEIGHT-sharded baseline).
        pytest.param(
            (1, 1, 128, 64),
            (1, 1, 128, 64),
            -1,
            lambda d: _height_sharded((1, 1, 128, 64), d),
            lambda d: _height_sharded((1, 1, 128, 64), d),
            _interleaved_l1,
            id="TILE_height_to_interleaved",
        ),
        # TILE interleaved → HEIGHT-sharded out (explicit spec).
        pytest.param(
            (1, 1, 128, 128),
            (1, 1, 128, 64),
            -1,
            _interleaved_l1,
            _interleaved_l1,
            lambda d: _height_sharded((1, 1, 128, 64), d),
            id="TILE_interleaved_to_height",
        ),
        # TILE interleaved → BLOCK-sharded out (explicit spec).
        pytest.param(
            (1, 1, 128, 128),
            (1, 1, 128, 64),
            -1,
            _interleaved_l1,
            _interleaved_l1,
            lambda d: _block_sharded((1, 1, 128, 64), d),
            id="TILE_interleaved_to_block",
        ),
        # TILE HEIGHT-sharded → HEIGHT-sharded (same layout, explicit specs).
        pytest.param(
            (1, 1, 128, 64),
            (1, 1, 128, 64),
            -1,
            lambda d: _height_sharded((1, 1, 128, 64), d),
            lambda d: _height_sharded((1, 1, 128, 64), d),
            lambda d: _height_sharded((1, 1, 128, 64), d),
            id="TILE_height_to_height",
        ),
        # TILE WIDTH-sharded → WIDTH-sharded.
        pytest.param(
            (1, 1, 64, 128),
            (1, 1, 64, 64),
            -1,
            lambda d: _width_sharded((1, 1, 64, 128), d),
            lambda d: _width_sharded((1, 1, 64, 64), d),
            lambda d: _width_sharded((1, 1, 64, 64), d),
            id="TILE_width_to_width",
        ),
        # TILE BLOCK-sharded → BLOCK-sharded.
        pytest.param(
            (1, 1, 128, 128),
            (1, 1, 128, 64),
            -1,
            lambda d: _block_sharded((1, 1, 128, 128), d),
            lambda d: _block_sharded((1, 1, 128, 64), d),
            lambda d: _block_sharded((1, 1, 128, 64), d),
            id="TILE_block_to_block",
        ),
        # Implicit-output — no memory_config arg, sharded input passes through.
        pytest.param(
            (1, 1, 128, 128),
            (1, 1, 128, 64),
            -1,
            lambda d: _block_sharded((1, 1, 128, 128), d),
            lambda d: _block_sharded((1, 1, 128, 64), d),
            None,
            id="TILE_block_default_output",
        ),
    ],
)
def test_gather_tile(input_shape, index_shape, dim, input_factory, index_factory, output_factory, dtype, device):
    _run_gather(
        input_shape,
        index_shape,
        dim,
        ttnn.TILE_LAYOUT,
        input_factory(device),
        index_factory(device),
        output_factory(device) if output_factory is not None else None,
        dtype,
        device,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group B: TILE — sharded output without explicit shard_spec (derivation path)
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "memory_layout",
    [
        pytest.param(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, id="height"),
        pytest.param(ttnn.TensorMemoryLayout.WIDTH_SHARDED, id="width"),
        pytest.param(ttnn.TensorMemoryLayout.BLOCK_SHARDED, id="block"),
    ],
)
def test_gather_tile_interleaved_to_sharded_no_spec(memory_layout, dtype, device):
    """TILE interleaved in/index → sharded output **without** explicit shard_spec.
    Exercises compute_output_specs derivation via generate_transpose_shard_spec."""
    out_mc = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1)
    _run_gather(
        (1, 1, 128, 128),
        (1, 1, 128, 64),
        -1,
        ttnn.TILE_LAYOUT,
        L1_INTERLEAVED,
        L1_INTERLEAVED,
        out_mc,
        dtype,
        device,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group C: RM — interleaved single-core + multi-core + sharded in/out, bf16 + f32.
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "input_shape, index_shape, dim, input_factory, index_factory, output_factory",
    [
        # Pure interleaved RM — single-core (Wt below threshold).
        pytest.param(
            (1, 1, 32, 64),
            (1, 1, 32, 32),
            -1,
            _interleaved_l1,
            _interleaved_l1,
            _interleaved_l1,
            id="RM_interleaved_baseline",
        ),
        # DRAM variant.
        pytest.param(
            (1, 1, 64, 128),
            (1, 1, 64, 64),
            -1,
            _interleaved_dram,
            _interleaved_dram,
            _interleaved_dram,
            id="RM_interleaved_DRAM",
        ),
        # Multi-core RM (W > 60 * TILE_W = 1920).
        pytest.param(
            (1, 1, 4, 2048),
            (1, 1, 4, 1024),
            -1,
            _interleaved_dram,
            _interleaved_dram,
            _interleaved_dram,
            id="RM_interleaved_multi_core",
        ),
        # RM HEIGHT-sharded in/index/out (matching).
        pytest.param(
            (1, 1, 64, 64),
            (1, 1, 64, 32),
            -1,
            lambda d: _height_sharded((1, 1, 64, 64), d, num_cores=8, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda d: _height_sharded((1, 1, 64, 32), d, num_cores=8, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda d: _height_sharded((1, 1, 64, 32), d, num_cores=8, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="RM_height_to_height",
        ),
        # RM HEIGHT-sharded in → INTERLEAVED out.
        pytest.param(
            (1, 1, 64, 64),
            (1, 1, 64, 32),
            -1,
            lambda d: _height_sharded((1, 1, 64, 64), d, num_cores=8, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda d: _height_sharded((1, 1, 64, 32), d, num_cores=8, layout=ttnn.ROW_MAJOR_LAYOUT),
            _interleaved_l1,
            id="RM_height_to_interleaved",
        ),
        # RM WIDTH-sharded regular (W tile-aligned) — per-shard page-size override engages.
        pytest.param(
            (1, 1, 32, 128),
            (1, 1, 32, 64),
            -1,
            lambda d: _width_sharded((1, 1, 32, 128), d, num_cores=4, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda d: _width_sharded((1, 1, 32, 64), d, num_cores=4, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda d: _width_sharded((1, 1, 32, 64), d, num_cores=4, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="RM_width_regular_to_width",
        ),
        # RM BLOCK-sharded regular — same per-shard page-size mechanism.
        pytest.param(
            (1, 1, 64, 128),
            (1, 1, 64, 64),
            -1,
            lambda d: _block_sharded((1, 1, 64, 128), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda d: _block_sharded((1, 1, 64, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda d: _block_sharded((1, 1, 64, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="RM_block_regular_to_block",
        ),
    ],
)
def test_gather_rm(input_shape, index_shape, dim, input_factory, index_factory, output_factory, dtype, device):
    _run_gather(
        input_shape,
        index_shape,
        dim,
        ttnn.ROW_MAJOR_LAYOUT,
        input_factory(device),
        index_factory(device),
        output_factory(device) if output_factory is not None else None,
        dtype,
        device,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group D: RM — sharded output without explicit shard_spec (derivation path)
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "memory_layout",
    [
        pytest.param(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, id="height"),
        pytest.param(ttnn.TensorMemoryLayout.BLOCK_SHARDED, id="block"),
    ],
)
def test_gather_rm_interleaved_to_sharded_no_spec(memory_layout, dtype, device):
    """RM interleaved in/index → sharded output without shard_spec → compute_output_specs derives."""
    out_mc = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1)
    _run_gather(
        (1, 1, 32, 128),
        (1, 1, 32, 64),
        -1,
        ttnn.ROW_MAJOR_LAYOUT,
        L1_INTERLEAVED,
        L1_INTERLEAVED,
        out_mc,
        dtype,
        device,
    )


def test_gather_rm_sharded_in_sharded_no_spec_out(device):
    """RM HEIGHT-sharded in → HEIGHT no-spec out: derivation must work when input is already sharded."""
    in_shape = (1, 1, 64, 64)
    idx_shape = (1, 1, 64, 32)
    in_mc = _height_sharded(in_shape, device, num_cores=4, layout=ttnn.ROW_MAJOR_LAYOUT)
    idx_mc = _height_sharded(idx_shape, device, num_cores=4, layout=ttnn.ROW_MAJOR_LAYOUT)
    out_no_spec = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    _run_gather(
        in_shape,
        idx_shape,
        -1,
        ttnn.ROW_MAJOR_LAYOUT,
        in_mc,
        idx_mc,
        out_no_spec,
        ttnn.bfloat16,
        device,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group E: Irregular logical shapes — mirrors transpose's irregular matrix.
# ────────────────────────────────────────────────────────────────────────────────


# NoC-aligned shapes only. W=97 deferred pending the cross-op helper fix in #47299.
_IRREGULAR_SHAPES = [
    pytest.param((1, 1, 65, 96), (1, 1, 65, 48), id="shape_65x96"),
    pytest.param((1, 1, 33, 64), (1, 1, 33, 32), id="shape_33x64"),
]


@pytest.mark.parametrize("input_shape, index_shape", _IRREGULAR_SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["TILE", "RM"])
def test_gather_irregular_shapes_interleaved(input_shape, index_shape, layout, device):
    _run_gather(
        input_shape,
        index_shape,
        -1,
        layout,
        L1_INTERLEAVED,
        L1_INTERLEAVED,
        L1_INTERLEAVED,
        ttnn.bfloat16,
        device,
    )


@pytest.mark.parametrize("input_shape, index_shape", _IRREGULAR_SHAPES)
@pytest.mark.parametrize(
    "shard_factory",
    [
        pytest.param(_height_sharded, id="height"),
        pytest.param(_width_sharded, id="width"),
        pytest.param(_block_sharded, id="block"),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["TILE", "RM"])
def test_gather_irregular_shapes_sharded(input_shape, index_shape, shard_factory, layout, device):
    """2 shapes × 3 shard layouts × 2 element layouts; RM B/W-sharded routes via composite arm."""
    # block_sharded helper has no num_cores arg.
    if shard_factory is _block_sharded:
        in_mc = shard_factory(input_shape, device, layout=layout)
        idx_mc = shard_factory(index_shape, device, layout=layout)
    else:
        in_mc = shard_factory(input_shape, device, num_cores=4, layout=layout)
        idx_mc = shard_factory(index_shape, device, num_cores=4, layout=layout)
    _run_gather(
        input_shape,
        index_shape,
        -1,
        layout,
        in_mc,
        idx_mc,
        L1_INTERLEAVED,
        ttnn.bfloat16,
        device,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group F: Sharded inputs with explicit grids/shapes — irregular, uneven splits
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "input_shape, index_shape, dim, in_mc_factory, idx_mc_factory",
    [
        # TILE HEIGHT-sharded with explicit 3-core grid (uneven split of H).
        pytest.param(
            (1, 1, 96, 64),
            (1, 1, 96, 32),
            -1,
            lambda d: _explicit_height_shard_config(d, 3, 32, 64),
            lambda d: _explicit_height_shard_config(d, 3, 32, 32),
            id="TILE_height_3_uneven",
        ),
        # TILE WIDTH-sharded with explicit 4-core grid — width evenly split per core.
        pytest.param(
            (1, 1, 32, 128),
            (1, 1, 32, 128),
            -1,
            lambda d: _explicit_width_shard_config(d, 4, 32, 32),
            lambda d: _explicit_width_shard_config(d, 4, 32, 32),
            id="TILE_width_4_even",
        ),
    ],
)
def test_gather_tile_explicit_sharded_inputs(input_shape, index_shape, dim, in_mc_factory, idx_mc_factory, device):
    _run_gather(
        input_shape,
        index_shape,
        dim,
        ttnn.TILE_LAYOUT,
        in_mc_factory(device),
        idx_mc_factory(device),
        L1_INTERLEAVED,
        ttnn.bfloat16,
        device,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group G: Multi-batch / multi-channel sharded inputs (N>1 or C>1)
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "input_shape, index_shape, dim, layout, in_mc_factory, idx_mc_factory",
    [
        # TILE C=2 BLOCK-sharded. Both input W=128 and index W=64 split 2x2 stay tile-aligned.
        pytest.param(
            (1, 2, 64, 128),
            (1, 2, 64, 64),
            -1,
            ttnn.TILE_LAYOUT,
            lambda d: _explicit_block_shard_config(d, 2, 2, 64, 64),
            lambda d: _explicit_block_shard_config(d, 2, 2, 64, 32),
            id="TILE_C2_block",
        ),
        # RM N=2 C=2 HEIGHT-sharded.
        pytest.param(
            (2, 2, 32, 64),
            (2, 2, 32, 32),
            -1,
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d: _explicit_height_shard_config(d, 4, 32, 64),
            lambda d: _explicit_height_shard_config(d, 4, 32, 32),
            id="RM_N2C2_height",
        ),
    ],
)
def test_gather_multi_batch_channel(input_shape, index_shape, dim, layout, in_mc_factory, idx_mc_factory, device):
    _run_gather(
        input_shape,
        index_shape,
        dim,
        layout,
        in_mc_factory(device),
        idx_mc_factory(device),
        L1_INTERLEAVED,
        ttnn.bfloat16,
        device,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group H: Higher-rank inputs (rank > 4) — host pipeline collapses to 4D
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "input_shape, index_shape, dim, layout, input_factory, index_factory",
    [
        # Rank-5 with HEIGHT-sharded inputs (parity with slice's rank5 coverage).
        pytest.param(
            (1, 2, 4, 32, 64),
            (1, 2, 4, 32, 32),
            -1,
            ttnn.TILE_LAYOUT,
            lambda d: _height_sharded((1, 2, 4, 32, 64), d),
            lambda d: _height_sharded((1, 2, 4, 32, 32), d),
            id="TILE_rank5_height_sharded",
        ),
        pytest.param(
            (1, 1, 2, 8, 2, 64),
            (1, 1, 2, 8, 2, 32),
            -1,
            ttnn.TILE_LAYOUT,
            _interleaved_l1,
            _interleaved_l1,
            id="TILE_rank6_interleaved",
        ),
    ],
)
def test_gather_higher_rank(input_shape, index_shape, dim, layout, input_factory, index_factory, dtype, device):
    _run_gather(
        input_shape,
        index_shape,
        dim,
        layout,
        input_factory(device),
        index_factory(device),
        L1_INTERLEAVED,
        dtype,
        device,
    )


def test_gather_higher_rank_to_sharded_no_spec(device):
    """Rank-5 TILE in/idx → HEIGHT-sharded out without spec — exercises spec derivation."""
    _run_gather(
        (1, 2, 4, 32, 64),
        (1, 2, 4, 32, 32),
        -1,
        ttnn.TILE_LAYOUT,
        L1_INTERLEAVED,
        L1_INTERLEAVED,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1),
        ttnn.bfloat16,
        device,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group I: DRAM-sharded input — TensorAccessor is buffer-type agnostic at tile granularity.
# ────────────────────────────────────────────────────────────────────────────────


def test_gather_dram_sharded_input(device):
    """TILE HEIGHT-sharded in DRAM."""
    shape = (1, 1, 128, 64)
    idx_shape = (1, 1, 128, 32)
    dram_sharded = _height_sharded(shape, device, buffer_type=ttnn.BufferType.DRAM)
    idx_dram_sharded = _height_sharded(idx_shape, device, buffer_type=ttnn.BufferType.DRAM)
    _run_gather(
        shape,
        idx_shape,
        -1,
        ttnn.TILE_LAYOUT,
        dram_sharded,
        idx_dram_sharded,
        L1_INTERLEAVED,
        ttnn.bfloat16,
        device,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group J: Default-memory-config regression guards
# ────────────────────────────────────────────────────────────────────────────────


def test_gather_block_sharded_default_mc_no_longer_fatal(device):
    """Regression guard for the lifted sharded-output TT_FATAL: implicit out must inherit BLOCK_SHARDED."""
    in_shape = (1, 1, 64, 64)
    idx_shape = (1, 1, 64, 32)
    in_mc = _block_sharded(in_shape, device)
    idx_mc = _block_sharded(idx_shape, device)

    torch.manual_seed(0)
    x = torch.randn(in_shape, dtype=torch.bfloat16)
    idx = torch.randint(0, in_shape[-1], idx_shape, dtype=torch.int64)
    ttnn_in = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=in_mc)
    ttnn_idx = ttnn.from_torch(idx, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32, device=device, memory_config=idx_mc)

    result = ttnn.gather(ttnn_in, -1, index=ttnn_idx)
    assert result.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED

    ref = torch.gather(x, -1, idx)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert_equal(ref, got)


def test_gather_rm_irregular_width_composite_arm_fires(device):
    """Regression guard for needs_rm_irregular_composite — RM WIDTH-sharded W=49 must route via composite."""
    in_shape = (1, 1, 32, 49)
    idx_shape = (1, 1, 32, 49)
    in_mc = _width_sharded(in_shape, device, num_cores=1, layout=ttnn.ROW_MAJOR_LAYOUT)
    idx_mc = _width_sharded(idx_shape, device, num_cores=1, layout=ttnn.ROW_MAJOR_LAYOUT)
    _run_gather(
        in_shape,
        idx_shape,
        -1,
        ttnn.ROW_MAJOR_LAYOUT,
        in_mc,
        idx_mc,
        L1_INTERLEAVED,
        ttnn.bfloat16,
        device,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group K: Multi-core RM sharded — RmSingleRowMultiCore on sharded buffers.
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "input_shape, index_shape, in_factory, idx_factory, out_factory",
    [
        # Wide W triggers the multi-core kernel; HEIGHT-sharded in/out.
        pytest.param(
            (1, 1, 8, 2048),
            (1, 1, 8, 1024),
            lambda d: _height_sharded((1, 1, 8, 2048), d, num_cores=8, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda d: _height_sharded((1, 1, 8, 1024), d, num_cores=8, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda d: _height_sharded((1, 1, 8, 1024), d, num_cores=8, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="RM_multicore_height_sharded",
        ),
        # Wide W + WIDTH-sharded (regular) → multi-core + per-shard page-size override.
        pytest.param(
            (1, 1, 8, 2048),
            (1, 1, 8, 1024),
            lambda d: _width_sharded((1, 1, 8, 2048), d, num_cores=8, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda d: _width_sharded((1, 1, 8, 1024), d, num_cores=8, layout=ttnn.ROW_MAJOR_LAYOUT),
            _interleaved_l1,
            id="RM_multicore_width_to_interleaved",
        ),
        # Wide W + WIDTH-sharded OUT → exercises noc_async_write_sharded with offset=w_start*elem_size
        # into a WIDTH-sharded buffer where w_per_core != shard_W in general.
        pytest.param(
            (1, 1, 8, 2048),
            (1, 1, 8, 1024),
            lambda d: _width_sharded((1, 1, 8, 2048), d, num_cores=8, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda d: _width_sharded((1, 1, 8, 1024), d, num_cores=8, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda d: _width_sharded((1, 1, 8, 1024), d, num_cores=8, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="RM_multicore_width_sharded_in_out",
        ),
    ],
)
def test_gather_rm_multicore_sharded(input_shape, index_shape, in_factory, idx_factory, out_factory, device):
    _run_gather(
        input_shape,
        index_shape,
        -1,
        ttnn.ROW_MAJOR_LAYOUT,
        in_factory(device),
        idx_factory(device),
        out_factory(device),
        ttnn.bfloat16,
        device,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group L: dim variations — non-last dims via the pre/post-gather transform pipeline.
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "input_shape, index_shape, dim, layout",
    [
        # dim=2 (H): pre-gather pipeline transposes H↔W before the device op.
        pytest.param((1, 1, 128, 64), (1, 1, 32, 64), 2, ttnn.TILE_LAYOUT, id="TILE_dim2_H"),
        pytest.param((1, 1, 128, 64), (1, 1, 32, 64), 2, ttnn.ROW_MAJOR_LAYOUT, id="RM_dim2_H"),
        # dim=1 (C): pre-gather pipeline transposes C to the W position.
        pytest.param((1, 8, 32, 64), (1, 4, 32, 64), 1, ttnn.TILE_LAYOUT, id="TILE_dim1_C"),
        # dim=0 (N): pre-gather pipeline transposes N to the W position.
        pytest.param((4, 1, 32, 64), (2, 1, 32, 64), 0, ttnn.TILE_LAYOUT, id="TILE_dim0_N"),
    ],
)
def test_gather_dim_variations(input_shape, index_shape, dim, layout, dtype, device):
    # dim != -1 routes through perform_transpose, which downcasts f32 → bf16 internally
    # (transpose kernel doesn't have a native f32 path); PCC for f32, bit-exact for bf16.
    _run_gather(
        input_shape,
        index_shape,
        dim,
        layout,
        L1_INTERLEAVED,
        L1_INTERLEAVED,
        L1_INTERLEAVED,
        dtype,
        device,
        expect_bit_exact=(dtype == ttnn.bfloat16),
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group M: Preallocated `out=` — exercises the optional-output-tensor pipeline on
# TILE and RM (regression guard for #46565 review finding #8).
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "layout",
    [
        pytest.param(ttnn.TILE_LAYOUT, id="TILE"),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="RM"),
    ],
)
def test_gather_preallocated_output(layout, device):
    """Pre-allocated `out=` tensor must be filled in place and returned."""
    in_shape = (1, 1, 64, 128)
    idx_shape = (1, 1, 64, 64)
    torch.manual_seed(0)
    x = torch.randn(in_shape, dtype=torch.bfloat16)
    idx = torch.randint(0, in_shape[-1], idx_shape, dtype=torch.int64)
    ttnn_in = ttnn.from_torch(x, layout=layout, dtype=ttnn.bfloat16, device=device, memory_config=L1_INTERLEAVED)
    ttnn_idx = ttnn.from_torch(idx, layout=layout, dtype=ttnn.uint32, device=device, memory_config=L1_INTERLEAVED)
    out_alloc = ttnn.from_torch(
        torch.zeros(idx_shape, dtype=torch.bfloat16),
        layout=layout,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=L1_INTERLEAVED,
    )

    result = ttnn.gather(ttnn_in, -1, index=ttnn_idx, out=out_alloc)

    ref = torch.gather(x, -1, idx)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert_equal(ref, got)


# ────────────────────────────────────────────────────────────────────────────────
# Group N: COL_MAJOR shard-orientation preservation (mirrors #48025 for gather).
# ────────────────────────────────────────────────────────────────────────────────


_COL = ttnn.ShardOrientation.COL_MAJOR


@pytest.mark.parametrize(
    "input_shape, index_shape, element_layout, input_factory, index_factory, out_mem_layout",
    [
        # Same-layout HEIGHT→HEIGHT: adjust_shard_spec_to_shape fast path (already preserved orientation
        # pre-#48025; kept as regression guard).
        pytest.param(
            (1, 1, 128, 128),
            (1, 1, 128, 64),
            ttnn.TILE_LAYOUT,
            lambda s, d: _height_sharded(s, d, orientation=_COL),
            lambda s, d: _height_sharded(s, d, orientation=_COL),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            id="TILE_same_layout_H_to_H",
        ),
        # Cross-layout TILE HEIGHT→WIDTH: adjust skipped, generate_transpose_shard_spec via hint.
        pytest.param(
            (1, 1, 128, 128),
            (1, 1, 128, 128),
            ttnn.TILE_LAYOUT,
            lambda s, d: _height_sharded(s, d, orientation=_COL),
            lambda s, d: _height_sharded(s, d, orientation=_COL),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            id="TILE_cross_layout_H_to_W",
        ),
        # Cross-layout RM HEIGHT→WIDTH: same fix, RM factories.
        pytest.param(
            (1, 1, 128, 128),
            (1, 1, 128, 128),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda s, d: _height_sharded(s, d, layout=ttnn.ROW_MAJOR_LAYOUT, orientation=_COL),
            lambda s, d: _height_sharded(s, d, layout=ttnn.ROW_MAJOR_LAYOUT, orientation=_COL),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            id="RM_cross_layout_H_to_W",
        ),
        # Composite arm (W=49 irregular): pre-hop orientation snapshot in gather.cpp.
        pytest.param(
            (1, 1, 32, 49),
            (1, 1, 32, 49),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda s, d: _width_sharded(s, d, num_cores=1, layout=ttnn.ROW_MAJOR_LAYOUT, orientation=_COL),
            lambda s, d: _width_sharded(s, d, num_cores=1, layout=ttnn.ROW_MAJOR_LAYOUT, orientation=_COL),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            id="RM_composite_W49_W_to_H",
        ),
    ],
)
def test_gather_col_major_orientation_preserved(
    input_shape, index_shape, element_layout, input_factory, index_factory, out_mem_layout, device
):
    """COL_MAJOR orientation is preserved through native and composite synthesis paths (#48025 pattern)."""
    _run_gather(
        input_shape,
        index_shape,
        -1,
        element_layout,
        input_factory(input_shape, device),
        index_factory(index_shape, device),
        ttnn.MemoryConfig(out_mem_layout, ttnn.BufferType.L1),
        ttnn.bfloat16,
        device,
        expected_shard_orientation=_COL,
    )
