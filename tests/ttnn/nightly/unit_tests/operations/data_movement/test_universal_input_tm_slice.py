# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tests for slice universal input/output support — one case per routing path.

Routing summary (verified below):
  - TILE + INTERLEAVED in/out                  → SliceTileProgramFactory (native)
  - TILE + sharded in/out (with spec)          → SliceTileProgramFactory (native)
  - TILE + sharded out without shard_spec      → composite via L1 interleaved + reshard
  - RM + HEIGHT-sh in + HEIGHT-sh out, no step → SliceRmShardedProgramFactory (native)
  - RM + (sharded in or out), no step          → SliceRmProgramFactory (native); irregular B/W
                                                 composes via L1 interleaved
  - RM + has step                              → SliceRmStrideProgramFactory (native)
  - DRAM-sharded                               → to_memory_config-style fallback
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp

_TTNN_TO_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.bfloat8_b: torch.bfloat16,
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


def _sharded_no_spec(layout):
    return lambda _d: ttnn.MemoryConfig(layout, ttnn.BufferType.L1)


def _height_sharded(shape, device, num_cores=4, layout=ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1):
    grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, grid.x * grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    total_h, w = _layout_hw_for_shard(shape, layout)
    shard_shape = _tile_align(((total_h + num_cores - 1) // num_cores, w), layout)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type,
        ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )


def _width_sharded(shape, device, num_cores=4, layout=ttnn.TILE_LAYOUT):
    grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, grid.x * grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    total_h, w = _layout_hw_for_shard(shape, layout)
    shard_shape = _tile_align((total_h, (w + num_cores - 1) // num_cores), layout)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )


def _block_sharded(shape, device, layout=ttnn.TILE_LAYOUT):
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
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


# Explicit shard helpers — caller-controlled grid + shard shape (irregular, uneven, sub-tile).


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


# ────────────────────────────────────────────────────────────────────────────────
# Shared runner
# ────────────────────────────────────────────────────────────────────────────────


def _run_slice(
    shape,
    begins,
    ends,
    step,
    layout,
    input_mem_config,
    output_mem_config,
    dtype,
    device,
    ulp_when_exact=True,
    expected_shard_orientation=None,
):
    """Run slice and assert output memory_layout matches request; verifies shard_spec is set when sharded."""
    torch.manual_seed(12345)
    torch_dtype = _TTNN_TO_TORCH_DTYPE[dtype]
    x = torch.rand(shape, dtype=torch_dtype)

    ttnn_in = ttnn.from_torch(x, layout=layout, dtype=dtype, device=device, memory_config=input_mem_config)
    slices = tuple(slice(b, e, s) for b, e, s in zip(begins, ends, step))
    result = ttnn.slice(ttnn_in, begins, ends, step, memory_config=output_mem_config)

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
                assert actual.shard_spec.orientation == expected_shard_orientation, (
                    f"Expected output shard orientation {expected_shard_orientation}, "
                    f"got {actual.shard_spec.orientation}"
                )

    ref = x[slices]
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))

    # bf16 native → bit-exact; bf8_b/f32 → PCC (quantization or composite bf16 intermediates).
    if dtype == ttnn.bfloat16 and ulp_when_exact:
        assert_with_ulp(ref, got, ulp_threshold=0)
    else:
        assert_with_pcc(ref.float(), got.float(), 0.9999)


# ────────────────────────────────────────────────────────────────────────────────
# Group A: TILE — interleaved & sharded in/out (native via SliceTileProgramFactory)
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "shape, begins, ends, step, input_factory, output_factory",
    [
        # Baselines: pure interleaved.
        pytest.param(
            (1, 1, 128, 128),
            (0, 0, 0, 0),
            (1, 1, 64, 64),
            (1, 1, 1, 1),
            _interleaved_l1,
            _interleaved_l1,
            id="L1_interleaved_baseline",
        ),
        pytest.param(
            (1, 1, 128, 128),
            (0, 0, 0, 0),
            (1, 1, 64, 64),
            (1, 1, 1, 1),
            _interleaved_dram,
            _interleaved_dram,
            id="DRAM_interleaved",
        ),
        # TILE sharded in → interleaved (height + block; width is symmetric).
        pytest.param(
            (1, 1, 128, 64),
            (0, 0, 0, 0),
            (1, 1, 64, 64),
            (1, 1, 1, 1),
            lambda d: _height_sharded((1, 1, 128, 64), d),
            _interleaved_l1,
            id="TILE_height_to_interleaved",
        ),
        pytest.param(
            (1, 1, 128, 128),
            (0, 0, 0, 0),
            (1, 1, 64, 64),
            (1, 1, 1, 1),
            lambda d: _block_sharded((1, 1, 128, 128), d),
            _interleaved_l1,
            id="TILE_block_to_interleaved",
        ),
        # TILE interleaved → sharded (height + block; width is symmetric).
        pytest.param(
            (1, 1, 128, 128),
            (0, 0, 0, 0),
            (1, 1, 64, 64),
            (1, 1, 1, 1),
            _interleaved_l1,
            lambda d: _height_sharded((1, 1, 64, 64), d, num_cores=2),
            id="TILE_interleaved_to_height",
        ),
        pytest.param(
            (1, 1, 128, 128),
            (0, 0, 0, 0),
            (1, 1, 64, 64),
            (1, 1, 1, 1),
            _interleaved_l1,
            lambda d: _block_sharded((1, 1, 64, 64), d),
            id="TILE_interleaved_to_block",
        ),
        # TILE sharded → sharded (same layout, explicit spec).
        pytest.param(
            (1, 1, 128, 64),
            (0, 0, 0, 0),
            (1, 1, 64, 64),
            (1, 1, 1, 1),
            lambda d: _height_sharded((1, 1, 128, 64), d),
            lambda d: _height_sharded((1, 1, 64, 64), d, num_cores=2),
            id="TILE_height_to_height",
        ),
        pytest.param(
            (1, 1, 64, 128),
            (0, 0, 0, 0),
            (1, 1, 64, 64),
            (1, 1, 1, 1),
            lambda d: _width_sharded((1, 1, 64, 128), d),
            lambda d: _width_sharded((1, 1, 64, 64), d, num_cores=2),
            id="TILE_width_to_width",
        ),
        pytest.param(
            (1, 1, 128, 128),
            (0, 0, 0, 0),
            (1, 1, 64, 64),
            (1, 1, 1, 1),
            lambda d: _block_sharded((1, 1, 128, 128), d),
            lambda d: _block_sharded((1, 1, 64, 64), d),
            id="TILE_block_to_block",
        ),
        # TILE sharded → sharded (cross-layout) exercises the spec recomputation block.
        pytest.param(
            (1, 1, 128, 128),
            (0, 0, 0, 0),
            (1, 1, 64, 64),
            (1, 1, 1, 1),
            lambda d: _height_sharded((1, 1, 128, 128), d),
            lambda d: _width_sharded((1, 1, 64, 64), d, num_cores=2),
            id="TILE_height_to_width",
        ),
        # Implicit inheritance — no memory_config arg, input is sharded (block stands in for all).
        pytest.param(
            (1, 1, 128, 128),
            (0, 0, 0, 0),
            (1, 1, 64, 64),
            (1, 1, 1, 1),
            lambda d: _block_sharded((1, 1, 128, 128), d),
            None,
            id="TILE_block_default_output",
        ),
    ],
)
def test_slice_tile(shape, begins, ends, step, input_factory, output_factory, dtype, device):
    _run_slice(
        shape,
        begins,
        ends,
        step,
        ttnn.TILE_LAYOUT,
        input_factory(device),
        output_factory(device) if output_factory is not None else None,
        dtype,
        device,
        ulp_when_exact=(dtype != ttnn.bfloat8_b),
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group B: TILE — sharded output without explicit shard_spec (composite reshard)
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "memory_layout",
    [
        pytest.param(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, id="height"),
        pytest.param(ttnn.TensorMemoryLayout.BLOCK_SHARDED, id="block"),
    ],
)
def test_slice_tile_interleaved_to_sharded_no_spec(memory_layout, dtype, device):
    shape = (1, 1, 128, 128)
    out_mc = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1)
    _run_slice(
        shape,
        (0, 0, 0, 0),
        (1, 1, 64, 64),
        (1, 1, 1, 1),
        ttnn.TILE_LAYOUT,
        L1_INTERLEAVED,
        out_mc,
        dtype,
        device,
        ulp_when_exact=False,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group C: RM — interleaved, HEIGHT-sharded fast path, mixed sharded I/O
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "shape, begins, ends, step, input_factory, output_factory",
    [
        # Pure interleaved RM.
        pytest.param(
            (1, 1, 64, 64),
            (0, 0, 0, 0),
            (1, 1, 32, 32),
            (1, 1, 1, 1),
            _interleaved_l1,
            _interleaved_l1,
            id="RM_interleaved_baseline",
        ),
        # RM HEIGHT-sh in → HEIGHT-sh out (fast path — SliceRmShardedProgramFactory).
        pytest.param(
            (1, 1, 128, 64),
            (0, 0, 0, 0),
            (1, 1, 96, 64),
            (1, 1, 1, 1),
            lambda d: _height_sharded((1, 1, 128, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda d: _height_sharded((1, 1, 96, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="RM_height_to_height_fastpath",
        ),
        # RM HEIGHT-sh in → INTERLEAVED out (native via SliceRmProgramFactory).
        pytest.param(
            (1, 1, 128, 64),
            (0, 0, 0, 0),
            (1, 1, 96, 64),
            (1, 1, 1, 1),
            lambda d: _height_sharded((1, 1, 128, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            _interleaved_l1,
            id="RM_height_to_interleaved",
        ),
        # RM WIDTH-sh in → INTERLEAVED out (native: reader passes shard_W as the page-size override).
        pytest.param(
            (1, 1, 32, 128),
            (0, 0, 0, 0),
            (1, 1, 32, 64),
            (1, 1, 1, 1),
            lambda d: _width_sharded((1, 1, 32, 128), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            _interleaved_l1,
            id="RM_width_to_interleaved_native",
        ),
        # RM BLOCK-sh in → INTERLEAVED out (native; same mechanism).
        pytest.param(
            (1, 1, 64, 64),
            (0, 0, 0, 0),
            (1, 1, 32, 32),
            (1, 1, 1, 1),
            lambda d: _block_sharded((1, 1, 64, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            _interleaved_l1,
            id="RM_block_to_interleaved_native",
        ),
        # RM INTERLEAVED in → BLOCK-sh out (covers interleaved→sharded scale-spec for all 3).
        pytest.param(
            (1, 1, 128, 128),
            (0, 0, 0, 0),
            (1, 1, 64, 64),
            (1, 1, 1, 1),
            _interleaved_l1,
            lambda d: _block_sharded((1, 1, 64, 64), d),
            id="RM_interleaved_to_block",
        ),
        # RM WIDTH-sh in → HEIGHT-sh out (cross-layout; composite path via L1 interleaved).
        pytest.param(
            (1, 1, 64, 128),
            (0, 0, 0, 0),
            (1, 1, 64, 64),
            (1, 1, 1, 1),
            lambda d: _width_sharded((1, 1, 64, 128), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            lambda d: _height_sharded((1, 1, 64, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            id="RM_width_to_height",
        ),
        # Strided + sharded inputs/outputs (native; SliceRmStrideProgramFactory).
        pytest.param(
            (1, 1, 64, 64),
            (0, 0, 0, 0),
            (1, 1, 64, 64),
            (1, 1, 2, 1),
            lambda d: _height_sharded((1, 1, 64, 64), d, layout=ttnn.ROW_MAJOR_LAYOUT),
            _interleaved_l1,
            id="RM_strided_height_to_interleaved",
        ),
        pytest.param(
            (1, 1, 64, 64),
            (0, 0, 0, 0),
            (1, 1, 64, 64),
            (1, 1, 2, 2),
            _interleaved_l1,
            _interleaved_l1,
            id="RM_strided_interleaved_baseline",
        ),
    ],
)
def test_slice_rm(shape, begins, ends, step, input_factory, output_factory, dtype, device):
    _run_slice(
        shape,
        begins,
        ends,
        step,
        ttnn.ROW_MAJOR_LAYOUT,
        input_factory(device),
        output_factory(device) if output_factory is not None else None,
        dtype,
        device,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group D: RM — sharded output without explicit shard_spec
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "memory_layout",
    [
        pytest.param(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, id="height"),
        pytest.param(ttnn.TensorMemoryLayout.BLOCK_SHARDED, id="block"),
    ],
)
def test_slice_rm_interleaved_to_sharded_no_spec(memory_layout, dtype, device):
    shape = (1, 1, 64, 64)
    out_mc = ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1)
    _run_slice(
        shape,
        (0, 0, 0, 0),
        (1, 1, 32, 32),
        (1, 1, 1, 1),
        ttnn.ROW_MAJOR_LAYOUT,
        L1_INTERLEAVED,
        out_mc,
        dtype,
        device,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group E: Irregular logical shapes (last-two not tile multiples)
# ────────────────────────────────────────────────────────────────────────────────


_IRREGULAR_SHAPES = [
    # Logical H/W not tile multiples — one shape covers all irregular paths.
    ((1, 1, 65, 97), (0, 0, 0, 0), (1, 1, 32, 49), (1, 1, 1, 1)),
]


@pytest.mark.parametrize("shape, begins, ends, step", _IRREGULAR_SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_slice_irregular_shapes_interleaved(shape, begins, ends, step, layout, device):
    _run_slice(shape, begins, ends, step, layout, L1_INTERLEAVED, L1_INTERLEAVED, ttnn.bfloat16, device)


@pytest.mark.parametrize("shape, begins, ends, step", _IRREGULAR_SHAPES)
@pytest.mark.parametrize(
    "shard_factory",
    [
        pytest.param(_height_sharded, id="height"),
        pytest.param(_width_sharded, id="width"),
        pytest.param(_block_sharded, id="block"),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_slice_irregular_shapes_sharded(shape, begins, ends, step, shard_factory, layout, device):
    _run_slice(
        shape,
        begins,
        ends,
        step,
        layout,
        shard_factory(shape, device, layout=layout),
        L1_INTERLEAVED,
        ttnn.bfloat16,
        device,
        ulp_when_exact=False,  # composite paths may go through tile-rounded intermediates
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group F: Sharded inputs with explicit grids/shapes — irregular logical dims, uneven splits.
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, begins, ends, step, mc_factory",
    [
        # Keep one of each shard layout (height/width/block) with irregular dims.
        pytest.param(
            (1, 1, 65, 64),
            (0, 0, 0, 0),
            (1, 1, 32, 32),
            (1, 1, 1, 1),
            lambda d: _explicit_height_shard_config(d, 3, 32, 64),
            id="TILE_height_3_irregular_h",
        ),
        pytest.param(
            (1, 1, 32, 97),
            (0, 0, 0, 0),
            (1, 1, 32, 49),
            (1, 1, 1, 1),
            lambda d: _explicit_width_shard_config(d, 4, 32, 32),
            id="TILE_width_4_irregular_w",
        ),
    ],
)
def test_slice_tile_explicit_sharded_inputs(shape, begins, ends, step, mc_factory, device):
    _run_slice(
        shape,
        begins,
        ends,
        step,
        ttnn.TILE_LAYOUT,
        mc_factory(device),
        L1_INTERLEAVED,
        ttnn.bfloat16,
        device,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group G: Multi-batch / multi-channel sharded inputs (N>1 or C>1)
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, begins, ends, step, layout, mc_factory",
    [
        pytest.param(
            (1, 2, 64, 64),
            (0, 0, 0, 0),
            (1, 1, 32, 32),
            (1, 1, 1, 1),
            ttnn.TILE_LAYOUT,
            lambda d: _explicit_block_shard_config(d, 2, 2, 64, 32),
            id="TILE_C2_block",
        ),
        pytest.param(
            (2, 2, 32, 64),
            (0, 0, 0, 0),
            (1, 1, 32, 32),
            (1, 1, 1, 1),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda d: _explicit_height_shard_config(d, 4, 32, 64),
            id="RM_N2C2_height",
        ),
    ],
)
def test_slice_multi_batch_channel(shape, begins, ends, step, layout, mc_factory, device):
    _run_slice(
        shape,
        begins,
        ends,
        step,
        layout,
        mc_factory(device),
        L1_INTERLEAVED,
        ttnn.bfloat16,
        device,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group H: Sub-tile slices and non-tile-aligned RM HEIGHT shards
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, begins, ends, step, ncores, shard_shape",
    [
        pytest.param((1, 1, 64, 32), (0, 0, 0, 0), (1, 1, 32, 32), (1, 1, 1, 1), 2, (32, 32), id="TILE_64x32_hs2"),
        pytest.param((2, 1, 64, 32), (0, 0, 0, 0), (1, 1, 32, 32), (1, 1, 1, 1), 4, (32, 32), id="TILE_2x1_64x32_hs4"),
    ],
)
def test_slice_tile_subtile_height_sharded(shape, begins, ends, step, ncores, shard_shape, device):
    imc = _explicit_height_shard_config(device, ncores, shard_shape[0], shard_shape[1])
    _run_slice(
        shape,
        begins,
        ends,
        step,
        ttnn.TILE_LAYOUT,
        imc,
        L1_INTERLEAVED,
        ttnn.bfloat16,
        device,
    )


@pytest.mark.parametrize(
    "shape, begins, ends, step, shard_shape",
    [
        pytest.param((1, 1, 52, 64), (0, 0, 0, 0), (1, 1, 26, 64), (1, 1, 1, 1), (13, 64), id="RM_hs_13x64_nontile"),
    ],
)
def test_slice_row_major_height_sharded_nontile_aligned(shape, begins, ends, step, shard_shape, device):
    imc = _explicit_height_shard_config(device, 4, shard_shape[0], shard_shape[1])
    _run_slice(
        shape,
        begins,
        ends,
        step,
        ttnn.ROW_MAJOR_LAYOUT,
        imc,
        L1_INTERLEAVED,
        ttnn.bfloat16,
        device,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group I: Higher-rank inputs (rank > 4) — slice supports up to nD
# ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "shape, begins, ends, step, layout, input_factory",
    [
        pytest.param(
            (1, 2, 1, 64, 64),
            (0, 0, 0, 0, 0),
            (1, 2, 1, 32, 64),
            (1, 1, 1, 1, 1),
            ttnn.TILE_LAYOUT,
            lambda d: _height_sharded((1, 2, 1, 64, 64), d),
            id="TILE_rank5_height",
        ),
        pytest.param(
            (1, 1, 2, 2, 32, 32),
            (0, 0, 0, 0, 0, 0),
            (1, 1, 2, 2, 32, 32),
            (1, 1, 1, 1, 1, 1),
            ttnn.TILE_LAYOUT,
            _interleaved_l1,
            id="TILE_rank6_interleaved_noop",
        ),
    ],
)
def test_slice_higher_rank(shape, begins, ends, step, layout, input_factory, dtype, device):
    _run_slice(
        shape,
        begins,
        ends,
        step,
        layout,
        input_factory(device),
        L1_INTERLEAVED,
        dtype,
        device,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Group J: DRAM-sharded fallback — DRAM-sharded inputs route through L1 interleaved.
# ────────────────────────────────────────────────────────────────────────────────


def test_slice_dram_sharded_fallback(device):
    shape = (1, 1, 128, 64)
    dram_sharded = _height_sharded(shape, device, buffer_type=ttnn.BufferType.DRAM)
    torch.manual_seed(12345)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(
        x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=dram_sharded
    )
    result = ttnn.slice(ttnn_in, (0, 0, 0, 0), (1, 1, 64, 64), (1, 1, 1, 1), memory_config=L1_INTERLEAVED)
    assert result.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    ref = x[:, :, :64, :]
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert_with_pcc(ref.float(), got.float(), 0.9999)


# ────────────────────────────────────────────────────────────────────────────────
# Group K: Default-memory-config regression guards
# ────────────────────────────────────────────────────────────────────────────────


def test_slice_block_sharded_default_mc_no_longer_fatal(device):
    """BLOCK_SHARDED input + implicit output mc — must produce a rescaled BLOCK_SHARDED output."""
    shape = (1, 1, 64, 64)
    in_mc = _block_sharded(shape, device)
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=in_mc)
    result = ttnn.slice(ttnn_in, (0, 0, 0, 0), (1, 1, 32, 32), (1, 1, 1, 1))
    assert result.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    ref = x[:1, :1, :32, :32]
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert_with_pcc(ref.float(), got.float(), 0.9999)


def test_slice_rm_irregular_width_in_bad_fallback(device):
    """RM WIDTH-sharded input with irregular W=97: needs_rm_composite_input → composite via L1 interleaved."""
    shape = (1, 1, 64, 97)
    _run_slice(
        shape,
        (0, 0, 0, 0),
        (1, 1, 64, 49),
        (1, 1, 1, 1),
        ttnn.ROW_MAJOR_LAYOUT,
        _width_sharded(shape, device, num_cores=4, layout=ttnn.ROW_MAJOR_LAYOUT),
        L1_INTERLEAVED,
        ttnn.bfloat16,
        device,
        ulp_when_exact=False,
    )


@pytest.mark.parametrize(
    "shard_factory, mem_layout",
    [
        pytest.param(_width_sharded, ttnn.TensorMemoryLayout.WIDTH_SHARDED, id="width"),
    ],
)
def test_slice_rm_irregular_bw_output_out_bad_fallback(shard_factory, mem_layout, device):
    """RM interleaved → B/W-sharded output with irregular W=97: needs_rm_composite_output → composite then reshard."""
    in_shape = (1, 1, 64, 97)
    out_shape = (1, 1, 64, 49)
    _run_slice(
        in_shape,
        (0, 0, 0, 0),
        out_shape,
        (1, 1, 1, 1),
        ttnn.ROW_MAJOR_LAYOUT,
        L1_INTERLEAVED,
        shard_factory(out_shape, device, num_cores=4, layout=ttnn.ROW_MAJOR_LAYOUT)
        if shard_factory is _width_sharded
        else shard_factory(out_shape, device, layout=ttnn.ROW_MAJOR_LAYOUT),
        ttnn.bfloat16,
        device,
        ulp_when_exact=False,
    )


def test_slice_rm_sharded_in_sharded_no_spec_out(device):
    """HEIGHT-sharded in → HEIGHT no-spec out + step>1. Regression: ret_adjustment must not FATAL
    when the output_mc has no shard_spec."""
    in_shape = (1, 1, 96, 64)
    in_mc = _height_sharded(in_shape, device, num_cores=4, layout=ttnn.ROW_MAJOR_LAYOUT)
    out_no_spec_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    _run_slice(
        in_shape,
        (0, 0, 0, 0),
        (1, 1, 64, 64),
        (1, 1, 2, 1),
        ttnn.ROW_MAJOR_LAYOUT,
        in_mc,
        out_no_spec_mc,
        ttnn.bfloat16,
        device,
        ulp_when_exact=False,
    )


# COL_MAJOR cases cover orientation propagation through slice's composite-reshard path.
@pytest.mark.parametrize(
    "requested_out_layout",
    [
        pytest.param(ttnn.TensorMemoryLayout.BLOCK_SHARDED, id="col_major_block_in_block_out_nospec"),
        pytest.param(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, id="col_major_block_in_height_out_nospec"),
    ],
)
def test_slice_block_sharded_input_to_sharded_nospec(requested_out_layout, device):
    in_shape = (1, 1, 64, 64)
    grid = device.compute_with_storage_grid_size()
    gx, gy = min(2, grid.x), min(2, grid.y)
    in_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))}),
            (32, 32),
            ttnn.ShardOrientation.COL_MAJOR,
        ),
    )
    out_no_spec_mc = ttnn.MemoryConfig(requested_out_layout, ttnn.BufferType.L1)
    _run_slice(
        in_shape,
        (0, 0, 0, 0),
        (1, 1, 32, 64),
        (1, 1, 1, 1),
        ttnn.TILE_LAYOUT,
        in_mc,
        out_no_spec_mc,
        ttnn.bfloat16,
        device,
        ulp_when_exact=True,
        expected_shard_orientation=ttnn.ShardOrientation.COL_MAJOR,
    )


def test_slice_tile_same_layout_sharded_no_spec_halving(device):
    """TILE HEIGHT-sharded → halving slice → HEIGHT no-spec out. Regression: adjust_shard_spec_to_shape
    must fall back to generate_transpose_shard_spec when the scaled shard would be sub-tile."""
    in_shape = (1, 1, 64, 64)
    in_mc = _height_sharded(in_shape, device, num_cores=2, layout=ttnn.TILE_LAYOUT)
    out_no_spec_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    _run_slice(
        in_shape,
        (0, 0, 0, 0),
        (1, 1, 32, 32),
        (1, 1, 1, 1),
        ttnn.TILE_LAYOUT,
        in_mc,
        out_no_spec_mc,
        ttnn.bfloat16,
        device,
        ulp_when_exact=True,
    )


def test_slice_tile_same_layout_sharded_no_spec_halving_col_major(device):
    """Native adjust-failed fallback path: preserves COL_MAJOR input orientation."""
    in_shape = (1, 1, 64, 64)
    grid = device.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(2, grid, True)
    in_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, (32, 64), ttnn.ShardOrientation.COL_MAJOR),
    )
    out_no_spec_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    _run_slice(
        in_shape,
        (0, 0, 0, 0),
        (1, 1, 32, 32),
        (1, 1, 1, 1),
        ttnn.TILE_LAYOUT,
        in_mc,
        out_no_spec_mc,
        ttnn.bfloat16,
        device,
        ulp_when_exact=True,
        expected_shard_orientation=ttnn.ShardOrientation.COL_MAJOR,
    )


@pytest.mark.parametrize(
    "shard_factory",
    [pytest.param(_width_sharded, id="width")],
)
def test_slice_rm_bw_sharded_nonzero_width_begin(shard_factory, device):
    """RM B/W-sharded input with non-zero width-begin: needs_rm_composite_input must route through
    L1 interleaved because the native reader can't address non-contiguous shard pages."""
    in_shape = (1, 1, 64, 128)
    in_mc = (
        shard_factory(in_shape, device, num_cores=4, layout=ttnn.ROW_MAJOR_LAYOUT)
        if shard_factory is _width_sharded
        else shard_factory(in_shape, device, layout=ttnn.ROW_MAJOR_LAYOUT)
    )
    _run_slice(
        in_shape,
        (0, 0, 0, 32),
        (1, 1, 64, 96),
        (1, 1, 1, 1),
        ttnn.ROW_MAJOR_LAYOUT,
        in_mc,
        L1_INTERLEAVED,
        ttnn.bfloat16,
        device,
        ulp_when_exact=True,
    )


# Focused dtype spot-checks on representative paths. The core suites above run bf16 only to keep
# the file under budget; this hits f32 across TILE+RM paths and bf8b on the TILE-only path.
@pytest.mark.parametrize(
    "shape, layout, in_mc_fn, dtype",
    [
        pytest.param((1, 1, 128, 128), ttnn.TILE_LAYOUT, _interleaved_l1, ttnn.float32, id="f32_tile_interleaved"),
        pytest.param(
            (1, 1, 128, 64),
            ttnn.TILE_LAYOUT,
            lambda d: _height_sharded((1, 1, 128, 64), d),
            ttnn.float32,
            id="f32_tile_height_to_interleaved",
        ),
        pytest.param((1, 1, 128, 128), ttnn.TILE_LAYOUT, _interleaved_l1, ttnn.bfloat8_b, id="bf8b_tile_interleaved"),
    ],
)
def test_slice_dtype_coverage(shape, layout, in_mc_fn, dtype, device):
    _run_slice(
        shape,
        (0, 0, 0, 0),
        (1, 1, 64, 64),
        (1, 1, 1, 1),
        layout,
        in_mc_fn(device),
        L1_INTERLEAVED if in_mc_fn is not _interleaved_dram else DRAM_INTERLEAVED,
        dtype,
        device,
        ulp_when_exact=(dtype != ttnn.bfloat8_b),
    )
