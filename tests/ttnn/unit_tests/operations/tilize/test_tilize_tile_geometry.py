# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 2: tile geometry — tiny output tiles and retile of a Layout::TILE input.

* Tiny tiles: a Layout::ROW_MAJOR input tilized to ttnn.Tile([tile_h, 32]) with
  tile_h in {16, 8, 4, 2, 1}. Both CBs carry TileDescriptor(tile_h, 32), the
  helper leaves the fast path, and the output is allocated through a TensorSpec
  carrying the tile.
* Retile (`retile_l1_facewalk`): a Layout::TILE input at in_tile_h re-tiled to
  tile_h in the same dispatch. The reader reads whole input tiles (or reads a
  resident input shard in place) and face-walks them into sticks.

Every case is an identity check (bit-exact at bf16). Device is module-scoped by
this directory's conftest.py.
"""
import os

import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize
import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize.tilize_program_descriptor import _core_assignment, _tile_grid

L1 = ttnn.BufferType.L1
ROW = ttnn.ShardOrientation.ROW_MAJOR
HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED


def _crs(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


def _legacy(scheme, grid, shard):
    return ttnn.MemoryConfig(scheme, L1, ttnn.ShardSpec(grid, shard, ROW))


def _run(device, shape, *, tile_h, in_tile_h=None, in_mc=ttnn.DRAM_MEMORY_CONFIG, out_mc=None):
    torch.manual_seed(0)
    x = torch.randn(shape).bfloat16()
    if in_tile_h is None:
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc)
    else:
        t = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile([in_tile_h, 32]),
            device=device,
            memory_config=in_mc,
        )
    out = tilize(t, memory_config=out_mc, tile=ttnn.Tile([tile_h, 32]))
    assert out.layout == ttnn.TILE_LAYOUT
    assert list(out.tile.tile_shape) == [tile_h, 32]
    if out_mc is not None:
        assert out.memory_config().memory_layout == out_mc.memory_layout
    y = ttnn.to_torch(out)
    assert list(y.shape) == list(shape)
    # TILIZE_ABLATION=1: payload-stubbed kernels (ablation profiling) are wrong by construction.
    if os.environ.get("TILIZE_ABLATION") != "1":
        assert torch.equal(x, y), f"max diff {(x.float() - y.float()).abs().max()}"
    return t, out


@pytest.mark.parametrize("tile_h", [16, 8, 4, 2, 1])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param([1, 1, 64, 128], id="small"),
        pytest.param([2, 3, 64, 64], id="leading_dims"),
        pytest.param([1, 1, 4096, 64], id="tall_narrow_64_cores"),
    ],
)
def test_tiny_tile_dram(device, shape, tile_h):
    _run(device, shape, tile_h=tile_h)


@pytest.mark.parametrize(
    "in_tile_h,tile_h",
    [(32, 16), (16, 32), (8, 4), (4, 2), (2, 1), (1, 32), (32, 1), (1, 8), (32, 32)],
)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param([1, 1, 64, 128], id="small"),
        pytest.param([4, 3, 64, 64], id="leading_dims"),
        pytest.param([1, 1, 4096, 64], id="tall_narrow_64_cores"),
    ],
)
def test_retile_dram(device, shape, in_tile_h, tile_h):
    _run(device, shape, tile_h=tile_h, in_tile_h=in_tile_h)


@pytest.mark.parametrize(
    "shape,in_tile_h,tile_h",
    [
        # H = 48 is not a whole number of 32-row input tiles: each image's last input
        # tile-row is H-padded, so row_align falls back to 1 and the fold is per image.
        pytest.param([2, 1, 48, 64], 32, 16, id="h_padded_input_32_to_16"),
        pytest.param([3, 2, 48, 96], 32, 8, id="h_padded_input_32_to_8"),
    ],
)
def test_retile_h_padded_input(device, shape, in_tile_h, tile_h):
    _run(device, shape, tile_h=tile_h, in_tile_h=in_tile_h)


def test_retile_l1_interleaved(device):
    _run(device, [1, 1, 256, 64], tile_h=16, in_tile_h=32, in_mc=ttnn.L1_MEMORY_CONFIG, out_mc=ttnn.L1_MEMORY_CONFIG)


_SHARD_CASES = [
    pytest.param([1, 1, 32, 1024], WIDTH, _crs(0, 0, 7, 3), id="width"),
    pytest.param([1, 1, 1024, 32], HEIGHT, _crs(0, 0, 7, 3), id="height"),
    pytest.param([1, 1, 256, 256], BLOCK, _crs(0, 0, 7, 7), id="block"),
]


@pytest.mark.parametrize("tile_h", [16, 4, 1])
@pytest.mark.parametrize("shape,scheme,grid", _SHARD_CASES)
def test_tiny_tile_sharded_same_spec(device, shape, scheme, grid, tile_h):
    mc = _legacy(scheme, grid, [32, 32])
    _run(device, shape, tile_h=tile_h, in_mc=mc, out_mc=mc)


def _residency(t_in, t_out, tile_h):
    shape = list(t_in.shape)
    R, _ = _tile_grid(shape, tile_h)
    _, in_res, out_res, _ = _core_assignment(
        t_in, t_out, rows_total=R * tile_h, width=int(shape[-1]), tile_h=tile_h, max_block_width=64
    )
    return in_res, out_res


@pytest.mark.parametrize("in_tile_h,tile_h", [(32, 16), (16, 32), (8, 1), (1, 8)])
@pytest.mark.parametrize("shape,scheme,grid", _SHARD_CASES)
def test_retile_sharded_same_spec(device, shape, scheme, grid, in_tile_h, tile_h):
    """Same spec both sides: the input TILE shard backs cb_retile_staging (read in place)."""
    mc = _legacy(scheme, grid, [32, 32])
    t_in, t_out = _run(device, shape, tile_h=tile_h, in_tile_h=in_tile_h, in_mc=mc, out_mc=mc)
    assert _residency(t_in, t_out, tile_h) == (True, True)


def test_retile_crossovers(device):
    grid = _crs(0, 0, 7, 3)
    # Resident TILE input shard -> interleaved DRAM output.
    _run(device, [1, 1, 1024, 64], tile_h=8, in_tile_h=32, in_mc=_legacy(HEIGHT, grid, [32, 64]))
    # Streamed DRAM input -> resident output whose 16-row shards cut 32-row input tiles
    # (row_align falls back to 1: every core reads its input tile-rows whole).
    out16 = _legacy(HEIGHT, grid, [16, 64])
    _run(device, [1, 1, 512, 64], tile_h=16, in_tile_h=32, out_mc=out16)
    _run(device, [1, 1, 512, 64], tile_h=8, in_tile_h=32, out_mc=out16)


@pytest.mark.parametrize(
    "in_tile_h,tile_h",
    [pytest.param(None, 16, id="tiny_tile_16"), pytest.param(32, 16, id="retile_32_to_16")],
)
def test_tile_geometry_program_cache(device, in_tile_h, tile_h):
    """Same config on fresh allocations: at most one program, then cache hits (addresses are RT args)."""
    device.enable_program_cache()
    n0 = device.num_program_cache_entries()
    keep_alive = []
    first_delta = None
    for i in range(4):
        keep_alive.append(_run(device, [1, 1, 32, 64], tile_h=tile_h, in_tile_h=in_tile_h))
        delta = device.num_program_cache_entries() - n0
        if i == 0:
            first_delta = delta
            assert first_delta <= 1, f"first call built {first_delta} programs"
        else:
            assert delta == first_delta, f"call {i + 1} added {delta - first_delta} program(s)"


# Retile knobs (tilize_program_descriptor.py): every parked / non-default setting stays bit-exact.
RETILE_KNOBS = {
    "facewalk_riscv": dict(RETILE_FACEWALK_NOC=False),
    "stage1": dict(RETILE_STAGE_DEPTH=1),
    "stage3": dict(RETILE_STAGE_DEPTH=3),
    # one tile-row per CB quantum: a unit of row_align > 1 tile-rows pushes mid-unit
    "quantum1": dict(QUANTUM_MIN_TILES=1),
    "quantum1_riscv": dict(QUANTUM_MIN_TILES=1, RETILE_FACEWALK_NOC=False),
    # block_width capped at 2: C = 5 -> 3 column blocks, the last one ragged
    "narrow_blocks": dict(FAST_TILIZE_MAX_BLOCK_WIDTH=2),
}


@pytest.mark.parametrize("knob", list(RETILE_KNOBS), ids=list(RETILE_KNOBS))
@pytest.mark.parametrize(
    "shape,in_tile_h,tile_h",
    [
        pytest.param([1, 1, 64, 160], 32, 4, id="64x160_32to4"),
        pytest.param([1, 1, 4320, 160], 32, 16, id="4320x160_32to16"),
        pytest.param([2, 3, 64, 96], 8, 32, id="2x3x64x96_8to32"),
        pytest.param([2, 1, 48, 160], 32, 16, id="h_padded_32to16"),
    ],
)
def test_retile_knob(device, monkeypatch, shape, in_tile_h, tile_h, knob):
    for name, value in RETILE_KNOBS[knob].items():
        monkeypatch.setattr(pd, name, value)
    _run(device, shape, tile_h=tile_h, in_tile_h=in_tile_h)


@pytest.mark.parametrize("knob", ["facewalk_riscv", "quantum1"])
def test_retile_knob_resident(device, monkeypatch, knob):
    for name, value in RETILE_KNOBS[knob].items():
        monkeypatch.setattr(pd, name, value)
    mc = _legacy(HEIGHT, _crs(0, 0, 7, 3), [32, 32])
    _run(device, [1, 1, 1024, 32], tile_h=4, in_tile_h=32, in_mc=mc, out_mc=mc)


@pytest.mark.parametrize(
    "in_tile_h,tile_h",
    [
        pytest.param(None, 16, id="tiny16"),
        pytest.param(None, 8, id="tiny8"),
        pytest.param(None, 1, id="tiny1"),
        pytest.param(32, 32, id="retile32_32"),
        pytest.param(32, 16, id="retile32_16"),
        pytest.param(16, 32, id="retile16_32"),
        pytest.param(1, 32, id="retile1_32"),
    ],
)
def test_tile_geometry_perf_shape(device, in_tile_h, tile_h):
    """Perf reference on the perf-focus shape [1,1,16384,64] (run with run_safe_pytest.sh --profile)."""
    _run(device, [1, 1, 16384, 64], tile_h=tile_h, in_tile_h=in_tile_h)
