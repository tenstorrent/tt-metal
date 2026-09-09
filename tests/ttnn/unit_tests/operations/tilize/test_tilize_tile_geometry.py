# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tile-geometry coverage for `tilize` (Refinement 4) — tiny tiles and retile.

DO NOT DELETE.

Two regimes share the `tile=` surface and are pinned here:

* **tiny tile** — ROW_MAJOR in, sub-32 tile out. A pure knob turn: `tile_h` is
  already a plan quantity everywhere (`in_page_bytes`, `rows_per_image`, both
  CBs' `TileDescriptor`, the reader's stick count), and the only behavioural
  change is that `can_use_fast_tilize` needs 32x32 output tiles, so a tiny tile
  takes the regular `tilize_init`/`tilize_block` path. Not arch-gated.

* **retile** — TILE in at one height, TILE out at another. The reader's
  `retile_block` walks FACES, not sticks: it gathers `retile_copy_unit`'s runs
  (the largest byte run contiguous in BOTH tile layouts) straight from the
  source tile's faces into the destination tile's faces, so there is no
  row-major intermediate and no compute kernel at all.

  These run on WORMHOLE even though the golden suite arch-gates the group to
  Blackhole: that gate is on the tiny-tile LLK, which a pure NoC face walk
  never touches. `test_retile_all_height_pairs` is the exhaustive witness.

The oracle is `torch.equal` throughout, not a PCC: tilize is a byte re-lay, so
anything short of bit-exact is a bug.
"""

import pytest
import torch

import ttnn

from ttnn.operations._op_contract import ExcludedCell
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize.tilize_program_descriptor import LEGAL_TILE_HEIGHTS, retile_copy_unit

TINY_TILE_HEIGHTS = [h for h in LEGAL_TILE_HEIGHTS if h != 32]


def _torch_input(shape):
    """Monotone-mod values: every element is distinguishable inside a tile, so a
    face landing at the wrong offset shows up as a mismatch rather than as an
    equal-valued coincidence."""
    n = 1
    for d in shape:
        n *= d
    return (torch.arange(n) % 4093).reshape(shape).to(torch.bfloat16)


def _to_device(t, device, *, tile_height=None, memory_config=None):
    kwargs = {}
    layout = ttnn.ROW_MAJOR_LAYOUT
    if tile_height is not None:
        layout = ttnn.TILE_LAYOUT
        kwargs["tile"] = ttnn.Tile([tile_height, 32])
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        **kwargs,
    )


def _check(out, t, out_tile_h):
    assert out.layout == ttnn.TILE_LAYOUT
    assert tuple(out.tile.tile_shape) == (out_tile_h, 32)
    assert list(out.shape) == list(t.shape)
    rb = ttnn.to_torch(out)
    assert torch.equal(rb.float(), t.float()), f"max diff {(rb.float() - t.float()).abs().max()}"


# ---------------------------------------------------------------------------
# tiny tile — ROW_MAJOR in, sub-32 out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tile_h", TINY_TILE_HEIGHTS)
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 32, 64), (1, 1, 64, 128), (1, 1, 2048, 64), (1, 1, 32, 2048)],
    ids=["single_block", "small", "tall_narrow", "short_wide"],
)
def test_tiny_tile(device, tile_h, shape):
    """The three golden `tile_geometry_tiny` shapes plus the two grid-scale
    geometries, at every sub-32 tile height."""
    t = _torch_input(shape)
    out = tilize(_to_device(t, device), dtype=ttnn.bfloat16, tile=ttnn.Tile([tile_h, 32]))
    _check(out, t, tile_h)


@pytest.mark.parametrize("tile_h,shape", [(16, (1, 1, 48, 64)), (8, (1, 1, 40, 64)), (4, (1, 1, 12, 96))])
def test_tiny_tile_realigns_h(device, tile_h, shape):
    """`alignment` is measured against the OUTPUT tile height, so a shape that
    is `h_non_aligned` at 32 can be `tile_aligned` at a tiny tile — H=48 is
    three whole tile-rows at 16 and needs no padding argument at all. This pins
    that re-partition of the `alignment` axis."""
    t = _torch_input(shape)
    out = tilize(_to_device(t, device), dtype=ttnn.bfloat16, tile=ttnn.Tile([tile_h, 32]))
    _check(out, t, tile_h)


@pytest.mark.parametrize("tile_h", [16, 4, 1])
def test_tiny_tile_padded(device, tile_h):
    """A tiny tile crossed with Refinement 2's H and W tails: the tail arithmetic
    is stated in units of `tile_h`, not of a literal 32."""
    shape = (1, 1, 33, 50)
    t = _torch_input(shape)
    out = tilize(_to_device(t, device), dtype=ttnn.bfloat16, tile=ttnn.Tile([tile_h, 32]), pad_value=0.0)
    _check(out, t, tile_h)


# ---------------------------------------------------------------------------
# retile — TILE in at one height, TILE out at another
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("out_tile_h", LEGAL_TILE_HEIGHTS)
@pytest.mark.parametrize("in_tile_h", LEGAL_TILE_HEIGHTS)
def test_retile_all_height_pairs(device, in_tile_h, out_tile_h):
    """Every legal (in, out) tile-height pair, including the equal-height
    identity re-lay (`tile=` must be honored on a TILE input) — 36 cells, which
    is the exhaustive witness for `retile_copy_unit`'s two cases."""
    shape = (2, 1, 64, 96)
    t = _torch_input(shape)
    tt = _to_device(t, device, tile_height=in_tile_h)
    out = tilize(tt, dtype=ttnn.bfloat16, tile=ttnn.Tile([out_tile_h, 32]))
    _check(out, t, out_tile_h)


@pytest.mark.parametrize(
    "shape,in_tile_h,out_tile_h",
    [
        ((1, 1, 32, 64), 32, 16),
        ((1, 1, 32, 64), 16, 32),
        ((1, 1, 64, 128), 8, 4),
        ((1, 1, 32, 64), 4, 2),
        ((1, 1, 32, 64), 2, 1),
        ((1, 1, 32, 64), 1, 32),
    ],
    ids=["32_to_16", "16_to_32", "8_to_4", "4_to_2", "2_to_1", "1_to_32"],
)
def test_retile_golden_scenarios(device, shape, in_tile_h, out_tile_h):
    """The six `tile_geometry_retile` golden scenarios verbatim. The golden suite
    skips them on Wormhole (the tiny-tile LLK gate); the face walk uses no LLK,
    so they are real coverage here."""
    t = _torch_input(shape)
    tt = _to_device(t, device, tile_height=in_tile_h)
    out = tilize(tt, dtype=ttnn.bfloat16, tile=ttnn.Tile([out_tile_h, 32]))
    _check(out, t, out_tile_h)


@pytest.mark.parametrize(
    "shape,in_tile_h,out_tile_h",
    [
        ((1, 1, 2048, 64), 32, 16),
        ((1, 1, 32, 4096), 32, 16),
        ((1, 1, 512, 512), 16, 32),
        ((1, 1, 512, 512), 8, 1),
        ((1, 1, 2048, 64), 1, 32),
    ],
    ids=["tall", "wide", "square_up", "square_tiny", "tall_1_to_32"],
)
def test_retile_grid_scale(device, shape, in_tile_h, out_tile_h):
    """Retile at grid scale: `block_width_tiles > 1`, several blocks per core,
    and both the column-cut and the row-cut work splits."""
    t = _torch_input(shape)
    tt = _to_device(t, device, tile_height=in_tile_h)
    out = tilize(tt, dtype=ttnn.bfloat16, tile=ttnn.Tile([out_tile_h, 32]))
    _check(out, t, out_tile_h)


@pytest.mark.parametrize(
    "shape,in_tile_h,out_tile_h",
    [
        ((1, 1, 20, 64), 8, 4),
        ((2, 3, 20, 64), 8, 4),
        ((3, 1, 12, 96), 8, 2),
        ((1, 1, 40, 64), 16, 8),
    ],
    ids=["h20", "h20_multi_image", "h12_multi_image", "h40"],
)
def test_retile_per_image_split(device, shape, in_tile_h, out_tile_h):
    """H a multiple of the OUTPUT tile height but not of the INPUT's: the two
    sides' padded per-image heights then differ (H=20 is three input tile-rows
    at 8 and five output tile-rows at 4), so a folded output tile-row is only a
    folded input tile-row after the image is factored out. This is the case the
    reader's per-image split exists for."""
    t = _torch_input(shape)
    tt = _to_device(t, device, tile_height=in_tile_h)
    out = tilize(tt, dtype=ttnn.bfloat16, tile=ttnn.Tile([out_tile_h, 32]))
    _check(out, t, out_tile_h)


@pytest.mark.parametrize(
    "in_buffer,out_buffer",
    [
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        (ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ],
    ids=["dram_to_l1", "l1_to_l1", "l1_to_dram"],
)
def test_retile_buffer_transitions(device, in_buffer, out_buffer):
    """Placement is a `TensorAccessor` concern on both legs of the face walk too."""
    t = _torch_input((1, 1, 64, 128))
    tt = _to_device(t, device, tile_height=32, memory_config=in_buffer)
    out = tilize(tt, out_buffer, dtype=ttnn.bfloat16, tile=ttnn.Tile([16, 32]))
    _check(out, t, 16)


# ---------------------------------------------------------------------------
# the two retile crossings EXCLUSIONS refuses (see tilize.EXCLUSIONS)
# ---------------------------------------------------------------------------


def test_retile_sharded_is_excluded(device, expect_error):
    """Retile x sharded is an `ExcludedCell`, not a wrong answer: the face walk
    addresses the source by interleaved TILE page index, and half-wiring a
    native sharded path that no test on this arch can verify is worse than
    refusing it."""
    t = _torch_input((1, 1, 64, 64))
    tt = _to_device(t, device, tile_height=32)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    sharded = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, 64], ttnn.ShardOrientation.ROW_MAJOR),
    )
    with expect_error(ExcludedCell, "unsupported combination"):
        tilize(tt, sharded, dtype=ttnn.bfloat16, tile=ttnn.Tile([16, 32]))


def test_retile_padded_is_excluded(device, expect_error):
    """Retile x padding is an `ExcludedCell`: the fill would have to land in
    output FACES the face walk never sources."""
    t = _torch_input((1, 1, 40, 64))
    tt = _to_device(t, device, tile_height=8)
    with expect_error(ExcludedCell, "unsupported combination"):
        tilize(tt, dtype=ttnn.bfloat16, tile=ttnn.Tile([32, 32]), pad_value=0.0)
