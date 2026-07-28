# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — the cross-core W-split: partial-sum combine + 1/rms multicast.

Pins the three things about this scheme that a correlational check cannot see:

1. **The split actually engages.** A wide/few-row shape must spread the
   DEPENDENT W axis over many cores; a correctness-only kernel that quietly ran
   on one core would pass every PCC gate in the suite.
2. **The combine gathers the RAW accumulator, not the reduced tile.** An
   absolute element-count assertion (all-ones input => mean(x^2) == 1 exactly),
   not a PCC: AccumulateViaAdd's finalize leaves the surviving x^2 lanes in
   columns 1..31 next to the row sum in column 0, so combining *reduced* tiles
   double-counts them. That produced mean 8.75 instead of 1.0 on W=64 — a 3x
   error that PCC still scored 0.9999, because rescaling every row by the same
   factor is invisible to a scale-invariant metric.
3. **Both W-splitting placements are consumed natively.** WIDTH/BLOCK sharded
   inputs are read from the core's own L1 shard (a zero-copy CB), including the
   geometries auto_shard_config actually emits — ragged core grids and shard
   grids that over-cover W with padding tiles.

Refinement 3 added a fourth, for the same reason as (2): the combine's fan-in
tree became a knob (flat root vs two-stage row-leaders, `COMBINE_MAX_FLAT_FANIN`),
and BOTH topologies must sum every element exactly once. A leader that finalized
its row sum instead of keeping the raw accumulator would re-introduce the (2)
bug one level up, and PCC would again score it 0.9999.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd


def _ref(x, gamma=None, eps=1e-6):
    xf = x.to(torch.float32)
    out = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


def _pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _placement(device, shape, sharded=False, tensor=None):
    grid = device.compute_with_storage_grid_size()
    t = tensor
    if t is None:
        t = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
    ht_total, wt_global = pd._tile_geometry(t)
    return pd._select_placement(device, grid, t, ht_total, wt_global, sharded)


# ---------------------------------------------------------------------------
# 1. the split engages, and the grid stops idling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 32, 16384), (1, 1, 32, 32768), (1, 1, 64, 12288), (1, 1, 32, 7168)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_wide_few_row_shapes_use_many_cores(device, shape):
    """The `_WIDE` / decode profiles: ht_total <= 2, so the INDEPENDENT row axis
    offers at most 2 units of work. Without the W-split these run on 1-2 of 110
    cores."""
    p = _placement(device, shape)
    assert p.w_split, f"{shape}: W-split did not engage"
    assert p.cw > 1
    assert p.num_cores > 4, f"{shape}: only {p.num_cores} cores"


def test_prefill_keeps_the_row_split(device):
    """A shape whose row axis already fills the grid must NOT pay for a combine."""
    p = _placement(device, (1, 1, 8192, 5120))
    assert not p.w_split
    assert p.cw == 1


def test_narrow_w_keeps_the_row_split(device):
    """W below the split threshold stays on the proven single-core-per-row path."""
    p = _placement(device, (1, 1, 32, 128))  # Wt = 4 < W_SPLIT_MIN_WT
    assert not p.w_split


def test_every_combine_group_is_one_virtual_rectangle_per_family(device):
    """A multicast addresses a VIRTUAL rectangle, and the logical grid is not
    virtually contiguous. Every family's sub-rectangle must map to a contiguous
    virtual x-range, or the broadcast lands on non-worker endpoints."""
    grid = device.compute_with_storage_grid_size()
    runs = pd._virtual_x_runs(device, grid)
    p = _placement(device, (1, 1, 32, 16384))
    for grp in p.groups:
        assert len(grp["subrects"]) <= pd.MAX_MCAST_FAMILIES
        for x0, _y0, x1, _y1 in grp["subrects"]:
            assert any(
                lo <= x0 and x1 <= hi for lo, hi in runs
            ), f"sub-rect x {x0}..{x1} straddles the virtual seam {runs}"


# ---------------------------------------------------------------------------
# 2. the combine sums every element exactly once (absolute, not correlational)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("W", [1024, 2304, 5120, 7168, 16384], ids=lambda w: f"W{w}")
def test_cross_core_combine_counts_every_element(device, W):
    """All-ones input => mean(x^2) == 1 => out == 1/sqrt(1+eps) on every element.

    Any mis-weighted combine (double-counted lanes, a dropped core's partial, a
    divisor that is the per-core slice instead of the grand total W) shifts this
    by a constant factor per row, which PCC cannot see.
    """
    shape = (1, 1, 32, W)
    assert _placement(device, shape).w_split, "shape must exercise the cross-core combine"

    x = torch.ones(shape, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(tt_x, epsilon=1e-6)).to(torch.float32)

    assert torch.allclose(
        out, torch.ones_like(out), rtol=2e-3, atol=2e-3
    ), f"W={W}: implied mean(x^2) = {1.0 / (out[0, 0, 0, 0].item() ** 2):.4f}, expected 1.0"


@pytest.mark.parametrize("W", [1024, 2304, 5120, 7168], ids=lambda w: f"W{w}")
def test_combine_topologies_agree(device, W):
    """Refinement 3: the STAGED gather must equal the FLAT one, bit-for-bit-close.

    Both topologies fold the same raw slice-accumulators over the same core set —
    only the fan-in tree differs — so they must produce the same answer. Staging
    is the exact place a second, premature within-tile fold would creep back in
    (a leader that *finalized* its row sum would double-count the surviving x^2
    lanes downstream), and PCC is blind to it because the error is one scale
    factor per row. So this is an all-ones ABSOLUTE check on both paths plus an
    agreement check between them, never a correlation.
    """
    shape = (1, 1, 32, W)
    saved = pd.COMBINE_MAX_FLAT_FANIN
    try:
        pd.COMBINE_MAX_FLAT_FANIN = 24  # the shipped default -> staged here
        staged = _placement(device, shape)
        assert staged.two_stage, f"W={W} must exercise the staged combine (CW={staged.cw})"
        assert staged.cw1 * staged.cw2 == staged.cw

        x = torch.ones(shape, dtype=torch.bfloat16)
        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out_staged = ttnn.to_torch(rms_norm(tt_x, epsilon=1e-6)).to(torch.float32)

        pd.COMBINE_MAX_FLAT_FANIN = 1 << 30  # force the Refinement 2 flat root
        flat = _placement(device, shape)
        assert not flat.two_stage and flat.cw == staged.cw, "the A/B must hold CW fixed"
        out_flat = ttnn.to_torch(rms_norm(tt_x, epsilon=1e-6)).to(torch.float32)
    finally:
        pd.COMBINE_MAX_FLAT_FANIN = saved

    ones = torch.ones_like(out_staged)
    for name, out in (("staged", out_staged), ("flat", out_flat)):
        assert torch.allclose(
            out, ones, rtol=2e-3, atol=2e-3
        ), f"W={W} {name}: implied mean(x^2) = {1.0 / (out[0, 0, 0, 0].item() ** 2):.4f}, expected 1.0"
    assert torch.allclose(out_staged, out_flat, rtol=2e-3, atol=2e-3), "topologies disagree"


@pytest.mark.parametrize(
    "shape,has_gamma",
    [
        ((1, 1, 32, 4096), True),
        ((1, 1, 32, 8192), False),
        ((1, 1, 32, 50), True),  # w_non_aligned: the mask rides ONE core's last tile
        ((1, 1, 17, 64), True),  # h_non_aligned
        ((2, 1, 64, 4096), True),  # several groups, several row-blocks
    ],
    ids=["4096", "8192-nogamma", "w50", "h17", "multigroup"],
)
def test_w_split_matches_reference(device, shape, has_gamma):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16) if has_gamma else None
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_g = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) if has_gamma else None
    out = ttnn.to_torch(rms_norm(tt_x, gamma=tt_g)).to(torch.float32)
    assert _pcc(out, _ref(x, g)) > 0.999


# ---------------------------------------------------------------------------
# 3. WIDTH / BLOCK sharded: native, zero-copy, on the geometries actually emitted
# ---------------------------------------------------------------------------

_SHARD_CASES = [
    # (shape, memory_layout, shard_shape, core_grid, id)
    ((1, 1, 32, 1024), ttnn.TensorMemoryLayout.WIDTH_SHARDED, [32, 128], (8, 1), "width-8x1"),
    ((1, 1, 32, 5120), ttnn.TensorMemoryLayout.WIDTH_SHARDED, [32, 160], (8, 4), "width-8x4"),
    # 7 wide x 4: the widest grid that still fits one virtual column run
    ((1, 1, 32, 7168), ttnn.TensorMemoryLayout.WIDTH_SHARDED, [32, 256], (7, 4), "width-7x4"),
    # 8 wide: the group STRADDLES the virtual seam -> two multicast families
    ((1, 1, 256, 512), ttnn.TensorMemoryLayout.BLOCK_SHARDED, [32, 64], (8, 8), "block-8x8"),
    ((1, 1, 8192, 1024), ttnn.TensorMemoryLayout.BLOCK_SHARDED, [1024, 128], (8, 8), "block-rows"),
]


@pytest.mark.parametrize(
    "shape,memory_layout,shard_shape,core_grid",
    [c[:4] for c in _SHARD_CASES],
    ids=[c[4] for c in _SHARD_CASES],
)
def test_sharded_pinned_geometry(device, shape, memory_layout, shard_shape, core_grid):
    from eval.sharding import shard_config

    torch.manual_seed(0)
    mc = shard_config(
        shard_shape,
        core_grid,
        memory_layout,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tt_g = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(tt_x, gamma=tt_g, memory_config=tt_x.memory_config())).to(torch.float32)
    assert _pcc(out, _ref(x, g)) > 0.999


@pytest.mark.parametrize(
    "shape,memory_layout",
    [
        ((1, 1, 32, 2048), ttnn.TensorMemoryLayout.WIDTH_SHARDED),  # 64 cores, RAGGED grid
        ((1, 1, 32, 8192), ttnn.TensorMemoryLayout.WIDTH_SHARDED),  # 86 cores, PADDING tiles
        ((1, 1, 64, 17), ttnn.TensorMemoryLayout.WIDTH_SHARDED),  # degenerate: 1 core, no combine
        ((4, 8, 47, 256), ttnn.TensorMemoryLayout.BLOCK_SHARDED),  # h_non_aligned
        ((1, 1, 32, 50), ttnn.TensorMemoryLayout.BLOCK_SHARDED),  # w_non_aligned
    ],
    ids=["ragged", "padding-tiles", "degenerate", "h-nonaligned", "w-nonaligned"],
)
def test_sharded_auto_geometry(device, shape, memory_layout):
    """The geometries `eval.sharding.auto_shard_config` really emits, including
    the two the bounding box makes awkward: a ragged core grid (padded with
    zero-work filler cores so the broadcast rectangle stays legal) and a shard
    grid that over-covers W (whose trailing padding tiles the reader zeroes, so
    they contribute nothing to sum(x^2))."""
    from eval.sharding import auto_shard_config

    torch.manual_seed(0)
    mc = auto_shard_config(list(shape), memory_layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tt_g = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(tt_x, gamma=tt_g, memory_config=tt_x.memory_config())).to(torch.float32)
    assert _pcc(out, _ref(x, g)) > 0.999


def test_sharded_input_is_consumed_in_place(device):
    """The shard IS the per-core block and it is already in this core's L1, so
    its CB must be BACKED BY THE TENSOR (zero-copy) rather than re-read through
    a TensorAccessor. Checked on the descriptor, not on the test colour: an
    accessor read of a core's own shard passes every numerical gate."""
    from eval.sharding import auto_shard_config

    shape = (1, 1, 32, 2048)
    mc = auto_shard_config(
        list(shape),
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    tt_x = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    from ttnn.operations.rms_norm import default_compute_kernel_config

    desc = pd.create_program_descriptor(
        tt_x, tt_out, device=device, compute_kernel_config=default_compute_kernel_config()
    )
    in_cb = next(cb for cb in desc.cbs if cb.format_descriptors[0].buffer_index == pd.CB_INPUT_TILES)
    out_cb = next(cb for cb in desc.cbs if cb.format_descriptors[0].buffer_index == pd.CB_OUTPUT_TILES)
    assert ttnn.get_cb_address(in_cb) == tt_x.buffer_address(), "input CB is not aliased onto the shard"
    assert ttnn.get_cb_address(out_cb) == tt_out.buffer_address(), "output CB is not aliased onto the shard"


def test_sharded_output_keeps_the_input_shard_spec(device):
    """The harness passes memory_config=input.memory_config() for every sharded
    cell, so the output must come back with that exact placement."""
    from eval.sharding import auto_shard_config

    shape = (1, 1, 256, 512)
    mc = auto_shard_config(
        list(shape),
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    tt_x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    out = rms_norm(tt_x, memory_config=mc)
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    assert list(out.memory_config().shard_spec.shape) == list(mc.shard_spec.shape)
