# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 5 — the FUSED square-accumulate datapath (`FUSE_SQUARE_ACCUM`).

Phases 2 and 3 used to be two FPU passes: square the block into `cb_x_squared`,
then elementwise-accumulate that block back out. Refinement 5 collapses them
into one by using the FPU's accumulate-into-DEST mode — `mul_tiles(x, x,
acc_to_dest)` over a sticky D0 leaves `Sum_w x_w^2` in DEST, which is the same
raw elementwise accumulator the pairwise-add datapath published.

**Why this file exists at all, rather than trusting the PCC gates.** This is a
new *reduction* datapath, and every accumulation bug this op has shipped was
invisible to PCC — three times now:

  * R1: a block-float reduce made the partial-W mask decode as all-zeros, so the
    last reduce-dim tile contributed nothing. All-ones `W=49` summed to **32**.
    PCC scored **0.9998** and the golden gate passed.
  * R2: combining *finalized* partials instead of raw accumulators double-counted
    the surviving x^2 lanes. All-ones `W=64` gave `mean(x^2) = 8.75`, not 1.0 —
    a 3x error. PCC scored **0.9999**.
  * R4: a wrong row->core map or a dropped W-tile would only rescale rows.

The common cause is that miscounting elements in a reduction *rescales each row
by one factor*, and PCC is scale-invariant. So the guard has to be **absolute**
(all-ones input => `mean(x^2) == 1` exactly => `out == 1/sqrt(1+eps)`) and it has
to be an **agreement check against the datapath being replaced**, never a
correlation. That is the R2/R3 discipline extended to Refinement 5's lever.

The knob is `pd.FUSE_SQUARE_ACCUM` (env `RMS_NORM_FUSE_SQ`), which is what makes
the A/B expressible: the same input, the same placement, the same block factors,
only the accumulation datapath differs.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd


# Shapes chosen so that between them they exercise every regime the fused path
# can be selected in, plus the two it must REFUSE to be selected in.
#
#   (1,1,32,1024)  W-split, cw > 1, NW == 1        -> fused, cross-core combine
#   (1,1,32,7168)  W-split, two-stage combine      -> fused, staged combine
#   (1,1,64,128)   plain row split, NW == 1        -> fused, LOCAL finalize
#   (1,1,8192,1024) grid full, NW == 1             -> fused, local, many rows
#   (1,1,8192,2304) NW == 3 (chunked reduce)       -> NOT fused (precondition)
#   (1,1,17,50)    partial W and partial H         -> NOT fused (mask precondition)
#
# Note (1,1,32,8192) is deliberately NOT in the "not fused" list even though its
# whole-tensor Wt is 256: the W-split hands each core only 8 W-tiles, so NW == 1
# and it IS fused. The precondition is on the PER-CORE chunk count, not on W.
_FUSED = [(1, 1, 32, 1024), (1, 1, 32, 7168), (1, 1, 64, 128), (1, 1, 8192, 1024)]
_NOT_FUSED = [(1, 1, 8192, 2304), (1, 1, 17, 50)]


def _blocking(device, shape, layout=ttnn.TILE_LAYOUT):
    grid = device.compute_with_storage_grid_size()
    dt = torch.bfloat16
    tt_x = ttnn.from_torch(torch.zeros(shape, dtype=dt), dtype=ttnn.bfloat16, layout=layout, device=device)
    tt_g = ttnn.from_torch(
        torch.zeros(1, 1, 1, shape[-1], dtype=dt), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ht_total, wt_global = pd._tile_geometry(tt_x)
    p = pd._select_placement(device, grid, tt_x, ht_total, wt_global, False)
    blk = pd._derive_blocking(tt_x, tt_g, grid.x * grid.y, p, l1_total_budget=pd._l1_total_budget(device))
    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_g)
    return blk


@pytest.mark.parametrize("shape", _FUSED, ids=lambda s: "x".join(map(str, s)))
def test_fused_path_is_selected(shape, device):
    """The lever must actually engage on these shapes.

    Without this, every assertion below could be passing because the op quietly
    fell back to the pairwise datapath — the same "it went green but the feature
    never ran" trap R2 and R4 both had to guard against on the placement side.
    """
    blk = _blocking(device, shape)
    assert blk.fuse_sq, f"{shape}: expected the fused datapath (NW={blk.nw}, partial_w={blk.has_partial_w})"
    assert blk.nw == 1


@pytest.mark.parametrize("shape", _NOT_FUSED, ids=lambda s: "x".join(map(str, s)))
def test_fused_path_respects_its_preconditions(shape, device):
    """...and must NOT engage where it is structurally wrong.

    A DEST accumulator dies at the next `tile_regs_acquire`, so it cannot carry a
    sum ACROSS chunks (`NW > 1`); and the 0/1 mask that zeroes a short last
    W-tile rides the reduce helper's partial-scaler hook, which the fused chain
    does not go through (`HAS_PARTIAL_W`). Both are static_asserted in the
    kernel, so a host-side mistake here is a compile error rather than silent
    wrong output — this test is what keeps it from becoming one.
    """
    blk = _blocking(device, shape)
    assert not blk.fuse_sq
    assert blk.nw > 1 or blk.has_partial_w


def _all_ones_implied_mean(device, shape, fuse):
    """Run all-ones through the op and invert the output back into mean(x^2).

    out = 1 / sqrt(mean + eps), so mean = 1/out^2 - eps. Any element the reduce
    dropped, double-counted, or divided by the wrong n_reduced shows up here as
    a number that is not 1.0 — the check PCC cannot make.
    """
    saved = pd.FUSE_SQUARE_ACCUM
    try:
        pd.FUSE_SQUARE_ACCUM = fuse
        assert _blocking(device, shape).fuse_sq == fuse, "the knob did not take"
        tt_x = ttnn.from_torch(
            torch.ones(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        out = ttnn.to_torch(rms_norm(tt_x, epsilon=1e-6)).to(torch.float32)
        ttnn.deallocate(tt_x)
    finally:
        pd.FUSE_SQUARE_ACCUM = saved
    return out


@pytest.mark.parametrize("shape", _FUSED, ids=lambda s: "x".join(map(str, s)))
def test_fused_reduce_counts_every_element(shape, device):
    """ABSOLUTE, not correlational: all-ones must give mean(x^2) == 1 exactly."""
    out = _all_ones_implied_mean(device, shape, fuse=True)
    implied = 1.0 / (out.flatten()[0].item() ** 2)
    assert torch.allclose(out, torch.ones_like(out), rtol=3e-3, atol=3e-3), (
        f"{shape}: implied mean(x^2) = {implied:.4f}, expected 1.0 — the fused "
        f"accumulate is not summing every element exactly once"
    )


@pytest.mark.parametrize("shape", _FUSED, ids=lambda s: "x".join(map(str, s)))
def test_fused_and_pairwise_datapaths_agree(shape, device):
    """The two accumulation datapaths must produce the same answer.

    They sum the same W tiles over the same core set; only *where* the running
    sum lives differs (a sticky DEST register vs an fp32 L1 accumulator reloaded
    per pair). So this is an equivalence check on random data, which is where a
    systematic factor — the failure mode of every accumulation bug this op has
    had — would show up as a constant ratio rather than as noise.
    """
    torch.manual_seed(7)
    x = torch.randn(shape, dtype=torch.bfloat16)
    gamma = torch.randn(shape[-1], dtype=torch.bfloat16)

    def run(fuse):
        saved = pd.FUSE_SQUARE_ACCUM
        try:
            pd.FUSE_SQUARE_ACCUM = fuse
            tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            tt_g = ttnn.from_torch(
                gamma.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            out = ttnn.to_torch(rms_norm(tt_x, gamma=tt_g, epsilon=1e-6)).to(torch.float32)
            ttnn.deallocate(tt_x)
            ttnn.deallocate(tt_g)
            return out
        finally:
            pd.FUSE_SQUARE_ACCUM = saved

    fused, pairwise = run(True), run(False)

    # A dropped / double-counted element is a constant scale factor per row, so
    # compare the RATIO's spread, not a correlation. Masked (never clamped — the
    # data is signed, and clamping a negative denominator inverts the sign) and
    # taken well away from zero so bf16 quantization of a near-zero element
    # cannot dominate the ratio.
    mask = pairwise.abs() > 1e-2
    ratio = fused[mask] / pairwise[mask]
    assert ratio.numel() > 0
    assert abs(ratio.median().item() - 1.0) < 5e-3, (
        f"{shape}: fused/pairwise median ratio {ratio.median().item():.5f} != 1.0 — "
        f"the fused datapath is systematically off by a scale factor"
    )
    torch.testing.assert_close(fused, pairwise, rtol=3e-2, atol=3e-2)
