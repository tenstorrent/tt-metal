# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Phase 1 (box-sparse tile-skip): the per-q-tile packed k-tile band must cover EVERY live key tile.

The fused gather packs a block chunk's neighborhood box densely (T-outer/H-mid/W-inner) and the flash
masks the over-included cells. :func:`qtile_k_band` gives, per 32-query sub-tile, the contiguous packed
k-tile band [lo, hi) that can hold a live key; the kernel will skip tiles outside it. The skip is only
LOSSLESS if the band covers every genuinely-live (masked-in) tile -- this test brute-forces the true
live map from the mask definition and asserts the band is a strict superset (no false skip), for every
chunk (interior AND grid edges) and every q-tile. It also reports the realized skip fraction.

This is the host spec for the C++ twin ``block_qtile_k_band`` (windowed_loop_geometry.hpp), validated on
device in Phase 2/3 -- same pattern as test_block_permute.py -> test_block_sdpa_op.py.
"""
import pytest

from models.tt_dit.layers.block_permute import TILE, nbr_shift_start, neighborhood_box_block, qtile_k_band


def _live_ktiles_bruteforce(qc, wid0, block, grid, kernel, hb, wb, w_origin, box):
    """The set of packed k-tiles that hold >=1 genuinely-live cell for the sub-tile [wid0, wid0+TILE)."""
    bt, bh, bw = block
    T, H, W = grid
    kt, kh, kw = kernel
    t0, t1, h_lo, h_hi, w_lo, w_hi = box
    bh_box, bw_box = h_hi - h_lo, w_hi - w_lo
    hw_box, n_box = bh_box * bw_box, (t1 - t0) * bh_box * bw_box
    ker_t, ker_h, ker_w = min(kt, T), min(kh, H), min(kw, W)
    vol = bt * bh * bw
    bti, bhi, bwi = qc // (wb * hb), (qc // wb) % hb, qc % wb

    # precompute each query's window in physical coords
    windows = []
    for wid in range(wid0, min(wid0 + TILE, vol)):
        dw, dh, dt = wid % bw, (wid // bw) % bh, wid // (bw * bh)
        q_t, q_h, q_w = w_origin + bti * bt + dt, bhi * bh + dh, bwi * bw + dw
        wt, wh, ww = nbr_shift_start(q_t, T, kt), nbr_shift_start(q_h, H, kh), nbr_shift_start(q_w, W, kw)
        windows.append((wt, wt + ker_t, wh, wh + ker_h, ww, ww + ker_w))

    live = set()
    for j in range(n_box):
        jt, rem = divmod(j, hw_box)
        jh, jw = divmod(rem, bw_box)
        ct, ch, cw = t0 + jt, h_lo + jh, w_lo + jw  # cell physical coord
        for wt0, wt1, wh0, wh1, ww0, ww1 in windows:
            if wt0 <= ct < wt1 and wh0 <= ch < wh1 and ww0 <= cw < ww1:
                live.add(j // TILE)
                break
    return live, (n_box + TILE - 1) // TILE


@pytest.mark.parametrize(
    "grid,block,kernel",
    [
        ((10, 16, 8), (5, 8, 4), (5, 5, 5)),  # bw<ker,bh>=ker : T+partial-H sparsity
        ((15, 16, 8), (5, 8, 4), (5, 5, 5)),  # 3 T-blocks -> exercises interior + both edges on T
        ((10, 32, 16), (5, 16, 8), (11, 11, 11)),  # bigger box, real-ish kernel
        ((20, 34, 12), (5, 17, 6), (11, 11, 11)),  # 6s-like aspect
    ],
    ids=["10x16x8", "15x16x8", "10x32x16", "20x34x12"],
)
@pytest.mark.parametrize("w_origin", [0, 5], ids=["worig0", "worig5"])
def test_qtile_band_covers_all_live(grid, block, kernel, w_origin):
    bt, bh, bw = block
    T, H, W = grid
    hb, wb = H // bh, W // bw
    n_chunks = (T // bt) * hb * wb
    vol = bt * bh * bw
    n_qtiles = (vol + TILE - 1) // TILE

    total_tiles = kept_tiles = 0
    for qc in range(n_chunks):
        box = neighborhood_box_block(qc, block, grid, kernel, hb, wb, w_origin)
        for qi in range(n_qtiles):
            wid0 = qi * TILE
            lo, hi = qtile_k_band(qc, wid0, block, grid, kernel, hb, wb, w_origin, box=box)
            live, n_ktiles = _live_ktiles_bruteforce(qc, wid0, block, grid, kernel, hb, wb, w_origin, box)
            # CORRECTNESS: every genuinely-live tile is inside the kept band (no lossy skip).
            missed = {kt for kt in live if not (lo <= kt < hi)}
            assert not missed, f"chunk {qc} qtile {qi}: band [{lo},{hi}) skips LIVE tiles {sorted(missed)}"
            assert 0 <= lo <= hi <= n_ktiles, f"band [{lo},{hi}) out of range [0,{n_ktiles}]"
            total_tiles += n_ktiles
            kept_tiles += hi - lo
    skip = 1 - kept_tiles / total_tiles
    print(
        f"\n  grid={grid} block={block} k={kernel} worig={w_origin}: "
        f"band skip = {skip*100:.1f}%  ({total_tiles-kept_tiles}/{total_tiles} tiles)"
    )
    assert skip >= 0.0  # informational; correctness is the assert above
