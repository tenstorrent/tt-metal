# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""GNA halo-gather geometry: per-block BOX gather indices + the fixed [vol, box] neighborhood window mask.

The accurate+fast GNA path = each query-block attends to its BOX (block dilated by the kernel = block + halo)
through a plain DENSE attention, with a mask restricting each query to its exact +-k//2 window inside the box.
This reproduces EXACT neighborhood attention (interior blocks) while running on the efficient dense path, unlike
the block-diagonal "blocked attention" (M1/M2) which drops the halo.

Layout contract (matches na3d t_inner / W-outer K/V order): a token at grid (t,h,w) sits at flat index
  fidx(t,h,w) = ((w*H + h)*T + t)                         [T innermost, W outermost]
The box for block (bti,bhi,bwi) is enumerated T-outer/H-mid/W-inner into [0, box_vol); an out-of-grid box cell
(edge blocks) maps to a sentinel and is masked. Queries in a block are enumerated dw-innermost (block_permute
within-block order) into [0, vol).

Host-only (numpy/torch); the ttnn path uses `box_gather_indices` for ttnn.embedding and `window_mask` as the
SDPA attn_mask. Validated against exact neighborhood in test (this file's __main__ / a unit test)."""
from __future__ import annotations

import torch

SENTINEL = -1  # out-of-grid box cell -> gather clamps to 0 and the mask -inf's it


def _nbr_start(q, L, ker):
    ker = min(ker, L)
    half = ker // 2
    return min(max(q - half, 0), L - ker)


def box_dims(block, kernel, grid):
    (bt, bh, bw), (kt, kh, kw), (T, H, W) = block, kernel, grid
    return (bt + min(kt, T) - 1, bh + min(kh, H) - 1, bw + min(kw, W) - 1)


def box_gather_indices(grid, block, kernel):
    """[num_blocks, box_vol] int64: flat (t_inner) sequence index of each box cell for each block; SENTINEL if
    the cell falls outside the grid (edge blocks). Box origin = block origin - (k-1)//2, clamped to the grid so
    the box always covers every in-block query's window."""
    T, H, W = grid
    bt, bh, bw = block
    Tb, Hb, Wb = T // bt, H // bh, W // bw
    ext_t, ext_h, ext_w = box_dims(block, kernel, grid)
    kt, kh, kw = kernel
    half_t, half_h, half_w = min(kt, T) // 2, min(kh, H) // 2, min(kw, W) // 2
    nb = Tb * Hb * Wb
    idx = torch.full((nb, ext_t * ext_h * ext_w), SENTINEL, dtype=torch.int64)
    for b in range(nb):
        bti, bhi, bwi = b // (Hb * Wb), (b // Wb) % Hb, b % Wb
        # box origin: block origin shifted by -half, clamped so [origin, origin+ext) stays in grid
        t0 = min(max(bti * bt - half_t, 0), T - ext_t)
        h0 = min(max(bhi * bh - half_h, 0), H - ext_h)
        w0 = min(max(bwi * bw - half_w, 0), W - ext_w)
        c = 0
        for jt in range(ext_t):
            for jh in range(ext_h):
                for jw in range(ext_w):
                    t, h, w = t0 + jt, h0 + jh, w0 + jw
                    idx[b, c] = ((w * H + h) * T + t) if (0 <= t < T and 0 <= h < H and 0 <= w < W) else SENTINEL
                    c += 1
    return idx, (t0 if nb == 1 else None)  # (origins vary; kept per-block internally)


def window_mask(grid, block, kernel):
    """[num_blocks, vol, box_vol] bool: query (in block) attends box cell iff within its +-k//2 window. Depends
    on the block's clamped box origin, so it's per-block (edge blocks differ). vol = bt*bh*bw."""
    T, H, W = grid
    bt, bh, bw = block
    kt, kh, kw = kernel
    Tb, Hb, Wb = T // bt, H // bh, W // bw
    ext_t, ext_h, ext_w = box_dims(block, kernel, grid)
    half_t, half_h, half_w = min(kt, T) // 2, min(kh, H) // 2, min(kw, W) // 2
    vol = bt * bh * bw
    nb = Tb * Hb * Wb
    m = torch.zeros(nb, vol, ext_t * ext_h * ext_w, dtype=torch.bool)
    for b in range(nb):
        bti, bhi, bwi = b // (Hb * Wb), (b // Wb) % Hb, b % Wb
        t0 = min(max(bti * bt - half_t, 0), T - ext_t)
        h0 = min(max(bhi * bh - half_h, 0), H - ext_h)
        w0 = min(max(bwi * bw - half_w, 0), W - ext_w)
        for wid in range(vol):
            dt, dh, dw = wid // (bh * bw), (wid // bw) % bh, wid % bw
            qt, qh, qw = bti * bt + dt, bhi * bh + dh, bwi * bw + dw
            wt, wh, ww = _nbr_start(qt, T, kt), _nbr_start(qh, H, kh), _nbr_start(qw, W, kw)
            c = 0
            for jt in range(ext_t):
                for jh in range(ext_h):
                    for jw in range(ext_w):
                        t, h, w = t0 + jt, h0 + jh, w0 + jw
                        m[b, wid, c] = (
                            (wt <= t < wt + min(kt, T)) and (wh <= h < wh + min(kh, H)) and (ww <= w < ww + min(kw, W))
                        )
                        c += 1
    return m
