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


def box_gather_indices_unclamped(grid, block, kernel):
    """[num_blocks, box_vol] int64: like box_gather_indices but the box origin is NOT clamped (= block origin
    - k//2), so every block uses the SAME box-relative layout and out-of-grid cells are SENTINEL. This lets the
    device build the per-batch mask cheaply as fixed_window[vol,box] AND valid[nb,box] (valid = idx != SENTINEL),
    at the cost of edge queries using the drop-out-of-grid window instead of the shifted one (accuracy check in
    test)."""
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
        t0, h0, w0 = bti * bt - half_t, bhi * bh - half_h, bwi * bw - half_w  # UNclamped
        c = 0
        for jt in range(ext_t):
            for jh in range(ext_h):
                for jw in range(ext_w):
                    t, h, w = t0 + jt, h0 + jh, w0 + jw
                    idx[b, c] = ((w * H + h) * T + t) if (0 <= t < T and 0 <= h < H and 0 <= w < W) else SENTINEL
                    c += 1
    return idx


def fixed_window_mask(block, kernel):
    """[vol, box_vol] bool, uniform across blocks (unclamped box): query wid attends box cell iff the cell's
    box-relative coord is in wid's +-k//2 window. Box origin = block origin - k//2, so query wid=(dt,dh,dw) has
    its window at box-relative [dt, dt+kt) x [dh, dh+kh) x [dw, dw+kw). Combine with per-block valid (not-sentinel)
    to get the full mask."""
    bt, bh, bw = block
    kt, kh, kw = kernel
    ext_t, ext_h, ext_w = bt + kt - 1, bh + kh - 1, bw + kw - 1
    vol = bt * bh * bw
    m = torch.zeros(vol, ext_t * ext_h * ext_w, dtype=torch.bool)
    for wid in range(vol):
        dt, dh, dw = wid // (bh * bw), (wid // bw) % bh, wid % bw
        c = 0
        for jt in range(ext_t):
            for jh in range(ext_h):
                for jw in range(ext_w):
                    m[wid, c] = (dt <= jt < dt + kt) and (dh <= jh < dh + kh) and (dw <= jw < dw + kw)
                    c += 1
    return m


def _clamp_origin(bi, b, half, ext, L):
    return min(max(bi * b - half, 0), L - ext)


def mask_table(grid, block, kernel):
    """Deduped clamped masks for the GNA halo path. The per-block window mask depends only on each axis's
    CLAMP CLASS (low-edge / interior / high-edge), so there are few distinct [vol, box] masks (interior + edge
    combinations, typically 27). Returns:
      box_idx   [num_blocks, box_vol] int64 clamped gather indices (in-grid, no sentinels; for ttnn.embedding)
      table     [n_distinct, vol, box_vol] bool window masks
      mask_id   [num_blocks] int -> row of `table` for each block
    box_idx + table[mask_id] == the full per-block (box_gather_indices, window_mask), so it reproduces exact
    neighborhood attention (validated) while staying memory-bounded (table is tiny; box_idx is num_blocks x box)."""
    T, H, W = grid
    bt, bh, bw = block
    kt, kh, kw = kernel
    Tb, Hb, Wb = T // bt, H // bh, W // bw
    ext_t, ext_h, ext_w = box_dims(block, kernel, grid)
    ht, hh, hw = min(kt, T) // 2, min(kh, H) // 2, min(kw, W) // 2
    ker_t, ker_h, ker_w = min(kt, T), min(kh, H), min(kw, W)
    vol, box_vol = bt * bh * bw, ext_t * ext_h * ext_w
    nb = Tb * Hb * Wb

    def clampL(bi, b, half, ext, L):  # vectorized clamped box origin per block-axis-index
        return (bi * b - half).clamp(0, L - ext)

    # per-block axis indices + clamped origins (vectorized)
    bidx = torch.arange(nb)
    bti, bhi, bwi = bidx // (Hb * Wb), (bidx // Wb) % Hb, bidx % Wb
    t0 = clampL(bti, bt, ht, ext_t, T)
    h0 = clampL(bhi, bh, hh, ext_h, H)
    w0 = clampL(bwi, bw, hw, ext_w, W)
    # box cell offsets (T-outer/H-mid/W-inner), flat [box_vol]
    jt = torch.arange(ext_t).view(-1, 1, 1).expand(ext_t, ext_h, ext_w).reshape(-1)
    jh = torch.arange(ext_h).view(1, -1, 1).expand(ext_t, ext_h, ext_w).reshape(-1)
    jw = torch.arange(ext_w).view(1, 1, -1).expand(ext_t, ext_h, ext_w).reshape(-1)
    box_idx = ((w0[:, None] + jw[None, :]) * H + (h0[:, None] + jh[None, :])) * T + (
        t0[:, None] + jt[None, :]
    )  # [nb, box_vol]

    # class per block = (clamp offset per axis); dedup -> mask_id + representative block per class
    ot = t0 - (bti * bt - ht)
    oh = h0 - (bhi * bh - hh)
    ow = w0 - (bwi * bw - hw)
    keys = ot * 1_000_000 + oh * 1000 + ow  # unique key per (ot,oh,ow) combo
    uniq, mask_id = torch.unique(keys, return_inverse=True)  # [n_distinct], [nb]
    reps = torch.stack([(keys == u).nonzero()[0, 0] for u in uniq])  # one rep block per class
    # build table [n_distinct, vol, box_vol] from reps (vectorized over vol x box_vol)
    dt = torch.arange(vol) // (bh * bw)
    dh = (torch.arange(vol) // bw) % bh
    dw = torch.arange(vol) % bw
    rti, rhi, rwi = reps // (Hb * Wb), (reps // Wb) % Hb, reps % Wb
    rt0, rh0, rw0 = clampL(rti, bt, ht, ext_t, T), clampL(rhi, bh, hh, ext_h, H), clampL(rwi, bw, hw, ext_w, W)

    def nbr_start_vec(q, L, ker):
        ker = min(ker, L)
        return (q - ker // 2).clamp(0, L - ker)

    def axis_mask(rbi, rb, r0, d, L, ker, ext, jc):  # [n_distinct, vol, box_vol] bool for one axis
        w = nbr_start_vec(rbi[:, None] * rb + d[None, :], L, ker)  # [n_distinct, vol]
        cell = r0[:, None, None] + jc[None, None, :]  # [n_distinct,1,box_vol]
        return (w[:, :, None] <= cell) & (cell < w[:, :, None] + min(ker, L))

    table = (
        axis_mask(rti, bt, rt0, dt, T, kt, ext_t, jt)
        & axis_mask(rhi, bh, rh0, dh, H, kh, ext_h, jh)
        & axis_mask(rwi, bw, rw0, dw, W, kw, ext_w, jw)
    )  # [n_distinct, vol, box_vol]
    return box_idx, table, mask_id


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
