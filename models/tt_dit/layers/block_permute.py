# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-side 3-D block token permutation for the DiffVAE fused-sdpa (Workstream: block-permute, v1).

Phase 1 -- the reorder machinery plus the coordinate map it shares with the (Phase-2) kernel, verified
in isolation before the kernel learns the block layout.

The stage-5 attention flattens the (T, H, W) volume into a 1-D token sequence. The current STRIDED
flatten (t outer, w inner, or the t_inner variant) makes a contiguous query chunk a *strip* along one
axis, so the fused-sdpa neighborhood box stretches with that axis -> super-linear at long T. Reordering
tokens into (bt, bh, bw) BLOCKS instead makes a contiguous chunk a compact 3-D cube, so its window union
-- the box the kernel gathers -- stays small and grid-size-independent. See ``box_model.py`` and the 6s
decode artifact for the sizing.

v1 permutes only the QUERY order: self-attention is per-query, so this reorders the output rows and
leaves K/V (and RoPE, which commutes with a per-position permute) untouched. The kernel's Q-coordinate
decode (Phase 2) mirrors :func:`token_to_block_coords` exactly.
"""
from __future__ import annotations

import math

import torch


def padded_grid(grid: tuple[int, int, int], block: tuple[int, int, int]):
    """The grid rounded up so each axis is a whole number of blocks, plus the per-axis block counts."""
    (t, h, w), (bt, bh, bw) = grid, block
    tp, hp, wp = math.ceil(t / bt) * bt, math.ceil(h / bh) * bh, math.ceil(w / bw) * bw
    return (tp, hp, wp), (tp // bt, hp // bh, wp // bw)


def block_to_token_index(t: int, h: int, w: int, block: tuple[int, int, int], counts: tuple[int, int, int]) -> int:
    """Physical ``(t, h, w)`` on the PADDED grid -> its position in the block-order sequence.

    Block-major, within-block-minor: ``[(block) | (within)]``. The Phase-2 kernel inverts this to
    recover each query's physical coord from its block-order index."""
    bt, bh, bw = block
    _, hb, wb = counts[0], counts[1], counts[2]
    block_id = (t // bt * hb + h // bh) * wb + w // bw
    within = (t % bt * bh + h % bh) * bw + w % bw
    return block_id * (bt * bh * bw) + within


def token_to_block_coords(idx: int, block: tuple[int, int, int], counts: tuple[int, int, int]) -> tuple[int, int, int]:
    """Inverse of :func:`block_to_token_index`: block-order index -> physical ``(t, h, w)`` on the padded grid.

    This is the exact math the kernel runs in ``neighborhood_box`` / ``windowed_mask_gen`` to place each
    query's neighborhood, so it is the spec for the C++ port."""
    bt, bh, bw = block
    _, hb, wb = counts
    vol = bt * bh * bw
    block_id, within = divmod(idx, vol)
    bw_i = block_id % wb
    bh_i = (block_id // wb) % hb
    bt_i = block_id // (wb * hb)
    dw = within % bw
    dh = (within // bw) % bh
    dt = within // (bw * bh)
    return bt_i * bt + dt, bh_i * bh + dh, bw_i * bw + dw


def _perm_axes(nlead: int, order: tuple[int, ...]) -> list[int]:
    """Leading dims kept in place; the six split spatial dims reordered by ``order``; C last."""
    return list(range(nlead)) + [nlead + o for o in order] + [nlead + 6]


def to_block_order(x: torch.Tensor, grid: tuple[int, int, int], block: tuple[int, int, int]) -> torch.Tensor:
    """``(*lead, S, C)`` strided (``S = T*H*W``, t outer / w inner) -> ``(*lead, Sp, C)`` block order (Sp padded).

    Pure reshape + pad + permute, so it maps 1:1 onto ttnn ops for the device path."""
    *lead, s, c = x.shape
    t, h, w = grid
    assert s == t * h * w, f"S={s} != T*H*W={t * h * w}"
    (tp, hp, wp), (tb, hb, wb) = padded_grid(grid, block)
    bt, bh, bw = block
    x = x.reshape(*lead, t, h, w, c)
    x = torch.nn.functional.pad(x, (0, 0, 0, wp - w, 0, hp - h, 0, tp - t))  # pad W, H, T (C untouched)
    x = x.reshape(*lead, tb, bt, hb, bh, wb, bw, c)
    x = x.permute(*_perm_axes(len(lead), (0, 2, 4, 1, 3, 5))).contiguous()  # Tb,Hb,Wb, bt,bh,bw
    return x.reshape(*lead, tp * hp * wp, c)


def from_block_order(x: torch.Tensor, grid: tuple[int, int, int], block: tuple[int, int, int]) -> torch.Tensor:
    """Inverse of :func:`to_block_order`: block order -> strided, cropping the padding back off."""
    *lead, sp, c = x.shape
    t, h, w = grid
    (tp, hp, wp), (tb, hb, wb) = padded_grid(grid, block)
    bt, bh, bw = block
    x = x.reshape(*lead, tb, hb, wb, bt, bh, bw, c)
    x = x.permute(*_perm_axes(len(lead), (0, 3, 1, 4, 2, 5))).contiguous()  # Tb,bt,Hb,bh,Wb,bw
    x = x.reshape(*lead, tp, hp, wp, c)
    x = x[..., :t, :h, :w, :]
    return x.reshape(*lead, t * h * w, c)


# ---------------------------------------------------------------------------
# Device (ttnn) reorder -- the same math as the torch reference, in ttnn ops. Q is (B, NH, S, HD);
# B and NH are folded to keep the spatial permute at rank 8. RM throughout (a reorder is a stride
# change), TILE at the ends. Used by stage 5; validated bit-for-bit against the torch twin.
# ---------------------------------------------------------------------------
def to_block_order_tt(x, grid: tuple[int, int, int], block: tuple[int, int, int]):
    """``(B, NH, S, HD)`` strided -> ``(B, NH, Sp, HD)`` block order (Sp padded). Consumes nothing."""
    import ttnn

    b, nh, s, c = tuple(x.shape)
    t, h, w = grid
    (tp, hp, wp), (tb, hb, wb) = padded_grid(grid, block)
    bt, bh, bw = block
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (b * nh, t, h, w, c))

    # Pad each spatial axis up to a whole number of blocks. ttnn.pad only reaches the lowest 3 dims of a
    # rank-5 tensor, so append zero slabs with concat instead (works on any dim, order T then H then W).
    def _append_zeros(x, dim, extent, shape):
        z = ttnn.zeros(shape, dtype=x.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=x.device())
        return ttnn.concat([x, z], dim=dim)

    if tp > t:
        x = _append_zeros(x, 1, tp - t, (b * nh, tp - t, h, w, c))
    if hp > h:
        x = _append_zeros(x, 2, hp - h, (b * nh, tp, hp - h, w, c))
    if wp > w:
        x = _append_zeros(x, 3, wp - w, (b * nh, tp, hp, wp - w, c))
    x = ttnn.reshape(x, (b * nh, tb, bt, hb, bh, wb, bw, c))
    x = ttnn.permute(x, (0, 1, 3, 5, 2, 4, 6, 7))  # BNH, Tb,Hb,Wb, bt,bh,bw, C
    x = ttnn.reshape(x, (b, nh, tp * hp * wp, c))
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


def from_block_order_tt(x, grid: tuple[int, int, int], block: tuple[int, int, int]):
    """Inverse of :func:`to_block_order_tt`: ``(B, NH, Sp, HD)`` block order -> ``(B, NH, S, HD)`` strided."""
    import ttnn

    b, nh, sp, c = tuple(x.shape)
    t, h, w = grid
    (tp, hp, wp), (tb, hb, wb) = padded_grid(grid, block)
    bt, bh, bw = block
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (b * nh, tb, hb, wb, bt, bh, bw, c))
    x = ttnn.permute(x, (0, 1, 4, 2, 5, 3, 6, 7))  # BNH, Tb,bt,Hb,bh,Wb,bw, C
    x = ttnn.reshape(x, (b * nh, tp, hp, wp, c))
    if (tp, hp, wp) != (t, h, w):
        x = ttnn.slice(x, [0, 0, 0, 0, 0], [b * nh, t, h, w, c])
    x = ttnn.reshape(x, (b, nh, t * h * w, c))
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


# ---------------------------------------------------------------------------
# Box-sparse tile-skip (Phase 1): per-q-tile packed k-tile band.
#
# The fused gather packs a block chunk's neighborhood BOX densely (T-outer/H-mid/W-inner) and the flash
# runs dense over it, masking the over-included cells. For a block whose H,W are >= the kernel, most of
# that box is dead for any given 32-query sub-tile -> those k-tiles are entirely -inf (exp(-inf)=0) and
# can be skipped LOSSLESSLY. This computes, per q-sub-tile, the contiguous packed k-tile band [lo, hi)
# that can hold a live key; everything outside is skippable. Host twin of block_qtile_k_band in
# windowed_loop_geometry.hpp -- the kernel runs the identical arithmetic. See tile_skip_phase0.py.
# ---------------------------------------------------------------------------
TILE = 32


def nbr_shift_start(q: int, length: int, ker: int) -> int:
    """Edge-clamped neighborhood window start: the window is [start, start+min(ker,length)). Mirrors
    ``nbr_shift_start`` in windowed_loop_geometry.hpp exactly."""
    if ker > length:
        ker = length
    half = ker // 2
    last = length - ker
    start = 0 if q < half else q - half
    return min(start, last)


def neighborhood_box_block(qc, block, grid, kernel, hb, wb, w_origin):
    """Host twin of ``neighborhood_box_block``: the box (t0,t1,h_lo,h_hi,w_lo,w_hi) for block chunk ``qc``.
    Coords are op-order (T = op-outer, carrying ``w_origin``; H; W). The block is dilated by the kernel,
    edge-clamped by :func:`nbr_shift_start`."""
    bt, bh, bw = block
    T, H, W = grid
    kt, kh, kw = kernel
    bti = qc // (wb * hb)
    bhi = (qc // wb) % hb
    bwi = qc % wb
    t_lo, t_hi = w_origin + bti * bt, w_origin + bti * bt + bt - 1
    h_lo, h_hi = bhi * bh, bhi * bh + bh - 1
    w_lo, w_hi = bwi * bw, bwi * bw + bw - 1
    ker_t, ker_h, ker_w = min(kt, T), min(kh, H), min(kw, W)
    return (
        nbr_shift_start(t_lo, T, kt),
        nbr_shift_start(t_hi, T, kt) + ker_t,
        nbr_shift_start(h_lo, H, kh),
        nbr_shift_start(h_hi, H, kh) + ker_h,
        nbr_shift_start(w_lo, W, kw),
        nbr_shift_start(w_hi, W, kw) + ker_w,
    )


def qtile_k_band(qc, wid0, block, grid, kernel, hb, wb, w_origin, box=None):
    """Packed k-tile band [lo, hi) that can hold a live key for the ``TILE``-query sub-tile starting at
    within-block index ``wid0`` of block chunk ``qc``. Tiles outside [lo, hi) are entirely -inf for every
    query in the sub-tile -> losslessly skippable. Host twin of block_qtile_k_band."""
    if box is None:
        box = neighborhood_box_block(qc, block, grid, kernel, hb, wb, w_origin)
    bt, bh, bw = block
    T, H, W = grid
    kt, kh, kw = kernel
    t0, _, h_lo, _, w_lo, _ = box
    bh_box, bw_box = box[3] - box[2], box[5] - box[4]
    hw_box = bh_box * bw_box
    vol = bt * bh * bw
    ker_t, ker_h, ker_w = min(kt, T), min(kh, H), min(kw, W)
    bti = qc // (wb * hb)
    bhi = (qc // wb) % hb
    bwi = qc % wb
    jmin, jmax = None, None
    for wid in range(wid0, min(wid0 + TILE, vol)):
        dw = wid % bw
        dh = (wid // bw) % bh
        dt = wid // (bw * bh)
        wt = nbr_shift_start(w_origin + bti * bt + dt, T, kt) - t0
        wh = nbr_shift_start(bhi * bh + dh, H, kh) - h_lo
        ww = nbr_shift_start(bwi * bw + dw, W, kw) - w_lo
        j_min_q = wt * hw_box + wh * bw_box + ww
        j_max_q = (wt + ker_t - 1) * hw_box + (wh + ker_h - 1) * bw_box + (ww + ker_w - 1)
        jmin = j_min_q if jmin is None else min(jmin, j_min_q)
        jmax = j_max_q if jmax is None else max(jmax, j_max_q)
    return jmin // TILE, jmax // TILE + 1
