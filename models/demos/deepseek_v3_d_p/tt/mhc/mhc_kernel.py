# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host wrapper for the fused mHC parametrization kernel
(ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn).

The kernel is pure tile ops; the scalars a and biases b, and the Sinkhorn row/col-sum
selection matrices, are baked host-side into a single [8,32,32] `consts` tensor. Tile order:
    0 SEL_pre   1 SEL_post   2 SEL_comb   3 base_pre   4 base_post   5 base_comb   6 RB   7 CB

SEL_* extract + left-align each group from the 24-wide mixes and fold in the scalar a, so
`mixes @ SEL_g` gives `a_g * raw_g` left-aligned. RB/CB turn the Sinkhorn row/col sums into
same-shape `m @ K` divides (see tt/mhc_ttnn.py for the derivation).
"""

from __future__ import annotations

import torch

import ttnn

W = 32


def _selection_row_col(n):
    RB = torch.zeros(W, W, dtype=torch.float32)
    CB = torch.zeros(W, W, dtype=torch.float32)
    for p in range(n * n):
        pi, pj = divmod(p, n)
        for q in range(n * n):
            qi, qj = divmod(q, n)
            if qi == pi:
                RB[q, p] = 1.0
            if qj == pj:
                CB[q, p] = 1.0
    for p in range(n * n, W):
        RB[p, p] = 1.0
        CB[p, p] = 1.0
    return RB, CB


def build_consts(cfg, scale, base) -> torch.Tensor:
    """-> [8, 32, 32] fp32 constant tiles for mhc_split_sinkhorn."""
    n = cfg.n
    a_pre, a_post, a_res = (float(scale[0]), float(scale[1]), float(scale[2]))
    base = base.float()

    sel_pre = torch.zeros(W, W)
    sel_post = torch.zeros(W, W)
    sel_comb = torch.zeros(W, W)
    for p in range(n):
        sel_pre[p, p] = a_pre
        sel_post[n + p, p] = a_post
    for p in range(n * n):
        sel_comb[2 * n + p, p] = a_res

    # base tiles: per-column bias replicated down every token row (add_tiles is tile+tile)
    base_pre = torch.zeros(W, W)
    base_post = torch.zeros(W, W)
    base_comb = torch.zeros(W, W)
    base_pre[:, :n] = base[0:n]
    base_post[:, :n] = base[n : 2 * n]
    base_comb[:, : n * n] = base[2 * n :]

    RB, CB = _selection_row_col(n)
    return torch.stack([sel_pre, sel_post, sel_comb, base_pre, base_post, base_comb, RB, CB], dim=0)


def mhc_split_sinkhorn(device, mixes_torch, scale, base, cfg):
    """mixes_torch: [T, (2+n)*n] -> (pre [T,n], post [T,n], comb [T,n,n]) torch tensors."""
    n, T = cfg.n, mixes_torch.shape[0]
    mixes_tt = ttnn.from_torch(mixes_torch.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    consts_tt = ttnn.from_torch(
        build_consts(cfg, scale, base), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32
    )

    pre_tt, post_tt, comb_tt = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
        mixes_tt, consts_tt, n, int(cfg.sinkhorn_iters), float(cfg.eps)
    )
    pre = ttnn.to_torch(pre_tt)
    post = ttnn.to_torch(post_tt)
    comb = ttnn.to_torch(comb_tt).reshape(T, n, n)
    return pre, post, comb
