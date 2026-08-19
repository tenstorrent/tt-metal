# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""GNA halo-gather geometry: box-gather + window-mask + dense attention must reproduce EXACT 3-D neighborhood
attention (interior AND edge blocks). This is the accuracy foundation for the fast GNA path (plain dense flash
over the gathered box, unlike block-diagonal "blocked attention" which drops the halo). Host/torch only."""
import pytest
import torch

from models.tt_dit.layers.gna_gather import box_dims, box_gather_indices, window_mask


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _fidx(t, h, w, T, H):
    return (w * H + h) * T + t  # t_inner / W-outer flatten (matches na3d K/V order)


def _exact(q, k, v, grid, kernel, d):
    T, H, W = grid
    kt, kh, kw = kernel

    def nbr(qc, L, ker):
        ker = min(ker, L)
        s = min(max(qc - ker // 2, 0), L - ker)
        return s, s + ker

    out = torch.zeros(T * H * W, d)
    for t in range(T):
        for h in range(H):
            for w in range(W):
                t0, t1 = nbr(t, T, kt)
                h0, h1 = nbr(h, H, kh)
                w0, w1 = nbr(w, W, kw)
                keys = torch.tensor(
                    [_fidx(a, b, c, T, H) for a in range(t0, t1) for b in range(h0, h1) for c in range(w0, w1)]
                )
                qi = _fidx(t, h, w, T, H)
                out[qi] = torch.softmax((q[qi] @ k[keys].T) / d**0.5, 0) @ v[keys]
    return out


def _gna(q, k, v, grid, kernel, block, d):
    T, H, W = grid
    bt, bh, bw = block
    Tb, Hb, Wb = T // bt, H // bh, W // bw
    vol, nb = bt * bh * bw, Tb * Hb * Wb
    idx, _ = box_gather_indices(grid, block, kernel)
    m = window_mask(grid, block, kernel)
    out = torch.zeros(T * H * W, d)
    for b in range(nb):
        bti, bhi, bwi = b // (Hb * Wb), (b // Wb) % Hb, b % Wb
        kb, vb = k[idx[b].clamp(min=0)], v[idx[b].clamp(min=0)]
        qids = torch.tensor(
            [
                _fidx(bti * bt + wid // (bh * bw), bhi * bh + (wid // bw) % bh, bwi * bw + wid % bw, T, H)
                for wid in range(vol)
            ]
        )
        s = ((q[qids] @ kb.T) / d**0.5).masked_fill(~m[b], float("-inf"))
        out[qids] = torch.softmax(s, 1) @ vb
    return out


@pytest.mark.parametrize(
    "grid,kernel,block",
    [
        ((10, 16, 8), (5, 5, 5), (5, 8, 4)),
        ((15, 16, 8), (5, 5, 5), (5, 8, 4)),  # T edges
        ((10, 16, 12), (5, 5, 5), (5, 8, 4)),  # W edges
        ((10, 16, 8), (3, 3, 3), (5, 8, 4)),
    ],
    ids=["10x16x8", "15x16x8", "10x16x12", "k3"],
)
def test_gna_halo_matches_exact(grid, kernel, block):
    torch.manual_seed(0)
    d = 32
    q, k, v = (torch.randn(grid[0] * grid[1] * grid[2], d) for _ in range(3))
    pcc = _pcc(_gna(q, k, v, grid, kernel, block, d), _exact(q, k, v, grid, kernel, d))
    print(f"\n  grid={grid} k={kernel} block={block} box={box_dims(block, kernel, grid)}: PCC={pcc * 100:.4f}%")
    assert pcc > 0.9999, f"GNA halo-gather != exact neighborhood (PCC {pcc})"
