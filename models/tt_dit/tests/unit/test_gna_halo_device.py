# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device validation of the GNA halo path: ttnn.embedding-gather the box K/V per block + batched dense SDPA
with the [vol, box] window mask == EXACT neighborhood attention, on the efficient plain path. Small grid (no
batching needed). Proves the ttnn mechanics before wiring into na3d / the decode."""
import pytest
import torch

import ttnn
from models.tt_dit.layers.gna_gather import box_dims, box_gather_indices, window_mask


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _fidx(t, h, w, T, H):
    return (w * H + h) * T + t


def _exact(q, k, v, grid, kernel, d):
    T, H, W = grid
    kt, kh, kw = kernel

    def nbr(qc, L, ker):
        ker = min(ker, L)
        return min(max(qc - ker // 2, 0), L - ker)

    out = torch.zeros(T * H * W, d)
    for t in range(T):
        for h in range(H):
            for w in range(W):
                t0, h0, w0 = nbr(t, T, kt), nbr(h, H, kh), nbr(w, W, kw)
                keys = torch.tensor(
                    [
                        _fidx(a, b, c, T, H)
                        for a in range(t0, min(t0 + kt, T))
                        for b in range(h0, min(h0 + kh, H))
                        for c in range(w0, min(w0 + kw, W))
                    ]
                )
                qi = _fidx(t, h, w, T, H)
                out[qi] = torch.softmax((q[qi] @ k[keys].T) / d**0.5, 0) @ v[keys]
    return out


def _gna_device(device, q_t, k_t, v_t, grid, kernel, block, heads, d):
    """q_t/k_t/v_t: torch [S, heads*d] in t_inner order. Returns device output scattered to [S, heads*d]."""
    T, H, W = grid
    bt, bh, bw = block
    Tb, Hb, Wb = T // bt, H // bh, W // bw
    vol, nb = bt * bh * bw, Tb * Hb * Wb
    box = box_dims(block, kernel, grid)
    box_vol = box[0] * box[1] * box[2]
    idx, _ = box_gather_indices(grid, block, kernel)  # [nb, box_vol]
    wm = window_mask(grid, block, kernel)  # [nb, vol, box_vol] bool

    # embedding-gather box K/V: weight [S, heads*d], ids [nb*box_vol] -> [nb*box_vol, heads*d]
    ids = ttnn.from_torch(
        idx.clamp(min=0).reshape(-1).to(torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    kw_ = ttnn.from_torch(k_t, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    vw_ = ttnn.from_torch(v_t, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    kb = ttnn.embedding(ids, kw_)  # [nb*box_vol, heads*d]
    vb = ttnn.embedding(ids, vw_)

    def to_bhsd(x):  # [nb*box_vol, heads*d] -> [nb, heads, box_vol, d]
        x = ttnn.reshape(x, (nb, box_vol, heads, d))
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    kb, vb = to_bhsd(kb), to_bhsd(vb)

    # Q is block-order: gather its rows too (same embedding trick) so this test stays self-contained.
    qids_t = torch.tensor(
        [
            [
                _fidx(
                    (b // (Hb * Wb)) * bt + wid // (bh * bw),
                    ((b // Wb) % Hb) * bh + (wid // bw) % bh,
                    (b % Wb) * bw + wid % bw,
                    T,
                    H,
                )
                for wid in range(vol)
            ]
            for b in range(nb)
        ]
    )  # [nb, vol]
    qw_ = ttnn.from_torch(q_t, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    qids = ttnn.from_torch(
        qids_t.reshape(-1).to(torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    qb = ttnn.embedding(qw_, ids=qids) if False else ttnn.embedding(qids, qw_)
    qb = ttnn.reshape(qb, (nb, vol, heads, d))
    qb = ttnn.to_layout(ttnn.permute(qb, (0, 2, 1, 3)), ttnn.TILE_LAYOUT)  # [nb, heads, vol, d]

    # additive mask [nb, 1, vol, box_vol]: 0 in-window, -inf out
    add = torch.where(wm, 0.0, float("-inf")).unsqueeze(1)  # [nb,1,vol,box_vol]
    mask = ttnn.from_torch(add.to(torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    g = device.compute_with_storage_grid_size()
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), exp_approx_mode=False, q_chunk_size=vol, k_chunk_size=256
    )
    out = ttnn.transformer.scaled_dot_product_attention(
        qb, kb, vb, attn_mask=mask, is_causal=False, scale=d**-0.5, program_config=pc
    )  # [nb, heads, vol, d]
    out = ttnn.to_torch(out).float()  # [nb, heads, vol, d]
    # scatter block-order -> [S, heads, d]
    res = torch.zeros(T * H * W, heads, d)
    for b in range(nb):
        res[qids_t[b]] = out[b].permute(1, 0, 2)  # [vol, heads, d]
    return res.reshape(T * H * W, heads * d)


@pytest.mark.parametrize(
    "grid,kernel,block",
    [((10, 16, 8), (5, 5, 5), (5, 8, 4)), ((15, 16, 8), (5, 5, 5), (5, 8, 4))],
    ids=["10x16x8", "15x16x8"],
)
def test_gna_halo_device(*, device, grid, kernel, block):
    torch.manual_seed(0)
    heads, d = 2, 64
    S = grid[0] * grid[1] * grid[2]
    q = torch.randn(S, heads * d)
    k = torch.randn(S, heads * d)
    v = torch.randn(S, heads * d)
    dev = _gna_device(device, q, k, v, grid, kernel, block, heads, d)
    # exact per head, then compare
    pccs = []
    for hh in range(heads):
        ex = _exact(
            q[:, hh * d : (hh + 1) * d], k[:, hh * d : (hh + 1) * d], v[:, hh * d : (hh + 1) * d], grid, kernel, d
        )
        pccs.append(_pcc(dev[:, hh * d : (hh + 1) * d], ex))
    print(
        f"\n  grid={grid} block={block} box={box_dims(block, kernel, grid)}: per-head PCC={[f'{p*100:.3f}' for p in pccs]}"
    )
    assert min(pccs) > 0.99, f"GNA-device != exact (PCC {min(pccs)})"
