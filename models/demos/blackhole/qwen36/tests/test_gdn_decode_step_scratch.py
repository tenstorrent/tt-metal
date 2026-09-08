# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: ttnn.experimental.kda.gdn_decode_step vs a torch reference of the GDN decode step (single device).

  pytest models/demos/blackhole/qwen36/tests/test_gdn_decode_step_scratch.py -s
"""
import pytest
import torch

import ttnn

Nv, Nk, Dk, Dv = 12, 4, 128, 128
KD, VD = Nk * Dk, Nv * Dv


def _reference(q, k, v, beta, g, h, w, scale, l2_eps=1e-6, norm_eps=1e-6):
    """q,k: [Nk,Dk]; v: [Nv,Dv]; beta,g: [Nv]; h: [Nv,Dk,Dv]; w: [Dv]. Returns (out [Nv*Dv], h_new)."""
    rf = Nv // Nk
    q = q.repeat_interleave(rf, dim=0)
    k = k.repeat_interleave(rf, dim=0)
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + l2_eps) * scale
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + l2_eps)
    h = h * torch.exp(g)[:, None, None]
    v_read = torch.einsum("hk,hkv->hv", kn, h)
    delta = beta[:, None] * (v - v_read)
    h = h + torch.einsum("hk,hv->hkv", kn, delta)
    o = torch.einsum("hk,hkv->hv", qn, h)
    on = o / torch.sqrt((o * o).mean(-1, keepdim=True) + norm_eps) * w[None, :]
    return on.reshape(-1), h


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


@pytest.mark.parametrize("steps", [3])
def test_gdn_decode_step(device, steps):
    torch.manual_seed(0)
    scale = Dk**-0.5
    h = (0.05 * torch.randn(Nv, Dk, Dv)).float()
    w = (1.0 + 0.1 * torch.randn(Dv)).bfloat16()
    to_dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    state = to_dev(h.reshape(1, Nv, Dk, Dv), ttnn.float32)
    w_dev = to_dev(w, ttnn.bfloat16)
    h_ref = h.clone()
    for step in range(steps):
        q = (0.5 * torch.randn(Nk, Dk)).bfloat16()
        k = (0.5 * torch.randn(Nk, Dk)).bfloat16()
        v = (0.5 * torch.randn(Nv, Dv)).bfloat16()
        beta = torch.sigmoid(torch.randn(Nv)).float()
        g = (-0.5 * torch.rand(Nv)).float()
        out_ref, h_ref = _reference(q.float(), k.float(), v.float(), beta, g, h_ref, w.float(), scale)
        qkv_row = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)]).reshape(1, 1, 2 * KD + VD)
        out = ttnn.experimental.kda.gdn_decode_step(
            to_dev(qkv_row, ttnn.bfloat16),
            to_dev(beta.reshape(1, 1, Nv), ttnn.float32),
            to_dev(g.reshape(1, 1, Nv), ttnn.float32),
            state,
            w_dev,
            Nv,
            Nk,
            Dk,
            Dv,
            scale=scale,
            output_dtype=ttnn.float32,
        )
        out_t = ttnn.to_torch(out).reshape(-1)
        h_t = ttnn.to_torch(state).reshape(Nv, Dk, Dv)
        p_out, p_h = _pcc(out_t, out_ref), _pcc(h_t, h_ref)
        d_out = (out_t - out_ref).abs().max().item()
        d_h = (h_t - h_ref).abs().max().item()
        print(f"step {step}: out pcc={p_out:.6f} max|d|={d_out:.4e}  state pcc={p_h:.6f} max|d|={d_h:.4e}")
        assert p_out > 0.999 and p_h > 0.9999, (p_out, p_h)


def _reference_conv(hist, new_row, taps, dtb, nea, h, w, scale, l2_eps=1e-6, norm_eps=1e-6):
    """Fused-conv mode reference. hist: 3 x [C] rows (oldest first), new_row: [W] = [q|k|v|z|a|b], taps: 4 x [C].
    Returns (gated [Nv*Dv], h_new, new_hist (list of 4 rows: cs0..cs3 after the shift))."""
    C = 2 * KD + VD
    conv = hist[0] * taps[0] + hist[1] * taps[1] + hist[2] * taps[2] + new_row[:C] * taps[3]
    conv = torch.nn.functional.silu(conv)
    q, k, v = conv[:KD].reshape(Nk, Dk), conv[KD : 2 * KD].reshape(Nk, Dk), conv[2 * KD : C].reshape(Nv, Dv)
    z = new_row[C : C + VD]
    a = new_row[C + VD : C + VD + Nv]
    b = new_row[C + VD + Nv : C + VD + 2 * Nv]
    beta = torch.sigmoid(b)
    g = nea * torch.nn.functional.softplus(a + dtb, beta=1.0, threshold=20.0)
    out, h_new = _reference(q, k, v, beta, g, h, w, scale, l2_eps, norm_eps)
    gated = out * torch.nn.functional.silu(z)
    new_hist = [hist[0], hist[1], hist[2], new_row[:C]]  # cs0 <- old cs1, ..., cs3 <- new qkv (hist holds old cs1..cs3)
    return gated, h_new, new_hist


def _pack_rows(rows):
    """4 torch [C] rows (oldest first) -> [Nv, 4, 32, 32] packed head tiles (row c = channel chunk c of [q|k|v])."""
    rf = Nv // Nk
    out = torch.zeros(Nv, 4, 32, 32, dtype=torch.bfloat16)
    for h in range(Nv):
        hk = h // rf
        for j, r in enumerate(rows):
            r = r.reshape(-1).to(torch.bfloat16)
            chunks = torch.cat(
                [
                    r[hk * Dk : (hk + 1) * Dk],
                    r[KD + hk * Dk : KD + (hk + 1) * Dk],
                    r[2 * KD + h * Dv : 2 * KD + (h + 1) * Dv],
                ]
            )
            n = chunks.numel() // 32
            out[h, j, 0 : 2 * n : 2, :] = chunks.reshape(-1, 32)  # chunk c in row 2c
    return out


@pytest.mark.parametrize("steps", [3])
def test_gdn_decode_step_conv(device, steps):
    torch.manual_seed(1)
    scale = Dk**-0.5
    C = 2 * KD + VD
    W = C + VD + 32  # [q|k|v | z | a b (padded to a tile)]
    az = C + VD
    h = (0.05 * torch.randn(Nv, Dk, Dv)).float()
    w = (1.0 + 0.1 * torch.randn(Dv)).bfloat16()
    taps = [(0.3 * torch.randn(C)).bfloat16() for _ in range(4)]
    dtb = (0.1 * torch.randn(Nv)).bfloat16()
    nea = (-torch.exp(0.2 * torch.randn(Nv))).bfloat16()
    cs = [(0.5 * torch.randn(C)).bfloat16() for _ in range(4)]  # cs0 (oldest) .. cs3 (newest)
    to_dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    state = to_dev(h.reshape(1, Nv, Dk, Dv), ttnn.float32)
    w_dev = to_dev(w, ttnn.bfloat16)
    taps_dev = to_dev(_pack_rows(taps), ttnn.bfloat16)
    hist_dev = to_dev(_pack_rows(cs), ttnn.bfloat16)
    h_ref = h.clone()
    cs_ref = [t.float() for t in cs]
    for step in range(steps):
        row = (0.5 * torch.randn(W)).bfloat16()
        row[az + 2 * Nv :] = 0
        gated_ref, h_ref, new_hist = _reference_conv(
            cs_ref[1:], row.float(), [t.float() for t in taps], dtb.float(), nea.float(), h_ref, w.float(), scale
        )
        out = ttnn.experimental.kda.gdn_decode_step(
            to_dev(row.reshape(1, 1, W), ttnn.bfloat16),
            to_dev(dtb, ttnn.bfloat16),
            to_dev(nea, ttnn.bfloat16),
            state,
            w_dev,
            Nv,
            Nk,
            Dk,
            Dv,
            scale=scale,
            output_dtype=ttnn.float32,
            conv_hist=hist_dev,
            conv_taps=taps_dev,
            qkvz_dim=az,
        )
        out_t = ttnn.to_torch(out).reshape(-1)
        h_t = ttnn.to_torch(state).reshape(Nv, Dk, Dv)
        hist_t = ttnn.to_torch(hist_dev)
        p_out, p_h = _pcc(out_t, gated_ref), _pcc(h_t, h_ref)
        hist_ok = torch.equal(hist_t, _pack_rows([t.bfloat16() for t in new_hist]))
        print(
            f"step {step}: out pcc={p_out:.6f} max|d|={(out_t - gated_ref).abs().max().item():.4e}  "
            f"state pcc={p_h:.6f} max|d|={(h_t - h_ref).abs().max().item():.4e}  packed-history shift exact={hist_ok}"
        )
        assert p_out > 0.999 and p_h > 0.9999 and hist_ok, (p_out, p_h, hist_ok)
        cs_ref = new_hist
