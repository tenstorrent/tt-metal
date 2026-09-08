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
