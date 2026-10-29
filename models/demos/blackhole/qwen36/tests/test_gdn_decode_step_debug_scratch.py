# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH debug for the packed fused-conv gdn_decode_step: isolate packing/scatter from the recurrence."""
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_gdn_decode_step_scratch import KD, VD, Dk, Dv, Nk, Nv, _pack_rows, _pcc


def test_packed_conv_isolation(device):
    torch.manual_seed(2)
    C = 2 * KD + VD
    W = C + VD + 32
    az = C + VD
    to_dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    h = (0.05 * torch.randn(Nv, Dk, Dv)).float()
    w = (1.0 + 0.1 * torch.randn(Dv)).bfloat16()
    row = (0.5 * torch.randn(W)).bfloat16()
    row[az + 2 * Nv :] = 0
    a = row[az : az + Nv].float()
    b = row[az + Nv : az + 2 * Nv].float()
    dtb = (0.1 * torch.randn(Nv)).bfloat16()
    nea = (-torch.exp(0.2 * torch.randn(Nv))).bfloat16()
    beta = torch.sigmoid(b)
    g = nea.float() * torch.nn.functional.softplus(a + dtb.float(), beta=1.0, threshold=20.0)
    # plain op on the equivalent conv output: taps [0,0,0,1], zero history -> conv = silu(new)
    conv_row = torch.nn.functional.silu(row[:C].float())
    state_a = to_dev(h.reshape(1, Nv, Dk, Dv), ttnn.float32)
    out_a = ttnn.experimental.kda.gdn_decode_step(
        to_dev(conv_row.bfloat16().reshape(1, 1, C), ttnn.bfloat16),
        to_dev(beta.reshape(1, 1, Nv), ttnn.float32),
        to_dev(g.reshape(1, 1, Nv), ttnn.float32),
        state_a,
        to_dev(w, ttnn.bfloat16),
        Nv,
        Nk,
        Dk,
        Dv,
        scale=Dk**-0.5,
        output_dtype=ttnn.float32,
    )
    out_a_t = ttnn.to_torch(out_a).reshape(-1) * torch.nn.functional.silu(row[C : C + VD].float())
    # packed op
    taps = [torch.zeros(C).bfloat16(), torch.zeros(C).bfloat16(), torch.zeros(C).bfloat16(), torch.ones(C).bfloat16()]
    cs = [torch.zeros(C).bfloat16() for _ in range(4)]
    state_b = to_dev(h.reshape(1, Nv, Dk, Dv), ttnn.float32)
    hist_dev = to_dev(_pack_rows(cs).unsqueeze(0), ttnn.bfloat16)
    out_b = ttnn.experimental.kda.gdn_decode_step(
        to_dev(row.reshape(1, 1, W), ttnn.bfloat16),
        to_dev(dtb, ttnn.bfloat16),
        to_dev(nea, ttnn.bfloat16),
        state_b,
        to_dev(w, ttnn.bfloat16),
        Nv,
        Nk,
        Dk,
        Dv,
        scale=Dk**-0.5,
        output_dtype=ttnn.float32,
        conv_hist=hist_dev,
        conv_taps=to_dev(_pack_rows(taps), ttnn.bfloat16),
        qkvz_dim=az,
    )
    out_b_t = ttnn.to_torch(out_b).reshape(-1)
    print(f"plain-vs-packed out pcc={_pcc(out_a_t, out_b_t):.6f} max|d|={(out_a_t - out_b_t).abs().max().item():.4e}")
    print(f"state pcc={_pcc(ttnn.to_torch(state_a), ttnn.to_torch(state_b)):.6f}")
    hist_t = ttnn.to_torch(hist_dev)[0]
    exp = _pack_rows([cs[1], cs[2], cs[3], row[:C]])
    for j in range(4):
        print(
            f"slot {j} exact={torch.equal(hist_t[:, j], exp[:, j])}  max|d|={(hist_t[:, j].float() - exp[:, j].float()).abs().max().item():.3e}"
        )
    # per-head check of slot 3 (the packed new token): which rows differ?
    bad = (hist_t[:, 3].float() != exp[:, 3].float()).any(dim=-1)  # [Nv, 32]
    print("slot3 rows differing (head 0):", bad[0].nonzero().flatten().tolist())
    print("slot3 head0 row0 dev:", hist_t[0, 3, 0, :8].float().tolist())
    print("slot3 head0 row0 exp:", exp[0, 3, 0, :8].float().tolist())
