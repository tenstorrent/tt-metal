# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: chunked KDA recurrence (ttnn.experimental.kda) accuracy vs log-decay magnitude, and the effect of clamping."""
from __future__ import annotations

import torch

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tests.utils import pcc
from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_recurrent_reference
from models.demos.deepseek_v3_d_p.tt.kda.config import KDARecurrenceProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import KDARecurrence


def _run(mesh_device, rec, q, k, v, g, beta, state):
    """q,k,v [1,T,H,D] torch -> device layout [1,T,H*D] bf16; g [1,T,H,K] bf16; beta [1,T,H] fp32; state [1,H,K,V] fp32."""
    T, H, K = q.shape[1], q.shape[2], q.shape[3]
    dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    new_state, out = rec(
        q=dev(q.reshape(1, T, H * K), ttnn.bfloat16),
        k=dev(k.reshape(1, T, H * K), ttnn.bfloat16),
        v=dev(v.reshape(1, T, H * K), ttnn.bfloat16),
        gate=dev(g.reshape(1, T, H * K), ttnn.bfloat16),
        beta=dev(beta, ttnn.float32),
        initial_state=dev(state, ttnn.float32),
    )
    o = ttnn.to_torch(out).float().reshape(H, T, K).permute(1, 0, 2).unsqueeze(0)  # [1,T,H,V]
    return o, ttnn.to_torch(new_state).float()


def test_recurrence_vs_gate_scale(mesh_device):
    torch.manual_seed(0)
    T, H, K = 64, 32, 128
    rec = KDARecurrence(
        mesh_device, KDARecurrenceProgramConfig(local_scan_strategy="direct"), sequence_parallel_axis=None
    )
    q, k, v = (
        (torch.randn(1, T, H, K) * 0.05).bfloat16().float(),
        (torch.randn(1, T, H, K) * 0.05).bfloat16().float(),
        (torch.randn(1, T, H, K) * 0.05).bfloat16().float(),
    )
    beta = torch.rand(1, T, H)
    state = torch.zeros(1, H, K, K)
    base = -torch.rand(1, T, H, K)
    for scale in (0.5, 2.0, 8.0, 32.0, 128.0):
        g = (base * scale).bfloat16().float()
        o_r, s_r = kda_recurrent_reference(q, k, v, g, beta, state)
        o_t, s_t = _run(mesh_device, rec, q, k, v, g, beta, state)
        print(
            f"[scale {scale:6.1f}] g range [{g.min():.1f}, {g.max():.2f}]  out pcc {pcc(o_r, o_t):.5f}  state pcc {pcc(s_r, s_t):.5f}  finite {torch.isfinite(o_t).all().item()}"
        )
    # clamp sweep on a heavy-tailed gate resembling layer 0 (exp(A_log) up to 200)
    heavy = -(torch.rand(1, T, H, K) ** 4) * 300.0
    for clamp in (None, -60.0, -30.0, -20.0, -12.0):
        g = heavy if clamp is None else heavy.clamp(min=clamp)
        g = g.bfloat16().float()
        o_r, s_r = kda_recurrent_reference(q, k, v, g, beta, state)
        o_t, s_t = _run(mesh_device, rec, q, k, v, g, beta, state)
        o_true, s_true = kda_recurrent_reference(q, k, v, heavy, beta, state)
        print(
            f"[clamp {clamp}] out pcc {pcc(o_r, o_t):.5f} state {pcc(s_r, s_t):.5f} | clamped-torch vs unclamped-torch out {pcc(o_true, o_r):.6f} state {pcc(s_true, s_r):.6f}"
        )
