# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-step GDN recurrent-decode drift probe (unit level, no full model).

Runs recurrent_gated_delta_rule_decode_ttnn for N steps carrying state, with the SAME per-step
random q/k/v/beta/g on TT and a torch fp32 reference (identical recurrence: L2-norm q/k, q*scale,
S=decay*S; mem=k@S; delta=(v-mem)*beta; S+=k⊗delta; o=q@S). Reports PCC(o_t^TT, o_t^ref) vs step.
If PCC decays with t → the recurrent decode numerics drift (a decode-drift cause); if stable → the
recurrence is sound and the full-model drift comes from the input feedback (attention/hidden state).

HIGH_PREC=1 (default) matches the model's fp32 decode; HIGH_PREC=0 tests bf16. N via GDN_STEPS.
Run: MESH_DEVICE=P150x4 pytest .../test_gdn_drift.py -s
"""
import os
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_decode_ttnn,
)


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _l2(x):
    return x / (x.pow(2).sum(-1, keepdim=True).sqrt() + 1e-12)


@parametrize_mesh_tp()
def test_gdn_drift(mesh_device):
    from loguru import logger

    H, Dk, Dv = 12, 128, 128
    B = 1
    N = int(os.environ.get("GDN_STEPS", "256"))
    hp = os.environ.get("HIGH_PREC", "1") == "1"
    scale = Dk**-0.5
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)

    def dev(t, dt=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)

    # torch fp32 reference state + TT state (start identical, zero)
    h_ref = torch.zeros(H, Dk, Dv, dtype=torch.float32)
    state_tt = dev(torch.zeros(B, H, Dk, Dv), ttnn.float32 if hp else ttnn.bfloat16)

    pccs = []
    for t in range(N):
        q = torch.randn(B, 1, H, Dk) * 0.3
        k = torch.randn(B, 1, H, Dk) * 0.3
        v = torch.randn(B, 1, H, Dv) * 0.3
        beta = torch.rand(B, 1, H)
        g = -torch.rand(B, 1, H) * 0.05  # decay = exp(g) ~ 0.95-1.0 (realistic, near 1)

        # ---- torch fp32 reference (single lane b=0) ----
        qn = (_l2(q[0, 0]) * scale)  # [H,Dk]
        kn = _l2(k[0, 0])            # [H,Dk]
        vt = v[0, 0]                 # [H,Dv]
        decay = torch.exp(g[0, 0])   # [H]
        S = h_ref * decay[:, None, None]
        mem = torch.einsum("hk,hkv->hv", kn, S)
        delta = (vt - mem) * beta[0, 0][:, None]
        S = S + torch.einsum("hk,hv->hkv", kn, delta)
        o_ref = torch.einsum("hk,hkv->hv", qn, S)
        h_ref = S

        # ---- TT recurrent decode ----
        o_tt, state_tt = recurrent_gated_delta_rule_decode_ttnn(
            dev(q.reshape(B, 1, H, Dk)), dev(k.reshape(B, 1, H, Dk)), dev(v.reshape(B, 1, H, Dv)),
            dev(beta.reshape(B, 1, H)), dev(g.reshape(B, 1, H)),
            scale=scale, initial_state=state_tt, device=mesh_device, high_precision=hp,
        )
        o_t = ttnn.to_torch(o_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].reshape(H, Dv)
        p = _pcc(o_t, o_ref)
        pccs.append(p)
        if (t + 1) % 32 == 0 or t < 4:
            logger.info(f"GDN_DRIFT step {t+1}/{N} hp={hp} PCC(o)={p:.5f}")
    logger.info(f"GDN_DRIFT_SUMMARY hp={hp} N={N} PCC first={pccs[0]:.5f} mid={pccs[N//2]:.5f} "
                f"last={pccs[-1]:.5f} min={min(pccs):.5f}")
