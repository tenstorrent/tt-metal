# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""§0 decisive branch: chunk-vs-torch AND recurrent-vs-torch on the SAME sequence.

test_gdn_drift showed recurrent ≈ torch (0.9998). test_gdn_chunk_vs_rec showed chunk ≠ recurrent
(diverging with position). This measures the missing number: chunk kernel vs the torch fp32 recurrence
(ground truth), on the identical raw q/k/v/beta/g fed to all three paths. Both kernels L2-norm+scale
q/k internally, matching the torch reference.

Decides the fix:
  - chunk-vs-torch ~0.999  -> both kernels accurate; drift is PATH INCONSISTENCY  -> fix = re-sync (A)
  - chunk-vs-torch low      -> chunk (prefill) kernel under-precise             -> fix = kernel precision (C)

Run: MESH_DEVICE=P150x4 pytest .../test_gdn_chunk_vs_torch.py -s
"""
import os
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_decode_ttnn,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import (
    chunk_gated_delta_rule_seq_adapter,
)


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _l2(x):
    return x / (x.pow(2).sum(-1, keepdim=True).sqrt() + 1e-12)


@parametrize_mesh_tp()
def test_gdn_chunk_vs_torch(mesh_device):
    from loguru import logger

    H, Dk, Dv = 12, 128, 128
    B, T = 1, 128
    scale = Dk**-0.5
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)

    def dev(t, dt=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)

    # full sequence of raw inputs (same scale/stats as test_gdn_drift)
    q = torch.randn(B, T, H, Dk) * 0.3
    k = torch.randn(B, T, H, Dk) * 0.3
    v = torch.randn(B, T, H, Dv) * 0.3
    beta = torch.rand(B, T, H)
    g = -torch.rand(B, T, H) * 0.05  # decay = exp(g) ~ 0.95-1.0

    # ---- torch fp32 reference recurrence (ground truth), per position ----
    h_ref = torch.zeros(H, Dk, Dv, dtype=torch.float32)
    o_ref = torch.zeros(T, H, Dv, dtype=torch.float32)
    for t in range(T):
        qn = _l2(q[0, t]) * scale
        kn = _l2(k[0, t])
        decay = torch.exp(g[0, t])
        S = h_ref * decay[:, None, None]
        mem = torch.einsum("hk,hkv->hv", kn, S)
        delta = (v[0, t] - mem) * beta[0, t][:, None]
        S = S + torch.einsum("hk,hv->hkv", kn, delta)
        o_ref[t] = torch.einsum("hk,hkv->hv", qn, S)
        h_ref = S

    # ---- TT recurrent decode, step by step (carry state) ----
    state_tt = dev(torch.zeros(B, H, Dk, Dv), ttnn.float32)
    o_rec = torch.zeros(T, H, Dv)
    for t in range(T):
        o_tt, state_tt = recurrent_gated_delta_rule_decode_ttnn(
            dev(q[:, t:t + 1]), dev(k[:, t:t + 1]), dev(v[:, t:t + 1]),
            dev(beta[:, t:t + 1]), dev(g[:, t:t + 1]),
            scale=scale, initial_state=state_tt, device=mesh_device, high_precision=True,
        )
        o_rec[t] = ttnn.to_torch(o_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].reshape(H, Dv)

    # ---- TT chunk kernel over the whole sequence ----
    o_ck, _ = chunk_gated_delta_rule_seq_adapter(
        dev(q), dev(k), dev(v), dev(beta), dev(g),
        chunk_size=128, scale=scale, initial_state=None, device=mesh_device, valid_len=T,
    )
    o_chunk = ttnn.to_torch(o_ck, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].reshape(T, H, Dv)

    logger.info(f"GDNCVT overall  recurrent_vs_torch={_pcc(o_rec, o_ref):.5f}  chunk_vs_torch={_pcc(o_chunk, o_ref):.5f}  "
                f"chunk_vs_recurrent={_pcc(o_chunk, o_rec):.5f}")
    for lo in range(0, T, 32):
        hi = min(lo + 32, T)
        rvt = sum(_pcc(o_rec[p], o_ref[p]) for p in range(lo, hi)) / (hi - lo)
        cvt = sum(_pcc(o_chunk[p], o_ref[p]) for p in range(lo, hi)) / (hi - lo)
        logger.info(f"GDNCVT pos[{lo}-{hi-1}] recurrent_vs_torch={rvt:.5f} chunk_vs_torch={cvt:.5f}")
