# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model-free probe: does the commit's chunked masked flash SDPA match torch?

Isolates the commit's ``_sdpa_causal_masked`` (chunked flash SDPA + additive causal
mask) from the full model. The denoise path only validated the *maskless* flash
SDPA, so this checks the new heavy-causal-mask regime directly against a torch
reference — no 46GB checkpoint load.
"""
from __future__ import annotations

import os
import sys

import torch
import ttnn

from models.experimental.diffusion_gemma.reference.attention_mask import build_canvas_denoise_mask
from models.experimental.diffusion_gemma.tt.commit_batched import _sdpa_causal_masked, NEG


def _pcc(a, b):
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if torch.allclose(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else 0.0


def torch_ref(q, k, v, mask, q_heads, kv_heads):
    # q [1,qh,C,hd], k/v [1,kvh,K,hd], mask [1,1,C,K] additive
    rep = q_heads // kv_heads
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2))  # scale=1.0
    scores = scores + mask.float()
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float())


def main():
    P = int(os.environ.get("PROBE_P", "32"))
    C = int(os.environ.get("PROBE_C", "256"))
    hd = 256
    qh, kvh = 4, 2
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
    try:
        torch.manual_seed(0)
        q = torch.randn(1, qh, C, hd, dtype=torch.bfloat16)
        k = torch.randn(1, kvh, P + C, hd, dtype=torch.bfloat16)
        v = torch.randn(1, kvh, P + C, hd, dtype=torch.bfloat16)
        mask = build_canvas_denoise_mask(
            P, C, layer_type="full_attention", causal=True, neg_inf=NEG, dtype=torch.float32
        ).view(1, 1, C, P + C)

        ref = torch_ref(q, k, v, mask, qh, kvh)

        def to_dev(t, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                t,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        tt_q, tt_k, tt_v = to_dev(q), to_dev(k), to_dev(v)
        tt_mask = to_dev(mask)
        tt_out = _sdpa_causal_masked(tt_q, tt_k, tt_v, attn_mask=tt_mask, head_dim=hd)
        got = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])

        pcc = _pcc(ref, got)
        print(
            f"PROBE P={P} C={C}: masked-flash-SDPA vs torch  PCC={pcc:.6f}  "
            f"max_abs_diff={float((ref.float()-got.float()).abs().max()):.4e}"
        )
        print("PROBE_RESULT", "PASS" if pcc >= 0.99 else "FAIL")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
