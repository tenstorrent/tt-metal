# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 router on a 1x4 mesh, replicated (every chip computes the same routing from replicated h_mid, no CCL).

    h      = rms_norm(h_mid)                        no weight, fp32
    logits = h @ W'                                 W' = (proj.weight * router.scale * H^-0.5)^T, fp32 [H, 128]
    probs  = softmax(logits)                        fp32
    ti     = topk(probs, 8)                         fp32 sort
    tw     = gather(probs * pes, ti) / sum(gather(probs, ti))    = renorm(top-8) * per_expert_scale[ti]
    dense  = scatter(zeros bf16, ti, tw bf16)       [S, 128] (boundary / dense-routing consumers)

router.scale (per hidden channel) and H^-0.5 are folded into the proj weight on the host: (h * s) @ W^T = h @ (W * s)^T.
per_expert_scale is applied after the renormalization by gathering probs * per_expert_scale at the same indices, so no
per-row index lookup is needed. Adapted from models/demos/gemma4/tt/router.py and models/demos/ernie45_d_p/tt/moe.py:TtRouter
(fp32 selection and renorm, bf16 scatter: ttnn.scatter has no fp32 TILE path).
"""

from __future__ import annotations

import torch

import ttnn


def _hifi4():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


class TtRouter:
    def __init__(
        self,
        mesh,
        proj: torch.Tensor,
        scale: torch.Tensor,
        per_expert_scale: torch.Tensor,
        top_k: int = 8,
        eps: float = 1e-6,
    ):
        """HF weights: proj [E, H], scale [H], per_expert_scale [E]."""
        self.mesh, self.top_k, self.eps = mesh, top_k, eps
        E, H = proj.shape
        self.num_experts = E
        w = (proj.float() * scale.float()[None, :] * (H**-0.5)).T.contiguous()  # [H, E]
        self.w = self._rep(w[None, None])
        self.pes = self._rep(per_expert_scale.float().reshape(1, 1, 1, E))

    def _rep(self, t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def __call__(self, x: ttnn.Tensor):
        """x: replicated [1, 1, S, H] TILE (bf16 or fp32). Returns (dense [1, 1, S, E] bf16, idx [1, 1, S, k] uint16,
        weights [1, 1, S, k] fp32), all replicated."""
        cfg = _hifi4()
        xf = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
        h = ttnn.rms_norm(xf, epsilon=self.eps, compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if xf is not x:
            ttnn.deallocate(xf)
        logits = ttnn.linear(h, self.w, compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(h)
        probs = ttnn.softmax(logits, dim=-1, numeric_stable=True, compute_kernel_config=cfg)
        ttnn.deallocate(logits)
        _, idx = ttnn.topk(probs, k=self.top_k, dim=-1, largest=True, sorted=True)
        top = ttnn.gather(probs, dim=-1, index=idx)
        scaled = ttnn.mul(probs, self.pes)
        top_scaled = ttnn.gather(scaled, dim=-1, index=idx)
        ttnn.deallocate(scaled)
        tsum = ttnn.sum(top, dim=-1, keepdim=True)
        ttnn.deallocate(top)
        wts = ttnn.div(top_scaled, tsum)
        ttnn.deallocate(top_scaled)
        ttnn.deallocate(tsum)
        zeros = ttnn.typecast(ttnn.zeros_like(probs), ttnn.bfloat16)
        ttnn.deallocate(probs)
        wts_bf16 = ttnn.typecast(wts, ttnn.bfloat16)
        dense = ttnn.scatter(zeros, dim=-1, index=idx, src=wts_bf16)
        ttnn.deallocate(zeros)
        ttnn.deallocate(wts_bf16)
        return dense, idx, wts
