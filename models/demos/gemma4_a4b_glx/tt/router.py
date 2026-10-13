# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 router for prefill on the 8x4 Galaxy mesh, replicated (every chip of a row routes its own tokens).

Reference: h = rms_norm(x) * scale * H^-0.5; probs = softmax(h @ W^T); top-8 of probs, renormalized over the 8,
times per_expert_scale[id], scattered into a dense [S, 128] matrix.

On device, all fp32 after the input: rms_norm (no weight), linear with W' = (W * scale * H^-0.5)^T (folded at load),
softmax, topk(8) ids, a 0/1 mask = bf16 scatter of ones at the ids (exact; scatter has no fp32 TILE path), then
dense = probs * per_expert_scale * mask / sum(probs * mask), all fp32 and 128 wide, so no reduction runs over a padded
8-wide tile. Off-top-8 entries are exactly 0. A mask from `probs >= 8th topk value` is wrong: on Blackhole the topk
values are not bit-equal to the fp32 probs, and ~30 of 2048 rows then select 7 or 9 experts.
bf16 weights / bf16 logits fail the selection checks (overlap 0.9947 with bfp8 W, matched rel 0.0065 with bf16
logits on the layer-0 golden), so W' stays fp32 by default.
"""


import ttnn

from .settings import settings

_FIDELITY = {"HiFi4": ttnn.MathFidelity.HiFi4, "HiFi2": ttnn.MathFidelity.HiFi2, "LoFi": ttnn.MathFidelity.LoFi}
_DTYPE = {"float32": ttnn.float32, "bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b}


class TtRouter:
    def __init__(self, mesh, w_proj, scale, per_expert_scale, top_k: int, eps: float = 1e-6):
        """w_proj: HF [E, H]; scale: [H]; per_expert_scale: [E]."""
        self.mesh = mesh
        self.top_k = top_k
        self.eps = eps
        hidden = w_proj.shape[1]
        w = (w_proj.float() * scale.float()[None, :] * hidden**-0.5).t().contiguous()  # [H, E]
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.w = ttnn.from_torch(
            w.reshape(1, 1, *w.shape),
            dtype=_DTYPE[settings.get("ROUTER_WEIGHT_DTYPE")],
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.pes = ttnn.from_torch(
            per_expert_scale.float().reshape(1, 1, 1, -1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self._consts = {}
        self.cfg = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=_FIDELITY[settings.get("ROUTER_FIDELITY")],
            math_approx_mode=False,
            fp32_dest_acc_en=settings.get("ROUTER_FP32_ACC"),
            packer_l1_acc=False,
        )

    def _scatter_operands(self, probs, vals):
        """bf16 zeros [.., S, E] and ones [.., S, k] for the selection mask, built once per S on the device (scatter
        returns a new tensor, so they are never written)."""
        key = tuple(probs.shape)
        if key not in self._consts:
            self._consts[key] = (
                ttnn.zeros_like(probs, dtype=ttnn.bfloat16),
                ttnn.ones_like(vals, dtype=ttnn.bfloat16),
            )
        return self._consts[key]

    def __call__(self, x):
        """x: [1, 1, S, H] (bf16 or fp32), replicated within each row. Returns (dense [1, 1, S, E] fp32,
        top-k ids [1, 1, S, k] uint32)."""
        xf = ttnn.typecast(x, ttnn.float32) if x.dtype != ttnn.float32 else x
        h = ttnn.rms_norm(xf, epsilon=self.eps, compute_kernel_config=self.cfg)
        if xf is not x:
            ttnn.deallocate(xf)
        logits = ttnn.linear(h, self.w, dtype=ttnn.float32, compute_kernel_config=self.cfg)
        ttnn.deallocate(h)
        probs = ttnn.softmax(logits, dim=-1, numeric_stable=True, compute_kernel_config=self.cfg)
        ttnn.deallocate(logits)
        vals, ids = ttnn.topk(probs, k=self.top_k, dim=-1, largest=True, sorted=True)
        zeros, ones = self._scatter_operands(probs, vals)
        ttnn.deallocate(vals)
        mask16 = ttnn.scatter(zeros, -1, ids, ones)
        mask = ttnn.typecast(mask16, ttnn.float32)
        ttnn.deallocate(mask16)
        sel = ttnn.mul(probs, mask, dtype=ttnn.float32)
        ttnn.deallocate(probs)
        ttnn.deallocate(mask)
        den = ttnn.sum(sel, dim=-1, keepdim=True, compute_kernel_config=self.cfg)
        num = ttnn.mul(sel, self.pes, dtype=ttnn.float32)
        ttnn.deallocate(sel)
        dense = ttnn.div(num, den, dtype=ttnn.float32)
        ttnn.deallocate(num)
        ttnn.deallocate(den)
        return dense, ids
