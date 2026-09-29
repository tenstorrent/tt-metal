# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hy4 MoE router (HF HYV4TopkRouter: sigmoid, noaux_tc, n_group 1, norm_topk_prob, x routed_scaling_factor 2.827)
on the 2x2 mesh, per plan.md: the gate weight and the correction bias are replicated on every chip, and each chip
routes its row's S/2 tokens of the ffn_norm output (row-split over axis 0, replicated over axis 1). Both chips of a
row compute the same routing; there is no collective.

    logits = x @ W^T                          fp32 x, fp32 weight [H, E] replicated, HiFi4 + fp32 acc, fp32 out
    scores = sigmoid(logits)                  fp32 (SFPU)
    choice = scores + bias                    fp32 + fp32 [1, E] broadcast (SFPU)
    idx    = topk(choice, 8)                  fp32 keys
    top    = gather(scores, idx)              the unbiased sigmoids of the chosen 8
    wts    = top / (sum(top) / 2.827)         renormalized, x route_scale

Precision (known issues): the fused moe_grouped_topk sorts TF32 keys, and bf16 logits / scores / choice keys flip
near-tie rows (the layer-1 median 8th/9th gap is 0.0028), so every stage stays fp32 on the SFPU path. The Hy4
layer-1 bias is small (-0.097..0.032), but it is kept fp32 anyway.

From models/demos/mimo_v2_6_d_p_2x2/tt/router.py:TtRouter (fp32 path) with route_scale 2.827 and without the device
dense scatter: the router returns (idx, wts) [1, 1, S/2, 8]; the dense [S, E] routing matrix is built only at the
harness boundary (hooks.py). Nothing in __call__ touches the host, and there are no per-call constants.
"""

from __future__ import annotations

import torch

import ttnn

TILE = 32


class TtHy4Router:
    def __init__(self, mesh, weight: torch.Tensor, bias: torch.Tensor, top_k: int = 8, route_scale: float = 2.827):
        """HF weights: mlp.gate.weight [E, H], mlp.gate.e_score_correction_bias [E]."""
        self.mesh, self.top_k, self.route_scale = mesh, int(top_k), float(route_scale)
        E, H = weight.shape
        assert E % TILE == 0 and H % TILE == 0, (E, H)
        self.num_experts, self.hidden = E, H
        self.w = self._rep(weight.float().T.contiguous().reshape(1, 1, H, E))
        self.bias = self._rep(bias.float().reshape(1, 1, 1, E))
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _rep(self, t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def __call__(self, x: ttnn.Tensor):
        """x: ffn_norm output [1, 1, S/2, H] TILE per chip (bf16 or fp32), S/2 a multiple of 32.
        Returns (idx [1, 1, S/2, k] (ttnn.topk's index dtype), wts [1, 1, S/2, k] fp32), same placement as x."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        assert x.shape[-1] == self.hidden and x.shape[-2] % TILE == 0, x.shape
        xf = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32, memory_config=dram)
        logits = ttnn.linear(xf, self.w, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=dram)
        if xf is not x:
            ttnn.deallocate(xf)
        scores = ttnn.sigmoid(logits, memory_config=dram)
        ttnn.deallocate(logits)
        choice = ttnn.add(scores, self.bias, memory_config=dram)
        _, idx = ttnn.topk(choice, k=self.top_k, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(choice)
        top = ttnn.gather(scores, dim=-1, index=idx)
        ttnn.deallocate(scores)
        tsum = ttnn.sum(top, dim=-1, keepdim=True)
        tsum2 = ttnn.multiply(tsum, 1.0 / self.route_scale, memory_config=dram)
        ttnn.deallocate(tsum)
        wts = ttnn.div(top, tsum2, memory_config=dram)
        ttnn.deallocate(top)
        ttnn.deallocate(tsum2)
        return idx, wts
