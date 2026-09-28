# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 MoE router (HF MiMoV2MoEGate: noaux_tc, sigmoid, n_group 1), replicated on the 2x2 mesh.

Copied from models/demos/mimo_v2_6_d_p/tt/router.py (1x4) unchanged in behaviour: the weight and every constant are
replicated (ReplicateTensorToMesh works for any mesh shape), every chip routes all S tokens of the replicated ffn_norm
output, and there is no CCL. The experts step takes its own row half later (mesh_partition inside the experts module).

    logits = x @ W^T                          fp32 weight [H, E], HiFi4 + fp32 acc, fp32 output
    scores = sigmoid(logits)                  fp32 (SFPU)
    choice = scores + bias                    fp32 + fp32 [1, E] broadcast (SFPU, exact to 2e-7)
    ti     = topk(choice, 8)                  fp32 keys
    tw     = gather(scores, ti) / sum(...)    unbiased sigmoid of the chosen 8, renormalized, x route_scale (1.0)
    dense  = scatter(zeros bf16, ti, tw bf16) [S, E] (boundary / dense-routing consumers)

Precision: the correction bias is 1.72..2.18 (8th choice score median 2.33), and the median gap between the 8th and 9th
choice is 0.0017. The fused ttnn.experimental.deepseek_prefill.moe_grouped_topk (the components entry) computes the
choice with an FPU add and sorts keys unpacked as TF32 (10-bit mantissa, step 0.002 near 2): on the layer-1 golden it
scores PCC 0.9855 even on exact fp32 logits, and 0.9975 with the bias recentred by its mean. The fp32 SFPU path above
scores 0.99933, the same as the CPU reference. The fused path stays selectable (mode="fused", bias recentred by its
mean, which does not change top-k) for comparison. routed_scaling_factor is null -> route_scale 1.0 (not DeepSeek's 2.5).

Constants that depend only on shapes (the zero tensor for the scatter; the fused path's full-shape bias) are built once
at load for max_rows and sliced on the device per chunk. Adapted from
models/demos/deepseek_v3_d_p/tt/moe/tt_moe_gate_prefill.py (DEVICE_FP32: fp32 gate matmul + moe_grouped_topk, without
the TP all-reduce: the router weight is replicated) and models/demos/gemma4_a4b_d_p/tt/router.py (fp32 topk / gather /
renorm, bf16 scatter boundary: ttnn.scatter has no fp32 TILE path).
"""

from __future__ import annotations

import torch

import ttnn

TILE = 32


def _hifi4():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


class TtRouter:
    MODES = ("fp32", "fused")

    def __init__(
        self,
        mesh,
        weight: torch.Tensor,
        bias: torch.Tensor,
        max_rows: int,
        top_k: int = 8,
        route_scale: float = 1.0,
        mode: str = "fp32",
    ):
        """HF weights: gate.weight [E, H], e_score_correction_bias [E]. max_rows: the largest chunk the router sees.
        mode: "fp32" (SFPU choice score + ttnn.topk, default) or "fused" (moe_grouped_topk, TF32 keys)."""
        assert mode in self.MODES, mode
        self.mesh, self.top_k, self.route_scale, self.mode = mesh, top_k, float(route_scale), mode
        E, H = weight.shape
        assert E % TILE == 0, f"{E} experts not tile aligned"
        self.num_experts = E
        self.max_rows = -(-int(max_rows) // TILE) * TILE
        self.w = self._rep(weight.float().T.contiguous().reshape(1, 1, H, E), ttnn.float32)
        b = bias.float().reshape(1, 1, 1, E)
        if mode == "fp32":
            self.bias = self._rep(b, ttnn.float32)
        else:
            # moe_grouped_topk wants the bias in the scores' shape; recentre it (top-k is shift invariant) so the
            # TF32 sort keys sit near 0 instead of 2.
            b = (b - b.mean()).expand(1, 1, self.max_rows, E).contiguous()
            self.bias = self._rep(b, ttnn.float32)
        self.zeros = self._rep(torch.zeros(1, 1, self.max_rows, E), ttnn.bfloat16)
        self.cfg = _hifi4()

    def _rep(self, t, dtype):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def _rows(self, t, s):
        if s == self.max_rows:
            return t
        return ttnn.slice(t, [0, 0, 0, 0], [1, 1, s, self.num_experts])

    def _select_fp32(self, logits):
        mc = ttnn.DRAM_MEMORY_CONFIG
        scores = ttnn.sigmoid(logits, memory_config=mc)
        choice = ttnn.add(scores, self.bias, memory_config=mc)
        _, idx = ttnn.topk(choice, k=self.top_k, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(choice)
        top = ttnn.gather(scores, dim=-1, index=idx)
        ttnn.deallocate(scores)
        tsum = ttnn.sum(top, dim=-1, keepdim=True)
        if self.route_scale != 1.0:
            tsum = ttnn.multiply(tsum, 1.0 / self.route_scale)
        wts = ttnn.div(top, tsum)
        ttnn.deallocate(top)
        ttnn.deallocate(tsum)
        return wts, idx

    def _select_fused(self, logits, s):
        bias = self._rows(self.bias, s)
        wts, idx = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            logits,
            bias,
            n_groups=1,
            summed_experts_per_group=1,
            topk_groups=1,
            n_activated_experts=self.top_k,
            route_scale=self.route_scale,
            epsilon=1e-20,
            stable_sort=True,
            score_func="sigmoid",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            weights_layout=ttnn.TILE_LAYOUT,
        )
        if bias is not self.bias:
            ttnn.deallocate(bias)
        return wts, idx

    def __call__(self, x: ttnn.Tensor):
        """x: replicated [1, 1, S, H] TILE (bf16 or fp32), S a multiple of 32 and <= max_rows.
        Returns (dense [1, 1, S, E] bf16, idx [1, 1, S, k] (uint32 fp32 path, uint16 fused), weights [1, 1, S, k]
        (fp32 path fp32, fused bf16)), all replicated."""
        s = x.shape[-2]
        assert s <= self.max_rows and s % TILE == 0, f"router rows {s} (max {self.max_rows}, tile aligned)"
        xf = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
        logits = ttnn.linear(
            xf, self.w, dtype=ttnn.float32, compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if xf is not x:
            ttnn.deallocate(xf)
        wts, idx = self._select_fp32(logits) if self.mode == "fp32" else self._select_fused(logits, s)
        ttnn.deallocate(logits)
        src = wts if wts.dtype == ttnn.bfloat16 else ttnn.typecast(wts, ttnn.bfloat16)
        zeros = self._rows(self.zeros, s)
        dense = ttnn.scatter(zeros, dim=-1, index=idx, src=src)
        if zeros is not self.zeros:
            ttnn.deallocate(zeros)
        if src is not wts:
            ttnn.deallocate(src)
        return dense, idx, wts
