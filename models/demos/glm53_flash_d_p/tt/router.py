# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 MoE router (HF Glm5NextTextTopkRouter: noaux_tc, sigmoid, one group, top-8 of 288) on a 2x2 mesh,
replicated: every chip routes all S tokens of the replicated ffn_norm output, no CCL.

    logits = x @ W^T                          fp32 weight [H, E], HiFi4 + fp32 acc, fp32 output
    scores = sigmoid(logits)                  fp32 (SFPU)
    choice = scores + (bias - mean(bias))     fp32 [1, E] broadcast; recentred at load (top-k is shift invariant)
    ti     = topk(choice, 8)                  fp32 keys, width 288 (9 tiles)
    tw     = gather(scores, ti) / sum(...) x 2.5   unbiased sigmoid of the chosen 8, renormalized, routed_scaling_factor
    dense  = scatter(zeros bf16, ti, tw bf16) [S, E] (component boundary / dense-routing consumers)

The layer-3 bias is 7.42..7.79, so the raw choice score reaches 8.8 (fp32 step 1e-6, fine; bf16 / TF32 are not, see
known issues); recentring moves it near 1 for extra margin. Constants (weight, bias, the scatter's zeros for max_rows)
are built once at load and sliced on the device per chunk.
From models/demos/mimo_v2_6_d_p/tt/router.py:TtRouter (fp32 mode).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.glm53_flash_d_p.reference.weights import PREFIX
from models.demos.glm53_flash_d_p.tt.common import hifi4_config, replicate

TILE = 32


class TtRouter:
    def __init__(
        self,
        mesh,
        weight: torch.Tensor,
        bias: torch.Tensor,
        max_rows: int,
        top_k: int = 8,
        route_scale: float = 2.5,
        recentre: bool = True,
    ):
        """HF weights: gate.weight [E, H], e_score_correction_bias [E]. max_rows: the largest chunk the router sees."""
        self.mesh, self.top_k, self.route_scale = mesh, int(top_k), float(route_scale)
        E, H = weight.shape
        assert E % TILE == 0, f"{E} experts not tile aligned"
        self.num_experts = E
        self.max_rows = -(-int(max_rows) // TILE) * TILE
        self.w = replicate(mesh, weight.float().T.contiguous().reshape(1, 1, H, E), dtype=ttnn.float32)
        b = bias.float().reshape(1, 1, 1, E)
        if recentre:
            b = b - b.mean()
        self.bias = replicate(mesh, b, dtype=ttnn.float32)
        self.zeros = replicate(mesh, torch.zeros(1, 1, self.max_rows, E), dtype=ttnn.bfloat16)
        self.cfg = hifi4_config()

    def _rows(self, t, s):
        if s == self.max_rows:
            return t
        return ttnn.slice(t, [0, 0, 0, 0], [1, 1, s, self.num_experts])

    def __call__(self, x: ttnn.Tensor):
        """x: replicated [1, 1, S, H] TILE (bf16 or fp32), S a multiple of 32 and <= max_rows.
        Returns (dense [1, 1, S, E] bf16, idx [1, 1, S, k] uint16/uint32, weights [1, 1, S, k] fp32), replicated."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        s = x.shape[-2]
        assert s <= self.max_rows and s % TILE == 0, f"router rows {s} (max {self.max_rows}, tile aligned)"
        xf = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32, memory_config=mc)
        logits = ttnn.linear(xf, self.w, dtype=ttnn.float32, compute_kernel_config=self.cfg, memory_config=mc)
        if xf is not x:
            ttnn.deallocate(xf)
        scores = ttnn.sigmoid(logits, memory_config=mc)
        ttnn.deallocate(logits)
        choice = ttnn.add(scores, self.bias, memory_config=mc)
        _, idx = ttnn.topk(choice, k=self.top_k, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(choice)
        top = ttnn.gather(scores, dim=-1, index=idx)
        ttnn.deallocate(scores)
        tsum = ttnn.sum(top, dim=-1, keepdim=True)
        tsum_s = ttnn.multiply(tsum, 1.0 / self.route_scale, memory_config=mc)
        ttnn.deallocate(tsum)
        wts = ttnn.div(top, tsum_s, memory_config=mc)
        ttnn.deallocate(top)
        ttnn.deallocate(tsum_s)
        src = ttnn.typecast(wts, ttnn.bfloat16, memory_config=mc)
        zeros = self._rows(self.zeros, s)
        dense = ttnn.scatter(zeros, dim=-1, index=idx, src=src)
        if zeros is not self.zeros:
            ttnn.deallocate(zeros)
        ttnn.deallocate(src)
        return dense, idx, wts


def build_router(mesh, loader, cfg, layer: int, max_rows: int) -> TtRouter:
    p = f"{PREFIX}layers.{layer}.mlp.gate."
    assert cfg.norm_topk_prob, "router assumes norm_topk_prob"
    return TtRouter(
        mesh,
        loader.weight(p + "weight"),
        loader.weight(p + "e_score_correction_bias"),
        max_rows,
        top_k=cfg.num_experts_per_tok,
        route_scale=cfg.routed_scaling_factor,
    )
