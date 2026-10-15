# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 MoE router (reference/ds41_ref.py:_route; inference Gate: score_func sqrtsoftplus, one group, top 6
of 384, norm_topk_prob, route_scale 1.5) on the SP x TP mesh, replicated per mesh row: no CCL.

Per chip (r, c) the input is ffn_norm's gathered output (tt/norm.py, gather=True): [1, 1, S/SP, H] fp32 TILE, the
row's whole token rows, replicated over axis 1, so every chip of a row computes the same routing.

    logits = x @ gate^T                       gate [384, 5120] bf16 as stored (exact), x fp32, HiFi4 + fp32
                                              accumulation, fp32 out. The FPU reads fp32 at about TF32 (K-30) and rounds
                                              each in-tile partial dot to ~TF32 (DS41-16), so x as bf16 hi + lo halves
                                              does not help: layer-0 golden, 2048 rows, 4 rows select another expert
                                              than fp64 either way (near ties, gap 6th / 7th down to 1.6e-6; K-39),
                                              routing weights rel 4.5e-5 on the rest; one matmul 0.72 ms vs 1.02 ms
    scores = sqrt(softplus(logits))           fp32 (SFPU)
    choice = scores + (bias - mean(bias))     fp32 [1, E] broadcast (SFPU, K-23); the bias (9.59..9.88 on layer 0) is
                                              recentred at load: top-k is shift invariant, keys stay near the scores
    ti     = topk(choice, 6)                  fp32 keys, width 384 (12 tiles)
    tw     = gather(scores, ti) / ((sum + 1e-20) / 1.5)   the unbiased scores of the chosen 6 (K-23), renormalized
    dense  = scatter(zeros fp32 RM, ti, tw)   [S/SP, 384] fp32 ROW_MAJOR (K-4: no fp32 TILE scatter): the component
                                              boundary and dense-routing consumers

Constants (weight, bias, the scatter's zeros for max_rows) are built once at load and sliced on the device per chunk.
Adapted from models/demos/xing40_a4b_d_p/tt/router.py:TtRouter (and mimo_v2_6_d_p/tt/router.py's ttnn.topk path).
"""

from __future__ import annotations

import torch

import ttnn

from .common import DRAM, SP_AXIS, compute_config

TILE = 32


def _replicate(mesh, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


class TtRouter:
    def __init__(
        self,
        mesh,
        weight: torch.Tensor,
        bias: torch.Tensor,
        max_rows: int,
        top_k: int = 6,
        route_scale: float = 1.5,
    ):
        """weight: ffn.gate.weight [E, H] (bf16 as stored), bias: ffn.gate.bias [E] fp32. max_rows: the largest
        per-chip row count (chunk / SP)."""
        self.mesh, self.top_k, self.route_scale = mesh, int(top_k), float(route_scale)
        E, H = weight.shape
        assert E % TILE == 0, f"{E} experts not tile aligned"
        assert weight.dtype == torch.bfloat16, f"router weight stored as {weight.dtype}, expected bf16"
        self.num_experts, self.hidden = E, H
        self.max_rows = -(-int(max_rows) // TILE) * TILE
        self.w = _replicate(mesh, weight.T.reshape(1, 1, H, E), ttnn.bfloat16)
        b = bias.float().reshape(1, 1, 1, E)
        self.bias = _replicate(mesh, b - b.mean(), ttnn.float32)
        self.zeros = _replicate(mesh, torch.zeros(1, 1, self.max_rows, E), ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.cfg = compute_config(mesh)

    def _rows(self, t, s):
        if s == self.max_rows:
            return t
        return ttnn.slice(t, [0, 0, 0, 0], [1, 1, s, self.num_experts])

    def __call__(self, x: ttnn.Tensor):
        """x: [1, 1, S/SP, H] fp32 TILE per chip, row split over axis 0, replicated over axis 1; S/SP a multiple of 32.
        Returns (dense [1, 1, S/SP, E] fp32 ROW_MAJOR, idx [1, 1, S/SP, k] (topk's index dtype), weights
        [1, 1, S/SP, k] fp32 TILE), same layout."""
        s = x.shape[-2]
        assert s <= self.max_rows and s % TILE == 0, f"router rows {s} (max {self.max_rows}, tile aligned)"
        assert x.shape[-1] == self.hidden, (tuple(x.shape), self.hidden)
        logits = ttnn.linear(x, self.w, dtype=ttnn.float32, compute_kernel_config=self.cfg, memory_config=DRAM)
        sp = ttnn.softplus(logits, beta=1.0, threshold=20.0, memory_config=DRAM)
        ttnn.deallocate(logits)
        scores = ttnn.sqrt(sp, memory_config=DRAM)
        ttnn.deallocate(sp)
        choice = ttnn.add(scores, self.bias, dtype=ttnn.float32, memory_config=DRAM)
        _, idx = ttnn.topk(choice, k=self.top_k, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(choice)
        top = ttnn.gather(scores, dim=-1, index=idx)
        ttnn.deallocate(scores)
        tsum = ttnn.sum(top, dim=-1, keepdim=True)
        # (sum + 1e-20) / route_scale: the reference's eps, then the routed scale folded into the divisor
        t1 = ttnn.add(tsum, 1e-20, memory_config=DRAM)
        ttnn.deallocate(tsum)
        tsum_s = ttnn.multiply(t1, 1.0 / self.route_scale, memory_config=DRAM)
        ttnn.deallocate(t1)
        wts = ttnn.div(top, tsum_s, memory_config=DRAM)
        ttnn.deallocate(top)
        ttnn.deallocate(tsum_s)
        zeros = self._rows(self.zeros, s)
        idx_rm = ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        wts_rm = ttnn.to_layout(wts, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        dense = ttnn.scatter(zeros, dim=-1, index=idx_rm, src=wts_rm)
        ttnn.deallocate(idx_rm)
        ttnn.deallocate(wts_rm)
        if zeros is not self.zeros:
            ttnn.deallocate(zeros)
        return dense, idx, wts


def build_router(mesh, ck, cfg, layer: int, chunk: int) -> TtRouter:
    """chunk: the largest chunk the router sees; each chip routes chunk / SP rows."""
    sp = mesh.shape[SP_AXIS]
    p = f"layers.{layer}.ffn.gate."
    return TtRouter(
        mesh,
        ck.get(p + "weight"),
        ck.weight(p + "bias", torch.float32),
        -(-int(chunk) // sp),
        top_k=cfg.topk_experts,
        route_scale=cfg.route_scale,
    )
