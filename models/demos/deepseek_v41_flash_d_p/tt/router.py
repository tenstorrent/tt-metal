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

ROUTER_IMPL=fused (P.22, the default; tt/settings.py): everything after the gate matmul is ONE
ttnn.bringup.moe_grouped_topk(n_groups=1, score_func="sqrtsoftplus", fp32_selection=True, fp32_activation=True)
(the fork of Marko Staletovic's fused gate, GLM-5.3's router; fp32_activation compiles sqrt(softplus) with the fp32
SFPU polynomials, as ttnn.softplus does for fp32: the source op's bf16-grade softplus is ~8e-4 relative off on [-3, 9],
0 below -5, and swapped up to 11 of 2048 rows' experts at layer 24, setting ROUTER_SOFTPLUS, DS41-105):
sqrt(softplus) on the SFPU, the recentred fp32 bias added on the
SFPU for the selection only, top 6 on full fp32 keys, the unbiased scores gathered and normalized x route_scale /
(sum + 1e-20) in the fp32 DEST (K-23). Outputs: weights bf16 TILE (the reference's own dtype, model.py Gate
`weights.type_as(x)`; the experts take bf16 weights either way, tt/weighted_combine.py:routing), ids uint16 TILE.
The op takes a full-shape bias: [max_rows, E] fp32 built at load. The dense routing (hybrid / component boundary only)
is the scatter of those weights (bf16 values in fp32). ROUTER_IMPL=ops = the chain above (P.21).

Constants (weight, bias, the scatter's zeros for max_rows) are built once at load and sliced on the device per chunk.
Adapted from models/demos/xing40_a4b_d_p/tt/router.py:TtRouter (and mimo_v2_6_d_p/tt/router.py's ttnn.topk path).
"""

from __future__ import annotations

import torch

import ttnn

from . import mm_configs as mm
from .ccl import dealloc
from .common import DRAM, SP_AXIS, compute_config
from .cores import from_torch, sub_cores, sub_cores_tiled, to_rm
from .settings import settings

TILE = 32


def _replicate(mesh, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return from_torch(
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
        self.fused = settings.get("ROUTER_IMPL") == "fused"
        self.fp32_activation = settings.get("ROUTER_SOFTPLUS") == "fp32"
        if self.fused:  # (P.22) the fused gate takes a bias of the scores' shape: [max_rows, E], sliced per row count
            self.bias = _replicate(mesh, (b - b.mean()).expand(1, 1, self.max_rows, E), ttnn.float32)
            self._bias_rows = {self.max_rows: self.bias}
        else:
            self.bias = _replicate(mesh, b - b.mean(), ttnn.float32)
        self.zeros = _replicate(mesh, torch.zeros(1, 1, self.max_rows, E), ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.cfg = compute_config(mesh)
        self.dense = True  # the dense [S/SP, E] routing; tt/model.py turns it off when its only reader takes wts

    def _rows(self, t, s):
        if s == self.max_rows:
            return t
        return ttnn.slice(t, [0, 0, 0, 0], [1, 1, s, self.num_experts], sub_core_grids=sub_cores(t))

    def __call__(self, x: ttnn.Tensor):
        """x: [1, 1, S/SP, H] fp32 TILE per chip, row split over axis 0, replicated over axis 1; S/SP a multiple of 32.
        Returns (dense [1, 1, S/SP, E] fp32 ROW_MAJOR, idx [1, 1, S/SP, k] (topk's index dtype; fused: uint16),
        weights [1, 1, S/SP, k] fp32 TILE (fused: bf16)), same layout; dense None when self.dense is off."""
        s = x.shape[-2]
        assert s <= self.max_rows and s % TILE == 0, f"router rows {s} (max {self.max_rows}, tile aligned)"
        assert x.shape[-1] == self.hidden, (tuple(x.shape), self.hidden)
        logits = mm.linear(
            x, self.w, site="router", dtype=ttnn.float32, compute_kernel_config=self.cfg, memory_config=DRAM
        )
        if self.fused:
            return self._fused(logits, s)
        sp = ttnn.softplus(logits, beta=1.0, threshold=20.0, memory_config=DRAM)
        dealloc(logits)
        scores = ttnn.sqrt(sp, memory_config=DRAM, sub_core_grids=sub_cores(sp))
        dealloc(sp)
        choice = ttnn.add(scores, self.bias, dtype=ttnn.float32, memory_config=DRAM, sub_core_grids=sub_cores(scores))
        _, idx = ttnn.topk(choice, k=self.top_k, dim=-1, largest=True, sorted=True)
        dealloc(choice)
        top = ttnn.gather(scores, dim=-1, index=idx)
        dealloc(scores)
        tsum = ttnn.sum(top, dim=-1, keepdim=True, sub_core_grids=sub_cores(top))
        # (sum + 1e-20) / route_scale: the reference's eps, then the routed scale folded into the divisor
        t1 = ttnn.add(tsum, 1e-20, memory_config=DRAM, sub_core_grids=sub_cores(tsum))
        dealloc(tsum)
        tsum_s = ttnn.multiply(t1, 1.0 / self.route_scale, memory_config=DRAM, sub_core_grids=sub_cores(t1))
        dealloc(t1)
        wts = ttnn.div(top, tsum_s, memory_config=DRAM, sub_core_grids=sub_cores(top))
        dealloc(top)
        dealloc(tsum_s)
        if not self.dense:  # INDEXER_GLUE=on + weighted experts (P.16): they take wts itself; no dense scatter
            return None, idx, wts
        return self._scatter(idx, wts, s), idx, wts

    def _fused(self, logits, s: int):
        """ROUTER_IMPL=fused (P.22): one moe_grouped_topk after the gate matmul. Returns (dense or None, idx uint16,
        weights bf16), both [1, 1, s, k] TILE."""
        bias = self._bias_rows.get(s)
        if bias is None:  # (a row count other than the chunk's: sliced once and kept)
            bias = self._bias_rows[s] = self._rows(self.bias, s)
        wts, idx = ttnn.bringup.moe_grouped_topk(
            logits,
            bias,
            n_groups=1,
            summed_experts_per_group=1,
            topk_groups=1,
            n_activated_experts=self.top_k,
            route_scale=self.route_scale,
            epsilon=1e-20,
            stable_sort=True,
            score_func="sqrtsoftplus",
            fp32_selection=True,
            fp32_activation=self.fp32_activation,
            memory_config=DRAM,
        )
        dealloc(logits)
        if not self.dense:
            return None, idx, wts
        w32 = ttnn.typecast(
            wts, ttnn.float32, memory_config=DRAM, sub_core_grids=sub_cores_tiled(wts)
        )  # (exact: the scatter writes into fp32 zeros)
        dense = self._scatter(idx, w32, s)
        dealloc(w32)
        return dense, idx, wts

    def _scatter(self, idx, wts, s: int):
        """dense [1, 1, s, E] fp32 ROW_MAJOR = zeros with wts (fp32 TILE) at idx (K-4: fp32 scatter in ROW_MAJOR)."""
        zeros = self._rows(self.zeros, s)
        idx_rm = to_rm(idx, memory_config=DRAM)
        wts_rm = to_rm(wts, memory_config=DRAM)
        dense = ttnn.scatter(zeros, dim=-1, index=idx_rm, src=wts_rm, sub_core_grids=sub_cores(zeros))
        dealloc(idx_rm)
        dealloc(wts_rm)
        if zeros is not self.zeros:
            dealloc(zeros)
        return dense


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
