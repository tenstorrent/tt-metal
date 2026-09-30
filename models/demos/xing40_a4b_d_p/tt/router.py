# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Xing4.0 MoE router (HF Xing4_0TopkRouter: sigmoid, noaux_tc, one group, top-4 of 64, norm_topk_prob,
routed_scaling_factor 2.0) on the 4x2 mesh.

Replicated weights: chip (r, c) routes its row's S/4 tokens of the gathered ffn_norm output ([1, 1, S/4, 3584],
replicated over axis 1), so both chips of a row compute the same routing and no collective runs.

    logits = x @ W^T                          fp32 weight [H, E], HiFi4 + fp32 dest, fp32 output
    scores = sigmoid(logits)                  fp32 (SFPU)
    choice = scores + (bias - mean(bias))     fp32 [1, E] broadcast; recentred at load (top-k is shift invariant)
    ti     = topk(choice, 4)                  fp32 keys, width 64
    tw     = gather(scores, ti) / ((sum + 1e-20) / 2.0)   unbiased sigmoid of the chosen 4, renormalized, x 2.0
    dense  = scatter(zeros fp32 RM, ti, tw)   [S/4, E] ROW_MAJOR (component boundary / dense-routing consumers)

Constants (weight, bias, the scatter's zeros for max_rows) are built once at load and sliced on the device per chunk.
From models/demos/glm53_flash_d_p/tt/router.py:TtRouter (route_scale folded into the renorm sum).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.xing40_a4b_d_p.tt.mhc import hifi4

TILE = 32


def _replicate(mesh, t: torch.Tensor, dtype) -> ttnn.Tensor:
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


class TtRouter:
    def __init__(
        self,
        mesh,
        weight: torch.Tensor,
        bias: torch.Tensor,
        max_rows: int,
        top_k: int = 4,
        route_scale: float = 2.0,
        recentre: bool = True,
        dense_dtype=ttnn.float32,
    ):
        """HF weights: gate.weight [E, H], e_score_correction_bias [E]. max_rows: the largest per-chip row count."""
        self.mesh, self.top_k, self.route_scale = mesh, int(top_k), float(route_scale)
        E, H = weight.shape
        assert E % TILE == 0, f"{E} experts not tile aligned"
        self.num_experts = E
        self.max_rows = -(-int(max_rows) // TILE) * TILE
        self.w = _replicate(mesh, weight.float().T.contiguous().reshape(1, 1, H, E), ttnn.float32)
        b = bias.float().reshape(1, 1, 1, E)
        if recentre:
            b = b - b.mean()
        self.bias = _replicate(mesh, b, ttnn.float32)
        self.dense_dtype = dense_dtype
        # ttnn.scatter rejects fp32 TILE inputs (scatter.cpp), so the fp32 dense matrix is scattered in ROW_MAJOR.
        self.dense_layout = ttnn.ROW_MAJOR_LAYOUT if dense_dtype == ttnn.float32 else ttnn.TILE_LAYOUT
        self.zeros = ttnn.from_torch(
            torch.zeros(1, 1, self.max_rows, E),
            dtype=dense_dtype,
            layout=self.dense_layout,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.cfg = hifi4(mesh)

    def _rows(self, t, s):
        if s == self.max_rows:
            return t
        return ttnn.slice(t, [0, 0, 0, 0], [1, 1, s, self.num_experts])

    def __call__(self, x: ttnn.Tensor):
        """x: [1, 1, S/4, H] TILE per chip (bf16 or fp32), row split, replicated over axis 1; S/4 a multiple of 32.
        Returns (dense [1, 1, S/4, E] fp32 ROW_MAJOR, idx [1, 1, S/4, k] uint16, weights [1, 1, S/4, k] fp32), same layout.
        """
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
        # (sum + 1e-20) / route_scale: the reference's eps, then the routed scale folded into the divisor.
        tsum_s = ttnn.multiply(ttnn.add(tsum, 1e-20, memory_config=mc), 1.0 / self.route_scale, memory_config=mc)
        ttnn.deallocate(tsum)
        wts = ttnn.div(top, tsum_s, memory_config=mc)
        ttnn.deallocate(top)
        ttnn.deallocate(tsum_s)
        src = wts if self.dense_dtype == ttnn.float32 else ttnn.typecast(wts, self.dense_dtype, memory_config=mc)
        zeros = self._rows(self.zeros, s)
        if self.dense_layout == ttnn.ROW_MAJOR_LAYOUT:
            idx_s = ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
            src_s = ttnn.to_layout(src, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
        else:
            idx_s, src_s = idx, src
        dense = ttnn.scatter(zeros, dim=-1, index=idx_s, src=src_s)
        if idx_s is not idx:
            ttnn.deallocate(idx_s)
            ttnn.deallocate(src_s)
        if zeros is not self.zeros:
            ttnn.deallocate(zeros)
        if src is not wts:
            ttnn.deallocate(src)
        return dense, idx, wts


def build_router(mesh, loader, cfg, layer: int, max_rows: int) -> TtRouter:
    p = f"model.layers.{layer}.mlp.gate."
    assert cfg.norm_topk_prob, "router assumes norm_topk_prob"
    return TtRouter(
        mesh,
        loader.get(p + "weight"),
        loader.get(p + "e_score_correction_bias"),
        max_rows,
        top_k=cfg.num_experts_per_tok,
        route_scale=cfg.routed_scaling_factor,
    )
