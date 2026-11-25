# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FFN sublayers on a TP stage.

* Dense SwiGLU MLP (layers 0-2): gate / up column parallel, down row parallel, one all-reduce.
* MoE (layers 3+): the router runs replicated (fp32 logits, sigmoid + correction-bias top-8, normalized, x2.5); the 288
  routed experts are expert parallel, ``288 / tp`` per chip as bfloat4_b, each chip keeping its slice of the routing
  weights; the shared expert is tensor parallel. Routed and shared partial sums leave through a single all-reduce.
"""

from __future__ import annotations

import torch

import ttnn
from models.experimental.glm53_flash.tt.mlp import swiglu_clamped
from models.experimental.glm53_flash.tt.moe import _sparse_matmul_config

from .common import DRAM, RM, TILE, TPContext, WeightLoader, to_tile

EXPERT_BLOCK = 256  # tokens per grouped sparse_matmul (bounds the [E_local, block, D] intermediates)
PADDED_EXPERTS = 320  # router width rounded to a tile multiple for top-k


def _gate_up_tp(gate, up, tp):
    """[out, in] gate / up -> [in, tp * (2 * out / tp)] with each chip's [gate_c | up_c] adjacent."""
    g, u = gate.t().chunk(tp, 1), up.t().chunk(tp, 1)
    return torch.cat([torch.cat([g[i], u[i]], 1) for i in range(tp)], 1)


class SwiGLUTP:
    """SwiGLU (clamped) with gate / up column parallel and down row parallel. ``forward`` returns the chip's partial
    sum (callers all-reduce)."""

    def __init__(self, ctx: TPContext, src, prefix: str, load: WeightLoader, limit: float, name: str):
        self.ctx = ctx
        self.limit = limit
        tp = ctx.tp
        self.gate_up = load(
            f"{name}.gate_up",
            lambda: _gate_up_tp(src[f"{prefix}.gate_proj.weight"], src[f"{prefix}.up_proj.weight"], tp),
            dtype=ttnn.bfloat8_b,
            shard_dim=-1,
        )
        self.down = load(
            f"{name}.down", lambda: src[f"{prefix}.down_proj.weight"].t(), dtype=ttnn.bfloat8_b, shard_dim=0
        )
        self.inter = self.gate_up.shape[-1] // 2

    def partial(self, x: ttnn.Tensor) -> ttnn.Tensor:
        ctx = self.ctx
        gu = ttnn.matmul(x, self.gate_up, dtype=ttnn.bfloat16, compute_kernel_config=ctx.hifi2, memory_config=DRAM)
        act = swiglu_clamped(gu, self.inter, self.limit)
        ttnn.deallocate(gu)
        y = ttnn.matmul(act, self.down, dtype=ttnn.bfloat16, compute_kernel_config=ctx.hifi2, memory_config=DRAM)
        ttnn.deallocate(act)
        return y

    def forward(self, x, valid=None):
        return self.ctx.all_reduce(self.partial(x))


class MoEEP:
    def __init__(self, ctx: TPContext, cfg, src, load: WeightLoader):
        self.ctx = ctx
        tp = ctx.tp
        self.E = cfg.n_routed_experts
        assert self.E % tp == 0
        self.E_local = self.E // tp
        self.top_k = cfg.num_experts_per_tok
        self.scaling = cfg.routed_scaling_factor
        self.inter = cfg.moe_intermediate_size
        self.hidden = cfg.hidden_size
        self.limit = cfg.swiglu_limit
        pad = PADDED_EXPERTS - self.E

        def router_w():
            w = src["mlp.gate.weight"].t()
            return torch.cat([w, torch.zeros(w.shape[0], pad, dtype=w.dtype)], 1)

        def router_b():
            b = src["mlp.gate.e_score_correction_bias"].float()
            return torch.cat([b, torch.full((pad,), -1e9)]).reshape(1, 1, 1, -1)

        self.router_w = load("router_w", router_w, dtype=ttnn.bfloat16)
        self.router_b = load("router_b", router_b, dtype=ttnn.float32)
        experts = src.experts
        self.gate_up = load(
            "experts.gate_up", lambda: experts()["gate_up"][None], dtype=ttnn.bfloat4_b, shard_dim=1
        )  # [1, E_local, D, 2I] per chip
        self.down = load("experts.down", lambda: experts()["down"][None], dtype=ttnn.bfloat4_b, shard_dim=1)
        self.shared = SwiGLUTP(ctx, src, "mlp.shared_experts", load, self.limit, "shared")
        self.grid = ctx.device.compute_with_storage_grid_size()

    def route(self, x: ttnn.Tensor, valid=None) -> ttnn.Tensor:
        """x [1, 1, T, D] bf16 -> this chip's routing weights [1, 1, T, E_local] fp32 (0 = not selected). Rows where
        ``valid`` ([1, 1, T, 1]) is 0 route nowhere, so padding rows activate no experts."""
        ctx = self.ctx
        logits = ttnn.matmul(x, self.router_w, dtype=ttnn.float32, compute_kernel_config=ctx.hifi4, memory_config=DRAM)
        scores = ttnn.sigmoid(logits)
        choice = ttnn.add(scores, self.router_b)
        top_vals, top_idx = ttnn.topk(choice, k=self.top_k, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(top_idx)
        t = x.shape[-2]
        kth = ttnn.slice(top_vals, (0, 0, 0, self.top_k - 1), (1, 1, t, self.top_k))
        w = ttnn.multiply(scores, ttnn.ge(choice, kth))
        w = ttnn.multiply(ttnn.div(w, ttnn.add(ttnn.sum(w, dim=-1, keepdim=True), 1e-20)), self.scaling)
        if valid is not None:
            w = ttnn.multiply(w, valid)
        w = ttnn.slice(w, (0, 0, 0, 0), (1, 1, t, self.E))
        w = ttnn.to_layout(w, RM, memory_config=DRAM)
        return to_tile(ctx.local_slice(w, dim=3))

    def _routed_block(self, x: ttnn.Tensor, w: ttnn.Tensor) -> ttnn.Tensor:
        """x [1, 1, C, D] bf16, w [1, 1, C, E_local] fp32 -> partial [1, 1, C, D] bf16 (this chip's experts)."""
        ctx = self.ctx
        c = x.shape[-2]
        e = self.E_local
        groups = c // TILE
        w_bf = ttnn.typecast(w, ttnn.bfloat16)
        tile_mask = ttnn.max(ttnn.reshape(w_bf, (1, groups, TILE, e)), dim=2, keepdim=True)  # [1, G, 1, E]
        sparsity = ttnn.reshape(ttnn.to_layout(tile_mask, RM), (1, 1, groups, e))
        expert_mask = ttnn.to_layout(ttnn.max(w_bf, dim=2, keepdim=True), RM)  # [1, 1, 1, E]
        gu = ttnn.sparse_matmul(
            ttnn.reshape(x, (1, groups, TILE, self.hidden)),
            self.gate_up,
            sparsity=sparsity,
            nnz=None,
            memory_config=DRAM,
            output_tile=ttnn.Tile([32, 32]),
            program_config=_sparse_matmul_config(TILE, 2 * self.inter, self.grid),
            compute_kernel_config=ctx.hifi2,
            dtype=ttnn.bfloat16,
        )
        gu = ttnn.reshape(gu, (groups, e, TILE, 2 * self.inter))
        gu = ttnn.reshape(ttnn.permute(gu, (1, 0, 2, 3)), (1, e, c, 2 * self.inter))
        act = swiglu_clamped(gu, self.inter, self.limit)
        ttnn.deallocate(gu)
        down = ttnn.sparse_matmul(
            act,
            self.down,
            sparsity=expert_mask,
            nnz=None,
            memory_config=DRAM,
            output_tile=ttnn.Tile([32, 32]),
            program_config=_sparse_matmul_config(c, self.hidden, self.grid),
            is_input_a_sparse=True,
            compute_kernel_config=ctx.hifi2,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(act)
        down = ttnn.reshape(down, (1, e, c, self.hidden))
        out = ttnn.multiply(down, ttnn.permute(w, (0, 3, 2, 1)), dtype=ttnn.float32)  # [1, E, C, D] * [1, E, C, 1]
        ttnn.deallocate(down)
        out = ttnn.experimental.fast_reduce_nc(out, dims=[1])
        return ttnn.typecast(ttnn.reshape(out, (1, 1, c, self.hidden)), ttnn.bfloat16)

    def forward(self, x: ttnn.Tensor, valid=None) -> ttnn.Tensor:
        t = x.shape[-2]
        w = self.route(x, valid)
        parts = []
        for s0 in range(0, t, EXPERT_BLOCK):
            s1 = min(t, s0 + EXPERT_BLOCK)
            xs = x if (s0, s1) == (0, t) else ttnn.slice(x, (0, 0, s0, 0), (1, 1, s1, self.hidden))
            ws = w if (s0, s1) == (0, t) else ttnn.slice(w, (0, 0, s0, 0), (1, 1, s1, self.E_local))
            parts.append(self._routed_block(xs, ws))
        routed = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)
        y = ttnn.add(routed, self.shared.partial(x))
        return self.ctx.all_reduce(y)
