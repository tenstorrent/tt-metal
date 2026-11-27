# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash MoE: noaux_tc sigmoid router, 288 routed experts (top-8) via ttnn.sparse_matmul, ungated shared expert.

Router (modeling:157-182), fp32 end to end:
    s      = sigmoid(x W_r^T)                      (bf16 weight, fp32 accumulate + fp32 output)
    choice = s + e_score_correction_bias           (selection only; n_group = topk_group = 1 is a no-op)
    sel    = top8(choice)                          -> dense mask choice >= 8th largest value
    w      = 2.5 * s[sel] / (sum s[sel] + 1e-20)   (UNBIASED scores, renormalised)
The expert axis is padded 288 -> 320 for the top-k (padded logits are 0 and their bias is -1e9, so they are never
selected); the dense routing weights [T, 288] carry w at the selected experts and 0 elsewhere.

Routed experts follow the qwen36 / gpt-oss sparse_matmul pattern (models/demos/blackhole/qwen36/tt/moe): gate|up in one
sparse matmul over bfloat4_b `[1, 288, 4096, 4096]` (columns [gate | up]), the clamped SwiGLU, the down sparse matmul
over `[1, 288, 2048, 4096]`, routing-weight scaling and a reduction over experts. Output = routed + shared(x).
"""

from __future__ import annotations

import math

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.mlp import SwiGLUMLP, swiglu_clamped
from models.autoports.zai_org_glm_5_3_flash.tt.precision import hifi2_fp32, hifi4_fp32

DRAM = ttnn.DRAM_MEMORY_CONFIG
TILE = 32
PADDED_EXPERTS = 320  # 288 rounded up to a tile multiple for top-k
PREFILL_EXPERT_CHUNK = 512  # tokens per grouped sparse_matmul in prefill (bounds the [E, chunk, H] down output)


def _sparse_matmul_config(m: int, n: int, grid, in0_block_w: int = 1):
    """1D mcast program config for sparse_matmul: largest divisor of the N tiles that fits the core grid."""
    n_tiles = math.ceil(n / TILE)
    best, best_x, best_y = 1, 1, 1
    for cores in range(1, min(grid.x * grid.y, n_tiles) + 1):
        if n_tiles % cores:
            continue
        for cy in range(1, grid.y + 1):
            if cores % cy == 0 and cores // cy <= grid.x:
                if cores > best:
                    best, best_x, best_y = cores, cores // cy, cy
                break
    per_core_n = n_tiles // best
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(best_x, best_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=max(TILE, m) // TILE,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


class Router:
    def __init__(
        self, mesh_device, weight_t: ttnn.Tensor, bias: torch.Tensor, *, num_experts: int, top_k: int, scaling: float
    ):
        """weight_t [4096, 288] bf16 (cache layout), bias [288] fp32 host tensor."""
        self.num_experts = num_experts
        self.top_k = top_k
        self.scaling = scaling
        pad = PADDED_EXPERTS - num_experts
        self.weight = ttnn.pad(weight_t, ((0, 0), (0, pad)), 0.0) if pad else weight_t
        bias_pad = torch.cat([bias.float(), torch.full((pad,), -1e9)]).reshape(1, 1, 1, PADDED_EXPERTS)
        self.bias = ttnn.from_torch(
            bias_pad, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=DRAM
        )
        self.ckc = hifi4_fp32(mesh_device)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [1, 1, T, 4096] bf16 -> dense routing weights [1, 1, T, 288] fp32 (0 for unselected experts)."""
        logits = ttnn.linear(x, self.weight, compute_kernel_config=self.ckc, dtype=ttnn.float32, memory_config=DRAM)
        scores = ttnn.sigmoid(logits)
        ttnn.deallocate(logits)
        choice = ttnn.add(scores, self.bias)
        top_vals, top_idx = ttnn.topk(choice, k=self.top_k, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(top_idx)
        t = top_vals.shape[-2]
        kth = ttnn.slice(top_vals, (0, 0, 0, self.top_k - 1), (1, 1, t, self.top_k))
        selected = ttnn.ge(choice, kth)
        weights = ttnn.multiply(scores, selected)
        denom = ttnn.add(ttnn.sum(weights, dim=-1, keepdim=True), 1e-20)
        weights = ttnn.multiply(ttnn.div(weights, denom), self.scaling)
        return ttnn.slice(weights, (0, 0, 0, 0), (1, 1, t, self.num_experts))


class MoE:
    def __init__(
        self,
        mesh_device,
        router: Router,
        gate_up: ttnn.Tensor,
        down: ttnn.Tensor,
        shared: SwiGLUMLP,
        *,
        intermediate: int,
        hidden: int,
        limit: float,
    ):
        """gate_up [1, E, 4096, 2I] bfloat4_b ([gate | up] columns), down [1, E, I, 4096] bfloat4_b."""
        self.device = mesh_device
        self.router = router
        self.gate_up = gate_up
        self.down = down
        self.shared = shared
        self.num_experts = router.num_experts
        self.intermediate = intermediate
        self.hidden = hidden
        self.limit = limit
        self.ckc = hifi2_fp32(mesh_device)
        grid = mesh_device.compute_with_storage_grid_size()
        self.grid = grid
        self.all_experts = ttnn.from_torch(
            torch.ones(1, 1, 1, self.num_experts),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=DRAM,
        )

    # ---------------------------------------------------------------------------------------------------------------
    def _routed_decode(self, x: ttnn.Tensor, routing: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,B,H] bf16, routing [1,1,B,E] -> [1,1,B,H]. The sparsity is the per-expert union over the B users."""
        b = x.shape[-2]
        e = self.num_experts
        routing_bf = ttnn.typecast(routing, ttnn.bfloat16)
        union = ttnn.max(routing_bf, dim=2, keepdim=True) if b > 1 else routing_bf
        sparsity = ttnn.to_layout(union, ttnn.ROW_MAJOR_LAYOUT)
        up_gate = ttnn.sparse_matmul(
            x,
            self.gate_up,
            sparsity=sparsity,
            nnz=None,
            memory_config=DRAM,
            output_tile=ttnn.Tile([32, 32]),
            program_config=_sparse_matmul_config(b, 2 * self.intermediate, self.grid),
            compute_kernel_config=self.ckc,
            dtype=ttnn.bfloat16,
        )
        # [1,1,1,E,B,2I] (expert-major) -> [1,E,B,2I]
        up_gate = ttnn.reshape(up_gate, (1, e, b, 2 * self.intermediate))
        act = swiglu_clamped(up_gate, self.intermediate, self.limit)
        ttnn.deallocate(up_gate)
        down = ttnn.sparse_matmul(
            act,
            self.down,
            sparsity=sparsity,
            nnz=None,
            memory_config=DRAM,
            output_tile=ttnn.Tile([32, 32]),
            program_config=_sparse_matmul_config(b, self.hidden, self.grid),
            is_input_a_sparse=True,
            compute_kernel_config=self.ckc,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(act)
        down = ttnn.reshape(down, (1, e, b, self.hidden))
        w = ttnn.permute(ttnn.reshape(routing, (1, 1, b, e)), (0, 3, 2, 1))  # [1, E, B, 1]
        out = ttnn.multiply(ttnn.typecast(down, ttnn.float32), w)
        ttnn.deallocate(down)
        out = ttnn.sum(out, dim=1, keepdim=True)  # [1, 1, B, H]
        return ttnn.typecast(out, ttnn.bfloat16)

    def _routed_prefill_chunk(self, x: ttnn.Tensor, routing: ttnn.Tensor) -> ttnn.Tensor:
        """x [1,1,C,H] bf16 (C multiple of 32), routing [1,1,C,E] fp32 -> [1,1,C,H] bf16."""
        c = x.shape[-2]
        e = self.num_experts
        groups = c // TILE
        x_grouped = ttnn.reshape(x, (1, groups, TILE, self.hidden))
        routing_bf = ttnn.typecast(routing, ttnn.bfloat16)
        tile_mask = ttnn.max(ttnn.reshape(routing_bf, (1, groups, TILE, e)), dim=2, keepdim=True)  # [1,G,1,E]
        sparsity = ttnn.reshape(ttnn.to_layout(tile_mask, ttnn.ROW_MAJOR_LAYOUT), (1, 1, groups, e))
        up_gate = ttnn.sparse_matmul(
            x_grouped,
            self.gate_up,
            sparsity=sparsity,
            nnz=None,
            memory_config=DRAM,
            output_tile=ttnn.Tile([32, 32]),
            program_config=_sparse_matmul_config(TILE, 2 * self.intermediate, self.grid),
            compute_kernel_config=self.ckc,
            dtype=ttnn.bfloat16,
        )
        # [1, G, 1, E, 32, 2I] -> [1, E, C, 2I]
        up_gate = ttnn.reshape(up_gate, (groups, e, TILE, 2 * self.intermediate))
        up_gate = ttnn.permute(up_gate, (1, 0, 2, 3))
        up_gate = ttnn.reshape(up_gate, (1, e, c, 2 * self.intermediate))
        act = swiglu_clamped(up_gate, self.intermediate, self.limit)
        ttnn.deallocate(up_gate)
        down = ttnn.sparse_matmul(
            act,
            self.down,
            sparsity=self.all_experts,
            nnz=e,
            memory_config=DRAM,
            output_tile=ttnn.Tile([32, 32]),
            program_config=_sparse_matmul_config(c, self.hidden, self.grid),
            is_input_a_sparse=True,
            compute_kernel_config=self.ckc,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(act)
        down = ttnn.reshape(down, (1, e, c, self.hidden))
        w = ttnn.permute(routing, (0, 3, 2, 1))  # [1, E, C, 1] fp32
        out = ttnn.multiply(down, w, dtype=ttnn.float32)  # fp32 routing weights, as in decode
        ttnn.deallocate(down)
        out = ttnn.experimental.fast_reduce_nc(out, dims=[1])
        return ttnn.typecast(ttnn.reshape(out, (1, 1, c, self.hidden)), ttnn.bfloat16)

    def __call__(self, x: ttnn.Tensor, *, decode: bool) -> ttnn.Tensor:
        """x [1, 1, T, 4096] bf16 -> MoE(x) [1, 1, T, 4096] bf16."""
        routing = self.router(x)
        t = x.shape[-2]
        if decode:
            routed = self._routed_decode(x, routing)
        elif t <= PREFILL_EXPERT_CHUNK:
            routed = self._routed_prefill_chunk(x, routing)
        else:
            parts = []
            for s0 in range(0, t, PREFILL_EXPERT_CHUNK):
                s1 = min(t, s0 + PREFILL_EXPERT_CHUNK)
                xs = ttnn.slice(x, (0, 0, s0, 0), (1, 1, s1, self.hidden))
                rs = ttnn.slice(routing, (0, 0, s0, 0), (1, 1, s1, self.num_experts))
                parts.append(self._routed_prefill_chunk(xs, rs))
            routed = ttnn.concat(parts, dim=2)
        shared = self.shared(x)
        return ttnn.add(routed, shared)
