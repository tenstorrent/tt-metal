# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo routed MoE (256 experts, top-8, sigmoid ``noaux_tc``), tensor-parallel over one Galaxy row.

Routing runs replicated on every chip with ``moe_grouped_topk`` (single-group path: fp32 sigmoid,
+bias for selection only, gather the unbiased scores, normalize with eps 1e-20), then the top-8
weights are scattered into a dense ``[T, 256]`` routing matrix.

Experts are TP-sharded on the intermediate dim: each chip holds every expert's 256-wide slice of
``moe_intermediate_size`` (2048). Tokens are grouped 32 per tile row; ``sparse_matmul`` computes
an expert for a group only if some token in the group routes to it (the union mask). Rows of
tokens that did not pick that expert get routing weight 0. The weighted sum over experts is a
partial sum per chip and is all-reduced across the row.
"""

import torch

import ttnn

from .common import TILE, as_weight, cache_name, compute_config, mcast_1d_config


class TtMiMoMoE:
    PREFILL_TOKENS_PER_CALL = 128  # bounds the [groups, 256, 32, hidden] expert-output buffer

    def __init__(self, plan, cfg, layer_idx, router=None, experts=None, cache_dir=None, expert_dtype=ttnn.bfloat8_b):
        self.plan = plan
        self.cfg = cfg
        self.layer_idx = layer_idx
        tp = plan.tp
        self.num_experts = cfg.n_routed_experts
        self.hidden = cfg.hidden_size
        assert cfg.moe_intermediate_size % (tp * TILE) == 0
        self.inter_local = cfg.moe_intermediate_size // tp
        prefix = f"layer{layer_idx}_moe_tp{tp}"

        w_router = bias = w_gate = w_up = w_down = None
        if router is not None:
            w_router = router["weight"].T.contiguous().to(torch.bfloat16)  # [hidden, E]
            bias = router["e_score_correction_bias"].float().reshape(1, 1, 1, -1)
        if experts is not None:
            # gate/up [E, I, H] -> [1, E, H, I]: TP shards I. down [E, H, I] -> [1, E, I, H]: TP shards I.
            w_gate = experts["gate_proj"].transpose(1, 2).unsqueeze(0)
            w_up = experts["up_proj"].transpose(1, 2).unsqueeze(0)
            w_down = experts["down_proj"].transpose(1, 2).unsqueeze(0)

        rep = plan.replicate()
        self.w_router = as_weight(w_router, plan, rep, ttnn.bfloat16, cache_name(cache_dir, prefix, "router"))
        self.bias_row = as_weight(
            bias, plan, rep, ttnn.float32, cache_name(cache_dir, prefix, "score_bias"), layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.w_gate = as_weight(w_gate, plan, plan.shard(-1), expert_dtype, cache_name(cache_dir, prefix, "w_gate"))
        self.w_up = as_weight(w_up, plan, plan.shard(-1), expert_dtype, cache_name(cache_dir, prefix, "w_up"))
        self.w_down = as_weight(w_down, plan, plan.shard(-2), expert_dtype, cache_name(cache_dir, prefix, "w_down"))
        self._bias_wide = {}

        self.router_config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True)
        self.expert_config = compute_config(ttnn.MathFidelity.HiFi2, fp32_acc=True)
        self.gate_up_program = mcast_1d_config((4, 2), TILE, self.inter_local, self.hidden, in0_block_w=4)
        self.down_program = mcast_1d_config((8, 4), TILE, self.hidden, self.inter_local, in0_block_w=4)

    def _bias_for(self, num_tokens):
        """moe_grouped_topk needs the bias at the same logical shape as the scores: [1, 1, T, E]."""
        if num_tokens not in self._bias_wide:
            wide = ttnn.repeat(self.bias_row, ttnn.Shape((1, 1, num_tokens, 1)))
            self._bias_wide[num_tokens] = ttnn.to_layout(wide, ttnn.TILE_LAYOUT)
            wide.deallocate(True)
        return self._bias_wide[num_tokens]

    def route(self, x, token_mask=None):
        """x [1, 1, T, hidden] -> dense routing weights [1, 1, T, E] (bf16; zero off the top-8)."""
        num_tokens = x.shape[-2]
        logits = ttnn.linear(
            x,
            self.w_router,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.router_config,
        )
        weights, indices = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            logits,
            self._bias_for(num_tokens),
            n_groups=1,
            summed_experts_per_group=1,
            topk_groups=1,
            n_activated_experts=self.cfg.num_experts_per_tok,
            route_scale=self.cfg.routed_scaling_factor,
            epsilon=1e-20,
        )
        zeros = ttnn.zeros(
            ttnn.Shape((1, 1, num_tokens, self.num_experts)),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.plan.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        dense = ttnn.scatter(zeros, dim=-1, index=indices, src=weights)
        for t in (logits, weights, indices, zeros):
            t.deallocate(True)
        if token_mask is not None:
            masked = ttnn.mul(dense, token_mask)
            dense.deallocate(True)
            dense = masked
        return dense

    def _experts(self, x, dense):
        """x [1, 1, T, hidden], dense [1, 1, T, E] with T a multiple of 32 -> partial [1, 1, T, hidden]."""
        num_tokens = x.shape[-2]
        groups = num_tokens // TILE
        E, H, I = self.num_experts, self.hidden, self.inter_local

        union = ttnn.max(ttnn.reshape(dense, (1, groups, TILE, E)), dim=2, keepdim=True)  # [1, G, 1, E]
        sparsity = ttnn.to_layout(union, ttnn.ROW_MAJOR_LAYOUT)
        union.deallocate(True)

        x4 = ttnn.reshape(x, (1, groups, TILE, H))
        out_tile = ttnn.Tile([TILE, TILE])
        mm_kwargs = dict(
            sparsity=sparsity,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=out_tile,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.expert_config,
        )
        gate = ttnn.sparse_matmul(x4, self.w_gate, program_config=self.gate_up_program, **mm_kwargs)
        up = ttnn.sparse_matmul(x4, self.w_up, program_config=self.gate_up_program, **mm_kwargs)
        gate = ttnn.reshape(gate, (groups, E, TILE, I))
        up = ttnn.reshape(up, (groups, E, TILE, I))
        act = ttnn.mul(
            gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gate.deallocate(True)
        up.deallocate(True)

        down_sparsity = ttnn.reshape(sparsity, (1, 1, groups, E))
        down = ttnn.sparse_matmul(
            act,
            self.w_down,
            sparsity=down_sparsity,
            nnz=None,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=out_tile,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.expert_config,
            program_config=self.down_program,
        )  # [G, E, 32, H]
        act.deallocate(True)
        sparsity.deallocate(True)

        rw = ttnn.transpose(ttnn.reshape(dense, (groups, TILE, E)), -2, -1)  # [G, E, 32]
        rw = ttnn.reshape(rw, (groups, E, TILE, 1))
        weighted = ttnn.mul(down, rw, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        down.deallocate(True)
        rw.deallocate(True)
        out = ttnn.sum(weighted, dim=1, keepdim=True)  # [G, 1, 32, H]
        weighted.deallocate(True)
        return ttnn.reshape(out, (1, 1, num_tokens, H))

    def forward(self, x, token_mask=None):
        """x [1, 1, T, hidden] replicated across the row -> [1, 1, T, hidden] replicated."""
        num_tokens = x.shape[-2]
        assert num_tokens % TILE == 0, "pad tokens to a multiple of 32"
        dense = self.route(x, token_mask)
        step = self.PREFILL_TOKENS_PER_CALL
        if num_tokens <= step:
            out = self._experts(x, dense)
        else:
            parts = []
            for s in range(0, num_tokens, step):
                e = min(s + step, num_tokens)
                xs = ttnn.slice(x, (0, 0, s, 0), (1, 1, e, self.hidden))
                ds = ttnn.slice(dense, (0, 0, s, 0), (1, 1, e, self.num_experts))
                parts.append(self._experts(xs, ds))
                xs.deallocate(True)
                ds.deallocate(True)
            out = ttnn.concat(parts, dim=2)
            for p in parts:
                p.deallocate(True)
        dense.deallocate(True)
        return self.plan.all_reduce(out)
