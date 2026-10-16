# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo routed MoE (256 experts, top-8, sigmoid ``noaux_tc``), tensor-parallel over one Galaxy row.

Routing runs replicated on every chip, in fp32 end to end: router logits (HiFi4, fp32 out), sigmoid,
+bias for selection only, fp32 top-8, gather the unbiased scores, normalize with eps 1e-20, then
scatter into a dense ``[T, 256]`` routing matrix. MiMo's biased scores sit in [0.5, 1.5] with a
median 8th-vs-9th gap of ~1.6e-3, so a bf16 top-k flips experts: ``moe_grouped_topk`` fed exact
fp32 logits matched the reference top-8 set on only 80% of real layer-5 tokens, this path on 100%.

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

        w_router = bias = w_gate_up = w_down = None
        if router is not None:
            w_router = router["weight"].T.contiguous().to(torch.bfloat16)  # [hidden, E]
            bias = router["e_score_correction_bias"].float().reshape(1, 1, 1, -1)
        if experts is not None:
            # gate/up [E, I, H] -> one [1, E, H, tp * 2 * I_local] tensor: chip c's slice holds
            # [gate_c | up_c], so one sparse_matmul makes both (twice the N width, twice the cores).
            # down [E, H, I] -> [1, E, I, H]: TP shards I.
            g = experts["gate_proj"].transpose(1, 2)
            u = experts["up_proj"].transpose(1, 2)
            il = self.inter_local
            w_gate_up = torch.cat(
                [torch.cat([g[..., c * il : (c + 1) * il], u[..., c * il : (c + 1) * il]], dim=-1) for c in range(tp)],
                dim=-1,
            ).unsqueeze(0)
            del g, u
            w_down = experts["down_proj"].transpose(1, 2).unsqueeze(0)

        rep = plan.replicate()
        self.w_router = as_weight(w_router, plan, rep, ttnn.bfloat16, cache_name(cache_dir, prefix, "router"))
        self.score_bias = as_weight(bias, plan, rep, ttnn.float32, cache_name(cache_dir, prefix, "score_bias"))
        self.w_gate_up = as_weight(
            w_gate_up, plan, plan.shard(-1), expert_dtype, cache_name(cache_dir, prefix, "w_gate_up")
        )
        del w_gate_up
        self.w_down = as_weight(w_down, plan, plan.shard(-2), expert_dtype, cache_name(cache_dir, prefix, "w_down"))

        self._zero_cache = {}
        self.router_config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True)
        self.expert_config = compute_config(ttnn.MathFidelity.HiFi2, fp32_acc=True)
        # Measured on BH (decode, 32 experts): wider K blocks cut the per-expert handshakes; down over 8x8 cores.
        self.gate_up_program = mcast_1d_config((8, 2), TILE, 2 * self.inter_local, self.hidden, in0_block_w=64)
        self.down_program = mcast_1d_config((8, 8), TILE, self.hidden, self.inter_local, in0_block_w=8)
        self.decode_users = None

    def _zeros(self, num_tokens):
        """Cached [1, 1, T, E] zeros. ``ttnn.zeros`` writes from the host, which a trace capture cannot do,
        so the tensor is made on first use (warm-up) and reused."""
        if num_tokens not in self._zero_cache:
            self._zero_cache[num_tokens] = ttnn.zeros(
                ttnn.Shape((1, 1, num_tokens, self.num_experts)),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.plan.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._zero_cache[num_tokens]

    def _topk(self, logits):
        """fp32 router logits [1, 1, T, E] -> (top-k indices uint32, normalized unbiased weights fp32), [1, 1, T, k]."""
        scores = ttnn.sigmoid(logits)
        biased = ttnn.add(scores, self.score_bias)
        _, indices = ttnn.topk(biased, k=self.cfg.num_experts_per_tok, dim=-1, largest=True, sorted=True)
        biased.deallocate(True)
        weights = ttnn.gather(scores, dim=-1, index=indices)
        scores.deallocate(True)
        if self.cfg.norm_topk_prob:
            denom = ttnn.add(ttnn.sum(weights, dim=-1, keepdim=True), 1e-20)
            normed = ttnn.div(weights, denom)
            weights.deallocate(True)
            denom.deallocate(True)
            weights = normed
        if self.cfg.routed_scaling_factor != 1.0:
            weights = ttnn.mul(weights, self.cfg.routed_scaling_factor)
        return indices, weights

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
        indices, weights = self._topk(logits)
        logits.deallocate(True)
        weights_bf16 = ttnn.typecast(weights, ttnn.bfloat16)
        weights.deallocate(True)
        zeros = self._zeros(num_tokens)  # scatter is out of place, so one cached zero tensor per T is safe
        dense = ttnn.scatter(zeros, dim=-1, index=indices, src=weights_bf16)
        for t in (weights_bf16, indices):
            t.deallocate(True)
        if token_mask is not None:
            masked = ttnn.mul(dense, token_mask)
            dense.deallocate(True)
            dense = masked
        return dense

    def _swiglu(self, gate_up):
        """[..., 2 * I] (gate | up) -> silu(gate) * up, [..., I]."""
        shape = list(gate_up.shape)
        I = self.inter_local
        gate = ttnn.slice(gate_up, [0] * len(shape), shape[:-1] + [I])
        up = ttnn.slice(gate_up, [0] * (len(shape) - 1) + [I], shape)
        gate_up.deallocate(True)
        act = ttnn.mul(
            gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gate.deallocate(True)
        up.deallocate(True)
        return act

    def set_decode_users(self, num_users):
        """Decode serves users in slots [0, num_users): their (user, top-k) pairs are the experts computed."""
        self.decode_users = num_users
        pairs = num_users * self.cfg.num_experts_per_tok
        # pair j = user * k + slot -> one-hot row of its token in the 32-row tile: [1, pairs, 32, 1]
        onehot = torch.zeros(1, pairs, TILE, 1)
        onehot[0, torch.arange(pairs), torch.arange(pairs) // self.cfg.num_experts_per_tok, 0] = 1
        rep = self.plan.replicate()
        self.pair_onehot = ttnn.from_torch(
            onehot, device=self.plan.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep
        )
        # indexed sparse_matmul still takes a sparsity operand sized to the expert axis; it is not read
        self.all_experts = ttnn.from_torch(
            torch.ones(1, 1, 1, self.num_experts),
            device=self.plan.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=rep,
        )

    def forward_decode(self, x, token_mask):
        """Decode: x [1, 1, 32, hidden], users in rows [0, decode_users). Computes only the selected experts.

        Each (user, k) pair is one entry of an indexed sparse_matmul over the 256 experts: gate|up for the
        pair's expert on all 32 rows, the routing weight applied to the pair's own row (the other rows are
        zeroed), down for the same expert, then a sum over pairs. Nothing is materialized per expert."""
        U, k = self.decode_users, self.cfg.num_experts_per_tok
        pairs = U * k
        H, I = self.hidden, self.inter_local
        logits = ttnn.linear(
            x,
            self.w_router,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.router_config,
        )
        indices, weights = self._topk(logits)
        logits.deallocate(True)

        idx_u = ttnn.slice(indices, (0, 0, 0, 0), (1, 1, U, k))
        # reshape does not take uint16, so flatten while uint32, then narrow to the uint16 stick sparse_matmul wants
        ids = ttnn.typecast(ttnn.reshape(ttnn.to_layout(idx_u, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 1, pairs)), ttnn.uint16)
        w_u = ttnn.slice(weights, (0, 0, 0, 0), (1, 1, U, k))
        mask_u = ttnn.slice(token_mask, (0, 0, 0, 0), (1, 1, U, 1))
        w_u = ttnn.mul(w_u, ttnn.typecast(mask_u, ttnn.float32))
        w_col = ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(ttnn.typecast(w_u, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT), (1, pairs, 1, 1)),
            ttnn.TILE_LAYOUT,
        )
        row_weight = ttnn.mul(self.pair_onehot, w_col)  # [1, pairs, 32, 1]
        for t in (indices, weights, idx_u, w_u, mask_u, w_col):
            t.deallocate(True)

        out_tile = ttnn.Tile([TILE, TILE])
        gate_up = ttnn.sparse_matmul(
            x,
            self.w_gate_up,
            sparsity=self.all_experts,
            indices=ids,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=out_tile,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.expert_config,
            program_config=self.gate_up_program,
        )  # [1, 1, 1, pairs, 32, 2I]
        act = self._swiglu(ttnn.reshape(gate_up, (1, pairs, TILE, 2 * I)))
        act_w = ttnn.mul(act, row_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        act.deallocate(True)
        row_weight.deallocate(True)
        down = ttnn.sparse_matmul(
            act_w,
            self.w_down,
            sparsity=self.all_experts,
            indices=ids,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=out_tile,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.expert_config,
            program_config=self.down_program,
        )  # [1, pairs, 32, H]
        act_w.deallocate(True)
        ids.deallocate(True)
        out = ttnn.sum(down, dim=1, keepdim=True)  # [1, 1, 32, H]
        down.deallocate(True)
        return self.plan.all_reduce(out)

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
        gate_up = ttnn.sparse_matmul(x4, self.w_gate_up, program_config=self.gate_up_program, **mm_kwargs)
        act = self._swiglu(ttnn.reshape(gate_up, (groups, E, TILE, 2 * I)))

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
