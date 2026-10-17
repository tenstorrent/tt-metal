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

from .common import TILE, DramShardedMatmul, as_weight, cache_exists, cache_name, compute_config, mcast_1d_config
from .moe_stream import (
    DownStream,
    GateUpStream,
    PendingCompact,
    ReduceExpand,
    fabric_row_gather,
    fabric_transport_noc,
    is_dram_interleaved,
)
from .proj_stream import RouterScoreStream


class TtMiMoMoE:
    PREFILL_TOKENS_PER_CALL = 128  # bounds the [groups, 256, 32, hidden] expert-output buffer

    def __init__(self, plan, cfg, layer_idx, router=None, experts=None, cache_dir=None, expert_dtype=ttnn.bfloat8_b):
        self.plan = plan
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.expert_dtype = expert_dtype
        tp = plan.tp
        self.num_experts = cfg.n_routed_experts
        self.hidden = cfg.hidden_size
        assert cfg.moe_intermediate_size % (tp * TILE) == 0
        self.inter_local = cfg.moe_intermediate_size // tp
        prefix = f"layer{layer_idx}_moe_tp{tp}"

        self.router_config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True)
        # Router logits in fp32 from a DRAM-sharded bf16 weight (decode: 50 -> 11 us on BH).
        self.router_mm = DramShardedMatmul(
            plan, self.hidden, self.num_experts, out_dtype=ttnn.float32, compute_kernel_config=self.router_config
        )
        expert_cache = [cache_name(cache_dir, prefix, n) for n in ("w_gate_up", "w_down")]
        if all(cache_exists(f, expert_dtype) for f in expert_cache):
            experts = None  # already converted: skip dequantizing 256 experts on the host
        w_router = bias = w_gate_up = w_down = None
        if router is not None:
            w_router = router["weight"].T.contiguous().to(torch.bfloat16)  # [hidden, E], replicated
            w_router = self.router_mm.pad_chip_blocks(w_router, 1)
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
        self.w_router = as_weight(
            w_router,
            plan,
            rep,
            ttnn.bfloat16,
            cache_name(cache_dir, prefix, "router_ds"),
            memory_config=self.router_mm.weight_memory_config,
        )
        self.score_bias = as_weight(bias, plan, rep, ttnn.float32, cache_name(cache_dir, prefix, "score_bias"))
        self.w_gate_up = as_weight(
            w_gate_up, plan, plan.shard(-1), expert_dtype, cache_name(cache_dir, prefix, "w_gate_up")
        )
        del w_gate_up
        self.w_down = as_weight(w_down, plan, plan.shard(-2), expert_dtype, cache_name(cache_dir, prefix, "w_down"))

        self._zero_cache = {}
        self.expert_config = compute_config(ttnn.MathFidelity.HiFi2, fp32_acc=True)
        # Measured on BH (decode, 32 experts): wider K blocks cut the per-expert handshakes; down over 8x8 cores.
        self.gate_up_program = mcast_1d_config((8, 2), TILE, 2 * self.inter_local, self.hidden, in0_block_w=64)
        self.down_program = mcast_1d_config((8, 8), TILE, self.hidden, self.inter_local, in0_block_w=8)
        self.decode_users = None
        self.gate_up_stream = None
        self.router_scores = None
        self.fabric_gather = False  # set by the model: the block reductions all-gather over the fabric
        self.down_stream = None
        self.ids_buf = None

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

    def _router_logits(self, x):
        """fp32 logits [1, 1, T, E]: DRAM-sharded matmul for one tile row, else linear on an interleaved copy."""
        if x.shape[-2] == TILE:
            return self.router_mm(x, self.w_router)
        weight = ttnn.to_memory_config(self.w_router, ttnn.DRAM_MEMORY_CONFIG)
        logits = ttnn.linear(
            x,
            weight,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.router_config,
        )
        weight.deallocate(True)
        return logits

    def route(self, x, token_mask=None):
        """x [1, 1, T, hidden] -> dense routing weights [1, 1, T, E] (bf16; zero off the top-8)."""
        num_tokens = x.shape[-2]
        logits = self._router_logits(x)
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

    def _swiglu(self, gate_up, memory_config=None):
        """[..., 2 * I] (gate | up) -> silu(gate) * up, [..., I]."""
        mem = memory_config or ttnn.DRAM_MEMORY_CONFIG
        shape = list(gate_up.shape)
        I = self.inter_local
        gate = ttnn.slice(gate_up, [0] * len(shape), shape[:-1] + [I], memory_config=mem)
        up = ttnn.slice(gate_up, [0] * (len(shape) - 1) + [I], shape, memory_config=mem)
        gate_up.deallocate(True)
        act = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=mem)
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
        if self.expert_dtype == ttnn.bfloat4_b and num_users <= GateUpStream.MAX_USERS:
            # BFP4 decode: top-k routing, gate|up, SwiGLU and the routing weight in one bank-local DRAM-streaming
            # program. It writes only row u of pair (u, k) into this persistent zeroed activation (the other rows
            # stay zero) and the pairs' uint16 expert ids into ids_buf for the down sparse_matmul.
            self.gate_up_stream = GateUpStream(
                self.plan,
                self.hidden,
                self.inter_local,
                pairs,
                self.cfg.num_experts_per_tok,
                self.num_experts,
                norm_topk_prob=self.cfg.norm_topk_prob,
                routed_scaling_factor=self.cfg.routed_scaling_factor,
            )
            # Same row-major uint16 [1, 1, 1, pairs] stick the routing glue used to build with typecast.
            ids32 = ttnn.from_torch(
                torch.zeros(1, 1, 1, pairs, dtype=torch.int32),
                device=self.plan.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )
            self.ids_buf = ttnn.typecast(ids32, ttnn.uint16)
            ids32.deallocate(True)
            # Down + sum over pairs, bank-local from the interleaved w_down, into a persistent compact
            # [1, 1, 32, hidden / 8] output holding only the user rows; one row all-gather of it, then ReduceExpand
            # sums the chips' copies and restores [1, 1, 32, hidden] with zero padding rows.
            self.down_stream = DownStream(self.plan, self.hidden, self.inter_local, pairs)
            self.act_buf = self.down_stream.activation_buffer(self.plan)
            self.out_buf = self.down_stream.compact_output(self.plan)
            self.reduce_expand = ReduceExpand(
                self.plan, self.hidden, self.plan.tp, self.down_stream.compute_config, DownStream.TINY_ROWS
            )
            # Router logits, sigmoid and the bias add in one bank-local launch (scores and biased are persistent).
            try:
                bias = ttnn.to_torch(ttnn.get_device_tensors(self.score_bias)[0]).float().reshape(-1)
                self.router_scores = RouterScoreStream(self.plan, self.hidden, self.num_experts, bias)
            except AssertionError:
                self.router_scores = None

    def reduce_residual(self, gathered, residual=None, norm=None, compact=None):
        """Row all-gather of the chips' compact [1, 1, 32, H / 8] rows -> a new [1, 1, 32, H]: their chip-ordered sum
        in rows 0-3 and zeros elsewhere, or with ``residual`` the residual plus that (bf16-rounded) sum.
        ``norm`` = (compact gamma, persistent output, eps): returns (sum, RMSNorm(sum) * gamma in that output).
        ``compact`` (with ``norm``, in place of ``gathered``): this chip's compact rows, all-gathered over the fabric
        inside the reduce launch (FabricRowGather)."""
        full = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, TILE, self.hidden]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.plan.mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        res = residual
        if residual is not None and not is_dram_interleaved(residual):
            res = ttnn.to_memory_config(residual, ttnn.DRAM_MEMORY_CONFIG)
        if compact is not None:
            fabric = fabric_row_gather(self.plan, self.reduce_expand)
            normed = self.reduce_expand.call_fabric(compact, full, res, norm, fabric)
        else:
            normed = self.reduce_expand(gathered, full, residual=res, norm=norm)
        if res is not residual:
            res.deallocate(True)
        return normed if norm is not None else full

    def _forward_decode_stream(self, x, token_mask, residual=None, norm=None):
        """BFP4 decode: fp32 router logits, sigmoid and +bias, then one streaming program that selects every user's
        fp32 top-k, normalizes and masks the weights and computes gate|up + SwiGLU + routing weight; then the
        streaming down program (down and the sum over each user's pairs) on the expert ids that program wrote."""
        fused_router = self.router_scores is not None and self.router_scores.accepts(x, self.w_router)
        if fused_router:
            scores, biased = self.router_scores(x, self.w_router)  # persistent
        else:
            logits = self._router_logits(x)
            scores = ttnn.sigmoid(logits)
            logits.deallocate(True)
            biased = ttnn.add(scores, self.score_bias)

        def dram(t):
            if t is None or is_dram_interleaved(t):
                return t
            return ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)

        srcs = (x, scores, biased, token_mask)
        x_in, s_in, b_in, m_in = (dram(t) for t in srcs)
        act = self.gate_up_stream(x_in, s_in, b_in, m_in, self.w_gate_up, self.act_buf, self.ids_buf)
        for t, src in zip((x_in, s_in, b_in, m_in), srcs):
            if t is not src:
                t.deallocate(True)
        if not fused_router:
            scores.deallocate(True)
            biased.deallocate(True)

        plan = self.plan
        if norm is not None and self.fabric_gather:
            # Down runs in the same launch as the all-gather and the reduce (it signals the transport cores).
            section = self.down_stream(
                act,
                self.ids_buf,
                self.w_down,
                self.out_buf,
                signal=fabric_transport_noc(plan.mesh_device),
                launch=False,
            )
            pending = PendingCompact(section, self.out_buf, len(self.down_stream.cores))
            return self.reduce_residual(None, residual, norm=norm, compact=pending)
        out = self.down_stream(act, self.ids_buf, self.w_down, self.out_buf)  # compact [1, 1, 32, H / 8], persistent
        # The row all-reduce as one fabric phase: gather every chip's compact output (chip order along the row, the
        # same on every chip), then ReduceExpand sums the copies in that order. The persistent compact output is
        # rewritten every call, so it is not deallocated.
        gathered = plan.all_gather(out, dim=3)  # [1, 1, 32, tp * H / 8]
        full = self.reduce_residual(gathered, residual, norm=norm)
        if gathered is not self.out_buf:
            gathered.deallocate(True)
        return full

    def forward_decode(self, x, token_mask, residual=None, norm=None):
        """Decode: x [1, 1, 32, hidden], users in rows [0, decode_users). Computes only the selected experts.
        ``residual``: returns residual + the MoE output (fused into the streaming path's reduction); ``norm`` (streaming
        path with ``residual``): returns (that, its RMSNorm), see ``reduce_residual``.

        Each (user, k) pair is one entry of an indexed sparse_matmul over the 256 experts: gate|up for the
        pair's expert on all 32 rows, the routing weight applied to the pair's own row (the other rows are
        zeroed), down for the same expert, then a sum over pairs. Nothing is materialized per expert."""
        if self.gate_up_stream is not None:
            return self._forward_decode_stream(x, token_mask, residual, norm)
        assert norm is None
        U, k = self.decode_users, self.cfg.num_experts_per_tok
        pairs = U * k
        H, I = self.hidden, self.inter_local
        # Per-pair intermediates are small (<= 64 KiB per core interleaved): keep them in L1, off the DRAM
        # that the expert weights stream from.
        L1 = ttnn.L1_MEMORY_CONFIG
        logits = self._router_logits(x)
        indices, weights = self._topk(logits)
        logits.deallocate(True)

        idx_u = ttnn.slice(indices, (0, 0, 0, 0), (1, 1, U, k))
        # reshape does not take uint16, so flatten while uint32, then narrow to the uint16 stick sparse_matmul wants
        ids = ttnn.typecast(ttnn.reshape(ttnn.to_layout(idx_u, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 1, pairs)), ttnn.uint16)
        # The mask is 0/1, so rounding to bf16 before masking is bit-identical to masking in fp32 first;
        # masking the whole tile with a broadcast mul avoids slicing and typecasting the mask.
        w_bf16 = ttnn.typecast(weights, ttnn.bfloat16)
        w_masked = ttnn.mul(w_bf16, token_mask)
        w_u = ttnn.slice(w_masked, (0, 0, 0, 0), (1, 1, U, k))
        w_col = ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(w_u, ttnn.ROW_MAJOR_LAYOUT), (1, pairs, 1, 1)),
            ttnn.TILE_LAYOUT,
        )
        row_weight = ttnn.mul(self.pair_onehot, w_col, memory_config=L1)  # [1, pairs, 32, 1]
        for t in (indices, weights, idx_u, w_bf16, w_masked, w_u, w_col):
            t.deallocate(True)

        out_tile = ttnn.Tile([TILE, TILE])
        gate_up = ttnn.sparse_matmul(
            x,
            self.w_gate_up,
            sparsity=self.all_experts,
            indices=ids,
            memory_config=L1,
            output_tile=out_tile,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.expert_config,
            program_config=self.gate_up_program,
        )  # [1, 1, 1, pairs, 32, 2I]
        act = self._swiglu(ttnn.reshape(gate_up, (1, pairs, TILE, 2 * I)), memory_config=L1)
        act_w = ttnn.mul(act, row_weight, memory_config=L1)
        act.deallocate(True)
        row_weight.deallocate(True)
        down = ttnn.sparse_matmul(
            act_w,
            self.w_down,
            sparsity=self.all_experts,
            indices=ids,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            memory_config=L1,
            output_tile=out_tile,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.expert_config,
            program_config=self.down_program,
        )  # [1, pairs, 32, H]
        act_w.deallocate(True)
        ids.deallocate(True)
        out = ttnn.sum(down, dim=1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, 32, H]
        down.deallocate(True)
        out = self.plan.all_reduce(out)
        if residual is None:
            return out
        y = ttnn.add(residual, out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate(True)
        return y

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
