# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo routed MoE (256 experts, top-8, sigmoid ``noaux_tc``), tensor-parallel over one Galaxy row.

Routing runs replicated on every chip, in fp32 end to end: router logits (HiFi4, fp32 out), sigmoid,
+bias for selection only, fp32 top-8, gather the unbiased scores, normalize with eps 1e-20, then
scatter into a dense ``[T, 256]`` routing matrix. MiMo's biased scores sit in [0.5, 1.5] with a
median 8th-vs-9th gap of ~1.6e-3, so a bf16 top-k flips experts: ``moe_grouped_topk`` fed exact
fp32 logits matched the reference top-8 set on only 80% of real layer-5 tokens, this path on 100%.

Experts are TP-sharded on the intermediate dim: each chip holds every expert's 256-wide slice of
``moe_intermediate_size`` (2048). Prefill computes the experts of T tokens in moe_prefill.py; decode
streams the selected experts (moe_stream.py). The weighted sum over experts is a partial sum per
chip and is all-reduced across the row.
"""

import os

import torch

import ttnn

from .common import (
    TILE,
    DramShardedMatmul,
    as_weight,
    bfp_mse_prepare,
    cache_exists,
    cache_name,
    compute_config,
    mcast_1d_config,
)
from .moe_stream import (
    DownStream,
    FusedStream,
    GateUpStream,
    DEDUP,
    FUSED,
    PendingCompact,
    ReduceExpand,
    fabric_row_gather,
    fabric_transport_noc,
    is_dram_interleaved,
    stick_buffer,
)
from .moe_prefill import PrefillExperts
from .profiling import zone
from .proj_stream import RouterScoreStream

# Longest prefill chunk one MoE call takes; forward runs longer chunks in slices. The router's prefill program block
# (_router_program) grows with the rows: at 8192 its static circular buffers end at 1.32 MB, past the L1 buffers decode
# keeps allocated (from 1.21 MB), so after decode only 4096 fits. Without decode it fits to 8192, and the expert plan's
# row buffers (moe_prefill.DevicePlan) outgrow L1 a little above that. vLLM's chunks are 4096.
PREFILL_MAX_TOKENS = 4096


def expert_quant(expert_dtype):
    """How bfloat4_b experts are prepared: "mse" (default, bfp_mse_prepare), "gptq" (tests/perf/gptq_experts.py,
    read from MIMO_EXPERT_PREP_DIR) or "" (the stock packer). MIMO_EXPERT_QUANT picks it; MIMO_EXPERT_MSE=0 is
    "" (older switch). Other dtypes always use the stock packer."""
    if expert_dtype != ttnn.bfloat4_b or os.environ.get("MIMO_EXPERT_MSE", "1") == "0":
        return ""
    quant = os.environ.get("MIMO_EXPERT_QUANT", "mse")
    assert quant in ("", "stock", "mse", "gptq"), quant
    return "" if quant == "stock" else quant


def expert_weights_host(experts, tp, inter_local, expert_dtype, layer_idx=None, mse=True, quant=None):
    """Host expert weights in device layout: (w_gate_up [1, E, H, tp * 2 * I_local], w_down [1, E, I, H]).

    gate/up [E, I, H] -> one tensor whose chip-c slice holds [gate_c | up_c], so one matmul makes both; down
    [E, H, I] -> [1, E, I, H] with I sharded over TP. bfloat4_b weights are MSE-prepared (bfp_mse_prepare); a
    precomputed copy is taken from MIMO_EXPERT_PREP_DIR/layer<i>.pt when present (tests/perf/prep_experts.py).
    """
    prep_dir = os.environ.get("MIMO_EXPERT_PREP_DIR")
    path = os.path.join(prep_dir, f"layer{layer_idx}.pt") if prep_dir and layer_idx is not None else None
    mse = mse and expert_dtype == ttnn.bfloat4_b
    if quant == "gptq":
        assert path and os.path.exists(path), f"gptq experts need tests/perf/gptq_experts.py output at {path}"
    if (mse or quant == "gptq") and path and os.path.exists(path):
        saved = torch.load(path)
        return saved["w_gate_up"], saved["w_down"]
    g = experts["gate_proj"].transpose(1, 2)
    u = experts["up_proj"].transpose(1, 2)
    il = inter_local
    w_gate_up = torch.cat(
        [torch.cat([g[..., c * il : (c + 1) * il], u[..., c * il : (c + 1) * il]], dim=-1) for c in range(tp)],
        dim=-1,
    ).unsqueeze(0)
    del g, u
    w_down = experts["down_proj"].transpose(1, 2).unsqueeze(0)
    if mse:
        w_gate_up = bfp_mse_prepare(w_gate_up, expert_dtype)
        w_down = bfp_mse_prepare(w_down, expert_dtype)
    return w_gate_up, w_down


class TtMiMoMoE:
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
        # bfloat4_b experts are stored prepared (expert_quant): same format and speed, lower error.
        self.expert_quant = expert_quant(expert_dtype)
        tag = f"_{self.expert_quant}" if self.expert_quant else ""
        expert_cache = [cache_name(cache_dir, prefix, n + tag) for n in ("w_gate_up", "w_down")]
        if all(cache_exists(f, expert_dtype) for f in expert_cache):
            experts = None  # already converted: skip dequantizing 256 experts on the host
        w_router = bias = w_gate_up = w_down = None
        if router is not None:
            w_router = router["weight"].T.contiguous().to(torch.bfloat16)  # [hidden, E], replicated
            w_router = self.router_mm.pad_chip_blocks(w_router, 1)
            bias = router["e_score_correction_bias"].float().reshape(1, 1, 1, -1)
        if experts is not None:
            w_gate_up, w_down = expert_weights_host(
                experts, tp, self.inter_local, expert_dtype, layer_idx, self.expert_quant == "mse", self.expert_quant
            )

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
        self.w_gate_up = as_weight(w_gate_up, plan, plan.shard(-1), expert_dtype, expert_cache[0])
        del w_gate_up
        self.w_down = as_weight(w_down, plan, plan.shard(-2), expert_dtype, expert_cache[1])

        self.expert_config = compute_config(ttnn.MathFidelity.HiFi2, fp32_acc=True)
        # Measured on BH (decode, 32 experts): wider K blocks cut the per-expert handshakes; down over 8x8 cores.
        self.gate_up_program = mcast_1d_config((8, 2), TILE, 2 * self.inter_local, self.hidden, in0_block_w=64)
        self.down_program = mcast_1d_config((8, 8), TILE, self.hidden, self.inter_local, in0_block_w=8)
        self.decode_users = None
        self.gate_up_stream = None
        self.router_scores = None
        self.fabric_gather = False  # set by the model: the block reductions all-gather over the fabric
        self.down_stream = None
        self.fused_stream = None
        self.ids_buf = None
        self.prefill_experts = PrefillExperts(self)  # forward (prefill): the experts of T tokens

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

    def _router_logits(self, x, prefill=False):
        """fp32 logits [1, 1, T, E]: DRAM-sharded matmul for a decode tile row, else linear on an interleaved copy.
        Prefill always takes the linear, with a program config whose K blocking does not change with T, so a row's
        logits do not depend on the other rows (a prefill row in a batch equals the same row alone)."""
        if x.shape[-2] == TILE and not prefill:
            return self.router_mm(x, self.w_router)
        weight = ttnn.to_memory_config(self.w_router, ttnn.DRAM_MEMORY_CONFIG)
        logits = ttnn.linear(
            x,
            weight,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.router_config,
            program_config=self._router_program(x.shape[-2]) if prefill else None,
        )
        weight.deallocate(True)
        return logits

    def _router_program(self, rows):
        """2D multicast matmul: the experts' 8 tile columns over 8 core columns, the row tiles over up to 8 core
        rows, a fixed K block of 8 tiles (every row's K sum is added in the same order for any T)."""
        mt, nt = rows // TILE, self.num_experts // TILE
        ny = max(d for d in range(1, 9) if mt % d == 0)
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(nt, ny),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=mt // ny,
            out_block_w=1,
            per_core_M=mt // ny,
            per_core_N=1,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

    def route(self, x, token_mask=None):
        """x [1, 1, T, hidden] -> dense routing weights [1, 1, T, E] (bf16; zero off the top-8)."""
        logits = self._router_logits(x)
        indices, weights = self._topk(logits)
        # Zeros filled on device for every call. A zero tensor cached per T and first made after a decode trace capture
        # (a prefill length first seen while serving) would sit in memory the trace's replays write to.
        zeros32 = ttnn.zeros_like(logits)
        logits.deallocate(True)
        zeros = ttnn.typecast(zeros32, ttnn.bfloat16)
        zeros32.deallocate(True)
        weights_bf16 = ttnn.typecast(weights, ttnn.bfloat16)
        weights.deallocate(True)
        dense = ttnn.scatter(zeros, dim=-1, index=indices, src=weights_bf16)
        for t in (zeros, weights_bf16, indices):
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
        """Decode serves users in slots [0, num_users): their (user, top-k) pairs are the experts computed. The streaming
        programs and their persistent buffers are sized for that count, so it is set once."""
        if self.decode_users == num_users:
            return
        assert self.decode_users is None, f"decode users already set to {self.decode_users}"
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
        stream_users = GateUpStream.MAX_USERS_DEDUP if DEDUP else GateUpStream.MAX_USERS
        # 16 users (the 16-slot verify): FusedStream's two row halves of 8, on 8-user streams.
        halves = 2 if FUSED and DEDUP and num_users == 2 * GateUpStream.MAX_USERS_DEDUP else 1
        if self.expert_dtype == ttnn.bfloat4_b and (num_users <= stream_users or halves == 2):
            pairs //= halves
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
            # The row-major uint16 expert stick (the pairs' ids, or deduplicated [count, distinct ids...]).
            self.ids_buf = stick_buffer(self.plan, pairs, self.gate_up_stream.dedup)
            # Down + sum over pairs, bank-local from the interleaved w_down, into a persistent compact
            # [1, 1, 32, hidden / 8] output holding only the user rows; one row all-gather of it, then ReduceExpand
            # sums the chips' copies and restores [1, 1, 32, hidden] with zero padding rows.
            rows = self.gate_up_stream.rows
            self.down_stream = DownStream(
                self.plan, self.hidden, self.inter_local, pairs, self.gate_up_stream.dedup, rows=rows
            )
            self.act_buf = self.down_stream.activation_buffer(self.plan)
            self.out_buf = self.down_stream.compact_output(self.plan)
            # Deduplicated: gate|up and down in one launch, h exchanged between the bank cores' L1 (FusedStream).
            self.fused_stream = None
            if FUSED and self.gate_up_stream.dedup:
                self.fused_stream = FusedStream(self.plan, self.gate_up_stream, self.down_stream, halves)
                rows = self.fused_stream.rows
                if halves == 2:
                    self.out_buf.deallocate(True)
                    self.out_buf = self.fused_stream.compact_output(self.plan)
            assert halves == 1 or self.fused_stream is not None
            self.reduce_expand = ReduceExpand(
                self.plan, self.hidden, self.plan.tp, self.down_stream.compute_config, rows
            )
            # Router logits, sigmoid and the bias add in one bank-local launch (scores and biased are persistent).
            try:
                bias = ttnn.to_torch(ttnn.get_device_tensors(self.score_bias)[0]).float().reshape(-1)
                self.router_scores = RouterScoreStream(self.plan, self.hidden, self.num_experts, bias, rows=rows)
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
        if self.fused_stream is not None:
            return self._forward_decode_fused(srcs, (x_in, s_in, b_in, m_in), fused_router, residual, norm)
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

    def _forward_decode_fused(self, srcs, ins, fused_router, residual, norm):
        """_forward_decode_stream with FusedStream: routing, gate|up and down in one launch, then the same row
        all-reduce (in the down launch's place, with ``norm`` and the fabric gather, the reduce launch runs it)."""
        x_in, s_in, b_in, m_in = ins
        args = (x_in, s_in, b_in, m_in, self.w_gate_up, self.w_down, self.out_buf)

        def release():
            for t, src in zip(ins, srcs):
                if t is not src:
                    t.deallocate(True)
            if not fused_router:
                srcs[1].deallocate(True)
                srcs[2].deallocate(True)

        plan = self.plan
        if norm is not None and self.fabric_gather:
            section = self.fused_stream(*args, signal=fabric_transport_noc(plan.mesh_device), launch=False)
            pending = PendingCompact(section, self.out_buf, len(self.down_stream.cores))
            out = self.reduce_residual(None, residual, norm=norm, compact=pending)
            release()
            return out
        out = self.fused_stream(*args)
        release()
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

    def forward(self, x, token_mask=None, decode=False):
        """x [1, 1, T, hidden] replicated across the row -> [1, 1, T, hidden] replicated."""
        assert x.shape[-2] % TILE == 0, "pad tokens to a multiple of 32"
        if decode or x.shape[-2] <= PREFILL_MAX_TOKENS:
            out = self._partial(x, token_mask, decode)
        else:
            out = self._partial_slices(x, token_mask)
        with zone("all_reduce"):
            return self.plan.all_reduce(out)

    def _partial(self, x, token_mask, decode=False):
        """This chip's share of the routed-expert output [1, 1, T, hidden], before the row all-reduce."""
        with zone("router"):
            logits = self._router_logits(x, prefill=not decode)
            indices, weights = self._topk(logits)
            logits.deallocate(True)
            if token_mask is not None:
                masked = ttnn.mul(weights, ttnn.typecast(token_mask, weights.dtype))
                weights.deallocate(True)
                weights = masked
        with zone("experts"):
            out = self.prefill_experts(x, indices, weights)
        indices.deallocate(True)
        weights.deallocate(True)
        return out

    def _partial_slices(self, x, token_mask):
        """_partial of a prefill chunk longer than PREFILL_MAX_TOKENS, in slices of at most that many tokens. Each
        token's partial is the same as in one call: the router's K blocking does not change with T, and the experts
        work per token. The caller all-reduces the whole chunk once, so the reduce sees the same rows as without
        slicing (MeshPlan.prefill_reduce)."""
        T, H = x.shape[-2], x.shape[-1]
        outs = []
        for s in range(0, T, PREFILL_MAX_TOKENS):  # the slices' zones add up (router, experts)
            e = min(T, s + PREFILL_MAX_TOKENS)
            xs = ttnn.slice(x, [0, 0, s, 0], [1, 1, e, H], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ms = None
            if token_mask is not None:
                ms = ttnn.slice(
                    token_mask, [0, 0, s, 0], [1, 1, e, token_mask.shape[-1]], memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            outs.append(self._partial(xs, ms))
            xs.deallocate(True)
            if ms is not None:
                ms.deallocate(True)
        out = ttnn.concat(outs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for o in outs:
            o.deallocate(True)
        return out
