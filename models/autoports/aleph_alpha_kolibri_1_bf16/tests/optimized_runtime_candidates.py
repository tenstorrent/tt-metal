# SPDX-License-Identifier: Apache-2.0
"""Phase-specific matmul and activation experiments on the production topology."""
import json
import math
import os

import ttnn

from ..tt.optimized_decoder import DRAM, OptimizedDecoder


class RuntimeCandidate(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        self = super().from_state_dict(state_dict, **kwargs)
        self.options = json.loads(os.environ.get("OPT_RUNTIME", "{}"))
        if self.options.get("cached_masks"):
            import torch

            self.cached_masks = {
                id(self.mask_zero): ttnn.from_torch(
                    torch.zeros(1, 1, 4096, 384, dtype=torch.bfloat16),
                    device=self.device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=DRAM,
                ),
                id(self.mask_one): ttnn.from_torch(
                    torch.ones(1, 1, 4096, 6, dtype=torch.bfloat16),
                    device=self.device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=DRAM,
                ),
            }
        if self.options.get("router_weight_dtype"):
            weights = {key.removeprefix(f"model.layers.{self.layer_idx}."): value for key, value in state_dict.items()}
            self.router = ttnn.from_torch(
                weights["mlp.gate.weight"].T.contiguous(),
                device=self.device,
                dtype=getattr(ttnn, self.options["router_weight_dtype"]),
                layout=ttnn.TILE_LAYOUT,
                memory_config=DRAM,
            )
        return self

    def _moe(self, x, *, prefill=False):
        if self.options.get("route_choice_bf16"):
            original = ttnn.topk

            def lower_choice(t, *args, **kwargs):
                return original(ttnn.typecast(t, ttnn.bfloat16), *args, **kwargs)

            ttnn.topk = lower_choice
            try:
                return super()._moe(x, prefill=prefill)
            finally:
                ttnn.topk = original
        if prefill and x.shape[-2] == 32 and self.options.get("short_grouped"):
            x = ttnn.to_memory_config(x, DRAM)
            logits = self._linear(x, self.router, dtype=ttnn.float32)
            _, ids = ttnn.topk(ttnn.add(logits, self.expert_bias), k=6, dim=-1)
            return self._grouped_prefill_moe(x, logits, ids)
        return super()._moe(x, prefill=prefill)

    def prefill_chunk_forward(self, hidden_states, **kwargs):
        if self.options.get("prefill_compute"):
            from dataclasses import replace

            options = self.options["prefill_compute"]
            original_compute, original_policy = self.sdpa_compute, self.policy
            self.sdpa_compute = ttnn.WormholeComputeKernelConfig(
                math_fidelity=getattr(ttnn.MathFidelity, options.get("fidelity", "LoFi")),
                math_approx_mode=False,
                fp32_dest_acc_en=options.get("fp32", False),
                packer_l1_acc=True,
            )
            self.policy = replace(self.policy, sdpa_exp_approx=options.get("exp", True))
            try:
                return super().prefill_chunk_forward(hidden_states, **kwargs)
            finally:
                self.sdpa_compute, self.policy = original_compute, original_policy
        opts = self.options.get("prefill_sdpa")
        if not opts:
            return super().prefill_chunk_forward(hidden_states, **kwargs)
        original = ttnn.transformer.chunked_scaled_dot_product_attention

        def alternate(q, *args, **options):
            qchunk = max(n for n in (32, 64, 128, 256, 512) if n <= opts.get("q", 32) and q.shape[-2] % n == 0)
            options["program_config"] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=tuple(opts.get("grid", [8, 8])),
                q_chunk_size=qchunk,
                k_chunk_size=opts.get("k", 32),
                exp_approx_mode=False,
            )
            return original(q, *args, **options)

        ttnn.transformer.chunked_scaled_dot_product_attention = alternate
        try:
            return super().prefill_chunk_forward(hidden_states, **kwargs)
        finally:
            ttnn.transformer.chunked_scaled_dot_product_attention = original

    def _prefill_linear(self, x, w, dtype, compute):
        if self.options.get("prefill_input_l1"):
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        fidelity = self.options.get("prefill_router_fidelity")
        if w is self.router and fidelity:
            compute = ttnn.WormholeComputeKernelConfig(
                math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        return super()._prefill_linear(x, w, dtype, compute)

    def _linear(self, x, w, *, dtype=ttnn.bfloat16, activation=None):
        info = self.projection_info.get(id(w))
        role = info["role"] if info else "router"
        selected = self.options.get("activation")
        attention = role in ("qkv", "o_proj")
        if (selected == "attention" and attention) or (selected == "moe" and role in ("gate_up", "down_proj")):
            x = ttnn.typecast(x, ttnn.bfloat8_b)
        kind = self.options.get("prefill_kind")
        if x.shape[-2] <= 32 or not kind or role not in self.options.get("roles", [role]):
            return super()._linear(x, w, dtype=dtype, activation=activation)
        grid = self.options.get("grid", [8, 8])
        block = self.options.get("kblock", 8)
        sub = self.options.get("subblock", 4)
        compute = info["compute"] if info else self.compute
        if kind == "minimal":
            mblock = self.options.get("mblock", 1)
            nblock = self.options.get("nblock", 8)
            program = ttnn.MinimalMatmulConfig(
                M_block_size=mblock,
                K_block_size=block,
                N_block_size=nblock,
                subblock_h=1,
                subblock_w=min(sub, nblock),
                compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
            )
            return ttnn.experimental.minimal_matmul(
                x, w, config=program, dtype=dtype, memory_config=DRAM, compute_kernel_config=compute
            )
        pm = math.ceil(x.shape[-2] / 32 / grid[1])
        pn = math.ceil(w.shape[-1] / 32 / grid[0])
        bm = self.options.get("output_mblock", pm)
        bn = self.options.get("output_nblock", pn)
        pm = math.ceil(pm / bm) * bm
        pn = math.ceil(pn / bn) * bn
        sw = max(v for v in (1, 2, 4) if v <= sub and pn % v == 0)
        program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=tuple(grid),
            in0_block_w=block,
            out_subblock_h=1,
            out_subblock_w=sw,
            out_block_h=bm,
            out_block_w=bn,
            per_core_M=pm,
            per_core_N=pn,
            transpose_mcast=False,
            fused_activation=None,
        )
        return ttnn.linear(x, w, dtype=dtype, memory_config=DRAM, program_config=program, compute_kernel_config=compute)

    def _indexed_moe(self, x, logits, ids):
        if self.options.get("skip_index_cast"):
            original = ttnn.typecast

            def avoid_identity(tensor, dtype, *args, **kwargs):
                if tensor is ids and dtype == ids.dtype:
                    return tensor
                return original(tensor, dtype, *args, **kwargs)

            ttnn.typecast = avoid_identity
            try:
                return super()._indexed_moe(x, logits, ids)
            finally:
                ttnn.typecast = original
        if self.options.get("route_score_bf16"):
            logits = ttnn.typecast(logits, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        if self.options.get("activation") != "moe":
            return super()._indexed_moe(x, logits, ids)
        # Cast both routed matmul inputs, preserving FP32 router and BF16 residual/norm.
        original = ttnn.sparse_matmul

        def reduced(a, *args, **kwargs):
            return original(ttnn.typecast(a, ttnn.bfloat8_b), *args, **kwargs)

        ttnn.sparse_matmul = reduced
        try:
            return super()._indexed_moe(x, logits, ids)
        finally:
            ttnn.sparse_matmul = original

    def _grouped_prefill_moe(self, x, logits, ids):
        if self.options.get("cached_masks") and x.shape[-2] == 4096:
            original = ttnn.repeat

            def cached(tensor, *args, **kwargs):
                if id(tensor) in self.cached_masks:
                    return self.cached_masks[id(tensor)]
                return original(tensor, *args, **kwargs)

            ttnn.repeat = cached
            try:
                return super()._grouped_prefill_moe(x, logits, ids)
            finally:
                ttnn.repeat = original
        if self.options.get("route_score_bf16"):
            logits = ttnn.typecast(logits, ttnn.bfloat16)
        kind = self.options.get("expert_kernel")
        if not kind:
            return super()._grouped_prefill_moe(x, logits, ids)
        ns = ttnn.experimental.deepseek_prefill
        original = ns.moe_fused_swiglu

        def alternate(a, wg, wu, wd, counts, experts, **kwargs):
            arguments = (a, kwargs["expert_region_offsets"], counts, experts, wg, wu, wd)
            options = dict(
                max_dispatched_tokens_per_expert=kwargs["input_m_tiles"] * 32, compute_kernel_config=self.expert_compute
            )
            if kind == "unified":
                return ns.unified_routed_expert_moe(*arguments, **options)
            return ns.hybrid_routed_expert_moe(
                *arguments, hybrid_token_threshold=self.options.get("threshold", 32), **options
            )

        ns.moe_fused_swiglu = alternate
        try:
            return super()._grouped_prefill_moe(x, logits, ids)
        finally:
            ns.moe_fused_swiglu = original

    def decode_forward(self, hidden_states, *, kv_cache, page_table, current_pos, cos=None, sin=None):
        """One token per request: [1,1,B,2560], tensor positions [B], B table rows.

        Every op is device-only and can be captured in a TTNN execution trace.
        Refresh stable input/position/RoPE/page-table buffers before replay.
        """
        if not self.options.get("head_norm"):
            return super().decode_forward(
                hidden_states, kv_cache=kv_cache, page_table=page_table, current_pos=current_pos, cos=cos, sin=sin
            )
        batch = hidden_states.shape[-2]
        if tuple(hidden_states.shape) != (1, 1, batch, 2560) or not 1 <= batch <= 32:
            raise ValueError("Decode expects [1,1,B,2560], 1 <= B <= 32")
        if tuple(current_pos.shape) != (batch,) or page_table.shape[0] != batch:
            raise ValueError("Decode positions and page-table rows must match batch")
        grid = self.device.compute_with_storage_grid_size()
        # Sharded SDPA assigns batch i to (i % grid.x, i // grid.x).
        # Q/K/V and RoPE must use that exact physical row-major prefix.
        shard_grid = ttnn.num_cores_to_corerangeset(batch, grid, row_wise=True)
        mem = ttnn.create_sharded_memory_config(
            (32, 128), shard_grid, ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
        )
        n = self._norm(hidden_states, "input_layernorm")
        packed = self._linear(n, self.projections["qkv"])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(packed, num_heads=48, num_kv_heads=4, memory_config=mem)
        q = self._norm(q if self.options.get("head_norm") else ttnn.to_memory_config(q, DRAM), "self_attn.q_norm")
        k = self._norm(k if self.options.get("head_norm") else ttnn.to_memory_config(k, DRAM), "self_attn.k_norm")
        if self.sliding:
            qmem = ttnn.create_sharded_memory_config(
                (64, 128), shard_grid, ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True
            )
            q = ttnn.to_memory_config(q, qmem)
            k = ttnn.to_memory_config(k, mem)
            c = ttnn.to_memory_config(ttnn.permute(cos, (0, 2, 1, 3)), mem)
            s = ttnn.to_memory_config(ttnn.permute(sin, (0, 2, 1, 3)), mem)
            q = ttnn.experimental.rotary_embedding_hf(q, c, s, is_decode_mode=True, compute_kernel_config=self.compute)
            k = ttnn.experimental.rotary_embedding_hf(k, c, s, is_decode_mode=True, compute_kernel_config=self.compute)
        k = ttnn.to_memory_config(k, mem)
        v = ttnn.to_memory_config(v, mem)
        ttnn.experimental.paged_update_cache(kv_cache[0], k, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(kv_cache[1], v, update_idxs_tensor=current_pos, page_table=page_table)
        y = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv_cache[0],
            kv_cache[1],
            page_table,
            cur_pos_tensor=current_pos,
            sliding_window_size=513 if self.sliding else None,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                q_chunk_size=32,
                k_chunk_size=self.policy.decode_k_chunk,
                max_cores_per_head_batch=self.policy.decode_cores_per_head,
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.compute,
        )
        y = ttnn.reshape(y, (1, 1, batch, 6144))
        return self._finish(hidden_states, self._linear(y, self.projections["o_proj"]))

    def _norm(self, x, name):
        if self.options.get("fused_gamma"):
            if x.shape[-1] == 2560 and x.shape[-2] <= 32:
                x = ttnn.to_memory_config(x, self.residual_mem)
                return ttnn.rms_norm(
                    x,
                    epsilon=self.config.rms_norm_eps,
                    weight=self.norms[name],
                    program_config=self.norm_program,
                    compute_kernel_config=self.compute,
                    memory_config=self.residual_mem,
                )
            x = ttnn.to_memory_config(x, DRAM)
            return ttnn.rms_norm(
                x,
                epsilon=self.config.rms_norm_eps,
                weight=self.norms[name],
                compute_kernel_config=self.compute,
                memory_config=DRAM,
            )
        if self.options.get("head_norm") and x.shape[-1] == 128 and x.is_sharded():
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, x.memory_config().shard_spec
            )
            x = ttnn.to_memory_config(x, mem)
            shard = mem.shard_spec
            bbox = shard.grid.bounding_box()
            grid = (bbox.end.x + 1, bbox.end.y + 1)
            program = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=grid,
                subblock_w=4,
                block_h=shard.shape[0] // 32,
                block_w=4,
                inplace=False,
            )
            y = ttnn.rms_norm(
                x,
                epsilon=self.config.rms_norm_eps,
                program_config=program,
                compute_kernel_config=self.compute,
                memory_config=mem,
            )
            return ttnn.multiply(y, self.norms[name], memory_config=mem)
        return super()._norm(x, name)
