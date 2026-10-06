# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measured graph fusions for the single-device Gemma4 decoder."""

from dataclasses import replace

import ttnn
from models.demos.gemma4.tt.attention.operations import (
    chunked_prefill_sdpa,
    prefill_sdpa_program_config,
    split_qkv_heads_decode,
    split_qkv_heads_prefill,
)
from models.demos.gemma4_26b_a4b_qb2.tt.decode_attention import DecodeAttention
from models.demos.gemma4_26b_a4b_qb2.tt.functional_decoder import FunctionalDecoder
from models.demos.gemma4_26b_a4b_qb2.tt.precision_ops import norm_weight, rms_norm, rotary
from models.demos.gemma4_26b_a4b_qb2.tt.routing_precision import QKVLinear


class BroadcastQKV(QKVLinear):
    """Broadcast each activation row directly into FP32 projection products."""

    def __init__(self, source, group_size=256):
        self.group_size = group_size
        self.decode_memory = ttnn.DRAM_MEMORY_CONFIG
        self.weight = source.weight
        self.rows = source.rows
        self.compute = source.compute

    def __call__(self, hidden_states, compute_kernel_config=None, out_memory_config=None):
        if hidden_states.shape[-2] != 1:
            return ttnn.linear(
                hidden_states,
                self.weight,
                dtype=ttnn.float32,
                compute_kernel_config=self.compute,
                memory_config=out_memory_config,
            )
        outputs = []
        for start in range(0, self.rows.shape[-2], self.group_size):
            weight = self.rows[:, :, start : min(start + self.group_size, self.rows.shape[-2]), :]
            products = ttnn.mul(weight, hidden_states)
            outputs.append(
                ttnn.transpose(ttnn.sum(products, dim=-1, keepdim=True), -2, -1, memory_config=self.decode_memory)
            )
        return ttnn.concat(outputs, dim=-1, memory_config=self.decode_memory)


class FusedDecoder(FunctionalDecoder):
    DEFAULT_FUSIONS = {
        "sliding_attention": "broadcast_pack_gelu_router_tail_norm_attention_common_tied_ropepolicy_cache_kvnorm_shardedtail_preciseheads_expmerge_residualnorm_rsqrt_expertbatch64_mixmatmul_mixsharded_mixfp32_qkvl1_projectsharded_sharedshard_unarycastshard_directtail",
        "full_attention": "broadcast_pack_gelu_router_tail_norm_attention_common_tied_expmerge_residualnorm_shardedtail_kvnorm_expertbatch64_mixmatmul_mixsharded_qkvl1_projectsharded_sharedshard_directtail",
    }

    @classmethod
    def from_state_dict(cls, state_dict, *, fusion=None, group_size=16384, **kwargs):
        self = super().from_state_dict(state_dict, **kwargs)
        fusion = fusion or cls.DEFAULT_FUSIONS[self.config.layer_types[self.layer_idx]]
        self.fusion = fusion
        self.fuse_tail = "tail" in fusion
        self.sharded_tail = "shardedtail" in fusion
        self.residual_norm = "residualnorm" in fusion
        self.fuse_rsqrt = "rsqrt" in fusion
        self.direct_tail = "directtail" in fusion
        if self.sharded_tail:
            for norm in (
                self.layer.post_feedforward_layernorm_1,
                self.layer.post_feedforward_layernorm_2,
                self.layer.post_feedforward_layernorm,
            ):
                norm._sharded_cfg = norm._build_sharded_cfg(self.config.hidden_size)
                norm._sharded_dim = self.config.hidden_size
        self.fuse_norm = "norm" in fusion
        self.precise_heads = "preciseheads" in fusion and self.config.layer_types[self.layer_idx] == "sliding_attention"
        self.common_norm = "common" in fusion
        if self.common_norm:
            self.shared_norm_weight = norm_weight(
                self.layer.pre_feedforward_layernorm.tt_weight, self.config.hidden_size
            )
            self.expert_norm_weight = norm_weight(
                self.layer.pre_feedforward_layernorm_2.tt_weight, self.config.hidden_size
            )
        attention = self.layer.self_attn.source
        projection = BroadcastQKV(attention.weights.wqkv, group_size)
        if "tied" in fusion and attention.weights.is_global:
            projection = TiedQKV(projection, attention.config.num_key_value_heads * attention.config.head_dim)
        if "qkvl1" in fusion:
            projection.decode_memory = ttnn.L1_MEMORY_CONFIG
        attention.weights = replace(attention.weights, wqkv=projection)
        rope_mode = (
            (True if attention.weights.is_global else "decode")
            if "ropepolicy" in fusion
            else ("decode" if "ropedecode" in fusion else True)
        )
        self.layer.self_attn = FusedAttention(
            self.layer.self_attn,
            self.normalize,
            rope_mode if "rope" in fusion and ("ropefull" not in fusion or attention.weights.is_global) else False,
        )
        self.layer.self_attn.fuse_cache = "cache" in fusion
        self.layer.self_attn.common_kv = "kvnorm" in fusion
        self.layer.self_attn.pack_heads = "headpack" in fusion
        self.layer.self_attn.cast_shard = "castshard" in fusion
        self.layer.self_attn.unary_cast_shard = "unarycastshard" in fusion
        self.layer.self_attn.i2s_cast = "i2scast" in fusion
        self.layer.self_attn.project_sharded = "projectsharded" in fusion
        if "pack" in fusion:
            batch_tokens = next(
                (int(flag.removeprefix("expertbatch")) for flag in fusion.split("_") if flag.startswith("expertbatch")),
                32,
            )
            self.layer.moe.experts = PackedExperts(
                self.layer.moe.experts,
                fused_gelu="gelu" in fusion,
                prefill_batch_tokens=batch_tokens,
                matmul_mix="mixmatmul" in fusion,
                mix_fp32="mixfp32" in fusion,
            )
        if "mixsharded" in fusion:
            experts = self.layer.moe.experts
            experts.mix_memory = self.layer.post_feedforward_layernorm_2._sharded_cfg[0]
            spec = experts.mix_memory.shard_spec
            end = spec.grid.bounding_box().end
            experts.mix_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(end.x + 1, end.y + 1),
                in0_block_w=self.config.num_experts // 32,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=spec.shape[1] // 32,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        if "gelu" in fusion:
            shared_memory = self.layer.post_feedforward_layernorm_1._sharded_cfg[0] if "sharedshard" in fusion else None
            shared_block = next(
                (int(flag.removeprefix("sharedblock")) for flag in fusion.split("_") if flag.startswith("sharedblock")),
                6,
            )
            self.layer.shared_mlp = FusedSharedMLP(
                self.layer.shared_mlp,
                down_weight=state_dict["mlp.down_proj.weight"] if shared_memory else None,
                output_memory=shared_memory,
                block_w=shared_block,
            )
        if "router" in fusion:
            self.layer.moe.router = BroadcastRouter(self.layer.moe.router, self.normalize)
        if "attention" in fusion:
            self.layer.self_attn.decode_sdpa = BatchedPagedAttention(
                self.layer.self_attn.decode_sdpa, "softmax" in fusion, "expmerge" in fusion, "centered" in fusion
            )
        return self

    def normalize(self, value, epsilon, weight=None):
        if not self.fuse_norm or (self.precise_heads and value.shape[-1] != self.config.hidden_size):
            if self.fuse_rsqrt:
                value = ttnn.typecast(value, ttnn.float32)
                variance = ttnn.mean(ttnn.mul(value, value), dim=-1, keepdim=True)
                scale = ttnn.add(variance, epsilon, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RSQRT, 0.0)])
                result = ttnn.mul(value, scale)
                return result if weight is None else ttnn.mul(result, weight)
            return rms_norm(value, epsilon, weight)
        result = ttnn.rms_norm(
            ttnn.typecast(value, ttnn.float32), epsilon=epsilon, compute_kernel_config=self.layer.self_attn.compute
        )
        return result if weight is None else ttnn.mul(result, weight)

    def _forward(self, x, **attention_kwargs):
        layer = self.layer
        normed = self.normalize(x, self.config.rms_norm_eps, self.input_norm_weight)
        attention = layer.self_attn(normed, **attention_kwargs)
        post_attention = self.normalize(attention, self.config.rms_norm_eps, self.post_attention_norm_weight)
        residual = ttnn.add(ttnn.typecast(x, ttnn.float32), post_attention)
        if self.common_norm:
            normalized = self.normalize(residual, self.config.rms_norm_eps)
            shared_input = ttnn.mul(normalized, self.shared_norm_weight, dtype=ttnn.bfloat16)
            expert_input = ttnn.mul(normalized, self.expert_norm_weight, dtype=ttnn.bfloat16)
            routes = layer.moe.router(residual, normalized=normalized)
            routed = layer.moe.experts(expert_input, routes)
        else:
            shared_input = ttnn.typecast(layer.pre_feedforward_layernorm.forward(residual), ttnn.bfloat16)
            expert_input = ttnn.typecast(layer.pre_feedforward_layernorm_2.forward(residual), ttnn.bfloat16)
            routed = layer.moe(residual, expert_input)
        shared = layer.shared_mlp(shared_input)
        if self.sharded_tail and x.shape[-2] <= 32:

            def norm_sharded(value, norm):
                memory, program = norm._sharded_cfg
                return ttnn.rms_norm(
                    ttnn.to_memory_config(value, memory),
                    weight=norm.tt_weight,
                    epsilon=norm.eps,
                    program_config=program,
                )

            shared = norm_sharded(shared, layer.post_feedforward_layernorm_1)
            routed = norm_sharded(routed, layer.post_feedforward_layernorm_2)
            if self.residual_norm:
                norm = layer.post_feedforward_layernorm
                combined = ttnn.rms_norm(
                    shared,
                    residual_input_tensor=routed,
                    weight=norm.tt_weight,
                    epsilon=norm.eps,
                    program_config=norm._sharded_cfg[1],
                )
            else:
                combined = norm_sharded(ttnn.add(shared, routed), layer.post_feedforward_layernorm)
            if not self.direct_tail:
                combined = ttnn.to_memory_config(combined, ttnn.DRAM_MEMORY_CONFIG)
        else:
            shared = layer.post_feedforward_layernorm_1.forward(shared)
            routed = layer.post_feedforward_layernorm_2.forward(routed)
            if self.residual_norm and x.shape[-2] > 32:
                norm = layer.post_feedforward_layernorm
                combined = ttnn.rms_norm(shared, residual_input_tensor=routed, weight=norm.tt_weight, epsilon=norm.eps)
            else:
                combined = layer.post_feedforward_layernorm.forward(ttnn.add(shared, routed))
        if self.fuse_tail:
            return ttnn.add(
                residual,
                combined,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, layer.layer_scalar)],
            )
        return ttnn.typecast(ttnn.mul(ttnn.add(residual, combined), layer.layer_scalar), ttnn.bfloat16)


class PackedExperts:
    """One sparse projection produces both gate and up for every expert."""

    def __init__(
        self,
        source,
        fused_gelu=False,
        fuse_decode_gelu=False,
        prefill_batch_tokens=32,
        matmul_mix=False,
        mix_fp32=False,
    ):
        self.mix_memory = None
        self.mix_program = None
        self.matmul_mix = matmul_mix
        self.mix_compute = ttnn.init_device_compute_kernel_config(
            source.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=mix_fp32,
            packer_l1_acc=False,
        )
        self.fused_gelu = fused_gelu
        # Keep the slower decode merge available for reproducible component A/B
        # checks; the selected runtime only merges expert GELU during prefill.
        self.fuse_decode_gelu = fuse_decode_gelu
        from models.demos.gemma4.tt.experts.decode import _build_sparse_matmul_config

        self.config = source.config
        self.width = source.weights.intermediate_size_per_device
        self.gate_up = ttnn.concat((source.weights.gate_proj, source.weights.up_proj), dim=-1)
        self.down = source.weights.down_proj
        self.sparsity = source.prefill_sparsity
        self.gate_config = _build_sparse_matmul_config(32, 2 * self.width)
        self.down_config = _build_sparse_matmul_config(32, self.config.hidden_size)
        if prefill_batch_tokens <= 0 or prefill_batch_tokens % 32:
            raise ValueError("Internal expert batch must be a positive multiple of one tile")
        self.prefill_batch_tokens = prefill_batch_tokens
        # Prepare every possible physical tail; logical lengths are padded by
        # the inherited public orchestration, without an alignment restriction.
        self.prefill_configs = {
            rows: (
                _build_sparse_matmul_config(rows, 2 * self.width),
                _build_sparse_matmul_config(rows, self.config.hidden_size),
            )
            for rows in range(32, prefill_batch_tokens + 1, 32)
        }
        self.compute = ttnn.init_device_compute_kernel_config(
            source.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def _chunk(self, x, routing, decode):
        cfg = self.config
        sparsity = ttnn.to_layout(routing, ttnn.ROW_MAJOR_LAYOUT) if decode else self.sparsity
        nnz = cfg.top_k if decode else cfg.num_experts
        memory = ttnn.L1_MEMORY_CONFIG if decode else ttnn.DRAM_MEMORY_CONFIG
        kwargs = {} if decode else {"compute_kernel_config": self.compute}
        gate_config, down_config = (self.gate_config, self.down_config) if decode else self.prefill_configs[x.shape[-2]]
        gu = ttnn.sparse_matmul(
            x,
            self.gate_up,
            sparsity=sparsity,
            nnz=nnz,
            memory_config=memory,
            output_tile=ttnn.Tile([32, 32]),
            program_config=gate_config,
            dtype=ttnn.bfloat16,
            **kwargs,
        )
        # A single request's expert-major output already matches the
        # sparse down projection, so no transpose round trip is necessary.
        gu = ttnn.reshape(gu, (1, cfg.num_experts, x.shape[-2], 2 * self.width))
        gate, up = gu[..., : self.width], gu[..., self.width :]
        hidden = (
            geglu(gate, up)
            if self.fused_gelu and (not decode or self.fuse_decode_gelu)
            else ttnn.mul(ttnn.gelu(gate, variant=ttnn.GeluVariant.Accurate), up)
        )
        down = ttnn.sparse_matmul(
            hidden,
            self.down,
            sparsity=sparsity,
            nnz=nnz,
            memory_config=memory,
            output_tile=ttnn.Tile([32, 32]),
            program_config=down_config,
            is_input_a_sparse=True,
            dtype=ttnn.bfloat16,
            **kwargs,
        )
        if decode and self.matmul_mix:
            down = ttnn.reshape(down, (1, cfg.num_experts, 1, cfg.hidden_size))
            return ttnn.matmul(
                routing,
                ttnn.permute(down, (0, 2, 1, 3)),
                dtype=ttnn.bfloat16,
                memory_config=self.mix_memory or memory,
                program_config=self.mix_program,
                compute_kernel_config=self.mix_compute,
            )
        if decode:
            states = ttnn.reshape(ttnn.permute(down, (0, 2, 1, 3)), (1, cfg.num_experts, cfg.hidden_size))
            states = ttnn.mul(states, ttnn.reshape(routing, (1, cfg.num_experts, 1)))
            states = ttnn.unsqueeze_to_4D(ttnn.sum(states, dim=1))
            return ttnn.reshape(states, (1, 1, 1, cfg.hidden_size), (1, 1, 32, cfg.hidden_size))
        down = ttnn.reshape(down, (1, cfg.num_experts, x.shape[-2], cfg.hidden_size))
        weighted = ttnn.mul(down, ttnn.permute(routing, (0, 3, 2, 1)))
        return ttnn.reshape(ttnn.experimental.fast_reduce_nc(weighted, dims=[1]), (1, 1, x.shape[-2], cfg.hidden_size))

    def __call__(self, x, routing):
        if x.shape[-2] == 1:
            return self._chunk(x, routing, True)
        width = self.prefill_batch_tokens
        outputs = [
            self._chunk(
                x[:, :, i : min(i + width, x.shape[-2]), :], routing[:, :, i : min(i + width, x.shape[-2]), :], False
            )
            for i in range(0, x.shape[-2], width)
        ]
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)


def geglu(gate, up):
    return ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)])


class FusedSharedMLP:
    def __init__(self, source, down_weight=None, output_memory=None, block_w=6):
        self.source = source
        self.down_proj = source.down_proj
        if output_memory is not None:
            self.down_weight = ttnn.from_torch(
                down_weight.transpose(-2, -1).unsqueeze(0).unsqueeze(0),
                device=source.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            self.output_memory = output_memory
            spec = output_memory.shard_spec
            end = spec.grid.bounding_box().end
            self.down_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(end.x + 1, end.y + 1),
                in0_block_w=block_w,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=spec.shape[1] // 32,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
            # Replace the setup-only closure so its original device weight can
            # be released; both phases use this same reloaded BF16 weight.
            self.down_proj = self._down_project
            source.down_proj = None

    def _down_project(self, value):
        if value.shape[-2] != 1:
            return ttnn.linear(value, self.down_weight)
        return ttnn.linear(value, self.down_weight, program_config=self.down_program, memory_config=self.output_memory)

    def __call__(self, x):
        gu = self.source.gate_up_proj(x)
        width = self.source._inter_per_device
        return self.down_proj(geglu(gu[..., width:], gu[..., :width]))


class BroadcastRouter:
    def __init__(self, source, normalize=rms_norm):
        self.source = source
        self.normalize = normalize

    def __call__(self, x, normalized=None):
        source = self.source
        router = source.source
        if normalized is None:
            normalized = self.normalize(x, source.epsilon)
        scaled = ttnn.mul(
            normalized,
            source.scale,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, router.scalar_root_size)],
        )
        if x.shape[-2] == 1:
            products = ttnn.mul(source.projection_rows, scaled)
            scores = ttnn.transpose(ttnn.sum(products, dim=-1, keepdim=True), -2, -1)
        else:
            scores = ttnn.linear(scaled, router.proj_weight, dtype=ttnn.float32, compute_kernel_config=source.compute)
        selected, indices = ttnn.topk(scores, k=router.top_k, dim=-1)
        values = ttnn.softmax(selected, dim=-1)
        routing = ttnn.scatter(
            ttnn.zeros_like(ttnn.typecast(scores, ttnn.bfloat16)),
            dim=-1,
            index=indices,
            src=ttnn.typecast(values, ttnn.bfloat16),
        )
        return ttnn.mul(routing, router.per_expert_scale)


class BatchedPagedAttention:
    """Merge independent KV-head matmuls and softmax arithmetic into batches."""

    def __init__(self, source, native_softmax=False, exp_merge=False, centered=False):
        self.source = source
        self.native_softmax = native_softmax
        self.exp_merge = exp_merge
        self.centered = centered

    def __getattr__(self, name):
        return getattr(self.source, name)

    def __call__(self, q, k, v, *, cur_pos_tensor, page_table_tensor, **kwargs):
        cfg = self.config
        block = k.shape[-2]
        if block != 32:
            raise ValueError("Functional paged attention requires 32-token pages")
        pages = page_table_tensor.shape[-1]
        position = ttnn.reshape(ttnn.to_layout(ttnn.typecast(cur_pos_tensor, ttnn.float32), ttnn.TILE_LAYOUT), (1, 1))
        if cfg.is_sliding:
            selected = min(pages, cfg.sliding_window // block + 1)
            start = ttnn.floor(ttnn.mul(ttnn.maximum(ttnn.add(position, 1 - cfg.sliding_window), 0), 1 / block))
            page_ids = ttnn.add(start, self.page_offsets[:, :selected])
            page_ids = ttnn.typecast(ttnn.minimum(page_ids, pages - 1), ttnn.uint32)
            page_ids = ttnn.to_layout(page_ids, ttnn.ROW_MAJOR_LAYOUT)
            physical = ttnn.gather(page_table_tensor, dim=1, index=page_ids)
        else:
            selected = pages
            start = ttnn.mul(position, 0.0)
            # Full attention reads all logical pages in table order.
            physical = page_table_tensor
        rows = self.cache_row_indices(physical)

        def gather(cache):
            flat = ttnn.reshape(cache, (cache.shape[0] * cfg.num_key_value_heads * block, cfg.head_dim))
            gathered = ttnn.embedding(rows, flat, layout=ttnn.TILE_LAYOUT)
            gathered = ttnn.reshape(gathered, (selected, cfg.num_key_value_heads, block, cfg.head_dim))
            gathered = ttnn.permute(gathered, (1, 0, 2, 3))
            return ttnn.reshape(gathered, (1, cfg.num_key_value_heads, selected * block, cfg.head_dim))

        keys, values = gather(k), gather(v)
        absolute = ttnn.add(
            ttnn.reshape(ttnn.mul(start, block), (1, 1, 1, 1)), self.token_offsets[:, :, :, : selected * block]
        )
        pos = ttnn.reshape(position, (1, 1, 1, 1))
        allowed = ttnn.le(absolute, pos)
        if cfg.is_sliding:
            allowed = ttnn.logical_and(allowed, ttnn.gt(absolute, ttnn.subtract(pos, cfg.sliding_window)))
        group = cfg.num_attention_heads // cfg.num_key_value_heads
        query = ttnn.reshape(q, (1, cfg.num_key_value_heads, group, cfg.head_dim))
        scores = ttnn.matmul(query, keys, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=self.compute)
        scores = ttnn.where(allowed, scores, -1.0e30)
        if self.native_softmax:
            softmax_input = ttnn.subtract(scores, ttnn.max(scores, dim=-1, keepdim=True)) if self.centered else scores
            probabilities = ttnn.softmax(
                softmax_input, dim=-1, numeric_stable=not self.centered, compute_kernel_config=self.compute
            )
        else:
            maximum = ttnn.max(scores, dim=-1, keepdim=True)
            if self.exp_merge:
                probabilities = ttnn.subtract(
                    scores, maximum, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, 0.0)]
                )
            else:
                probabilities = ttnn.exp(ttnn.subtract(scores, maximum), fast_and_approximate_mode=False)
            probabilities = ttnn.div(probabilities, ttnn.sum(probabilities, dim=-1, keepdim=True))
        result = ttnn.matmul(probabilities, values, dtype=ttnn.float32, compute_kernel_config=self.compute)
        return ttnn.reshape(result, (1, 1, cfg.num_attention_heads, cfg.head_dim))


class FusedAttention(DecodeAttention):
    def __init__(self, source, normalize, fuse_rope):
        self.source = source.source
        self.compute = source.compute
        self.q_weight, self.k_weight = source.q_weight, source.k_weight
        self.decode_sdpa = source.decode_sdpa
        self.tail = None
        self.normalize = normalize
        self.fuse_rope = fuse_rope
        self.fuse_cache = False
        self.common_kv = False
        self.pack_heads = False
        self.update_memories = tuple(
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x, 0), ttnn.CoreCoord(x, 0))}),
                    [32, self.source.config.head_dim],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            for x in (0, 1)
        )

    def rotary(self, value, cos, sin, *, decode=False):
        if not self.fuse_rope or (self.fuse_rope == "decode" and not decode):
            return rotary(value, cos, sin, decode=decode)
        if decode:
            tables = tuple(ttnn.repeat(t[:, :, :1, :], (1, 1, value.shape[-2], 1)) for t in (cos, sin))
        else:
            tables = tuple(t[:, :, : value.shape[-2], :] for t in (cos, sin))
        tables = tuple(ttnn.typecast(t, ttnn.float32) for t in tables)
        return ttnn.experimental.rotary_embedding_hf(value, *tables, compute_kernel_config=self.compute)

    def project(self, attention, decode):
        if not decode or not self.project_sharded:
            return super().project(attention, decode)
        cfg = self.source.config
        memory = ttnn.create_sharded_memory_config(
            shape=(32, cfg.head_dim),
            core_grid=ttnn.CoreGrid(x=1, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        combined = ttnn.experimental.nlp_concat_heads_decode(
            ttnn.to_memory_config(attention, memory),
            num_heads=cfg.num_attention_heads,
        )
        combined = ttnn.reshape(
            combined,
            (1, 1, 1, cfg.num_attention_heads * cfg.head_dim),
            (1, 1, 32, cfg.num_attention_heads * cfg.head_dim),
        )
        return ttnn.linear(
            combined,
            self.source.weights.o_proj,
            dtype=ttnn.float32,
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def heads(self, hidden_states, decode):
        cfg = self.source.config
        qkv = self.source.weights.wqkv(hidden_states)
        split = split_qkv_heads_decode if decode else split_qkv_heads_prefill
        q, k, v = split(qkv, cfg, self.source.weights.is_global)
        memory = q.memory_config()
        if self.pack_heads:
            tensors = [ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in (q, k, v)]
            shapes = [t.shape for t in tensors]
            rows = [t.shape[-3] * t.shape[-2] for t in tensors]
            packed = ttnn.concat([ttnn.reshape(t, (1, 1, n, cfg.head_dim)) for t, n in zip(tensors, rows)], dim=-2)
            normalized = self.normalize(packed, cfg.rms_norm_eps)
            result, start = [], 0
            for n, shape in zip(rows, shapes):
                result.append(ttnn.reshape(normalized[:, :, start : start + n, :], shape))
                start += n
            q, k, v = result
            q, k = ttnn.mul(q, self.q_weight), ttnn.mul(k, self.k_weight)
        elif self.common_kv and self.source.weights.is_global:
            q, k = [ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in (q, k)]
            v = self.normalize(k, cfg.rms_norm_eps)
            k = ttnn.mul(v, self.k_weight)
            q = self.normalize(q, cfg.rms_norm_eps, self.q_weight)
        else:
            q, k, v = [ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in (q, k, v)]
            q = self.normalize(q, cfg.rms_norm_eps, self.q_weight)
            k = self.normalize(k, cfg.rms_norm_eps, self.k_weight)
            v = self.normalize(v, cfg.rms_norm_eps)
        return q, k, v, memory

    def cache_cast(self, value, dtype, memory):
        if self.i2s_cast:
            return ttnn.interleaved_to_sharded(value, memory, output_dtype=dtype)
        if self.unary_cast_shard:
            return ttnn.unary_chain(
                value,
                [ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, value.dtype.value, dtype.value)],
                memory_config=memory,
            )
        if self.cast_shard:
            return ttnn.typecast(value, dtype, memory_config=memory)
        return ttnn.to_memory_config(ttnn.typecast(value, dtype), memory)

    def decode(self, hidden_states, **kwargs):
        if hidden_states.shape[-2] != 1:
            raise ValueError("DecodeAttention expects one request slot")
        cfg = self.source.config
        q, k, v, cache_memory = self.heads(hidden_states, True)
        cos, sin = (
            ttnn.unsqueeze_to_4D(ttnn.embedding(kwargs["position_idx"], table, layout=ttnn.TILE_LAYOUT))
            for table in kwargs["rope_mats"]
        )
        q, k = self.rotary(q, cos, sin, decode=True), self.rotary(k, cos, sin, decode=True)
        cache_pos, page_table = kwargs["position_idx_cache"], kwargs["page_table"]
        k_cache, v_cache = kwargs["kv_cache"]
        if self.fuse_cache:
            updates = [self.cache_cast(t, ttnn.bfloat16, mem) for t, mem in zip((k, v), self.update_memories)]
            ttnn.experimental.paged_fused_update_cache(
                k_cache,
                updates[0],
                v_cache,
                updates[1],
                update_idxs_tensor=cache_pos,
                page_table=page_table,
            )
        else:
            for cache, update in ((k_cache, k), (v_cache, v)):
                update = self.cache_cast(update, cache.dtype, cache_memory)
                ttnn.experimental.paged_update_cache(
                    cache,
                    update,
                    update_idxs_tensor=cache_pos,
                    page_table=page_table,
                    block_size=cache.shape[-2],
                    num_kv_heads=cfg.num_key_value_heads,
                )
        attention = self.decode_sdpa(q, k_cache, v_cache, cur_pos_tensor=cache_pos, page_table_tensor=page_table)
        return self.project(attention, True)

    def prefill(self, hidden_states, **kwargs):
        cfg = self.source.config
        q, k, v, _ = self.heads(hidden_states, False)
        cos, sin = kwargs["rope_mats"]
        q, k = self.rotary(q, cos, sin), self.rotary(k, cos, sin)
        q, k, v = [ttnn.typecast(t, ttnn.bfloat16) for t in (q, k, v)]
        page_table, user_id = kwargs["page_table"], kwargs["user_id"]
        fill_table = kwargs.get("chunk_page_table")
        fill_table = page_table if fill_table is None else fill_table
        valid = kwargs["valid_seq_len"]
        fill_length = (valid + 31) // 32 * 32
        for cache, update in zip(kwargs["kv_cache"], (k, v)):
            ttnn.experimental.paged_fill_cache(
                cache, update[:, :, :fill_length, :], fill_table, batch_idx=user_id, block_size=cache.shape[-2]
            )

        start = kwargs.get("chunk_start_idx")
        if cfg.is_sliding:
            history = 0
            if self.tail is not None:
                history = self.tail[0].shape[-2]
                q_attention = ttnn.concat((q[:, :, :history, :], q), dim=2)
                k_attention = ttnn.concat((self.tail[0], k), dim=2)
                v_attention = ttnn.concat((self.tail[1], v), dim=2)
            else:
                q_attention, k_attention, v_attention = q, k, v
            attention = ttnn.transformer.scaled_dot_product_attention(
                q_attention,
                k_attention,
                v_attention,
                is_causal=True,
                scale=1.0,
                sliding_window_size=cfg.sliding_window,
                program_config=prefill_sdpa_program_config(cfg.head_dim, q_attention.shape[-2], cfg.sliding_window),
                compute_kernel_config=self.compute,
            )
            if history:
                attention = attention[:, :, history : history + q.shape[-2], :]
            # All non-final physical chunks cover a full sliding window. A
            # fresh request clears this tail; prefix continuation uses decode.
            tail_length = min(cfg.sliding_window, k.shape[-2])
            self.tail = tuple(ttnn.clone(t[:, :, -tail_length:, :]) for t in (k, v))
        elif start is not None and start != 0:
            attention = chunked_prefill_sdpa(
                q,
                *kwargs["kv_cache"],
                page_table,
                user_id,
                cfg.head_dim,
                scale=1.0,
                base_offset=start,
                num_kv_heads=cfg.num_key_value_heads,
            )
        else:
            attention = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=1.0,
                program_config=prefill_sdpa_program_config(cfg.head_dim, q.shape[-2]),
                compute_kernel_config=self.compute,
            )
        return self.project(attention, False)


class TiedQKV(BroadcastQKV):
    """Project tied K/V once, then restore the packed head-split contract."""

    def __init__(self, source, kv_width):
        self.group_size = source.group_size
        self.decode_memory = source.decode_memory
        self.compute = source.compute
        self.kv_width = kv_width
        self.width = source.weight.shape[-1] - kv_width
        self.weight = source.weight[..., : self.width]
        self.rows = source.rows[:, :, : self.width, :]

    def __call__(self, hidden_states, compute_kernel_config=None, out_memory_config=None):
        qk = super().__call__(hidden_states, compute_kernel_config, out_memory_config)
        return ttnn.concat(
            (qk, qk[..., self.width - self.kv_width :]),
            dim=-1,
            memory_config=self.decode_memory if hidden_states.shape[-2] == 1 else ttnn.DRAM_MEMORY_CONFIG,
        )
