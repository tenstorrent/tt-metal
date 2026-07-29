# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Optimized single-device North-Mini decoder.

This path starts from the graph-fused decoder and owns every measured decode
operation that is changed by Stage 03.  Candidate selection is construction
time only so traced replay never branches or falls back to the functional path.
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tt.fused_decoder import FusedDecoder


@dataclass(frozen=True)
class OptimizationPolicy:
    attention_weight_dtype: object = ttnn.bfloat16
    mlp_weight_dtype: object = ttnn.bfloat16
    cache_dtype: object = ttnn.bfloat16
    attention_fidelity: object = ttnn.MathFidelity.HiFi2
    mlp_fidelity: object = ttnn.MathFidelity.HiFi2
    advisor_seed: bool = False
    geometry_block: int | None = None
    sparse_geometry_block: int | None = None


POLICIES = {
    "baseline": OptimizationPolicy(),
    "advisor_seed": OptimizationPolicy(advisor_seed=True),
    "bfp8_hifi2": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_weight_dtype=ttnn.bfloat8_b,
    ),
    "bfp8_lofi": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_weight_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
    ),
    "bfp8_cache": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_weight_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
    ),
    "bfp4_attention": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_weight_dtype=ttnn.bfloat8_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
    ),
    "bfp4_mlp": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_weight_dtype=ttnn.bfloat4_b,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
    ),
    "all_bfp4": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat4_b,
        mlp_weight_dtype=ttnn.bfloat4_b,
        attention_fidelity=ttnn.MathFidelity.LoFi,
        mlp_fidelity=ttnn.MathFidelity.LoFi,
    ),
    "geometry_8": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_weight_dtype=ttnn.bfloat8_b,
        geometry_block=8,
    ),
    "geometry_2": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_weight_dtype=ttnn.bfloat8_b,
        geometry_block=2,
    ),
    "geometry_4": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_weight_dtype=ttnn.bfloat8_b,
        geometry_block=4,
    ),
    "geometry_16": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_weight_dtype=ttnn.bfloat8_b,
        geometry_block=16,
    ),
    "sparse_geometry_2": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_weight_dtype=ttnn.bfloat8_b,
        sparse_geometry_block=2,
    ),
    "sparse_geometry_4": OptimizationPolicy(
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_weight_dtype=ttnn.bfloat8_b,
        sparse_geometry_block=4,
    ),
}


def _dram_weight_memory_config(mesh_device, k, n):
    dram = mesh_device.dram_grid_size()
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, 0))})
    shard_n = ((n + dram.x * ttnn.TILE_SIZE - 1) // (dram.x * ttnn.TILE_SIZE)) * ttnn.TILE_SIZE
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (k, shard_n), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _l1_width_memory_config(mesh_device, width, cores):
    grid_size = mesh_device.compute_with_storage_grid_size()
    grid = ttnn.num_cores_to_corerangeset(cores, grid_size, row_wise=True)
    shard_width = ((width + cores * ttnn.TILE_SIZE - 1) // (cores * ttnn.TILE_SIZE)) * ttnn.TILE_SIZE
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, shard_width),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


class OptimizedDecoder(FusedDecoder):
    """Fused North-Mini graph with explicit optimization policy/configs."""

    @classmethod
    def from_state_dict(cls, state_dict, *, candidate="advisor_seed", **kwargs):
        if candidate not in POLICIES:
            raise ValueError(f"unknown optimization candidate {candidate!r}; expected one of {sorted(POLICIES)}")
        decoder = super().from_state_dict(state_dict, **kwargs)
        decoder.candidate = candidate
        decoder.policy = POLICIES[candidate]
        if decoder.policy.advisor_seed:
            decoder.policy = OptimizationPolicy(
                attention_weight_dtype=ttnn.bfloat8_b,
                mlp_weight_dtype=ttnn.bfloat8_b,
                advisor_seed=True,
            )
        decoder.use_advisor_decode = decoder.policy.advisor_seed and decoder.mlp_type == "dense" and decoder.batch == 1
        decoder.advisor_weights = {}
        attention_names = ("qkv", "o")
        mlp_names = ("gate_up", "down_proj", "expert_gate_up", "expert_down")
        for name in attention_names + mlp_names:
            if name not in decoder.weights:
                continue
            target_dtype = (
                decoder.policy.attention_weight_dtype if name in attention_names else decoder.policy.mlp_weight_dtype
            )
            if target_dtype != ttnn.bfloat16:
                original = decoder.weights[name]
                decoder.weights[name] = ttnn.typecast(original, target_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                original.deallocate(True)
            if decoder.use_advisor_decode and name in ("qkv", "o", "gate_up", "down_proj"):
                k, n = tuple(decoder.weights[name].shape)[-2:]
                decoder.advisor_weights[name] = ttnn.to_memory_config(
                    decoder.weights[name], _dram_weight_memory_config(decoder.mesh_device, k, n)
                )
        if "router" in decoder.weights and candidate != "baseline":
            original = decoder.weights["router"]
            decoder.weights["router"] = ttnn.typecast(original, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            original.deallocate(True)
        decoder.attention_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            decoder.mesh_device.arch(),
            math_fidelity=decoder.policy.attention_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        decoder.mlp_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            decoder.mesh_device.arch(),
            math_fidelity=decoder.policy.mlp_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        decoder.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )
        decoder.advisor_input_2048_memcfg = _l1_width_memory_config(decoder.mesh_device, 2048, 8)
        decoder.advisor_input_4096_memcfg = _l1_width_memory_config(decoder.mesh_device, 4096, 8)
        decoder.advisor_input_3072_memcfg = _l1_width_memory_config(decoder.mesh_device, 3072, 8)
        decoder.advisor_qkv_output_memcfg = _l1_width_memory_config(decoder.mesh_device, 5120, 80)
        decoder.advisor_hidden_output_memcfg = _l1_width_memory_config(decoder.mesh_device, 2048, 64)
        decoder.advisor_gate_up_output_memcfg = _l1_width_memory_config(decoder.mesh_device, 6144, 96)
        decoder.advisor_qkv_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=8, per_core_M=1, per_core_N=2
        )
        decoder.advisor_o_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=16, per_core_M=1, per_core_N=1
        )
        decoder.advisor_gate_up_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=8, per_core_M=1, per_core_N=2
        )
        decoder.advisor_down_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=12,
            per_core_M=1,
            per_core_N=1,
        )
        if decoder.policy.geometry_block is not None:
            block = decoder.policy.geometry_block
            common = dict(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=block,
                out_subblock_h=1,
                per_core_M=16,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
            decoder.geometry_wide_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                out_subblock_w=3, per_core_N=3, **common
            )
            decoder.geometry_narrow_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=1,
                per_core_N=2,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        if decoder.policy.sparse_geometry_block is not None:
            sparse_common = dict(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=decoder.policy.sparse_geometry_block,
                out_subblock_h=1,
                per_core_M=16,
                transpose_mcast=False,
                fused_activation=None,
            )
            decoder.sparse_gate_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                out_subblock_w=2, per_core_N=6, **sparse_common
            )
            decoder.sparse_down_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                out_subblock_w=4, per_core_N=8, **sparse_common
            )
        return decoder

    def create_paged_kv_cache(self, *, num_blocks: int | None = None):
        cache = super().create_paged_kv_cache(num_blocks=num_blocks)
        if self.policy.cache_dtype == ttnn.bfloat16:
            return cache
        converted = tuple(
            ttnn.typecast(tensor, self.policy.cache_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG) for tensor in cache
        )
        for tensor in cache:
            tensor.deallocate(True)
        return converted

    def _dense_mlp(self, normalized):
        use_advisor = self.use_advisor_decode and normalized.shape[2] == 1
        if use_advisor:
            normalized = ttnn.to_memory_config(normalized, self.advisor_input_2048_memcfg)
        packed = ttnn.linear(
            normalized,
            self.advisor_weights["gate_up"] if use_advisor else self.weights["gate_up"],
            dtype=ttnn.bfloat16,
            memory_config=self.advisor_gate_up_output_memcfg if use_advisor else ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.advisor_gate_up_program_config
            if use_advisor
            else self.geometry_wide_program_config
            if self.policy.geometry_block is not None and normalized.shape[2] == 1
            else None,
            compute_kernel_config=self.mlp_compute_kernel_config,
        )
        if self.dense_gate_up_variant == "packed_slice" and normalized.shape[2] != 1:
            gate = ttnn.slice(
                packed, (0, 0, 0, 0), (packed.shape[0], packed.shape[1], packed.shape[2], self.intermediate_size)
            )
            up = ttnn.slice(
                packed,
                (0, 0, 0, self.intermediate_size),
                (packed.shape[0], packed.shape[1], packed.shape[2], 2 * self.intermediate_size),
            )
        else:
            gate, up = ttnn.split(packed, self.intermediate_size, dim=-1)
        activated = self._swiglu(gate, up)
        if use_advisor:
            activated = ttnn.to_memory_config(activated, self.advisor_input_3072_memcfg)
        output = ttnn.linear(
            activated,
            self.advisor_weights["down_proj"] if use_advisor else self.weights["down_proj"],
            dtype=ttnn.bfloat16,
            memory_config=self.advisor_hidden_output_memcfg if use_advisor else ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.advisor_down_program_config
            if use_advisor
            else self.geometry_narrow_program_config
            if self.policy.geometry_block is not None and normalized.shape[2] == 1
            else None,
            compute_kernel_config=self.mlp_compute_kernel_config,
        )
        return output

    def _sparse_moe_chunk(self, normalized, token_count):
        """Expose expert matmul geometry candidates while retaining device routing."""
        flat = ttnn.reshape(normalized, (token_count, self.hidden_size))
        logits = ttnn.linear(flat, self.weights["router"], dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        top_values, top_indices = ttnn.topk(logits, k=self.top_k, dim=-1, sorted=True)
        top_values = ttnn.sigmoid(top_values)
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)
        expert_input = ttnn.reshape(flat, (1, token_count, self.hidden_size))
        expert_input = ttnn.repeat(expert_input, ttnn.Shape((self.num_experts, 1, 1)))
        use_sparse_geometry = self.policy.sparse_geometry_block is not None
        gate_up = ttnn.matmul(
            expert_input,
            self.weights["expert_gate_up"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.sparse_gate_program_config if use_sparse_geometry else None,
            compute_kernel_config=self.mlp_compute_kernel_config,
        )
        gate, up = ttnn.split(gate_up, self.intermediate_size, dim=-1)
        activated = self._swiglu(gate, up)
        expert_output = ttnn.matmul(
            activated,
            self.weights["expert_down"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.sparse_down_program_config if use_sparse_geometry else None,
            compute_kernel_config=self.mlp_compute_kernel_config,
        )
        routing = ttnn.permute(routing, (1, 0))
        routing = ttnn.reshape(routing, (self.num_experts, token_count, 1))
        return ttnn.sum(ttnn.multiply(expert_output, routing), dim=0)

    def _attention_prefill(
        self,
        normalized,
        *,
        key_cache,
        value_cache,
        page_table,
        position_cos,
        position_sin,
        seq_len,
    ):
        query, key, value = self._qkv_prefill(normalized, seq_len, position_cos, position_sin)
        cache_key = key
        cache_value = value
        if self.policy.cache_dtype != ttnn.bfloat16:
            cache_key = ttnn.typecast(key, self.policy.cache_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            cache_value = ttnn.typecast(value, self.policy.cache_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for user in range(self.batch):
            key_user = ttnn.slice(cache_key, (user, 0, 0, 0), (user + 1, self.num_kv_heads, seq_len, self.head_dim))
            value_user = ttnn.slice(cache_value, (user, 0, 0, 0), (user + 1, self.num_kv_heads, seq_len, self.head_dim))
            ttnn.experimental.paged_fill_cache(key_cache, key_user, page_table, batch_idx=user)
            ttnn.experimental.paged_fill_cache(value_cache, value_user, page_table, batch_idx=user)
        attended = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.transformer.concatenate_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(
            attended,
            self.weights["o"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_kernel_config,
        )

    def _qkv_prefill(self, normalized, seq_len, position_cos, position_sin):
        fused = ttnn.linear(
            normalized,
            self.weights["qkv"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        fused = ttnn.reshape(fused, (self.batch, seq_len, -1))
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.use_rope:
            query = ttnn.experimental.rotary_embedding(
                query, position_cos, position_sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            key = ttnn.experimental.rotary_embedding(
                key, position_cos, position_sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            query = ttnn.slice(query, (0, 0, 0, 0), (self.batch, self.num_heads, seq_len, self.head_dim))
            key = ttnn.slice(key, (0, 0, 0, 0), (self.batch, self.num_kv_heads, seq_len, self.head_dim))
        return query, key, value

    def _qkv_decode(self, normalized, position_cos, position_sin):
        use_advisor = self.use_advisor_decode
        if use_advisor:
            normalized = ttnn.to_memory_config(normalized, self.advisor_input_2048_memcfg)
        fused = ttnn.linear(
            normalized,
            self.advisor_weights["qkv"] if use_advisor else self.weights["qkv"],
            dtype=ttnn.bfloat16,
            memory_config=self.advisor_qkv_output_memcfg if use_advisor else ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.advisor_qkv_program_config
            if use_advisor
            else self.geometry_wide_program_config
            if self.policy.geometry_block is not None
            else None,
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        if use_advisor:
            fused = ttnn.to_memory_config(fused, ttnn.L1_MEMORY_CONFIG)
        fused = ttnn.reshape(fused, (1, 1, self.batch, -1))
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        if self.use_rope:
            position_cos = ttnn.interleaved_to_sharded(position_cos, self.decode_rope_memory_config)
            position_sin = ttnn.interleaved_to_sharded(position_sin, self.decode_rope_memory_config)
            query = ttnn.experimental.rotary_embedding_hf(
                query,
                position_cos,
                position_sin,
                is_decode_mode=True,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )
            key = ttnn.experimental.rotary_embedding_hf(
                key,
                position_cos,
                position_sin,
                is_decode_mode=True,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )
        return query, key, value

    def _attention_decode(
        self,
        normalized,
        *,
        key_cache,
        value_cache,
        page_table,
        current_positions,
        position_cos,
        position_sin,
    ):
        query, key, value = self._qkv_decode(normalized, position_cos, position_sin)
        ttnn.experimental.paged_update_cache(
            key_cache, key, update_idxs_tensor=current_positions, page_table=page_table
        )
        ttnn.experimental.paged_update_cache(
            value_cache, value, update_idxs_tensor=current_positions, page_table=page_table
        )
        attended = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_positions,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            program_config=self.decode_sdpa_program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.to_memory_config(attended, self.decode_concat_memory_config)
        attended = ttnn.experimental.nlp_concat_heads_decode(
            attended,
            num_heads=self.num_heads,
            sub_core_grids=self.decode_sub_core_grids,
        )
        use_advisor = self.use_advisor_decode
        if use_advisor:
            attended = ttnn.to_memory_config(attended, self.advisor_input_4096_memcfg)
        projected = ttnn.linear(
            attended,
            self.advisor_weights["o"] if use_advisor else self.weights["o"],
            dtype=ttnn.bfloat16,
            memory_config=self.advisor_hidden_output_memcfg if use_advisor else ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.advisor_o_program_config
            if use_advisor
            else self.geometry_narrow_program_config
            if self.policy.geometry_block is not None
            else None,
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        if self.policy.advisor_seed:
            projected = ttnn.to_memory_config(projected, ttnn.DRAM_MEMORY_CONFIG)
        if projected.shape[2] != self.batch:
            projected = ttnn.slice(projected, (0, 0, 0, 0), (1, 1, self.batch, self.hidden_size))
        if self.batch == 32:
            return ttnn.permute(projected, (0, 2, 1, 3))
        return ttnn.reshape(projected, (1, self.batch, 1, self.hidden_size))
