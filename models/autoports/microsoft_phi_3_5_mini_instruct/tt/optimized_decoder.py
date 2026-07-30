# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-device optimized Phi-3.5 decoder layer.

This class deliberately owns the optimization policy while reusing the
functional stage's validated paged-cache and LongRoPE contract. Runtime
overrides are added here as candidates are proven; the final optimized tests
reject construction of ``FunctionalDecoder`` and inspect this class directly.
"""

from __future__ import annotations

import math

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.functional_decoder import (
    PREFILL_SDPA_MAX_SEQ,
    FunctionalDecoder,
)


class OptimizedDecoder(FunctionalDecoder):
    """Phi-3.5 dense decoder with phase- and projection-specific policies."""

    WEIGHT_DTYPES = {
        "qkv": ttnn.bfloat8_b,
        "o_proj": ttnn.bfloat8_b,
        "gate_up": ttnn.bfloat4_b,
        "down": ttnn.bfloat8_b,
    }
    DECODE_WEIGHT_DTYPES = {
        "qkv": ttnn.bfloat4_b,
        "o_proj": ttnn.bfloat4_b,
        "gate_up": ttnn.bfloat4_b,
        "down": ttnn.bfloat4_b,
    }

    @staticmethod
    def _compute_kernel_config(fidelity):
        return ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    @staticmethod
    def _dram_weight_memory_config(k, n):
        dram_cores = 8
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
        shard_spec = ttnn.ShardSpec(
            cores,
            (k, math.ceil(n / (ttnn.TILE_SIZE * dram_cores)) * ttnn.TILE_SIZE),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    @staticmethod
    def _decode_input_memory_config(k, num_cores=8):
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, k // num_cores),
            core_grid=ttnn.CoreGrid(x=num_cores, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    @staticmethod
    def _decode_program_config(in0_block_w, per_core_n):
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w,
            per_core_M=1,
            per_core_N=per_core_n,
            fused_activation=None,
        )

    def _decode_output_memory_config(self, name):
        per_core_n = self.decode_config_values[name][1]
        n = {
            "qkv": 3 * self.hidden_size,
            "o_proj": self.hidden_size,
            "gate_up": 2 * self.intermediate_size,
            "down": self.hidden_size,
            "q": self.hidden_size,
            "k": self.hidden_size,
            "v": self.hidden_size,
            "gate": self.intermediate_size,
            "up": self.intermediate_size,
        }[name]
        num_cores = math.ceil(n / (ttnn.TILE_SIZE * per_core_n))
        core_grid = ttnn.num_cores_to_corerangeset(
            num_cores, self.mesh_device.compute_with_storage_grid_size(), row_wise=True
        )
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, per_core_n * ttnn.TILE_SIZE),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        """Reuse validated loading/LongRoPE setup, then materialize named policy.

        The base object is setup-only: the returned instance is an
        ``OptimizedDecoder`` and all measured calls dispatch through this
        class. Weight conversion remains outside runtime forwards.
        """
        weight_dtype_overrides = kwargs.pop("weight_dtype_overrides", None) or {}
        prefill_weight_dtype_overrides = kwargs.pop("prefill_weight_dtype_overrides", None) or {}
        kv_cache_dtype = kwargs.pop("kv_cache_dtype", ttnn.bfloat8_b)
        split_decode_projections = kwargs.pop("split_decode_projections", False)
        decode_config_overrides = kwargs.pop("decode_config_overrides", None) or {}
        decode_input_core_overrides = kwargs.pop("decode_input_core_overrides", None) or {}
        fidelity_overrides = kwargs.pop("fidelity_overrides", None) or {}
        functional = FunctionalDecoder.from_state_dict(state_dict, **kwargs)
        source_weights = dict(functional.weights)
        weights = dict(source_weights)
        selected_weight_dtypes = dict(cls.WEIGHT_DTYPES)
        selected_weight_dtypes.update(prefill_weight_dtype_overrides)
        for name, dtype in selected_weight_dtypes.items():
            weights[name] = ttnn.typecast(weights[name], dtype)
        selected_decode_weight_dtypes = dict(cls.DECODE_WEIGHT_DTYPES)
        selected_decode_weight_dtypes.update(weight_dtype_overrides)
        typed_decode_weights = {
            name: ttnn.typecast(source_weights[name], dtype) for name, dtype in selected_decode_weight_dtypes.items()
        }
        decode_weights = {
            "qkv": ttnn.to_memory_config(
                typed_decode_weights["qkv"],
                cls._dram_weight_memory_config(functional.hidden_size, 3 * functional.hidden_size),
            ),
            "o_proj": ttnn.to_memory_config(
                typed_decode_weights["o_proj"],
                cls._dram_weight_memory_config(functional.hidden_size, functional.hidden_size),
            ),
            "gate_up": ttnn.to_memory_config(
                typed_decode_weights["gate_up"],
                cls._dram_weight_memory_config(functional.hidden_size, 2 * functional.intermediate_size),
            ),
            "down": ttnn.to_memory_config(
                typed_decode_weights["down"],
                cls._dram_weight_memory_config(functional.intermediate_size, functional.hidden_size),
            ),
        }
        instance = cls(
            hf_config=functional.hf_config,
            layer_idx=functional.layer_idx,
            mesh_device=functional.mesh_device,
            batch=functional.batch,
            max_context=functional.max_context,
            page_size=functional.page_size,
            weights=weights,
            short_cos=functional.short_cos,
            short_sin=functional.short_sin,
            long_cos=functional.long_cos,
            long_sin=functional.long_sin,
        )
        instance.decode_weights = decode_weights
        selected_decode_configs = {
            "qkv": (12, 3),
            "o_proj": (12, 1),
            "gate_up": (6, 5),
            "down": (32, 1),
        }
        selected_decode_configs.update(decode_config_overrides)
        instance.decode_config_values = selected_decode_configs
        instance.decode_input_core_overrides = decode_input_core_overrides
        instance.decode_program_configs = {
            name: cls._decode_program_config(*values) for name, values in selected_decode_configs.items()
        }
        instance.decode_fidelities = {
            "qkv": ttnn.MathFidelity.LoFi,
            "o_proj": ttnn.MathFidelity.LoFi,
            "gate_up": ttnn.MathFidelity.LoFi,
            "down": ttnn.MathFidelity.LoFi,
        }
        instance.decode_fidelities.update(fidelity_overrides)
        instance.selected_weight_dtypes = selected_weight_dtypes
        instance.selected_decode_weight_dtypes = selected_decode_weight_dtypes
        instance.kv_cache_dtype = kv_cache_dtype
        instance.split_decode_projections = split_decode_projections
        if split_decode_projections:
            split_specs = {
                "q": ("qkv", 0, functional.hidden_size),
                "k": ("qkv", functional.hidden_size, 2 * functional.hidden_size),
                "v": ("qkv", 2 * functional.hidden_size, 3 * functional.hidden_size),
                "gate": ("gate_up", 0, functional.intermediate_size),
                "up": ("gate_up", functional.intermediate_size, 2 * functional.intermediate_size),
            }
            split_configs = {
                "q": (12, 1),
                "k": (12, 1),
                "v": (12, 1),
                "gate": (6, 4),
                "up": (6, 4),
            }
            for name, (packed_name, start, end) in split_specs.items():
                packed = typed_decode_weights[packed_name]
                split = ttnn.slice(packed, [0, start], [tuple(packed.shape)[-2], end])
                instance.decode_weights[name] = ttnn.to_memory_config(
                    split, cls._dram_weight_memory_config(tuple(split.shape)[-2], end - start)
                )
                instance.decode_config_values[name] = split_configs[name]
                instance.decode_program_configs[name] = cls._decode_program_config(*split_configs[name])
                instance.decode_fidelities[name] = ttnn.MathFidelity.LoFi
        return instance

    def create_paged_kv_cache(self, *, num_physical_blocks=None):
        blocks_per_user = math.ceil(self.max_context / self.page_size)
        num_physical_blocks = num_physical_blocks or self.batch * blocks_per_user
        shape = (num_physical_blocks, self.num_kv_heads, self.page_size, self.head_dim)
        cache_kwargs = dict(
            dtype=self.kv_cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.zeros(shape, **cache_kwargs), ttnn.zeros(shape, **cache_kwargs)

    def _mlp(self, hidden_states):
        """Packed gate/up candidate with group-specific reduced precision."""
        normalized = self._norm(hidden_states, self.weights["post_norm"])
        gate_up = ttnn.linear(
            normalized,
            self.weights["gate_up"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config(ttnn.MathFidelity.LoFi),
        )
        gate_up_shape = tuple(gate_up.shape)
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [*gate_up_shape[:-1], self.intermediate_size])
        up = ttnn.slice(
            gate_up,
            [0, 0, 0, self.intermediate_size],
            [*gate_up_shape[:-1], 2 * self.intermediate_size],
        )
        activated = ttnn.multiply(ttnn.silu(gate), up)
        down = ttnn.linear(
            activated,
            self.weights["down"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config(ttnn.MathFidelity.HiFi2),
        )
        return ttnn.add(hidden_states, down)

    def _decode_rope(self, query, key, current_positions, *, use_long_rope):
        """Explicit decode layout boundary; no runtime layout introspection."""
        cos_table = self.long_cos if use_long_rope else self.short_cos
        sin_table = self.long_sin if use_long_rope else self.short_sin
        rope_positions = ttnn.typecast(current_positions, ttnn.uint32)
        cos = ttnn.embedding(rope_positions, cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rope_positions, sin_table, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.reshape(cos, [1, 1, self.batch, self.head_dim])
        sin = ttnn.reshape(sin, [1, 1, self.batch, self.head_dim])
        query = self._apply_rope(ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
        key = self._apply_rope(ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
        cache_input_memory_config = self._decode_concat_memory_config()
        return (
            ttnn.to_memory_config(query, cache_input_memory_config),
            ttnn.to_memory_config(key, cache_input_memory_config),
        )

    def _decode_linear(self, name, hidden_states, fidelity):
        num_input_cores = self.decode_input_core_overrides.get(name, 8)
        hidden_states = ttnn.to_memory_config(
            hidden_states,
            self._decode_input_memory_config(tuple(self.decode_weights[name].shape)[-2], num_input_cores),
        )
        return ttnn.linear(
            hidden_states,
            self.decode_weights[name],
            dtype=ttnn.bfloat16,
            memory_config=self._decode_output_memory_config(name),
            program_config=self.decode_program_configs[name],
            compute_kernel_config=self._compute_kernel_config(fidelity),
        )

    def _decode_norm(self, hidden_states, weight):
        """Keep the residual in the QKV/MLP input shard through RMSNorm."""
        memory_config = self._decode_input_memory_config(self.hidden_size)
        hidden_states = ttnn.to_memory_config(hidden_states, memory_config)
        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 1],
            subblock_w=4,
            block_h=1,
            block_w=self.hidden_size // (8 * ttnn.TILE_SIZE),
            inplace=False,
        )
        return ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=weight,
            memory_config=memory_config,
            program_config=program_config,
            compute_kernel_config=self._compute_kernel_config(ttnn.MathFidelity.HiFi2),
        )

    def _decode_mlp(self, hidden_states):
        normalized = self._decode_norm(hidden_states, self.weights["post_norm"])
        if self.split_decode_projections:
            gate = self._decode_linear("gate", normalized, self.decode_fidelities["gate"])
            up = self._decode_linear("up", normalized, self.decode_fidelities["up"])
            gate_up = ttnn.concat(
                (
                    ttnn.to_memory_config(gate, ttnn.L1_MEMORY_CONFIG),
                    ttnn.to_memory_config(up, ttnn.L1_MEMORY_CONFIG),
                ),
                dim=-1,
            )
        else:
            gate_up = self._decode_linear("gate_up", normalized, self.decode_fidelities["gate_up"])
        gate_up = ttnn.to_memory_config(gate_up, ttnn.L1_MEMORY_CONFIG)
        gate_up_shape = tuple(gate_up.shape)
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [*gate_up_shape[:-1], self.intermediate_size])
        up = ttnn.slice(
            gate_up,
            [0, 0, 0, self.intermediate_size],
            [*gate_up_shape[:-1], 2 * self.intermediate_size],
        )
        activated = ttnn.multiply(ttnn.silu(gate), up)
        down = self._decode_linear("down", activated, self.decode_fidelities["down"])
        output_memory_config = self._decode_output_memory_config("down")
        residual = ttnn.to_memory_config(hidden_states, output_memory_config)
        return ttnn.add(residual, down, memory_config=output_memory_config)

    def prefill_forward(self, hidden_states, *, key_cache, value_cache, page_table, user_id=0):
        """Optimized-owned prefill with packed projections and paged cache fill."""
        shape = tuple(hidden_states.shape)
        if len(shape) != 4 or shape[:2] != (1, self.batch) or shape[3] != self.hidden_size:
            raise ValueError(f"prefill hidden_states must be [1,{self.batch},S,{self.hidden_size}], got {shape}")
        seq_len = shape[2]
        if not 1 < seq_len <= self.max_context:
            raise ValueError(f"prefill sequence must be in [2,{self.max_context}], got {seq_len}")

        residual = hidden_states
        normalized = self._norm(hidden_states, self.weights["input_norm"])
        fused = ttnn.linear(
            normalized,
            self.weights["qkv"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self._compute_kernel_config(ttnn.MathFidelity.HiFi2),
        )
        fused = ttnn.reshape(fused, [self.batch, seq_len, 3 * self.hidden_size])
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        query, key = self._prefill_rope(query, key, seq_len)
        query = ttnn.slice(query, [0, 0, 0, 0], [self.batch, self.num_heads, seq_len, self.head_dim])
        key = ttnn.slice(key, [0, 0, 0, 0], [self.batch, self.num_kv_heads, seq_len, self.head_dim])
        value = ttnn.slice(value, [0, 0, 0, 0], [self.batch, self.num_kv_heads, seq_len, self.head_dim])
        for batch_idx in range(self.batch):
            user_key = ttnn.slice(key, [batch_idx, 0, 0, 0], [batch_idx + 1, self.num_kv_heads, seq_len, self.head_dim])
            user_value = ttnn.slice(
                value, [batch_idx, 0, 0, 0], [batch_idx + 1, self.num_kv_heads, seq_len, self.head_dim]
            )
            if self.kv_cache_dtype != ttnn.bfloat16:
                user_key = ttnn.typecast(user_key, self.kv_cache_dtype)
                user_value = ttnn.typecast(user_value, self.kv_cache_dtype)
            ttnn.experimental.paged_fill_cache(
                key_cache, user_key, page_table, batch_idx=user_id + batch_idx, block_size=self.page_size
            )
            ttnn.experimental.paged_fill_cache(
                value_cache, user_value, page_table, batch_idx=user_id + batch_idx, block_size=self.page_size
            )

        if seq_len <= PREFILL_SDPA_MAX_SEQ:
            attended = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                scale=self.scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            attended_chunks = []
            chunk_start = 0
            while chunk_start < seq_len:
                chunk_capacity = PREFILL_SDPA_MAX_SEQ if chunk_start == 0 else 4 * ttnn.TILE_SIZE
                chunk_len = min(chunk_capacity, seq_len - chunk_start)
                padded_len = math.ceil(chunk_len / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
                query_chunk = ttnn.slice(
                    query,
                    [0, 0, chunk_start, 0],
                    [self.batch, self.num_heads, chunk_start + chunk_len, self.head_dim],
                )
                if padded_len != chunk_len:
                    query_chunk = ttnn.pad(
                        query_chunk, [(0, 0), (0, 0), (0, padded_len - chunk_len), (0, 0)], value=0.0
                    )
                if chunk_start == 0 and chunk_len == PREFILL_SDPA_MAX_SEQ:
                    prefix_key = ttnn.slice(
                        key, [0, 0, 0, 0], [self.batch, self.num_kv_heads, chunk_len, self.head_dim]
                    )
                    prefix_value = ttnn.slice(
                        value, [0, 0, 0, 0], [self.batch, self.num_kv_heads, chunk_len, self.head_dim]
                    )
                    output_chunk = ttnn.transformer.scaled_dot_product_attention(
                        query_chunk,
                        prefix_key,
                        prefix_value,
                        is_causal=True,
                        scale=self.scale,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                else:
                    mask = self._offset_causal_mask(chunk_start=chunk_start, query_len=padded_len, key_len=seq_len)
                    output_chunk = ttnn.transformer.scaled_dot_product_attention(
                        query_chunk,
                        key,
                        value,
                        attn_mask=mask,
                        is_causal=False,
                        scale=self.scale,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=ttnn.types.BlackholeComputeKernelConfig(
                            math_fidelity=ttnn.MathFidelity.HiFi4,
                            math_approx_mode=False,
                            fp32_dest_acc_en=True,
                            packer_l1_acc=False,
                        ),
                    )
                if padded_len != chunk_len:
                    output_chunk = ttnn.slice(
                        output_chunk,
                        [0, 0, 0, 0],
                        [self.batch, self.num_heads, chunk_len, self.head_dim],
                    )
                attended_chunks.append(output_chunk)
                chunk_start += chunk_len
            attended = attended_chunks[0] if len(attended_chunks) == 1 else ttnn.concat(attended_chunks, dim=2)

        attended = ttnn.transformer.concatenate_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        projected = ttnn.linear(
            attended,
            self.weights["o_proj"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self._compute_kernel_config(ttnn.MathFidelity.HiFi2),
        )
        projected = ttnn.reshape(projected, [1, self.batch, seq_len, self.hidden_size])
        return self._mlp(ttnn.add(residual, projected))

    def decode_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        page_table,
        current_positions,
        use_long_rope,
    ):
        """Run the advisor-seeded DRAM-sharded decode candidate."""
        shape = tuple(hidden_states.shape)
        if shape != (1, 1, self.batch, self.hidden_size):
            raise ValueError(f"decode hidden_states must be [1,1,{self.batch},{self.hidden_size}], got {shape}")
        if tuple(current_positions.shape) != (self.batch,):
            raise ValueError(f"current_positions must have shape [{self.batch}], got {tuple(current_positions.shape)}")

        residual = hidden_states
        normalized = self._decode_norm(hidden_states, self.weights["input_norm"])
        if self.split_decode_projections:
            split_qkv = [
                ttnn.to_memory_config(
                    self._decode_linear(name, normalized, self.decode_fidelities[name]), ttnn.L1_MEMORY_CONFIG
                )
                for name in ("q", "k", "v")
            ]
            fused = ttnn.concat(split_qkv, dim=-1)
        else:
            fused = self._decode_linear("qkv", normalized, self.decode_fidelities["qkv"])
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        query, key = self._decode_rope(query, key, current_positions, use_long_rope=use_long_rope)
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
            cur_pos_tensor=current_positions,
            page_table_tensor=page_table,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.to_memory_config(attended, self._decode_concat_memory_config())
        attended = ttnn.experimental.nlp_concat_heads_decode(attended, num_heads=self.num_heads)
        if self.batch < ttnn.TILE_SIZE:
            attended = ttnn.slice(attended, [0, 0, 0, 0], [1, 1, self.batch, self.hidden_size])
        projected = self._decode_linear("o_proj", attended, self.decode_fidelities["o_proj"])
        projected = ttnn.reshape(projected, [1, 1, self.batch, self.hidden_size])
        output_memory_config = self._decode_output_memory_config("o_proj")
        residual = ttnn.to_memory_config(residual, output_memory_config)
        hidden_states = ttnn.add(residual, projected, memory_config=output_memory_config)
        return ttnn.to_memory_config(self._decode_mlp(hidden_states), ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, hidden_states, *, mode, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
