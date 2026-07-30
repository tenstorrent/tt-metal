# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-device optimized Phi-3.5 decoder layer.

This stage starts from :class:`FusedDecoder` so the packed QKV, packed gate/up,
and fused SiLU-multiply topology are preserved.  Phase-specific memory,
precision, and program configurations are owned here; no runtime method
dispatches back to ``FunctionalDecoder._mlp``.
"""

from __future__ import annotations

import os

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.fused_decoder import FusedDecoder


class OptimizedDecoder(FusedDecoder):
    """Phi-3.5 decoder with an explicitly owned optimized dense block."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        """Materialize the independently swept decode precision/layout policy."""
        decoder = super().from_state_dict(state_dict, **kwargs)
        decoder.decode_matmul_family = os.environ.get("PHI_OPT_DECODE_MATMUL", "dram_sharded")
        decoder.decode_in0_block_w = int(os.environ.get("PHI_OPT_IN0_BLOCK_W", "4"))
        decoder.decode_math_fidelity = os.environ.get("PHI_OPT_MATH_FIDELITY", "lofi")
        dtype_by_name = {
            "qkv": os.environ.get("PHI_OPT_ATTENTION_DTYPE", "bfp4"),
            "o_proj": os.environ.get("PHI_OPT_ATTENTION_DTYPE", "bfp4"),
            "gate_up": os.environ.get("PHI_OPT_MLP_DTYPE", "bfp4"),
            "down": os.environ.get("PHI_OPT_DOWN_DTYPE", os.environ.get("PHI_OPT_MLP_DTYPE", "bfp4")),
        }
        dtype_map = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "bf16": ttnn.bfloat16}
        for name in ("qkv", "o_proj", "gate_up", "down"):
            decoder.weights[name] = ttnn.typecast(decoder.weights[name], dtype_map[dtype_by_name[name]])
        gate_up_shape = tuple(decoder.weights["gate_up"].shape)
        decoder.weights["gate"] = ttnn.slice(
            decoder.weights["gate_up"], [0, 0], [gate_up_shape[0], decoder.intermediate_size]
        )
        decoder.weights["up"] = ttnn.slice(
            decoder.weights["gate_up"],
            [0, decoder.intermediate_size],
            [gate_up_shape[0], 2 * decoder.intermediate_size],
        )
        decoder.decode_weights = decoder.weights
        if decoder.decode_matmul_family == "dram_sharded":
            decoder.decode_weights = dict(decoder.weights)
            dram_grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(decoder.mesh_device.dram_grid_size().x - 1, 0))}
            )
            for name, weight in tuple(decoder.weights.items()):
                if name not in ("qkv", "o_proj", "gate_up", "gate", "up", "down"):
                    continue
                k, n = tuple(weight.shape)
                shard_spec = ttnn.ShardSpec(
                    dram_grid,
                    (k, n // decoder.mesh_device.dram_grid_size().x),
                    ttnn.ShardOrientation.ROW_MAJOR,
                )
                memory_config = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    shard_spec,
                )
                decoder.decode_weights[name] = ttnn.to_memory_config(weight, memory_config)
        return decoder

    def _decode_linear(self, value, weight_name):
        if self.decode_matmul_family != "dram_sharded":
            return ttnn.linear(value, self.weights[weight_name], dtype=ttnn.bfloat16)
        weight = self.decode_weights[weight_name]
        k, n = tuple(weight.shape)
        num_cores = self.mesh_device.dram_grid_size().x
        core_grid = ttnn.CoreGrid(x=num_cores, y=1)
        value = ttnn.to_memory_config(
            value,
            ttnn.create_sharded_memory_config(
                (ttnn.TILE_SIZE, k // num_cores),
                core_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ),
        )
        legal_k_tiles = k // (ttnn.TILE_SIZE * num_cores)
        role_defaults = {"qkv": 12, "o_proj": 12, "gate_up": 6, "down": 32}
        role_override = int(
            os.environ.get(
                f"PHI_OPT_{weight_name.upper()}_IN0_BLOCK_W",
                str(role_defaults.get(weight_name, 0)),
            )
        )
        in0_block_w = role_override or self.decode_in0_block_w or legal_k_tiles
        if legal_k_tiles % in0_block_w:
            raise ValueError(f"in0_block_w={in0_block_w} must divide K tiles/core={legal_k_tiles} for {weight_name}")
        output_memory_config = self._decode_linear_memory_config(n)
        return ttnn.linear(
            value,
            weight,
            dtype=ttnn.bfloat16,
            memory_config=output_memory_config,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=in0_block_w,
                per_core_M=1,
                per_core_N=n // (ttnn.TILE_SIZE * num_cores),
                fused_activation=None,
            ),
            compute_kernel_config=ttnn.types.BlackholeComputeKernelConfig(
                math_fidelity=(
                    ttnn.MathFidelity.HiFi2 if self.decode_math_fidelity == "hifi2" else ttnn.MathFidelity.LoFi
                ),
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            ),
        )

    def _decode_linear_memory_config(self, width):
        num_cores = self.mesh_device.dram_grid_size().x
        return ttnn.create_sharded_memory_config(
            (ttnn.TILE_SIZE, width // num_cores),
            ttnn.CoreGrid(x=num_cores, y=1),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _decode_rope(self, query, key, current_positions, *, use_long_rope):
        """Keep the functional Phi rotate-half contract with explicit layouts.

        Explicit source/destination configs also make the graph analyzable by
        shard-advise; the functional implementation queried transient tensor
        metadata that is intentionally unknown during compiler analysis.
        """
        cos_table = self.long_cos if use_long_rope else self.short_cos
        sin_table = self.long_sin if use_long_rope else self.short_sin
        rope_positions = ttnn.typecast(current_positions, ttnn.uint32)
        cos = ttnn.embedding(rope_positions, cos_table, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rope_positions, sin_table, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.reshape(cos, [1, 1, self.batch, self.head_dim])
        sin = ttnn.reshape(sin, [1, 1, self.batch, self.head_dim])
        query = self._apply_rope(ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
        key = self._apply_rope(ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
        cache_update_memory_config = self._decode_concat_memory_config()
        return (
            ttnn.to_memory_config(query, cache_update_memory_config),
            ttnn.to_memory_config(key, cache_update_memory_config),
        )

    def _mlp(self, hidden_states, *, decode=False):
        normalized = self._norm(hidden_states, self.weights["post_norm"])
        split_gate_up = os.environ.get("PHI_OPT_SPLIT_GATE_UP", "0") == "1"
        if split_gate_up:
            linear = (
                self._decode_linear
                if decode
                else (lambda value, name: ttnn.linear(value, self.weights[name], dtype=ttnn.bfloat16))
            )
            gate = linear(normalized, "gate")
            up = linear(normalized, "up")
            activated = ttnn.multiply(
                gate,
                up,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            )
            down = (
                self._decode_linear(activated, "down")
                if decode
                else ttnn.linear(activated, self.weights["down"], dtype=ttnn.bfloat16)
            )
            if decode:
                output_memory_config = self._decode_linear_memory_config(self.hidden_size)
                hidden_states = ttnn.to_memory_config(hidden_states, output_memory_config)
                return ttnn.add(hidden_states, down, memory_config=output_memory_config)
            if down.memory_config() != hidden_states.memory_config():
                hidden_states = ttnn.to_memory_config(hidden_states, down.memory_config())
            return ttnn.add(hidden_states, down, memory_config=down.memory_config())
        gate_up = (
            self._decode_linear(normalized, "gate_up")
            if decode
            else ttnn.linear(normalized, self.weights["gate_up"], dtype=ttnn.bfloat16)
        )
        gate_up_shape = tuple(gate_up.shape)
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [*gate_up_shape[:-1], self.intermediate_size])
        up = ttnn.slice(
            gate_up,
            [0, 0, 0, self.intermediate_size],
            [*gate_up_shape[:-1], 2 * self.intermediate_size],
        )
        activated = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )
        down = (
            self._decode_linear(activated, "down")
            if decode
            else ttnn.linear(activated, self.weights["down"], dtype=ttnn.bfloat16)
        )
        if decode:
            output_memory_config = self._decode_linear_memory_config(self.hidden_size)
            hidden_states = ttnn.to_memory_config(hidden_states, output_memory_config)
            return ttnn.add(hidden_states, down, memory_config=output_memory_config)
        if down.memory_config() != hidden_states.memory_config():
            hidden_states = ttnn.to_memory_config(hidden_states, down.memory_config())
        return ttnn.add(hidden_states, down, memory_config=down.memory_config())

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
        if self.decode_matmul_family != "dram_sharded":
            return super().decode_forward(
                hidden_states,
                key_cache=key_cache,
                value_cache=value_cache,
                page_table=page_table,
                current_positions=current_positions,
                use_long_rope=use_long_rope,
            )
        residual = hidden_states
        normalized = self._norm(hidden_states, self.weights["input_norm"])
        fused = self._decode_linear(normalized, "qkv")
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
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                exp_approx_mode=False,
                q_chunk_size=0,
                k_chunk_size=0,
            ),
        )
        attended = ttnn.to_memory_config(attended, self._decode_concat_memory_config())
        attended = ttnn.experimental.nlp_concat_heads_decode(attended, num_heads=self.num_heads)
        if self.batch < ttnn.TILE_SIZE:
            attended = ttnn.slice(attended, [0, 0, 0, 0], [1, 1, self.batch, self.hidden_size])
        projected = self._decode_linear(attended, "o_proj")
        projected_memory_config = self._decode_linear_memory_config(self.hidden_size)
        residual = ttnn.to_memory_config(residual, projected_memory_config)
        post_attention = ttnn.add(residual, projected, memory_config=projected_memory_config)
        return ttnn.to_memory_config(self._mlp(post_attention, decode=True), ttnn.DRAM_MEMORY_CONFIG)
