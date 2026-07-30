# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Optimized single-device Phi-3.5 decoder layer.

The optimized path inherits the fused decoder topology and changes only
device-side storage/compute policy.  Weight conversion happens once during
construction; forward methods remain entirely on device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.fused_decoder import FusedDecoder


@dataclass(frozen=True)
class OptimizationPolicy:
    attention_weight_dtype: object = ttnn.bfloat4_b
    mlp_gate_up_weight_dtype: object = ttnn.bfloat4_b
    mlp_down_weight_dtype: object = ttnn.bfloat4_b


DEFAULT_OPTIMIZATION_POLICY = OptimizationPolicy()


class OptimizedDecoder(FusedDecoder):
    """Fused Phi decoder with an explicit per-tensor-group precision policy."""

    def __init__(self, *args, optimization_policy=DEFAULT_OPTIMIZATION_POLICY, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimization_policy = optimization_policy

    @classmethod
    def from_state_dict(cls, state_dict, *, optimization_policy=DEFAULT_OPTIMIZATION_POLICY, **kwargs):
        decoder = super().from_state_dict(
            state_dict,
            optimization_policy=optimization_policy,
            **kwargs,
        )
        # The base constructor ignores unknown subclass kwargs, so preserve
        # this subclass-specific policy explicitly.
        decoder.optimization_policy = optimization_policy
        policy = decoder.optimization_policy
        dtype_by_weight = {
            "qkv": policy.attention_weight_dtype,
            "o_proj": policy.attention_weight_dtype,
            "gate_up": policy.mlp_gate_up_weight_dtype,
            "down": policy.mlp_down_weight_dtype,
        }
        for name, dtype in dtype_by_weight.items():
            if decoder.weights[name].dtype != dtype:
                decoder.weights[name] = ttnn.typecast(decoder.weights[name], dtype)
        dram = decoder.mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))})

        decode_weights = {
            "qkv_decode": ("qkv", decoder.hidden_size, 3 * decoder.hidden_size),
            "o_proj_decode": ("o_proj", decoder.hidden_size, decoder.hidden_size),
            "gate_up_decode": ("gate_up", decoder.hidden_size, 2 * decoder.intermediate_size),
            "down_decode": ("down", decoder.intermediate_size, decoder.hidden_size),
        }
        for decode_name, (source_name, k, n) in decode_weights.items():
            shard_width = ((n + dram.x * ttnn.TILE_SIZE - 1) // (dram.x * ttnn.TILE_SIZE)) * ttnn.TILE_SIZE
            memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, (k, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
            )
            decoder.weights[decode_name] = ttnn.to_memory_config(decoder.weights[source_name], memory_config)
        return decoder

    def _decode_projection_memcfg(self, width):
        grid = self.mesh_device.compute_with_storage_grid_size()
        cores = ttnn.num_cores_to_corerangeset(16, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
        return ttnn.create_sharded_memory_config_(
            shape=(ttnn.TILE_SIZE, width // 16),
            core_grid=cores,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _decode_down(self, value):
        # Preserve the explicit shipped-weight contract: self.weights["down_decode"].
        return self._decode_dram_sharded_linear(
            value,
            weight_name="down_decode",
            output_width=self.hidden_size,
            in0_block_w=16,
        )

    def _decode_dram_sharded_linear(self, value, *, weight_name, output_width, in0_block_w):
        value = ttnn.to_memory_config(value, self._decode_projection_memcfg(tuple(value.shape)[-1]))
        output = ttnn.linear(
            value,
            self.weights[weight_name],
            dtype=ttnn.bfloat16,
            memory_config=self._decode_projection_memcfg(output_width),
            program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=in0_block_w,
                per_core_M=1,
                per_core_N=(output_width // ttnn.TILE_SIZE) // 16,
                fused_activation=None,
            ),
            compute_kernel_config=ttnn.types.BlackholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            ),
        )
        return ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)

    def _prefill_down(self, value):
        shape = tuple(value.shape)
        m_tiles = math.ceil(shape[-2] * shape[-3] / ttnn.TILE_SIZE)
        return ttnn.linear(
            value,
            self.weights["down"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=math.ceil(m_tiles / 8),
                per_core_N=(self.hidden_size // ttnn.TILE_SIZE) // 8,
                transpose_mcast=False,
                fused_activation=None,
            ),
        )

    def _mlp(self, hidden_states):
        normalized = self._norm(hidden_states, self.weights["post_norm"])
        is_decode = tuple(hidden_states.shape) == (1, 1, self.batch, self.hidden_size)
        if is_decode and self.batch == ttnn.TILE_SIZE:
            gate_up = self._decode_dram_sharded_linear(
                normalized,
                weight_name="gate_up_decode",
                output_width=2 * self.intermediate_size,
                in0_block_w=6,
            )
        else:
            gate_up = ttnn.linear(normalized, self.weights["gate_up"], dtype=ttnn.bfloat16)
        gate_up_shape = tuple(gate_up.shape)
        gate = ttnn.slice(gate_up, [0, 0, 0, 0], [*gate_up_shape[:-1], self.intermediate_size])
        up = ttnn.slice(
            gate_up,
            [0, 0, 0, self.intermediate_size],
            [*gate_up_shape[:-1], 2 * self.intermediate_size],
        )
        activated = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        if not is_decode:
            return ttnn.add(hidden_states, self._prefill_down(activated))
        projected = self._decode_down(activated)
        return ttnn.add(hidden_states, projected)

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
        if self.batch != ttnn.TILE_SIZE:
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
        fused = self._decode_dram_sharded_linear(
            normalized,
            weight_name="qkv_decode",
            output_width=3 * self.hidden_size,
            in0_block_w=3,
        )
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
        projected = self._decode_dram_sharded_linear(
            attended,
            weight_name="o_proj_decode",
            output_width=self.hidden_size,
            in0_block_w=3,
        )
        projected = ttnn.reshape(projected, [1, 1, self.batch, self.hidden_size])
        return self._mlp(ttnn.add(residual, projected))
