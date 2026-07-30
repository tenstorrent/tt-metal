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

        shard_width = (
            (decoder.hidden_size + dram.x * ttnn.TILE_SIZE - 1) // (dram.x * ttnn.TILE_SIZE)
        ) * ttnn.TILE_SIZE
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                dram_grid,
                (decoder.intermediate_size, shard_width),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        decoder.weights["down_decode"] = ttnn.to_memory_config(decoder.weights["down"], memory_config)
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
        value = ttnn.to_memory_config(value, self._decode_projection_memcfg(tuple(value.shape)[-1]))
        output = ttnn.linear(
            value,
            self.weights["down_decode"],
            dtype=ttnn.bfloat16,
            memory_config=self._decode_projection_memcfg(self.hidden_size),
            program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=16,
                per_core_M=1,
                per_core_N=(self.hidden_size // ttnn.TILE_SIZE) // 16,
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
