# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cumulative indexed experts, DRAM projections and residual/norm layout trial."""

import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_optimized_dram import OptimizedDecoder as Dram
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_optimized_indexed import OptimizedDecoder as Indexed


class OptimizedDecoder(Indexed, Dram):
    def _residual_memory(self):
        cores = self.options.get("residual_cores", 8)
        grid = ttnn.num_cores_to_corerangeset(cores, self.mesh_device.compute_with_storage_grid_size(), True)
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [32, 2560 // cores], ttnn.ShardOrientation.ROW_MAJOR),
        )

    def _norm(self, x, name):
        if x.shape[-2] != 32 or not self.options.get("sharded_norm"):
            return super()._norm(x, name)
        # Grouping moves each token's two1280-wide groups into separate rows.
        shape = tuple(x.shape)
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        if not self.options.get("bf16_norm"):
            x = ttnn.typecast(x, ttnn.float32)
        grouped = ttnn.reshape(x, (1, 1, 64, 1280))
        cores = self.options.get("norm_cores", 8)
        grid = ttnn.num_cores_to_corerangeset(cores, self.mesh_device.compute_with_storage_grid_size(), True)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [64, 1280 // cores], ttnn.ShardOrientation.ROW_MAJOR),
        )
        grouped = ttnn.to_memory_config(grouped, mem)
        width = 40 // cores
        sub = next(v for v in range(min(4, width), 0, -1) if width % v == 0)
        prog = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            block_h=2,
            block_w=width,
            subblock_w=sub,
            inplace=False,
        )
        y = ttnn.rms_norm(
            grouped,
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.compute,
            program_config=prog,
            memory_config=mem,
        )
        y = ttnn.reshape(ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG), shape)
        return ttnn.mul(y, self.weights[name + ".weight"], dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

    def _finish(self, residual, gate_logits, attention, *, valid_tokens=None):
        if residual.shape[-2] != 32 or not self.options.get("sharded_residual"):
            return super()._finish(residual, gate_logits, attention, valid_tokens=valid_tokens)
        mem = self._residual_memory()
        gated = ttnn.mul(
            attention,
            gate_logits,
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, math.log(2), 20.0)],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        projected = ttnn.to_memory_config(self._linear(gated, "self_attn.o_proj"), mem)
        residual = ttnn.to_memory_config(residual, mem)
        x = ttnn.add(residual, projected, memory_config=mem)
        mlp = self._mlp(self._norm(x, "post_attention_layernorm"), valid_tokens=valid_tokens)
        return ttnn.add(x, ttnn.to_memory_config(mlp, mem), memory_config=mem)
