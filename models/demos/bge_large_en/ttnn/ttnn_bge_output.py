# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.utility_functions import is_blackhole

if is_blackhole():
    pass
else:
    pass


class TtnnBGEOutput:
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config
        self.dense = ttnn.linear
        self.LayerNorm = ttnn.layer_norm

    def __call__(self, hidden_states: ttnn.Tensor, input_tensor: ttnn.Tensor):
        # Calculate per_core_M dynamically based on tensor dimensions
        *batch_sizes, height, width = hidden_states.shape
        batch_size = 1
        for bs in batch_sizes:
            batch_size *= bs
        M_tiles = (batch_size * height) // 32  # Convert to tiles (tile height = 32)
        core_grid_y = 8
        per_core_M = M_tiles // core_grid_y
        if per_core_M == 0:
            per_core_M = 1

        # Following metal_BERT_large_11 pattern: FF2 output ALWAYS goes to DRAM
        # This avoids L1 buffer clashes, especially for large sequence lengths
        # metal_BERT uses DRAM for FF2 output by default (see custom_matmuls.bert_large_ff2_matmul)
        ff2_output_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Create dynamic program config
        dynamic_ff2_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=16,  # Keep 16 (4096 / 32 / 8 = 16)
            out_subblock_h=1,  # Keep 1
            out_subblock_w=4,  # Keep 4 (max for FP32: 1*4=4, divides per_core_N=4)
            per_core_M=per_core_M,  # Calculate dynamically
            per_core_N=4,  # Keep same (1024 / 32 / 8 = 4)
            transpose_mcast=False,
            fused_activation=None,
        )

        bert_output_lin = self.dense(
            hidden_states,
            self.parameters.dense.weight,
            bias=self.parameters.dense.bias,
            memory_config=ff2_output_memory_config,  # Always use DRAM (following metal_BERT pattern)
            program_config=dynamic_ff2_program_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                fp32_dest_acc_en=True,  # FP32 accumulation only on output layer (most critical)
            ),
            dtype=ttnn.bfloat8_b,  # Keep BF8 for storage; compute will use BF16 with FP32 accumulation
        )

        # Convert from DRAM back to sharded memory config for layer norm
        # This matches metal_BERT pattern where FF2 output is in DRAM, then converted for layer norm
        bert_output_lin = ttnn.to_memory_config(bert_output_lin, input_tensor.memory_config())

        # Calculate LayerNorm program config dynamically based on tensor dimensions
        # block_h must equal M (in tiles) / num_cores_r
        *batch_sizes, height, width = bert_output_lin.shape
        batch_size = 1
        for bs in batch_sizes:
            batch_size *= bs
        M_tiles = (batch_size * height) // 32  # Convert to tiles (tile height = 32)
        core_grid_y = 8
        block_h = M_tiles // core_grid_y
        if block_h == 0:
            block_h = 1

        dynamic_layernorm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            subblock_w=4,
            block_h=block_h,
            block_w=4,
            inplace=True,
            legacy_reduction=True,
            legacy_rsqrt=True,
        )

        bert_output_lin = self.LayerNorm(
            bert_output_lin,
            residual_input_tensor=input_tensor,
            weight=self.parameters.LayerNorm.weight,
            bias=self.parameters.LayerNorm.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            epsilon=self.config.layer_norm_eps,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                fp32_dest_acc_en=True,  # FP32 accumulation per engineering guidance (layer norm)
            ),
            program_config=dynamic_layernorm_program_config,
        )
        return bert_output_lin
