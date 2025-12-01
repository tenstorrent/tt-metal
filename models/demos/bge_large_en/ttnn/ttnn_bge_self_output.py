# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.utility_functions import is_blackhole

if is_blackhole():
    pass
else:
    pass


class TtnnBGESelfOutput:
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

        # Create dynamic program config
        dynamic_self_out_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,  # Keep same
            out_subblock_h=2,  # Keep 2 (no FP32 accumulation on self-output)
            out_subblock_w=4,  # Keep 4
            per_core_M=per_core_M,  # Calculate dynamically
            per_core_N=4,  # Keep same (1024 / 32 / 8 = 4)
            transpose_mcast=False,
            fused_activation=None,
        )

        output = self.dense(
            hidden_states,
            self.parameters.dense.weight,
            bias=self.parameters.dense.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=dynamic_self_out_program_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
            ),
            dtype=ttnn.bfloat8_b,  # Keep BF8 for storage; compute will use BF16 with FP32 accumulation
        )
        input_tensor = ttnn.reshard(input_tensor, output.memory_config())

        # Calculate LayerNorm program config dynamically based on tensor dimensions
        # block_h must equal M (in tiles) / num_cores_r
        *batch_sizes, height, width = output.shape
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

        output = self.LayerNorm(
            output,
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
        return output
