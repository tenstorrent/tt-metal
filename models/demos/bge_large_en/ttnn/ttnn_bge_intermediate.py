# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.utility_functions import is_blackhole

if is_blackhole():
    pass
else:
    pass


class TtnnBGEIntermediate:
    def __init__(self, parameters):
        self.dense = ttnn.linear
        self.parameters = parameters

    def __call__(self, hidden_states: ttnn.Tensor):
        # Calculate per_core_M dynamically based on tensor dimensions
        # per_core_M = (batch_size * seq_len) / 32 / 8 (tiles per core)
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
        dynamic_ff1_matmul_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,  # Keep 4 (Kt=32, per_core=4, max is 4)
            out_subblock_h=1,
            out_subblock_w=8,  # Keep 8 (no FP32 accumulation, max for BF16 is 8)
            per_core_M=per_core_M,  # Calculate dynamically
            per_core_N=16,  # Keep same (4096 / 32 / 8 = 16)
            transpose_mcast=False,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
        )

        out_intermediate = self.dense(
            hidden_states,
            self.parameters.dense.weight,
            bias=self.parameters.dense.bias,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            program_config=dynamic_ff1_matmul_program_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                packer_l1_acc=False,
            ),
            dtype=ttnn.bfloat8_b,  # Keep BF8 for storage; compute will use BF16 with FP32 accumulation
        )
        return out_intermediate
