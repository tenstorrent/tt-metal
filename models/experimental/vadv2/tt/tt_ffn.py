# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


def get_ffn_matmul_program_configs(device):
    """Get specialized matmul program configs for FFN operations - optimized for VAD-v2"""
    # Calculate core grid - use 8x8 cores for VAD-v2 FFN operations
    core_grid_8x8 = ttnn.CoreGrid(y=8, x=8)

    # For VAD-v2 FFN: input dim 256 -> hidden dim 1024 -> output dim 256
    # Assuming tile size of 32, so dimensions in tiles:
    # 256/32 = 8 tiles, 1024/32 = 32 tiles

    return {
        "ff1_matmul_program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(core_grid_8x8.x, core_grid_8x8.y),
            in0_block_w=8,  # Input width in tiles (256/32)
            out_subblock_h=1,
            out_subblock_w=8,  # Must be <= 8 (hardware limit: out_subblock_w * out_subblock_h <= 8)
            per_core_M=8,  # Conservative per-core M dimension
            per_core_N=16,  # Conservative per-core N dimension (1024/32/2)
        ),
        "ff2_matmul_program_config": ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(core_grid_8x8.x, core_grid_8x8.y),
            in0_block_w=16,  # Hidden width in tiles (1024/32/2 for stability)
            out_subblock_h=1,
            out_subblock_w=8,  # Must be <= 8 (hardware limit)
            per_core_M=8,  # Conservative per-core M dimension
            per_core_N=8,  # Output dimension (256/32)
        ),
    }


class TtFFN:
    def __init__(self, params, device):
        self.device = device
        self.linear1_weight = params.linear1.weight
        self.linear2_weight = params.linear2.weight
        self.linear1_bias = params.linear1.bias
        self.linear2_bias = params.linear2.bias

        # STEP 3: Get specialized matmul program configs for FFN operations
        # These configs optimize memory usage and performance for VAD-v2 dimensions
        self.program_configs = get_ffn_matmul_program_configs(device)

    def __call__(self, x, identity=None):
        if identity is None:
            identity = x

        # STEP 2: Fuse GeLU activation with first linear operation
        x = ttnn.linear(x, self.linear1_weight, bias=self.linear1_bias, activation="gelu")

        # Second linear operation
        x = ttnn.linear(x, self.linear2_weight, bias=self.linear2_bias)

        # Residual connection
        x = ttnn.add(x, identity)
        ttnn.deallocate(identity)
        return x
