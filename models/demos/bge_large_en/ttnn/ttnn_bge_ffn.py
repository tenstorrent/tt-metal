# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn


def feed_forward(
    ff1_weight,
    ff1_bias,
    ff2_weight,
    ff2_bias,
):
    """
    Feed forward function following metal_BERT_large_11 pattern.
    Returns a function that performs OP9 (FF1) and OP10 (FF2).
    """

    def op9_ff1(activation):
        """
        OP9: FF1 intermediate with GELU activation
        """
        # Calculate per_core_M dynamically
        *batch_sizes, height, width = activation.shape
        batch_size = 1
        for bs in batch_sizes:
            batch_size *= bs
        M_tiles = (batch_size * height) // 32
        core_grid_y = 8
        per_core_M = M_tiles // core_grid_y
        if per_core_M == 0:
            per_core_M = 1

        # Determine if we need DRAM for FF1 output (for large sequences)
        # For seq_len >= 512, use DRAM to avoid L1 buffer clashes
        total_height = batch_size * height
        use_dram_for_ff1 = total_height >= 2048  # Threshold for large sequences

        dynamic_ff1_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=per_core_M,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=(ttnn.UnaryOpType.GELU, True),
        )

        # Use DRAM for large sequences, L1 for smaller ones
        ff1_output_mem_config = ttnn.DRAM_MEMORY_CONFIG if use_dram_for_ff1 else ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG

        ff1_output = ttnn.linear(
            activation,
            ff1_weight,
            bias=ff1_bias,
            memory_config=ff1_output_mem_config,
            program_config=dynamic_ff1_program_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                packer_l1_acc=False,
            ),
            dtype=ttnn.bfloat8_b,
        )
        return ff1_output, use_dram_for_ff1

    def op10_ff2(ff1_output, ff1_is_dram):
        """
        OP10: FF2 output (always DRAM, like metal_BERT)
        """
        # Calculate per_core_M dynamically
        # Constraint: M_tiles / per_core_M <= core_grid_y
        # So: per_core_M >= M_tiles / core_grid_y
        *batch_sizes, height, width = ff1_output.shape
        batch_size = 1
        for bs in batch_sizes:
            batch_size *= bs
        M_tiles = (batch_size * height) // 32
        core_grid_y = 8
        per_core_M = M_tiles // core_grid_y
        if per_core_M == 0:
            per_core_M = 1

        # Ensure per_core_M satisfies the constraint: M_tiles / per_core_M <= core_grid_y
        # For seq_len=512, batch=8: M_tiles=128, so per_core_M must be >= 16
        # We use DRAM for input/output to avoid L1 buffer clashes even with larger per_core_M

        # Hardware constraint: out_subblock_w * out_subblock_h <= 4
        # Options: (h=1, w=4), (h=2, w=2), (h=4, w=1)
        # Use (h=1, w=4) to match original BGE config and ensure compatibility
        # Note: metal_BERT uses (h=2, w=4) in model_config, but that's 8 which exceeds limit
        # metal_BERT's custom_matmuls uses (h=6, w=1) for different shape
        # Try reducing in0_block_w to reduce L1 buffer allocations
        # K_tiles = 4096 / 32 = 128, per core in X = 128 / 8 = 16
        # Options: 16, 8, 4, 2, 1. Try 8 to reduce L1 usage while maintaining efficiency
        dynamic_ff2_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=8,  # Reduced from 16 to reduce L1 buffer allocations
            out_subblock_h=1,  # Must be 1 to satisfy out_subblock_w * out_subblock_h <= 4
            out_subblock_w=4,  # 1 * 4 = 4, which satisfies the constraint
            per_core_M=per_core_M,  # Calculated as M_tiles // core_grid_y (must satisfy constraint)
            per_core_N=4,  # Match metal_BERT: per_core_N=4
            transpose_mcast=False,
            fused_activation=None,
        )

        # Convert FF1 output to DRAM if not already in DRAM
        # This ensures both input and output are in DRAM, avoiding L1 buffer clashes
        if not ff1_is_dram:
            ff1_output_dram = ttnn.to_memory_config(ff1_output, ttnn.DRAM_MEMORY_CONFIG)
            ff1_output.deallocate()
        else:
            ff1_output_dram = ff1_output

        # Ensure weights are in DRAM (metal_BERT always uses DRAM for weights)
        # This helps reduce L1 buffer allocations during matmul
        if ff2_weight.memory_config().buffer_type != ttnn.BufferType.DRAM:
            ff2_weight_dram = ttnn.to_memory_config(ff2_weight, ttnn.DRAM_MEMORY_CONFIG)
        else:
            ff2_weight_dram = ff2_weight

        if ff2_bias.memory_config().buffer_type != ttnn.BufferType.DRAM:
            ff2_bias_dram = ttnn.to_memory_config(ff2_bias, ttnn.DRAM_MEMORY_CONFIG)
        else:
            ff2_bias_dram = ff2_bias

        # Always use DRAM for FF2 output (following metal_BERT pattern)
        ff2_output = ttnn.linear(
            ff1_output_dram,
            ff2_weight_dram,
            bias=ff2_bias_dram,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Always DRAM like metal_BERT
            program_config=dynamic_ff2_program_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                fp32_dest_acc_en=True,
            ),
            dtype=ttnn.bfloat8_b,
        )

        # Deallocate DRAM intermediate
        ff1_output_dram.deallocate()

        return ff2_output

    def feed_forward_(activation: ttnn.Tensor) -> ttnn.Tensor:
        """
        Feed forward function following metal_BERT pattern:
        OP9 (FF1) -> OP10 (FF2)
        """
        ff1_output, ff1_is_dram = op9_ff1(activation)
        # Don't deallocate activation here since it is used by more ops in encoder

        ff2_output = op10_ff2(ff1_output, ff1_is_dram)

        return ff2_output

    return feed_forward_


class TtnnBGEFeedForwardModel:
    """
    BGE Feed Forward Model following metal_BERT_large_11 TtFeedForwardModel pattern.
    """

    def __init__(self, layer_params):
        # FF1 weights and bias (OP9)
        self.ff1_weight = layer_params.intermediate.dense.weight
        self.ff1_bias = layer_params.intermediate.dense.bias

        # FF2 weights and bias (OP10)
        self.ff2_weight = layer_params.output.dense.weight
        self.ff2_bias = layer_params.output.dense.bias

        # Create feed forward function
        self.ffn = feed_forward(
            self.ff1_weight,
            self.ff1_bias,
            self.ff2_weight,
            self.ff2_bias,
        )

    def __call__(self, activation: ttnn.Tensor) -> ttnn.Tensor:
        return self.ffn(activation)
