# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

program_configs = {
    "linear_1_config_1": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=6,
        per_core_M=8,
        per_core_N=12,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    ),
    "linear_1_config_2": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=6,
        per_core_M=2,
        per_core_N=24,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    ),
    "linear_1_config_4": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=96,
        per_core_M=1,
        per_core_N=96,
        transpose_mcast=False,
        fused_activation=None,
    ),
    "linear_2_config_1": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=8,
        per_core_N=3,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    ),
    "linear_2_config_2": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=2,
        per_core_N=6,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    ),
    "linear_2_config_3": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=4,
        per_core_N=12,
        transpose_mcast=False,
        fused_activation=None,
    ),
    "linear_2_config_4": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=24,
        per_core_M=1,
        per_core_N=24,
        transpose_mcast=False,
        fused_activation=None,
    ),
}


class TtMLP:
    def __init__(
        self,
        hidden_channels,
        device,
        parameters,
        inplace=None,
        activation_layer=ttnn.relu,
        norm_layer=None,
    ):
        self.params = {} if inplace is None else {"inplace": inplace}
        self.device = device
        self.parameters = parameters
        self.norm_layer = norm_layer
        self.hidden_channels = hidden_channels
        self.activation_layer = activation_layer

    def __call__(self, input_tensor):
        for hidden_dim in self.hidden_channels[:-1]:
            if input_tensor.shape[-1] == 96:
                input_tensor = ttnn.to_memory_config(
                    input_tensor,
                    memory_config=ttnn.create_sharded_memory_config(
                        input_tensor.shape,
                        core_grid=ttnn.CoreGrid(y=8, x=8),
                        strategy=ttnn.ShardStrategy.HEIGHT,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                    dtype=ttnn.bfloat16,
                )

                output_tensor = ttnn.linear(
                    input_tensor,
                    self.parameters[0].weight,
                    bias=self.parameters[0].bias,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                    ),
                    program_config=program_configs["linear_1_config_1"],
                )
            elif input_tensor.shape[-1] == 192:
                input_tensor = ttnn.to_memory_config(
                    input_tensor,
                    memory_config=ttnn.create_sharded_memory_config(
                        input_tensor.shape,
                        core_grid=ttnn.CoreGrid(y=8, x=8),
                        strategy=ttnn.ShardStrategy.HEIGHT,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                    dtype=ttnn.bfloat16,
                )

                output_tensor = ttnn.linear(
                    input_tensor,
                    self.parameters[0].weight,
                    bias=self.parameters[0].bias,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                    ),
                    program_config=program_configs["linear_1_config_2"],
                )
            elif input_tensor.shape[-1] == 384:
                output_tensor = ttnn.linear(
                    input_tensor,
                    self.parameters[0].weight,
                    bias=self.parameters[0].bias,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                    ),
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                )
            elif input_tensor.shape[-1] == 768:
                output_tensor = ttnn.linear(
                    input_tensor,
                    self.parameters[0].weight,
                    bias=self.parameters[0].bias,
                    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                    ),
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )

            output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
            if self.norm_layer is not None:
                output_tensor = ttnn.layer_norm(
                    output_tensor,
                    weight=self.parameters.norm_weight,
                    bias=self.parameters.norm_bias,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            output_tensor = self.activation_layer(
                output_tensor,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        if output_tensor.shape[-1] == 384:
            output_tensor = ttnn.to_memory_config(
                output_tensor,
                memory_config=ttnn.create_sharded_memory_config(
                    output_tensor.shape,
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat16,
            )

            output_tensor = ttnn.linear(
                output_tensor,
                self.parameters[3].weight,
                bias=self.parameters[3].bias,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                program_config=program_configs["linear_2_config_1"],
            )
        elif output_tensor.shape[-1] == 768:
            output_tensor = ttnn.to_memory_config(
                output_tensor,
                memory_config=ttnn.create_sharded_memory_config(
                    output_tensor.shape,
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat16,
            )

            output_tensor = ttnn.linear(
                output_tensor,
                self.parameters[3].weight,
                bias=self.parameters[3].bias,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                program_config=program_configs["linear_2_config_2"],
            )
        elif output_tensor.shape[-1] == 1536:
            output_tensor = ttnn.to_memory_config(
                output_tensor,
                memory_config=ttnn.create_sharded_memory_config(
                    output_tensor.shape,
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat16,
            )
            output_tensor = ttnn.linear(
                output_tensor,
                self.parameters[3].weight,
                bias=self.parameters[3].bias,
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                program_config=program_configs["linear_2_config_3"],
            )
        elif output_tensor.shape[-1] == 3072:
            output_tensor = ttnn.linear(
                output_tensor,
                self.parameters[3].weight,
                bias=self.parameters[3].bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                core_grid=ttnn.CoreGrid(y=8, x=8),
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
            )
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        return output_tensor
