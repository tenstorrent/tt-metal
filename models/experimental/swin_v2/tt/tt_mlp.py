# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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

    def __call__(self, x):
        for hidden_dim in self.hidden_channels[:-1]:
            if x.shape[-1] == 96:
                x = ttnn.to_memory_config(
                    x,
                    memory_config=ttnn.create_sharded_memory_config(
                        x.shape,
                        core_grid=ttnn.CoreGrid(y=8, x=8),
                        strategy=ttnn.ShardStrategy.HEIGHT,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                    dtype=ttnn.bfloat16,
                )

                x = ttnn.linear(
                    x,
                    self.parameters[0].weight,
                    bias=self.parameters[0].bias,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                    ),
                    program_config=program_configs["linear_1_config_1"],
                )
            elif x.shape[-1] == 192:
                x = ttnn.to_memory_config(
                    x,
                    memory_config=ttnn.create_sharded_memory_config(
                        x.shape,
                        core_grid=ttnn.CoreGrid(y=8, x=8),
                        strategy=ttnn.ShardStrategy.HEIGHT,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                    dtype=ttnn.bfloat16,
                )

                x = ttnn.linear(
                    x,
                    self.parameters[0].weight,
                    bias=self.parameters[0].bias,
                    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                    ),
                    program_config=program_configs["linear_1_config_2"],
                )
            elif x.shape[-1] == 384:
                x = ttnn.linear(
                    x,
                    self.parameters[0].weight,
                    bias=self.parameters[0].bias,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    # dtype=ttnn.bfloat8_b,
                    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                    ),
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                )
            elif x.shape[-1] == 768:
                x = ttnn.linear(
                    x,
                    self.parameters[0].weight,
                    bias=self.parameters[0].bias,
                    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                    ),
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )

            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
            if self.norm_layer is not None:
                x = ttnn.layer_norm(
                    x,
                    weight=self.parameters.norm_weight,
                    bias=self.parameters.norm_bias,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            x = self.activation_layer(
                x,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        if x.shape[-1] == 384:
            x = ttnn.to_memory_config(
                x,
                memory_config=ttnn.create_sharded_memory_config(
                    x.shape,
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat16,
            )

            x = ttnn.linear(
                x,
                self.parameters[3].weight,
                bias=self.parameters[3].bias,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                program_config=program_configs["linear_2_config_1"],
            )
        elif x.shape[-1] == 768:
            x = ttnn.to_memory_config(
                x,
                memory_config=ttnn.create_sharded_memory_config(
                    x.shape,
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat16,
            )

            x = ttnn.linear(
                x,
                self.parameters[3].weight,
                bias=self.parameters[3].bias,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                program_config=program_configs["linear_2_config_2"],
            )
        elif x.shape[-1] == 1536:
            x = ttnn.to_memory_config(
                x,
                memory_config=ttnn.create_sharded_memory_config(
                    x.shape,
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat16,
            )
            x = ttnn.linear(
                x,
                self.parameters[3].weight,
                bias=self.parameters[3].bias,
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                program_config=program_configs["linear_2_config_3"],
            )
        elif x.shape[-1] == 3072:
            x = ttnn.linear(
                x,
                self.parameters[3].weight,
                bias=self.parameters[3].bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                core_grid=ttnn.CoreGrid(y=8, x=8),
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
            )
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        ttnn.DumpDeviceProfiler(self.device)
        return x
