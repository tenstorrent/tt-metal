# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

program_configs = {
    "linear_config_1": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
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
    "linear_config_2": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=4,
        per_core_N=12,
        transpose_mcast=False,
        fused_activation=None,
    ),
}


class TtPatchMerging:
    def __init__(self, device, parameters, dim):
        self.dim = dim
        self.device = device
        self.parameters = parameters

    def __call__(self, input_tensor):
        _, H, W, _ = input_tensor.shape
        input_tensor = ttnn.pad(input_tensor, input_tensor.shape, [0, 0, 0, 0], 0)
        input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        input_tensor_0 = input_tensor[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        input_tensor_1 = input_tensor[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        input_tensor_2 = input_tensor[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        input_tensor_3 = input_tensor[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        output_tensor = ttnn.concat(
            [input_tensor_0, input_tensor_1, input_tensor_2, input_tensor_3], -1, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.layer_norm(
            output_tensor,
            weight=self.parameters.norm["weight"],
            bias=self.parameters.norm["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(input_tensor_0)
        ttnn.deallocate(input_tensor_1)
        ttnn.deallocate(input_tensor_2)
        ttnn.deallocate(input_tensor_3)
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
                self.parameters.reduction["weight"],
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                program_config=program_configs["linear_config_1"],
            )
        elif output_tensor.shape[-1] == 768:
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
                self.parameters.reduction["weight"],
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                program_config=program_configs["linear_config_2"],
            )
        else:
            output_tensor = ttnn.linear(
                output_tensor,
                self.parameters.reduction["weight"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                core_grid=ttnn.CoreGrid(y=8, x=8),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        return output_tensor
