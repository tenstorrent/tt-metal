# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


program_configs = {
    "linear_config_1": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2,
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


class TtPatchMergingV2:
    def __init__(self, device, parameters, dim):
        self.dim = dim
        self.device = device
        self.parameters = parameters

    def __call__(self, x):
        if use_signpost:
            signpost(header="swin_patch_merging_v2")

        B, H, W, C = x.shape
        x = ttnn.reshape(
            x,
            (B, H // 2, 2, W // 2, 2, C),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x = ttnn.permute(
            x,
            (0, 1, 3, 2, 4, 5),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        x = ttnn.reshape(
            x,
            (B, H // 2, W // 2, 4 * C),
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
                self.parameters.reduction["weight"],
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                ),
                program_config=program_configs["linear_config_1"],
            )
        elif x.shape[-1] == 768:
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
                self.parameters.reduction["weight"],
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                ),
                program_config=program_configs["linear_config_2"],
            )
        else:
            x = ttnn.linear(
                x,
                self.parameters.reduction["weight"],
                dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                ),
                core_grid=ttnn.CoreGrid(y=8, x=8),
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        x = ttnn.layer_norm(
            x,
            weight=self.parameters.norm["weight"],
            bias=self.parameters.norm["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return x
