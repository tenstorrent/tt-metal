# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tt_lib.fallback_ops import fallback_ops

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

    def __call__(self, x):
        _, H, W, _ = x.get_legacy_shape()
        x = ttnn.pad(
            x, x.shape, [0, 0, 0, 0], 0
        )  # This is not needed(check on this) x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2)) , No difference in shape
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x0 = ttnn.to_layout(x0, layout=ttnn.TILE_LAYOUT)
        x1 = ttnn.to_layout(x1, layout=ttnn.TILE_LAYOUT)
        x2 = ttnn.to_layout(x2, layout=ttnn.TILE_LAYOUT)
        x3 = ttnn.to_layout(x3, layout=ttnn.TILE_LAYOUT)
        x = ttnn.concat([x0, x1, x2, x3], -1)
        x = ttnn.layer_norm(x, weight=self.parameters.norm["weight"], bias=self.parameters.norm["bias"])
        if x.shape[-1] == 384:
            x = ttnn.to_memory_config(
                x,
                memory_config=ttnn.create_sharded_memory_config(
                    x.shape,
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                ),
                dtype=ttnn.bfloat8_b,
            )

            x = ttnn.linear(
                x,
                self.parameters.reduction["weight"],
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
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
                dtype=ttnn.bfloat8_b,
            )
            x = ttnn.linear(
                x,
                self.parameters.reduction["weight"],
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                program_config=program_configs["linear_config_2"],
            )
        else:
            x = ttnn.linear(
                x,
                self.parameters.reduction["weight"],
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                core_grid=ttnn.CoreGrid(y=8, x=8),
            )

        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        return x
