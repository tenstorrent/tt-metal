# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tt_lib.fallback_ops import fallback_ops


def torch_to_ttnn(input, device, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat16, device=device, layout=layout)
    return input


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


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
        x = ttnn_to_torch(x)
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C Issue #8920
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C Issue #8920
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C Issue #8920
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C Issue #8920
        x = torch_to_ttnn(x, self.device)
        x0 = torch_to_ttnn(x0, self.device)
        x1 = torch_to_ttnn(x1, self.device)
        x2 = torch_to_ttnn(x2, self.device)
        x3 = torch_to_ttnn(x3, self.device)
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
