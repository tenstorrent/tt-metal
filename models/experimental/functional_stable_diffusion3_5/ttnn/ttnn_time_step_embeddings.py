# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class ttnn_TimestepEmbedding:
    def __init__(self, parameters):
        self.linear_1_w = parameters.linear_1.weight
        self.linear_1_b = parameters.linear_1.bias
        self.linear_2_w = parameters.linear_2.weight
        self.linear_2_b = parameters.linear_2.bias

    def __call__(self, sample, device):
        hifi2_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
        )

        mm_a_y = 8
        mm_a_x = 6
        mm_a_x_strategy = ttnn.ShardStrategy.WIDTH
        mm_a_x_memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

        sample = ttnn.unsqueeze(sample, dim=1)
        sample = ttnn.unsqueeze(sample, dim=1)

        sample = ttnn.linear(
            sample,
            self.linear_1_w,
            bias=self.linear_1_b,
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
            compute_kernel_config=hifi2_kernel_config,
        )
        sample = ttnn.silu(sample)
        sample = ttnn.linear(
            sample,
            self.linear_2_w,
            bias=self.linear_2_b,
            memory_config=mm_a_x_memory_config,
            core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
            compute_kernel_config=hifi2_kernel_config,
        )
        return sample
