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
        sample = ttnn.linear(sample, self.linear_1_w, bias=self.linear_1_b, memory_config=ttnn.L1_MEMORY_CONFIG)
        sample = ttnn.silu(sample)
        sample = ttnn.linear(sample, self.linear_2_w, bias=self.linear_2_b, memory_config=ttnn.L1_MEMORY_CONFIG)
        return sample
