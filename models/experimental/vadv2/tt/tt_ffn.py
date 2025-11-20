# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.vadv2.tt.tt_opt_utils import get_high_perf_compute_config


class TtFFN:
    def __init__(self, params, device):
        self.device = device
        self.linear1_weight = params.linear1.weight
        self.linear2_weight = params.linear2.weight
        self.linear1_bias = params.linear1.bias
        self.linear2_bias = params.linear2.bias

        # LOAD OPTIMIZATION CONFIG
        self.compute_config = get_high_perf_compute_config()

    def __call__(self, x, identity=None):
        if identity is None:
            identity = x

        # First linear + ReLU
        # Optimization: Use HiFi2 compute config (faster math), but default memory (DRAM) to avoid OOM
        x = ttnn.linear(x, self.linear1_weight, bias=self.linear1_bias, compute_kernel_config=self.compute_config)
        x = ttnn.relu(x)

        # Second linear
        # Optimization: Use HiFi2 compute config
        x = ttnn.linear(x, self.linear2_weight, bias=self.linear2_bias, compute_kernel_config=self.compute_config)

        # Residual connection
        x = ttnn.add(x, identity)
        ttnn.deallocate(identity)
        return x
