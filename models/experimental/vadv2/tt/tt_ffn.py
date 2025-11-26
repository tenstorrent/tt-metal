# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtFFN:
    def __init__(self, params, device):
        self.device = device
        self.linear1_weight = params.linear1.weight
        self.linear2_weight = params.linear2.weight
        self.linear1_bias = params.linear1.bias
        self.linear2_bias = params.linear2.bias

    def __call__(self, x, identity=None):
        if identity is None:
            identity = x

        # STEP 2: Fuse GeLU activation with first linear operation
        # This eliminates the separate ttnn.relu() call and improves performance
        x = ttnn.linear(x, self.linear1_weight, bias=self.linear1_bias, activation="gelu")

        # Second linear
        x = ttnn.linear(x, self.linear2_weight, bias=self.linear2_bias)

        # Residual connection
        x = ttnn.add(x, identity)
        ttnn.deallocate(identity)
        return x
