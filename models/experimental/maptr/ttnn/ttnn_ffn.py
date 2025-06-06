# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtnnFFN:
    def __init__(self, conv_pth, device):
        self.device = device
        self.linear1_weight = conv_pth.linear1.weight
        self.linear2_weight = conv_pth.linear2.weight
        self.linear1_bias = conv_pth.linear1.bias
        self.linear2_bias = conv_pth.linear2.bias

    def __call__(self, x, identity=None):
        if identity is None:
            identity = x

        # First linear + ReLU
        x = ttnn.linear(x, self.linear1_weight, self.linear1_bias)
        x = ttnn.relu(x)

        # Second linear
        x = ttnn.linear(x, self.linear2_weight, self.linear2_bias)

        # Residual connection
        x = ttnn.add(x, identity)

        return x
