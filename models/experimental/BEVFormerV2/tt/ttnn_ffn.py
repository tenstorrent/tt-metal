# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtFFN:
    """TTNN implementation of FFN"""

    def __init__(self, params, device):
        self.device = device
        self.linear1_weight = params.linear1.weight
        self.linear2_weight = params.linear2.weight
        self.linear1_bias = params.linear1.bias
        self.linear2_bias = params.linear2.bias

    def __call__(self, x, identity=None):
        if identity is None:
            identity = x
        x = ttnn.linear(x, self.linear1_weight, bias=self.linear1_bias)
        x = ttnn.relu(x)
        out = ttnn.linear(x, self.linear2_weight, bias=self.linear2_bias)
        x = ttnn.add(out, identity)
        ttnn.deallocate(out)
        ttnn.deallocate(identity)
        return x
