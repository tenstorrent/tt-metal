# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class TtFFN:
    def __init__(self, params, device):
        self.device = device
        self.linear1_weight = params.linear1.weight
        self.linear2_weight = params.linear2.weight
        self.linear1_bias = params.linear1.bias
        self.linear2_bias = params.linear2.bias

    def __call__(self, x, identity=None):
        if use_signpost:
            signpost(header="TtFFN_call_start")
        if identity is None:
            identity = x

        # First linear + ReLU
        x = ttnn.linear(x, self.linear1_weight, bias=self.linear1_bias)
        x = ttnn.relu(x)

        # Second linear
        x = ttnn.linear(x, self.linear2_weight, bias=self.linear2_bias)

        # Residual connection
        x = ttnn.add(x, identity)
        ttnn.deallocate(identity)
        if use_signpost:
            signpost(header="TtFFN_call_end")
        return x
