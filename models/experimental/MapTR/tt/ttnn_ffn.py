# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtFFN:
    """TT implementation of FFN matching reference:
    out = Linear(ReLU(Linear(x)))
    return identity + out
    """

    def __init__(self, params, device):
        self.device = device
        self.linear1_weight = params.linear1.weight
        self.linear2_weight = params.linear2.weight
        self.linear1_bias = params.linear1.bias
        self.linear2_bias = params.linear2.bias

    def __call__(self, x, identity=None):
        # Store identity before modifying x (matches reference behavior)
        if identity is None:
            identity = x

        # Ensure input is in TILE_LAYOUT for linear operations
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # First linear: embed_dims -> feedforward_channels
        out = ttnn.linear(x, self.linear1_weight, bias=self.linear1_bias)

        # ReLU activation
        out = ttnn.relu(out)

        # Second linear: feedforward_channels -> embed_dims
        out = ttnn.linear(out, self.linear2_weight, bias=self.linear2_bias)

        # Residual connection: identity + out (matches reference)
        result = ttnn.add(identity, out)

        ttnn.deallocate(out)

        return result
