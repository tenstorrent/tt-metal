# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import os

try:
    from tracy import signpost

    use_signpost = os.getenv("USE_SIGNPOST", "False").lower() in ("true", "1", "yes")
except ModuleNotFoundError:
    use_signpost = False


class TtFFN:
    def __init__(self, params, device, embed_dims=256, feedforward_dims=512):
        self.device = device
        self.embed_dims = embed_dims
        self.feedforward_dims = feedforward_dims

        # Store weights
        self.linear1_weight = params.linear1.weight
        self.linear2_weight = params.linear2.weight
        self.linear1_bias = params.linear1.bias
        self.linear2_bias = params.linear2.bias

        # Note: VAD-v2 has variable sequence lengths (300-64,000) with limited L1 memory
        # Custom program configs cause L1 overflow, so we let TTNN auto-select
        # We keep fused ReLU activation which is supported without custom configs

    def __call__(self, x, identity=None):
        if use_signpost:
            signpost(header="TtFFN_call_start")
        if identity is None:
            identity = x

        # Ensure input is in TILE_LAYOUT
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # First linear: 256 -> 512 (with fused ReLU)
        x = ttnn.linear(
            x,
            self.linear1_weight,
            bias=self.linear1_bias,
            dtype=ttnn.bfloat16,
            activation="relu",  # Fused ReLU activation
        )

        # Second linear: 512 -> 256
        x = ttnn.linear(
            x,
            self.linear2_weight,
            bias=self.linear2_bias,
            dtype=ttnn.bfloat16,
        )

        # Residual connection
        x = ttnn.add(
            x,
            identity,
            dtype=ttnn.bfloat16,
        )

        ttnn.deallocate(identity)
        if use_signpost:
            signpost(header="TtFFN_call_end")
        return x
