# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
SwiGLU feed-forward network.

    output = down_proj( silu(gate_proj(x)) * up_proj(x) )

Every one of the 64 decoder layers has one of these, DeltaNet and full-attention
alike. It is also where most of the model's parameters live.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtQwen36MLP(LightweightModule):
    """
    SwiGLU MLP.

    Dimensions:
        D = 5120      hidden size
        I = 17408     intermediate size

    Weights (torch shapes, before the [out, in] -> [in, out] transpose):
        gate_proj  [17408, 5120]
        up_proj    [17408, 5120]
        down_proj  [ 5120, 17408]

    Shapes:
        x        [B, T, D]
        gate/up  [B, T, I]
        hidden   [B, T, I]      silu(gate) * up
        output   [B, T, D]

    To implement:
        1. gate = linear(x, gate_proj)   -- silu can fuse into the matmul via
                                            ttnn.linear(..., activation="silu")
        2. up   = linear(x, up_proj)
        3. hidden = gate * up
        4. output = linear(hidden, down_proj)
    """

    def __init__(self, device, layer_idx: int):
        self.device = device
        self.layer_idx = layer_idx

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[B, T, D] -> [B, T, D]."""
        return x
