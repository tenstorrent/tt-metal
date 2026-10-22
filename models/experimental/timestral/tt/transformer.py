# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN Timestral transformer block (attention sublayers of one layer kind, then the MLP).

reference : models/experimental/timestral/reference/src/timestral/layers.py (``TransformerBlock``)
    for attn in (time,) | (time, variate):   x = x + attn(x)   # residual adds run in the wo / down matmuls
    x = x + mlp(x)
"""

from __future__ import annotations

from models.experimental.timestral.tt.attention import TtAttention, TtAttentionInputs
from models.experimental.timestral.tt.common import TtTimestralPrecision
from models.experimental.timestral.tt.mlp import TtSwiGLU
from models.experimental.timestral.tt.weights import BlockWeights


class TtTransformerBlock:
    def __init__(self, device, weights: BlockWeights, precision: TtTimestralPrecision):
        self.kind = weights.kind
        self.attns = [TtAttention(device, a, precision) for a in weights.attns]
        self.mlp = TtSwiGLU(device, weights.mlp, precision)

    def __call__(self, x, inputs: TtAttentionInputs):
        """Flat device ``(1, 1, R * S_pad, D)``; consumes ``x`` and returns the new residual stream."""
        import ttnn

        for attn in self.attns:
            y = attn(x, inputs, residual=x)
            ttnn.deallocate(x)
            x = y
        y = self.mlp(x, residual=x)
        ttnn.deallocate(x)
        return y
