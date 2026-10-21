# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN Timestral transformer block (attention sublayers of one layer kind, then the MLP).

reference : models/experimental/timestral/reference/src/timestral/layers.py (``TransformerBlock``)
    for attn in (time,) | (time, variate):   x = x + attn(x)
    x = x + mlp(x)
"""

from __future__ import annotations

from models.experimental.timestral.tt.attention import TtAttention, TtAttentionInputs
from models.experimental.timestral.tt.common import TtTimestralPrecision, add_residual
from models.experimental.timestral.tt.mlp import TtSwiGLU
from models.experimental.timestral.tt.weights import BlockWeights


class TtTransformerBlock:
    def __init__(self, device, weights: BlockWeights, precision: TtTimestralPrecision):
        self.kind = weights.kind
        self.attns = [TtAttention(device, a, precision) for a in weights.attns]
        self.mlp = TtSwiGLU(device, weights.mlp, precision)

    def __call__(self, x, inputs: TtAttentionInputs):
        """Flat device ``(1, 1, R * S_pad, D)``; consumes ``x`` and returns the new residual stream."""
        for attn in self.attns:
            x = add_residual(x, attn(x, inputs))
        return add_residual(x, self.mlp(x))
