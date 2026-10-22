# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN Timestral transformer block (attention sublayers of one layer kind, then the MLP).

reference : models/experimental/timestral/reference/src/timestral/layers.py (``TransformerBlock``)
    for attn in (time,) | (time, variate):   x = x + attn(x)   # residual adds run in the wo / down matmuls
    x = x + mlp(x)
The last block keeps only the decoder slots after its last time attention.
"""

from __future__ import annotations

from models.experimental.timestral.tt.attention import DecoderSlots, TtAttention, TtAttentionInputs
from models.experimental.timestral.tt.common import TtTimestralPrecision
from models.experimental.timestral.tt.mlp import TtSwiGLU
from models.experimental.timestral.tt.weights import BlockWeights


class TtTransformerBlock:
    def __init__(self, device, weights: BlockWeights, precision: TtTimestralPrecision):
        self.kind = weights.kind
        self.attns = [TtAttention(device, a, precision) for a in weights.attns]
        self.mlp = TtSwiGLU(device, weights.mlp, precision)

    def __call__(self, x, inputs: TtAttentionInputs, decoder: DecoderSlots | None = None):
        """Flat device ``(1, 1, R * S_pad, D)``; consumes ``x`` and returns the new residual stream.
        With ``decoder`` (the last block) only the decoder slots come out, ``(1, 1, R * count, D)``,
        cut right after the block's last time attention: nothing later mixes token slots."""
        import ttnn

        cut = None
        if decoder is not None:
            cut = max((i for i, attn in enumerate(self.attns) if attn.is_time), default=None)
            if cut is None:
                y = decoder.take(x, inputs)
                ttnn.deallocate(x)
                x, inputs = y, decoder.inputs
        for i, attn in enumerate(self.attns):
            y = attn(x, inputs, residual=x, decoder=decoder if i == cut else None)
            ttnn.deallocate(x)
            x = y
            if i == cut:
                inputs = decoder.inputs
        y = self.mlp(x, residual=x)
        ttnn.deallocate(x)
        return y
