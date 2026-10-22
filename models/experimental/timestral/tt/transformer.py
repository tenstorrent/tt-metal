# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN Timestral transformer block (attention sublayers of one layer kind, then the MLP).

reference : models/experimental/timestral/reference/src/timestral/layers.py (``TransformerBlock``)
    for attn in (time,) | (time, variate):   x = x + attn(x)   # residual adds run in the wo / down matmuls
    x = x + mlp(x)
The last block keeps only the decoder slots after its last time attention.
"""

from __future__ import annotations

from models.experimental.timestral.tt.attention import DecoderSlots, SplitStreams, TtAttention, TtAttentionInputs
from models.experimental.timestral.tt.common import TtTimestralPrecision
from models.experimental.timestral.tt.mlp import TtSwiGLU
from models.experimental.timestral.tt.weights import BlockWeights


class TtTransformerBlock:
    def __init__(self, device, weights: BlockWeights, precision: TtTimestralPrecision, *, last: bool = False):
        """``last``: the block that ends the context stream (``split(last=True)``)."""
        self.kind = weights.kind
        self._cut = max((i for i, a in enumerate(weights.attns) if a.is_time), default=None)
        self.attns = [
            TtAttention(device, a, precision, kv_only=last and i == self._cut) for i, a in enumerate(weights.attns)
        ]
        self.mlp = TtSwiGLU(device, weights.mlp, precision)

    def __call__(self, x, inputs: TtAttentionInputs, decoder: DecoderSlots | None = None):
        """Flat device ``(1, 1, R * S_pad, D)``; consumes ``x`` and returns the new residual stream.
        With ``decoder`` (the last block) only the decoder slots come out, ``(1, 1, R * count, D)``,
        cut right after the block's last time attention: nothing later mixes token slots."""
        import ttnn

        cut = None
        if decoder is not None:
            cut = self._cut
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

    def split(self, x_c, x_d, streams: SplitStreams, *, last: bool = False):
        """Context and decoder streams through the block; consumes both. With ``last`` the context
        stream ends at the block's last time attention, which reads only its keys and values, and the
        returned context is None."""
        import ttnn

        cut = self._cut if last else None
        if last and cut is None:
            ttnn.deallocate(x_c)
            x_c = None
        for i, attn in enumerate(self.attns):
            if attn.is_time:
                y_c, y_d = attn.split_time(x_c, x_d, streams, context_out=i != cut)
            else:
                y_c = None if x_c is None else attn(x_c, streams.context, residual=x_c)
                y_d = attn(x_d, streams.decoder, residual=x_d)
            for t in (x_c, x_d):
                if t is not None:
                    ttnn.deallocate(t)
            x_c, x_d = y_c, y_d
        y_c = None if x_c is None else self.mlp(x_c, residual=x_c)
        y_d = self.mlp(x_d, residual=x_d)
        for t in (x_c, x_d):
            if t is not None:
                ttnn.deallocate(t)
        return y_c, y_d
