# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Token embedding lookup for a frozen table, without re-untilizing it every step.

``ttml.ops.embedding.embedding`` untilizes the whole weight on every forward before the lookup
(``ops/embedding_op.cpp``). For Mistral-Small-24B that is the 131072 x 5120 table, 1.34 GB, for a table
that never changes under LoRA. :func:`embed_tokens` converts the frozen table to ROW_MAJOR once, in place
of the TILE copy (nothing is duplicated), and then runs ``ttnn.embedding`` directly. The output is
identical. A trainable table goes through the ttml op as before, since its backward needs the TILE weight.
"""

from __future__ import annotations

import ttnn
import ttml
from ttml.autograd import Function
from ttml.modules import Embedding

__all__ = ["FrozenEmbedding", "embed_tokens"]


class FrozenEmbedding(Function):
    @staticmethod
    def forward(ctx, input_ids, weight):
        table = weight.get_value()
        if table.layout == ttnn.TILE_LAYOUT:
            # One-time: the frozen parameter now holds the row-major table.
            table = ttnn.untilize(table)
            weight.set_value(table)
        ids = input_ids.get_value()
        batch, _, _, seq = [int(d) for d in ids.shape]
        out = ttnn.embedding(ids, table, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(out, [batch, 1, seq, int(out.shape[-1])])

    @staticmethod
    def backward(ctx, grad_output):
        return None, None


def embed_tokens(embedding, input_ids):
    """``embedding(input_ids)``; on the no-untilize path when the table is a frozen plain ``Embedding``.

    A vocab- or feature-parallel embedding (tensor parallelism) masks ids and reduces across chips, so it keeps
    its own forward; its table is then untilized by the ttml op every step (a known cost under TP).
    """
    weight = embedding.weight.tensor
    if type(embedding) is not Embedding or weight.get_requires_grad():
        return embedding(input_ids)
    return FrozenEmbedding.apply(input_ids, weight)
