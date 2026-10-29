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
from ttml.modules import Embedding, VocabParallelEmbedding

__all__ = ["embed_tokens"]


def _row_major_table(weight):
    """The frozen table as a row-major tensor; converted once, the parameter then holds that layout."""
    table = weight.get_value()
    if table.layout == ttnn.TILE_LAYOUT:
        table = ttnn.untilize(table)
        weight.set_value(table)
    return table


def _frozen_embedding(input_ids, weight):
    """Plain lookup on the once-untilized table. Returns a tensor outside the autograd graph (``requires_grad``
    False): neither the ids nor the frozen table take a gradient, and an output that claimed to would make the
    first block's norm and qkv projection build a backward for it (review round 3, finding 1)."""
    table = _row_major_table(weight)
    ids = input_ids.get_value()
    batch, _, _, seq = [int(d) for d in ids.shape]
    out = ttnn.embedding(ids, table, layout=ttnn.TILE_LAYOUT)
    out = ttnn.reshape(out, [batch, 1, seq, int(out.shape[-1])])
    return ttml.autograd.create_tensor(out, requires_grad=False)


def _frozen_vocab_parallel_embedding(input_ids, weight, vocab_start, rows_per_rank, cluster_axis):
    """The vocab-parallel lookup (ids masked to the rank's shard, out-of-range rows zeroed, all-reduce over the TP
    axis) on a frozen table untilized once; the result is outside the autograd graph (see ``_frozen_embedding``)."""
    table = _row_major_table(weight)
    ids = input_ids.get_value(ttml.autograd.PreferredPrecision.NATIVE)
    local = ttnn.subtract(ttnn.typecast(ids, ttnn.DataType.INT32), vocab_start)
    in_range = ttnn.multiply(ttnn.ge(local, 0), ttnn.lt(local, rows_per_rank))
    local_ids = ttnn.typecast(ttnn.multiply(local, in_range), ttnn.DataType.UINT32)
    batch, _, _, seq = [int(d) for d in ids.shape]
    emb = ttnn.embedding(local_ids, table, layout=ttnn.TILE_LAYOUT)
    emb = ttnn.reshape(emb, [batch, 1, seq, int(emb.shape[-1])])
    mask = ttnn.transpose(ttnn.to_layout(ttnn.typecast(in_range, ttnn.DataType.FLOAT32), ttnn.Layout.TILE), 2, 3)
    emb = ttnn.multiply(emb, mask)
    emb = ttml.core.distributed.all_reduce(emb, cluster_axis=cluster_axis)
    return ttml.autograd.create_tensor(emb, requires_grad=False)


def embed_tokens(embedding, input_ids):
    """``embedding(input_ids)``; on the no-untilize path when the table is frozen: a plain ``Embedding`` or a
    ``VocabParallelEmbedding`` (iteration 8: its table was untilized by the ttml op on every step, 1.5 ms).
    A feature-parallel embedding keeps its own forward.
    """
    weight = embedding.weight.tensor
    if weight.get_requires_grad():
        return embedding(input_ids)
    if type(embedding) is Embedding:
        return _frozen_embedding(input_ids, weight)
    if type(embedding) is VocabParallelEmbedding:
        embedding._check_tp_replicated(input_ids)
        return _frozen_vocab_parallel_embedding(
            input_ids, weight, embedding._vocab_start, embedding.num_embeddings_per_partition, embedding.cluster_axis
        )
    return embedding(input_ids)
