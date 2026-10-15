# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer's temporal self-attention (``reference/temporal_self_attention.py``):
deformable attention of the BEV queries over the previous BEV and themselves, on the fused MSDA op.

The value stacks two BEV maps per sample, the previous BEV and the current queries, as upstream's
encoder builds it. The offsets and attention weights read both maps concatenated and give one set
of points per map; the two sampled results are averaged. The parameters must come from
``model_preprocessing.create_temporal_self_attention_parameters``, which orders the Linears' output
channels queue-major, so each map's channels are a contiguous half of the row.
"""

import torch

import ttnn
from models.experimental.bevformer.model_config import EMBED_DIMS, NUM_HEADS, TSA_NUM_POINTS
from models.experimental.bevformer.tt.tt_common import signpost
from models.experimental.bevformer.tt.tt_ms_deformable_attention import multi_scale_deformable_attn_fused

# BEV maps the value stacks per sample: the previous BEV and the current queries.
NUM_BEV_QUEUE = 2


class TTTemporalSelfAttention:
    """Batch-first temporal self-attention over one BEV level. It calls the fused op directly
    rather than through TTMSDeformableAttention: its Linears read both maps concatenated, and the op
    samples each map with its own half of their channels."""

    def __init__(
        self, params, *, spatial_shapes, embed_dims=EMBED_DIMS, num_heads=NUM_HEADS, num_points=TSA_NUM_POINTS
    ):
        """``params`` from ``model_preprocessing.create_temporal_self_attention_parameters``;
        ``spatial_shapes`` the BEV grid as one ``(h, w)`` level; ``num_points`` sampling points per
        head and map."""
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})")
        if len(spatial_shapes) != 1:
            raise ValueError(f"the self-attention samples one BEV level, got {len(spatial_shapes)}")
        self.params = params
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.spatial_shapes = torch.as_tensor(spatial_shapes)

    def __call__(self, query, value=None, query_pos=None, reference_points=None):
        """``query`` the BEV queries ``(bs, num_queries, C)``; ``value`` the previous BEV and the
        encoder's input query stacked per sample, ``(bs * NUM_BEV_QUEUE, num_queries, C)``, or None
        on the first frame for the query stacked with itself; ``query_pos`` their positional
        encoding or None; ``reference_points`` ``(bs * NUM_BEV_QUEUE, num_queries, 1, 2)`` in [0, 1],
        float32 ROW_MAJOR, the previous BEV's ego-shifted, the current query's as they are.
        Returns ``(bs, num_queries, C)``, the residual included."""
        signpost(header="TTNN TSA Forward Start")
        bs, num_queries, embed_dims = query.shape
        queue = NUM_BEV_QUEUE
        if value is None:
            value = ttnn.reshape(
                ttnn.concat([ttnn.unsqueeze(query, 1), ttnn.unsqueeze(query, 1)], dim=1),
                (bs * queue, num_queries, embed_dims),
            )

        identity = query
        if query_pos is not None:
            query = ttnn.add(query, query_pos)

        # value[:bs], as upstream: the first bs stacked maps, which is each sample's previous BEV
        # only at bs=1 (BEVFormer's inference batch).
        query = ttnn.concat([value[:bs], query], dim=-1)
        value = ttnn.linear(value, self.params.value_proj.weight, bias=self.params.value_proj.bias)

        # Queue-major channels to the batch, in ROW_MAJOR, where the queue axis never lands in a
        # tiled dimension: (bs, nq, queue * heads * width) -> (bs * queue, nq, heads, width).
        def split_queue(tensor):
            width = tensor.shape[-1] // (queue * self.num_heads)
            tensor = ttnn.reshape(ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT), (bs, num_queries, queue, -1))
            return ttnn.reshape(ttnn.permute(tensor, (0, 2, 1, 3)), (bs * queue, num_queries, self.num_heads, width))

        sampling_offsets = ttnn.linear(
            query, self.params.sampling_offsets.weight, bias=self.params.sampling_offsets.bias
        )
        attention_weights = ttnn.linear(
            query, self.params.attention_weights.weight, bias=self.params.attention_weights.bias
        )
        attention_weights = ttnn.softmax(
            ttnn.reshape(attention_weights, (bs, num_queries, queue * self.num_heads, self.num_points)), dim=-1
        )
        attention_weights = ttnn.reshape(attention_weights, (bs, num_queries, queue * self.num_heads * self.num_points))

        output = multi_scale_deformable_attn_fused(
            value,
            reference_points,
            split_queue(sampling_offsets),
            split_queue(attention_weights),
            self.spatial_shapes,
        )
        output = ttnn.reshape(output, (bs, queue, num_queries, embed_dims))
        output = ttnn.mean(ttnn.to_layout(output, ttnn.TILE_LAYOUT), dim=1)
        output = ttnn.linear(output, self.params.output_proj.weight, bias=self.params.output_proj.bias)
        output = ttnn.add(output, identity)
        signpost(header="TTNN TSA Forward End")
        return output
