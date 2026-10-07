# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Temporal self-attention (TSA) in PyTorch.

This module implements the encoder's self-attention over the BEV queries, as upstream's
``TemporalSelfAttention`` (``projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py``
in fundamentalvision/BEVFormer). It is the reference ``tt/tt_temporal_self_attention.py`` is
checked against. Parameter names follow upstream, so a checkpoint's
``pts_bbox_head.transformer.encoder.layers.<i>.attentions.0`` weights load unchanged.

The value holds ``num_bev_queue`` (2) BEV maps per sample: the previous BEV, aligned to the
current frame, and the current BEV queries; on the first frame the queries twice. The sampling
offsets and attention weights read the previous BEV and the queries concatenated, and give one
set of deformable-attention points per map, around the cell's reference point (shifted by the
ego motion for the previous BEV). The two sampled results are averaged, projected, and added to
the queries.

Only inference is kept: dropout is dropped.
"""

import math

import torch
import torch.nn as nn

from models.experimental.bevformer.model_config import EMBED_DIMS, NUM_HEADS, TSA_NUM_POINTS

from models.experimental.bevformer.reference.ms_deformable_attention import multi_scale_deformable_attn


class TemporalSelfAttention(nn.Module):
    """Deformable self-attention over ``num_bev_queue`` (2) stacked BEV maps per sample, the
    previous BEV and the current queries, averaged."""

    def __init__(
        self,
        embed_dims=EMBED_DIMS,
        num_heads=NUM_HEADS,
        num_levels=1,
        num_points=TSA_NUM_POINTS,
        num_bev_queue=2,
        batch_first=True,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})")
        assert num_bev_queue == 2, "the value stacks the previous BEV and the current query"
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.batch_first = batch_first
        self.sampling_offsets = nn.Linear(
            embed_dims * num_bev_queue, num_bev_queue * num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims * num_bev_queue, num_bev_queue * num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Upstream's init: offsets start on a ring of ``num_points`` steps per head, attention uniform."""
        nn.init.zeros_(self.sampling_offsets.weight)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2)
        grid = grid.repeat(1, self.num_levels * self.num_bev_queue, self.num_points, 1)
        for i in range(self.num_points):
            grid[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.view(-1))
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        for proj in (self.value_proj, self.output_proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, query, value=None, identity=None, query_pos=None, reference_points=None, spatial_shapes=None):
        """``query`` ``(bs, num_query, C)``; ``value`` ``(bs * 2, num_query, C)``, the previous BEV
        and the query stacked per sample, or None to stack the query with itself; ``reference_points``
        ``(bs * 2, num_query, num_levels, 2)`` in [0, 1], one set per stacked map."""
        assert self.batch_first
        if value is None:
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        # Upstream's value[:bs]: the first bs stacked maps, which is each sample's previous BEV
        # only at bs=1 (BEVFormer's inference batch); kept to match upstream.
        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value).reshape(bs * self.num_bev_queue, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points
        )
        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5).reshape(
            bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points
        )
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(
            bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )
        output = multi_scale_deformable_attn(value, spatial_shapes, sampling_locations, attention_weights)

        # (bs * queue, num_query, C) -> mean over the queue.
        output = output.permute(1, 2, 0).view(num_query, embed_dims, bs, self.num_bev_queue).mean(-1)
        output = self.output_proj(output.permute(2, 0, 1))
        return output + identity
