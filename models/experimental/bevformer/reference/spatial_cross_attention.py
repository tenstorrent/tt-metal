# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Spatial cross-attention (SCA) in PyTorch.

This module implements the encoder's cross-attention from the BEV queries into the camera
features, as upstream's ``SpatialCrossAttention`` and ``MSDeformableAttention3D``
(``projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py`` in
fundamentalvision/BEVFormer). It is the reference ``tt/tt_spatial_cross_attention.py`` is checked
against. Parameter names follow upstream, so a checkpoint's
``pts_bbox_head.transformer.encoder.layers.<i>.attentions.1`` weights load unchanged.

Each BEV query is gathered ("rebatched") into every camera whose image its pillar's points land
in (``bev_mask``). There it attends to the four FPN levels around those points: its 8 sampling
points are split over the pillar's 4 points. The per-camera results are scattered back,
averaged over the cameras that saw the query, projected, and added to the queries.

Only inference is kept: dropout is dropped.
"""

import math

import torch
import torch.nn as nn

from models.experimental.bevformer.model_config import EMBED_DIMS, NUM_CAMS, NUM_HEADS, NUM_LEVELS, SCA_NUM_POINTS

from models.experimental.bevformer.reference.ms_deformable_attention import multi_scale_deformable_attn


class MSDeformableAttention3D(nn.Module):
    """Deformable attention whose ``num_points`` are split over the pillar's ``num_Z_anchors``
    reference points: each anchor gets ``num_points // num_Z_anchors`` offsets. No output
    projection; SpatialCrossAttention applies one after the cameras are merged."""

    def __init__(
        self,
        embed_dims=EMBED_DIMS,
        num_heads=NUM_HEADS,
        num_levels=NUM_LEVELS,
        num_points=SCA_NUM_POINTS,
        batch_first=True,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims ({embed_dims}) must be divisible by num_heads ({num_heads})")
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.batch_first = batch_first
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Upstream's init: offsets start on a ring of ``num_points`` steps per head, attention uniform."""
        nn.init.zeros_(self.sampling_offsets.weight)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2)
        grid = grid.repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.view(-1))
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)

    def forward(self, query, value, reference_points, spatial_shapes):
        """``query`` ``(bs, num_query, C)``, ``value`` ``(bs, num_value, C)``, ``reference_points``
        ``(bs, num_query, num_Z_anchors, 2)`` in [0, 1]; ``spatial_shapes`` ``(num_levels, 2)`` as (h, w)."""
        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value).view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        num_z_anchors = reference_points.shape[2]
        sampling_offsets = (sampling_offsets / offset_normalizer[None, None, None, :, None, :]).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points // num_z_anchors, num_z_anchors, 2
        )
        sampling_locations = reference_points[:, :, None, None, None, :, :] + sampling_offsets
        sampling_locations = sampling_locations.view(bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        return multi_scale_deformable_attn(value, spatial_shapes, sampling_locations, attention_weights)


class SpatialCrossAttention(nn.Module):
    """Cross-attention from the BEV queries into the cameras that see them, through
    ``MSDeformableAttention3D``, averaged over those cameras and projected."""

    def __init__(
        self,
        embed_dims=EMBED_DIMS,
        num_cams=NUM_CAMS,
        num_heads=NUM_HEADS,
        num_levels=NUM_LEVELS,
        num_points=SCA_NUM_POINTS,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.deformable_attention = MSDeformableAttention3D(
            embed_dims=embed_dims, num_heads=num_heads, num_levels=num_levels, num_points=num_points
        )
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, query, value, reference_points_cam, bev_mask, spatial_shapes, query_pos=None):
        """``query`` ``(bs, num_query, C)``; ``value`` ``(num_cams, num_value, bs, C)``;
        ``reference_points_cam`` ``(num_cams, bs, num_query, D, 2)`` and ``bev_mask``
        ``(num_cams, bs, num_query, D)`` from point_sampling."""
        inp_residual = query
        slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape
        depth = reference_points_cam.size(3)
        # As upstream, the queries a camera sees come from the first sample's mask.
        indexes = [mask_per_img[0].sum(-1).nonzero().squeeze(-1) for mask_per_img in bev_mask]
        max_len = max(len(each) for each in indexes)

        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, depth, 2])
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, : len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, : len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img
                ]

        num_cams, num_value, bs, embed_dims = value.shape
        value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, num_value, self.embed_dims)
        queries = self.deformable_attention(
            query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
            value=value,
            reference_points=reference_points_rebatch.view(bs * self.num_cams, max_len, depth, 2),
            spatial_shapes=spatial_shapes,
        ).view(bs, self.num_cams, max_len, self.embed_dims)

        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, : len(index_query_per_img)]

        count = (bev_mask.sum(-1) > 0).permute(1, 2, 0).sum(-1)
        slots = slots / torch.clamp(count, min=1.0)[..., None]
        return self.output_proj(slots) + inp_residual
