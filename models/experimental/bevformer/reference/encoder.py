# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
BEVFormer encoder in PyTorch.

This module implements the encoder that builds the bird's-eye-view map: one query per BEV cell
(200x200 for BEVFormer-base, 50x50 for BEVFormer-tiny) gathers the previous frame's BEV and the
six cameras' FPN features. It follows upstream's ``BEVFormerEncoder`` and ``BEVFormerLayer``
(``projects/mmdet3d_plugin/bevformer/modules/encoder.py`` in fundamentalvision/BEVFormer) and is
the reference the TTNN encoder in ``tt/tt_encoder.py`` is checked against. Module names follow
mmcv's ``BaseTransformerLayer`` (``attentions``, ``ffns``, ``norms``), so the
``pts_bbox_head.transformer.encoder`` weights of a BEVFormer checkpoint load into it with that
prefix stripped.

Per frame, the encoder projects each BEV pillar's ``num_points_in_pillar`` points into the
cameras (``point_sampling``) and builds the cells' 2D reference points; for the previous BEV
these are shifted by the ego motion. Each of the six layers then performs, batch-first:
1. Temporal self-attention over the BEV queries and the previous BEV, the positional encoding
   added to the query, with a residual add
2. LayerNorm
3. Spatial cross-attention into the cameras each pillar projects into, with a residual add
4. LayerNorm
5. FFN (Linear-ReLU-Linear) with a residual add
6. LayerNorm

The previous BEV must already be rotated to the current frame; the detector does that before
calling the encoder. Only inference is kept: dropout is dropped.
"""

import torch
import torch.nn as nn

from models.experimental.bevformer.config.head_config import PC_RANGE

from .point_sampling_3d_2d import bev_reference_points, camera_geometry
from .spatial_cross_attention import SpatialCrossAttention
from .temporal_self_attention import TemporalSelfAttention


class FFN(nn.Module):
    """mmcv's two-layer ``FFN`` with its identity shortcut; dropout is a no-op in inference."""

    def __init__(self, embed_dims=256, feedforward_channels=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(embed_dims, feedforward_channels), nn.ReLU(inplace=True)),
            nn.Linear(feedforward_channels, embed_dims),
        )

    def forward(self, x):
        return x + self.layers(x)


class BEVFormerLayer(nn.Module):
    """One encoder layer: temporal self-attention (``attentions[0]``), spatial cross-attention
    (``attentions[1]``) and an FFN, each followed by its LayerNorm (``norms``)."""

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=8,
        num_cams=6,
        feedforward_channels=512,
        tsa_num_points=4,
    ):
        super().__init__()
        self.attentions = nn.ModuleList(
            [
                TemporalSelfAttention(
                    embed_dims=embed_dims, num_heads=num_heads, num_levels=1, num_points=tsa_num_points
                ),
                SpatialCrossAttention(
                    embed_dims=embed_dims,
                    num_cams=num_cams,
                    num_heads=num_heads,
                    num_levels=num_levels,
                    num_points=num_points,
                ),
            ]
        )
        self.ffns = nn.ModuleList([FFN(embed_dims, feedforward_channels)])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(3)])

    def forward(
        self, query, value, bev_pos, ref_2d, bev_shape, spatial_shapes, reference_points_cam, bev_mask, prev_bev
    ):
        """``query`` and ``bev_pos`` ``(bs, num_query, C)``; ``prev_bev`` ``(bs * 2, num_query, C)``
        (the previous BEV and the encoder's input query, stacked) or None; ``ref_2d``
        ``(bs * 2, num_query, 1, 2)``. The positional encoding enters the self-attention only."""
        query = self.attentions[0](
            query, value=prev_bev, query_pos=bev_pos, reference_points=ref_2d, spatial_shapes=bev_shape
        )
        query = self.norms[0](query)
        query = self.attentions[1](
            query,
            value=value,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            spatial_shapes=spatial_shapes,
        )
        query = self.norms[1](query)
        return self.norms[2](self.ffns[0](query))


class BEVFormerEncoder(nn.Module):
    """The six-layer encoder. ``num_points`` is the spatial cross-attention's sampling points per
    head and level, split over the ``num_points_in_pillar`` heights of each BEV cell's pillar;
    ``tsa_num_points`` the self-attention's. ``pc_range`` is the metric box the BEV grid and the
    pillars cover, shared with the head."""

    def __init__(
        self,
        num_layers=6,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=8,
        num_cams=6,
        feedforward_channels=512,
        tsa_num_points=4,
        num_points_in_pillar=4,
        pc_range=PC_RANGE,
    ):
        super().__init__()
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = list(pc_range)
        self.layers = nn.ModuleList(
            [
                BEVFormerLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_levels=num_levels,
                    num_points=num_points,
                    num_cams=num_cams,
                    feedforward_channels=feedforward_channels,
                    tsa_num_points=tsa_num_points,
                )
                for _ in range(num_layers)
            ]
        )

    def point_sampling(self, bev_h, bev_w, bs, img_metas):
        """Each BEV pillar's points in every camera and whether they land in the image; see
        ``point_sampling_3d_2d.camera_geometry``."""
        assert len(img_metas) == bs, f"{len(img_metas)} img_metas for batch size {bs}"
        return camera_geometry(img_metas, bev_h, bev_w, self.num_points_in_pillar, self.pc_range)

    def forward(
        self,
        bev_query,
        value,
        bev_h,
        bev_w,
        bev_pos,
        spatial_shapes,
        img_metas,
        prev_bev=None,
        shift=None,
    ):
        """Upstream's argument layout: ``bev_query`` and ``bev_pos`` ``(num_query, bs, C)``, ``value``
        ``(num_cams, num_value, bs, C)``, ``prev_bev`` ``(num_query, bs, C)`` already rotated to the
        current frame, ``shift`` ``(bs, 2)`` the ego translation in BEV fractions, and
        ``spatial_shapes`` ``(num_levels, 2)`` as (h, w). Returns ``(bs, num_query, C)``."""
        bs = bev_query.shape[1]
        ref_2d = bev_reference_points(bev_h, bev_w, bs, bev_query.dtype)
        reference_points_cam, bev_mask = self.point_sampling(bev_h, bev_w, bs, img_metas)

        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        _, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            shift_ref_2d = ref_2d + (0.0 if shift is None else shift[:, None, None, :])
            # The previous BEV is paired with the encoder's input query, not each layer's input.
            prev_bev = torch.stack([prev_bev.permute(1, 0, 2), bev_query], 1).reshape(bs * 2, len_bev, -1)
            hybrid_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(bs * 2, len_bev, num_bev_level, 2)
        else:
            hybrid_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(bs * 2, len_bev, num_bev_level, 2)

        bev_shape = torch.tensor([[bev_h, bev_w]])
        output = bev_query
        for layer in self.layers:
            output = layer(
                output,
                value=value,
                bev_pos=bev_pos,
                ref_2d=hybrid_ref_2d,
                bev_shape=bev_shape,
                spatial_shapes=spatial_shapes,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
            )
        return output
