# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Spatial Cross Attention (SCA) module for BEVFormer.

This module implements the spatial cross-attention mechanism that enables
BEV queries to extract spatial features from regions of interest across
multiple camera views using deformable attention.

Based on the original BEVFormer implementation:
https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py
"""

import torch
import torch.nn as nn
from typing import Optional
from .ms_deformable_attention import MSDeformableAttention
from models.experimental.bevformer.config import DeformableAttentionConfig


class SpatialCrossAttention(nn.Module):
    """
    Spatial Cross Attention module for BEVFormer.

    This attention mechanism allows BEV queries to extract spatial features
    from regions of interest across camera views using deformable attention.
    Each BEV query can attend to multiple camera features at different scales
    and locations.

    Note: This module expects pre-projected reference points and validity masks.
    Point sampling/projection from 3D to camera coordinates should be handled
    by the encoder before calling this attention module.

    Args:
        embed_dims (int): The embedding dimension.
        num_cams (int): Number of cameras.
        init_cfg (dict, optional): Initialization config dict.
        batch_first (bool): Whether the first dimension of input is batch_size.
        deformable_attention (dict): Config for MSDeformableAttention3D.
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_cams: int = 6,
        init_cfg: Optional[dict] = None,
        batch_first: bool = False,
        deformable_attention: Optional[dict] = None,
        **kwargs,
    ):
        super(SpatialCrossAttention, self).__init__()

        if deformable_attention is None:
            deformable_attention = dict(
                type="MSDeformableAttention3D", embed_dims=256, num_levels=4, num_points=8, num_heads=8
            )

        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.batch_first = batch_first

        # Initialize MSDeformableAttention using existing implementation
        # Store deformable attention config for flexible initialization
        self.deformable_attention_config = deformable_attention or {}

        # Initialize with default config - num_points will be validated in forward()
        deform_config = DeformableAttentionConfig(
            embed_dims=self.deformable_attention_config.get("embed_dims", embed_dims),
            num_heads=self.deformable_attention_config.get("num_heads", 8),
            num_levels=self.deformable_attention_config.get("num_levels", 4),  # Feature pyramid levels
            num_points=self.deformable_attention_config.get("num_points", 4),  # Will be validated against depth levels
            batch_first=batch_first,
        )
        self.deformable_attention = MSDeformableAttention(deform_config)

        # Output projection layer
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of Spatial Cross Attention.

        Args:
            query (torch.Tensor): BEV queries [B, num_query, embed_dims].
            key (torch.Tensor): Multi-camera features [num_cams, H*W, B, embed_dims].
            value (torch.Tensor): Same as key.
            residual (torch.Tensor): Residual connection input.
            query_pos (torch.Tensor): Query positional encoding.
            key_padding_mask (torch.Tensor): Key padding mask.
            reference_points_cam (torch.Tensor): Camera projected reference points [num_cams, B, num_query, num_points_in_pillar, 2].
            bev_mask (torch.Tensor): Valid mask for camera projections [num_cams, B, num_query, num_points_in_pillar].
            spatial_shapes (torch.Tensor): Spatial shapes of multi-scale features.
            level_start_index (torch.Tensor): Start index of each level.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Output features [B, num_query, embed_dims].
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            inp_residual = query
        else:
            inp_residual = residual

        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape

        num_points_in_pillar = reference_points_cam.size(3)

        # Find valid queries for each camera
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            # Sum over depth levels dimension and check if any point is valid
            index_query_per_img = mask_per_img.sum(-1) > 0  # [B, num_query]
            indexes.append(index_query_per_img)

        max_len = max([index.sum().max().item() for index in indexes])
        if max_len == 0:
            # No valid points, return original query
            return inp_residual

        # Initialize output accumulator
        slots = torch.zeros_like(query)

        # Rebatch queries and reference points for each camera
        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, num_points_in_pillar, 2])

        # Fill rebatched tensors with valid queries
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                valid_indices = torch.nonzero(index_query_per_img[j], as_tuple=False).squeeze(-1)
                if len(valid_indices) > 0:
                    num_valid = min(len(valid_indices), max_len)
                    queries_rebatch[j, i, :num_valid] = query[j, valid_indices[:num_valid]]
                    reference_points_rebatch[j, i, :num_valid] = reference_points_cam[i, j, valid_indices[:num_valid]]

        num_cams, L, bs, embed_dims = key.shape

        # [num_cams, L, bs, embed_dims] -> [bs * num_cams, L, embed_dims]
        key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, L, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, L, self.embed_dims)

        # [bs, num_cams, max_len, embed_dims] -> [bs * num_cams, max_len, embed_dims]
        queries_batched = queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims)

        # [bs, num_cams, max_len, num_points_in_pillar, 2] -> [bs * num_cams, max_len, num_points_in_pillar, 2]
        reference_points_batched = reference_points_rebatch.view(bs * self.num_cams, max_len, num_points_in_pillar, 2)

        # Apply deformable attention with 3D reference points (matching original BEVFormer)
        queries = self.deformable_attention(
            query=queries_batched,
            key=key,
            value=value,
            reference_points=reference_points_batched,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs,
        )

        # Reshape output back to [bs, num_cams, max_len, embed_dims]
        queries = queries.view(bs, self.num_cams, max_len, self.embed_dims)

        # Aggregate features back to original query positions
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                valid_indices = torch.nonzero(index_query_per_img[j], as_tuple=False).squeeze(-1)
                if len(valid_indices) > 0:
                    num_valid = min(len(valid_indices), max_len)
                    slots[j, valid_indices[:num_valid]] += queries[j, i, :num_valid]

        # Count valid queries per camera
        # Original: count = bev_mask.sum(-1) > 0; count = count.permute(1, 2, 0).sum(-1)
        count = bev_mask.sum(-1) > 0  # [num_cams, B, num_query]
        count = count.permute(1, 2, 0).sum(-1)  # [B, num_query]
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]

        # Output projection
        slots = self.output_proj(slots)

        return slots + inp_residual
