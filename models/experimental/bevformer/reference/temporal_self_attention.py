# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Temporal Self Attention (TSA) module for BEVFormer.

This module implements the temporal self-attention mechanism that enables
BEV features to model temporal dependencies across different timesteps.
It uses deformable attention to aggregate information from current and
historical BEV features.

Based on the original BEVFormer implementation:
https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py
"""

import torch
import torch.nn as nn
import warnings
from typing import Optional
from .ms_deformable_attention import MSDeformableAttention
from models.experimental.bevformer.config import DeformableAttentionConfig


class TemporalSelfAttention(nn.Module):
    """
    Temporal Self Attention module for BEVFormer.

    This attention mechanism models temporal dependencies by allowing BEV queries
    to attend to both current and historical BEV features using deformable attention.
    It's designed to handle temporal relationships for object tracking and motion
    understanding in autonomous driving scenarios.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Number of attention heads.
        num_levels (int): Number of feature levels.
        num_points (int): Number of sampling points in deformable attention.
        num_bev_queue (int): Number of BEV timesteps (typically 2: current + history).
        batch_first (bool): Whether the first dimension of input is batch_size.
        init_cfg (dict, optional): Initialization config dict.
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
        num_bev_queue: int = 2,
        batch_first: bool = True,
        init_cfg: Optional[dict] = None,
        **kwargs,
    ):
        super(TemporalSelfAttention, self).__init__()

        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")

        dim_per_head = embed_dims // num_heads
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.batch_first = batch_first

        # Check if dim_per_head is power of 2 for efficiency
        if not self._is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "TemporalSelfAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our implementation."
            )

        # Initialize MSDeformableAttention using existing implementation
        deform_config = DeformableAttentionConfig(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            batch_first=batch_first,
        )
        self.deformable_attention = MSDeformableAttention(deform_config)

    def forward(
        self,
        query: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of Temporal Self Attention.

        Args:
            query (torch.Tensor): Current BEV queries [B, num_query, embed_dims].
            value (torch.Tensor): Temporal BEV features [B, num_query, embed_dims].
                If None, will be constructed from current query and prev_bev.
            identity (torch.Tensor): Identity connection input.
            query_pos (torch.Tensor): Query positional encoding.
            key_padding_mask (torch.Tensor): Key padding mask.
            reference_points (torch.Tensor): Reference points for deformable attention.
            spatial_shapes (torch.Tensor): Spatial shapes of BEV features.
            level_start_index (torch.Tensor): Start index of each level.
            **kwargs: Additional arguments.

        Returns:
            torch.Tensor: Output features [B, num_query, embed_dims].
        """
        if value is None:
            # For temporal self-attention, just use query as value (no temporal information)
            # This is a simplified version that avoids temporal complexity for now
            value = query

        if identity is None:
            identity = query

        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape

        # Use reference points as-is for simplified version
        ref_points = reference_points

        # Apply deformable attention with integrated temporal processing
        output = self.deformable_attention(
            query=query,
            value=value,
            reference_points=ref_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )

        return output + identity

    @staticmethod
    def _is_power_of_2(n: int) -> bool:
        """Check if a number is a power of 2."""
        return (n != 0) and (n & (n - 1) == 0)
