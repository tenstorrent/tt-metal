# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


from ..config import DeformableAttentionConfig


def multi_scale_deformable_attn(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Reference version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, head_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, head_dim -> bs*num_heads, head_dim, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, head_dim, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 -> bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)

    # (bs, num_queries, num_heads, num_levels, num_points) -> (bs, num_heads, num_queries, num_levels, num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * head_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class MSDeformableAttention(nn.Module):
    """
    PyTorch implementation of Multi-Scale Deformable Attention.
    """

    def __init__(self, config: DeformableAttentionConfig, device=None):
        # Validate configuration
        if config.embed_dims % config.num_heads != 0:
            raise ValueError(f"embed_dims ({config.embed_dims}) must be divisible by num_heads ({config.num_heads})")

        super(MSDeformableAttention, self).__init__()

        # Set attributes before calling parent __init__
        self.embed_dims = config.embed_dims
        self.num_heads = config.num_heads
        self.num_levels = config.num_levels
        self.num_points = config.num_points
        self.im2col_step = config.im2col_step
        self.batch_first = config.batch_first

        self.head_dim = self.embed_dims // self.num_heads
        self.config = config
        self.device = device

        self._setup_layers()

    def _setup_layers(self):
        """Setup the linear projection layers following MMCV approach"""
        # Value projection
        self.value_proj = nn.Linear(self.embed_dims, self.embed_dims)

        # Sampling offsets projection
        self.sampling_offsets = nn.Linear(self.embed_dims, self.num_heads * self.num_levels * self.num_points * 2)

        # Attention weights projection
        self.attention_weights = nn.Linear(self.embed_dims, self.num_heads * self.num_levels * self.num_points)

        # Output projection
        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass following MMCV's approach.

        Args:
            query: [bs, num_queries, embed_dims] Query features
            key: [bs, num_keys, embed_dims] Key features (optional, defaults to query)
            value: [bs, num_keys, embed_dims] Value features (optional, defaults to key)
            identity: [bs, num_queries, embed_dims] Identity for residual connection
            query_pos: [bs, num_queries, embed_dims] Query positional encoding
            key_padding_mask: [bs, num_keys] Padding mask for keys
            reference_points: [bs, num_queries, num_levels, 2] Reference points
            spatial_shapes: [num_levels, 2] Spatial shapes (H, W) for each level

        Returns:
            output: [bs, num_queries, embed_dims]
        """
        # Handle input defaults
        if value is None:
            value = query if key is None else key
        if key is None:
            key = query
        if identity is None:
            identity = query

        # Add query positional encoding
        if query_pos is not None:
            query = query + query_pos

        # Handle batch_first format
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
            identity = identity.permute(1, 0, 2)

        bs, num_queries, _ = query.shape
        bs, num_keys, _ = value.shape

        # Validate required inputs
        assert spatial_shapes is not None, "spatial_shapes is required"
        assert reference_points is not None, "reference_points is required"

        # Verify spatial shapes consistency
        total_keys = spatial_shapes.prod(dim=1).sum()
        assert total_keys == num_keys, f"Inconsistent keys: {total_keys} != {num_keys}"

        # Project value and reshape to multi-head format
        value = self.value_proj(value)

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = value.masked_fill(mask, 0.0)

        value = value.view(bs, num_keys, self.num_heads, self.head_dim)

        # Generate sampling offsets
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(bs, num_queries, self.num_heads, self.num_levels, self.num_points, 2)

        # Generate attention weights
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(bs, num_queries, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(bs, num_queries, self.num_heads, self.num_levels, self.num_points)

        # Handle different reference point formats
        if reference_points.shape[-1] == 2:
            # 2D reference points: [bs, num_queries, num_levels, 2]
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)
            offset_normalizer = offset_normalizer[None, None, None, :, None, :]  # Broadcasting shape

            # Add reference points to sampling offsets
            reference_points = reference_points[:, :, None, :, None, :]  # Add head and point dimensions
            sampling_locations = reference_points + sampling_offsets / offset_normalizer

        elif reference_points.shape[-1] == 4:
            # 4D reference points format - not commonly used
            raise NotImplementedError("4D reference points not implemented")
        else:
            raise ValueError(f"Reference points must have 2 or 4 dimensions, got {reference_points.shape[-1]}")

        # Apply multi-scale deformable attention
        output = multi_scale_deformable_attn(
            value=value,
            value_spatial_shapes=spatial_shapes,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
        )

        # Apply output projection
        output = self.output_proj(output)

        # Add residual connection
        output = output + identity

        # Handle batch_first format for output
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output

    def extra_repr(self) -> str:
        """String representation for debugging"""
        return (
            f"embed_dims={self.embed_dims}, num_heads={self.num_heads}, "
            f"num_levels={self.num_levels}, num_points={self.num_points}, "
            f"im2col_step={self.im2col_step}, batch_first={self.batch_first}"
        )
