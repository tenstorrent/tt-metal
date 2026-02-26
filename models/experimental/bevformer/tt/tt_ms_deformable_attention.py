# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Multi-Scale Deformable Attention implementation for BEVFormer.

This module implements the multi-scale deformable attention mechanism using TTNN operations
to enable efficient feature sampling across multiple feature pyramid levels. The implementation
is optimized for TTNN execution and provides the core attention mechanism used in spatial
cross-attention and temporal self-attention modules.

Key components:
- multi_scale_deformable_attn_ttnn: Core attention computation function
- TTMSDeformableAttention: Main attention class with parameter management
"""

from typing import Optional
import ttnn

import torch
from ..config import DeformableAttentionConfig

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

from loguru import logger

# Enable/disable logging output
ENABLE_LOGGING = False


def multi_scale_deformable_attn_ttnn(
    value,
    value_spatial_shapes,
    sampling_locations,
    attention_weights,
    device,
):
    """
    ttnn implementation of multi-scale deformable attention core logic.

    Args:
        value (ttnn.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (ttnn.Tensor): The location of sampling points,
            has shape
            (bs, num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (ttnn.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs, num_queries, num_heads, num_levels, num_points),
        device: TTNN device

    Returns:
        ttnn.Tensor: Attended features with shape (bs, num_queries, embed_dims)
    """
    bs, _, num_heads, head_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape

    if ENABLE_LOGGING:
        logger.info("MSDA Start")

    # Split value into a list of tensors for each level
    value_list = ttnn.split(value, [H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    # Normalize sampling locations from [0,1] to [-1,1] for grid_sample
    sampling_grids = ttnn.mul(sampling_locations, 2.0)
    sampling_grids = ttnn.sub(sampling_grids, 1.0)

    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # [bs, H_*W_, num_heads, head_dim] -> [bs*num_heads, H_, W_, head_dim]
        value_l_ = ttnn.to_layout(value_list[level], layout=ttnn.ROW_MAJOR_LAYOUT)
        value_l_ = ttnn.permute(value_l_, (0, 2, 1, 3))  # Move heads to dimension 1
        value_l_ = ttnn.reshape(value_l_, (bs * num_heads, H_, W_, head_dim))

        sampling_grid_l_ = sampling_grids[:, :, :, level, :, :]  # [bs, num_queries, num_heads, num_points, 2]
        sampling_grid_l_ = ttnn.to_layout(sampling_grid_l_, layout=ttnn.ROW_MAJOR_LAYOUT)
        sampling_grid_l_ = ttnn.permute(
            sampling_grid_l_, (0, 2, 1, 3, 4)
        )  # [bs, num_heads, num_queries, num_points, 2]
        sampling_grid_l_ = ttnn.reshape(
            sampling_grid_l_, (bs * num_heads, num_queries * num_points, 1, 2)
        )  # [N, H_out, W_out, 2] = [bs*num_heads, num_queries*num_points, 1, 2]

        # Input: (bs*num_heads, H_, W_, head_dim), Grid: (bs*num_heads, num_queries*num_points, 1, 2)
        # Output: (bs*num_heads, num_queries*num_points, 1, head_dim)
        sampling_value_l_ = ttnn.grid_sample(value_l_, sampling_grid_l_)

        # (bs*num_heads, num_queries*num_points, 1, head_dim) -> (bs*num_heads, head_dim, num_queries, num_points)
        sampling_value_l_ = ttnn.squeeze(
            sampling_value_l_, 2
        )  # Remove the 1 dimension: (bs*num_heads, num_queries*num_points, head_dim)
        sampling_value_l_ = ttnn.reshape(sampling_value_l_, (bs * num_heads, num_queries, num_points, head_dim))
        sampling_value_l_ = ttnn.permute(
            sampling_value_l_, (0, 3, 1, 2)
        )  # (bs*num_heads, head_dim, num_queries, num_points)

        sampling_value_list.append(sampling_value_l_)

    # [bs, num_queries, num_heads, num_levels, num_points] -> [bs*num_heads, 1, num_queries, num_levels*num_points]
    attention_weights = ttnn.permute(attention_weights, (0, 2, 1, 3, 4))  # Move heads to dim 1
    attention_weights = ttnn.reshape(attention_weights, (bs * num_heads, 1, num_queries, num_levels * num_points))

    # Stack sampled values from all pyramid levels
    stacked_values = ttnn.stack(
        sampling_value_list, dim=-2
    )  # (bs*num_heads, head_dim, num_queries, num_levels, num_points)
    # Flatten level and point dimensions
    stacked_values = ttnn.reshape(stacked_values, (bs * num_heads, head_dim, num_queries, num_levels * num_points))

    output = ttnn.mul(stacked_values, attention_weights)
    # Aggregate across all sampling points and levels
    output = ttnn.sum(output, dim=-1)  # Final shape: (bs*num_heads, head_dim, num_queries)

    output = ttnn.reshape(output, (bs, num_heads * head_dim, num_queries))
    output = ttnn.permute(output, (0, 2, 1))  # [bs, num_queries, num_heads * head_dim]

    if ENABLE_LOGGING:
        logger.info("MSDA End")

    return output


class TTMSDeformableAttention:
    """
    ttnn implementation of Multi-Scale Deformable Attention.
    Based on the MMCV/BEVFormer approach.
    """

    def __init__(self, config: DeformableAttentionConfig, device, params=None):
        """
        Initialize TTNN Multi-Scale Deformable Attention module.

        Args:
            config (DeformableAttentionConfig): Configuration object containing:
                - embed_dims (int): Feature embedding dimensions
                - num_heads (int): Number of attention heads
                - num_levels (int): Number of feature pyramid levels
                - num_points (int): Number of sampling points per head
                - batch_first (bool): Whether batch dimension comes first
            device: TTNN device for tensor operations
            params: Pre-computed TTNN parameters containing linear layer weights and biases.
                Should include: value_proj, sampling_offsets, attention_weights, output_proj

        Raises:
            ValueError: If embed_dims is not divisible by num_heads
        """
        # Validate configuration
        if config.embed_dims % config.num_heads != 0:
            raise ValueError(f"embed_dims ({config.embed_dims}) must be divisible by num_heads ({config.num_heads})")

        # Set attributes
        self.embed_dims = config.embed_dims
        self.num_heads = config.num_heads
        self.num_levels = config.num_levels
        self.num_points = config.num_points
        self.batch_first = config.batch_first
        self.device = device
        self.params = params

        self.head_dim = self.embed_dims // self.num_heads

    def forward(
        self,
        query,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass using ttnn operations.

        Args:
            query: [bs, num_queries, embed_dims] Query features
            value: [bs, num_keys, embed_dims] Value features (optional, defaults to query)
            identity: [bs, num_queries, embed_dims] Identity for residual connection
            query_pos: [bs, num_queries, embed_dims] Query positional encoding
            key_padding_mask: [bs, num_keys] Padding mask for keys
            reference_points: [bs, num_queries, num_points_in_pillar, 2] Reference points
            spatial_shapes: [num_levels, 2] Spatial shapes (H, W) for each level

        Returns:
            output: [bs, num_queries, embed_dims]
        """

        # Handle input defaults
        if value is None:
            value = query
        if identity is None:
            identity = query

        # Add query positional encoding
        if query_pos is not None:
            query = ttnn.add(query, query_pos)

        if use_signpost:
            signpost(
                header=f"TT MS Deformable Attn Module Start, {query.shape[1]} - {spatial_shapes.prod(dim=1).sum()}"
            )

        # Handle batch_first format
        if not self.batch_first:
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))
            identity = ttnn.permute(identity, (1, 0, 2))

        bs, num_queries, _ = query.shape
        bs, num_keys, _ = value.shape
        bs, num_queries, D, _ = reference_points.shape

        # Validate required inputs
        assert spatial_shapes is not None, "spatial_shapes is required"
        assert reference_points is not None, "reference_points is required"

        # Verify spatial shapes consistency
        total_keys = spatial_shapes.prod(dim=1).sum()
        assert total_keys == num_keys, f"Inconsistent keys: {total_keys} != {num_keys}"

        if ENABLE_LOGGING:
            logger.info("MSDA Value Projection Start")

        # Project value and reshape to multi-head format
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, self.params.value_proj.weight, bias=self.params.value_proj.bias)

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            mask = ttnn.unsqueeze(key_padding_mask, -1)  # [bs, num_keys, 1]
            zeros_like_value = ttnn.zeros_like(value)
            value = ttnn.where(mask, zeros_like_value, value)

        value = ttnn.reshape(value, (bs, num_keys, self.num_heads, self.head_dim))

        if ENABLE_LOGGING:
            logger.info("MSDA Sampling Offset Generation")

        # Generate sampling offsets
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        sampling_offsets = ttnn.linear(
            query, self.params.sampling_offsets.weight, bias=self.params.sampling_offsets.bias
        )
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs * num_queries * self.num_heads, self.num_levels, self.num_points, 2)
        )

        if ENABLE_LOGGING:
            logger.info("MSDA Attention Weight Generation")

        # Generate attention weights
        attention_weights = ttnn.linear(
            query, self.params.attention_weights.weight, bias=self.params.attention_weights.bias
        )

        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_queries, self.num_heads, self.num_levels * self.num_points)
        )
        attention_weights = ttnn.softmax(attention_weights, dim=-1)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_queries, self.num_heads, self.num_levels, self.num_points)
        )

        if ENABLE_LOGGING:
            logger.info("MSDA Sampling Location Calculation")

        # Handle different reference point formats
        if reference_points.shape[-1] == 2:
            # D represents the number of depth levels in 3D point sampling (e.g., 4 points per pillar)
            D = reference_points.shape[2]

            spatial_shapes_tt = ttnn.from_torch(
                spatial_shapes, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

            # Create offset normalizer to convert pixel-space offsets to normalized coordinates [0,1]
            offset_normalizer = ttnn.stack([spatial_shapes_tt[..., 1], spatial_shapes_tt[..., 0]], dim=-1)

            # sampling_offsets: [bs*num_queries*num_heads, num_levels, num_points, 2]
            # offset_normalizer: [num_levels, 2] -> [1, num_levels, 1, 2] for broadcasting
            offset_normalizer = ttnn.unsqueeze(offset_normalizer, 0)  # Add batch * query * head dimension
            offset_normalizer = ttnn.unsqueeze(offset_normalizer, -2)  # Add point dimension

            sampling_offsets = ttnn.div(sampling_offsets, offset_normalizer)

            # reference_points: [bs, num_queries, D, 2] -> [bs, num_queries, 1, 1, 1, D, 2]
            reference_points_expanded = ttnn.unsqueeze(reference_points, 2)  # Add head dimension
            reference_points_expanded = ttnn.unsqueeze(reference_points_expanded, 3)  # Add level dimension
            reference_points_expanded = ttnn.unsqueeze(reference_points_expanded, 4)  # Add point dimension

            # Reshape sampling_offsets to separate depth dimension for proper addition with reference_points
            # From [bs*num_queries*num_heads, num_levels, num_points, 2]
            sampling_offsets = ttnn.reshape(
                sampling_offsets, (bs, num_queries, self.num_heads, self.num_levels, self.num_points // D, D, 2)
            )
            # Compute final sampling locations
            sampling_locations = ttnn.add(reference_points_expanded, sampling_offsets)
            # Flatten back to standard format for multi-scale deformable attention
            sampling_locations = ttnn.reshape(
                sampling_locations, (bs, num_queries, self.num_heads, self.num_levels, self.num_points, 2)
            )
        else:
            raise ValueError(f"Reference points must have 2 dimensions, got {reference_points.shape[-1]}")

        # Apply multi-scale deformable attention
        output = multi_scale_deformable_attn_ttnn(
            value=value,
            value_spatial_shapes=spatial_shapes,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
            device=self.device,
        )

        if ENABLE_LOGGING:
            logger.info("MSDA Core Attention Complete")

        # Apply output projection
        if hasattr(self.params, "output_proj"):
            output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
            output = ttnn.linear(output, self.params.output_proj.weight, bias=self.params.output_proj.bias)

        if ENABLE_LOGGING:
            logger.info("MSDA Adding Residual")

        # Add residual connection
        output = ttnn.add(output, identity)

        # Handle batch_first format for output
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        if use_signpost:
            signpost(header="TT MS Deformable Attn Module End")

        return output

    def __call__(self, *args, **kwargs):
        """Make the class callable"""
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        """String representation for debugging"""
        return (
            f"embed_dims={self.embed_dims}, num_heads={self.num_heads}, "
            f"num_levels={self.num_levels}, num_points={self.num_points}, "
            f"batch_first={self.batch_first}"
        )
