# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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
    """
    bs, _, num_heads, head_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape

    if use_signpost:
        signpost(header="MSDA Start")

    # Split value into a list of tensors for each level
    value_list = ttnn.split(value, [H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    # Convert sampling_locations to normalized grid coordinates [-1, 1]
    sampling_grids = ttnn.mul(sampling_locations, 2.0)
    sampling_grids = ttnn.sub(sampling_grids, 1.0)

    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, head_dim -> bs*num_heads, H_, W_, head_dim
        value_l_ = ttnn.to_layout(value_list[level], layout=ttnn.ROW_MAJOR_LAYOUT)
        value_l_ = ttnn.permute(value_l_, (0, 2, 1, 3))
        value_l_ = ttnn.reshape(value_l_, (bs * num_heads, H_, W_, head_dim))

        # bs, num_queries, num_heads, num_points, 2 -> bs*num_heads, num_queries*num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level, :, :]  # [bs, num_queries, num_heads, num_points, 2]
        sampling_grid_l_ = ttnn.to_layout(sampling_grid_l_, layout=ttnn.ROW_MAJOR_LAYOUT)
        sampling_grid_l_ = ttnn.permute(
            sampling_grid_l_, (0, 2, 1, 3, 4)
        )  # [bs, num_heads, num_queries, num_points, 2]
        sampling_grid_l_ = ttnn.reshape(sampling_grid_l_, (bs * num_heads, num_queries * num_points, 1, 2))

        logger.debug(f"Level {level}: grid shape {sampling_grid_l_.shape}, value_l shape {value_l_.shape}")

        # Perform bilinear sampling
        # Input: (bs*num_heads, H_, W_, head_dim), Grid: (bs*num_heads, num_queries*num_points, 1, 2)
        # Output: (bs*num_heads, num_queries*num_points, 1, head_dim)
        sampling_value_l_ = ttnn.grid_sample(value_l_, sampling_grid_l_)
        logger.debug(f"Raw sampled shape: {sampling_value_l_.shape}")

        # (bs*num_heads, num_queries*num_points, 1, head_dim) -> (bs*num_heads, head_dim, num_queries, num_points)
        sampling_value_l_ = ttnn.squeeze(
            sampling_value_l_, 2
        )  # Remove the 1 dimension: (bs*num_heads, num_queries*num_points, head_dim)
        sampling_value_l_ = ttnn.reshape(sampling_value_l_, (bs * num_heads, num_queries, num_points, head_dim))
        sampling_value_l_ = ttnn.permute(
            sampling_value_l_, (0, 3, 1, 2)
        )  # (bs*num_heads, head_dim, num_queries, num_points)
        logger.debug(f"Reshaped sampled shape: {sampling_value_l_.shape}")

        sampling_value_list.append(sampling_value_l_)

    # (bs, num_queries, num_heads, num_levels, num_points) -> (bs*num_heads, 1, num_queries, num_levels * num_points)
    attention_weights = ttnn.permute(attention_weights, (0, 2, 1, 3, 4))  # transpose(1, 2)
    attention_weights = ttnn.reshape(attention_weights, (bs * num_heads, 1, num_queries, num_levels * num_points))

    stacked_values = ttnn.stack(
        sampling_value_list, dim=-2
    )  # (bs*num_heads, head_dim, num_queries, num_levels, num_points)
    stacked_values = ttnn.reshape(
        stacked_values, (bs * num_heads, head_dim, num_queries, num_levels * num_points)
    )  # flatten(-2)

    # Apply attention weights: (bs*num_heads, head_dim, num_queries, num_levels*num_points) * (bs*num_heads, 1, num_queries, num_levels*num_points)
    output = ttnn.mul(stacked_values, attention_weights)
    output = ttnn.sum(output, dim=-1)  # sum over num_levels*num_points: (bs*num_heads, head_dim, num_queries)

    # Reshape and transpose
    output = ttnn.reshape(output, (bs, num_heads * head_dim, num_queries))
    output = ttnn.permute(output, (0, 2, 1))  # [bs, num_queries, num_heads * head_dim]

    if use_signpost:
        signpost(header="MSDA End")

    return output


class TTMSDeformableAttention:
    """
    ttnn implementation of Multi-Scale Deformable Attention.
    Based on the MMCV/BEVFormer approach but using ttnn operations.
    """

    def __init__(self, config: DeformableAttentionConfig, device, params=None):
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
        query,  # ttnn tensor or torch.Tensor
        key=None,  # Optional ttnn tensor
        value=None,  # Optional ttnn tensor
        identity=None,  # Optional ttnn tensor
        query_pos=None,  # Optional ttnn tensor
        key_padding_mask=None,  # Optional ttnn tensor
        reference_points=None,  # Optional ttnn tensor
        spatial_shapes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass using ttnn operations.

        Args:
            query: [bs, num_queries, embed_dims] Query features
            key: [bs, num_keys, embed_dims] Key features (optional, defaults to query)
            value: [bs, num_keys, embed_dims] Value features (optional, defaults to key)
            identity: [bs, num_queries, embed_dims] Identity for residual connection
            query_pos: [bs, num_queries, embed_dims] Query positional encoding
            key_padding_mask: [bs, num_keys] Padding mask for keys
            reference_points: [bs, num_queries, num_points_in_pillar, 2] Reference points
            spatial_shapes: [num_levels, 2] Spatial shapes (H, W) for each level

        Returns:
            output: [bs, num_queries, embed_dims]
        """
        # Convert torch tensors to ttnn tensors first
        if isinstance(query, torch.Tensor):
            query = ttnn.from_torch(query, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if key is not None and isinstance(key, torch.Tensor):
            key = ttnn.from_torch(key, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if value is not None and isinstance(value, torch.Tensor):
            value = ttnn.from_torch(value, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if identity is not None and isinstance(identity, torch.Tensor):
            identity = ttnn.from_torch(identity, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if reference_points is not None and isinstance(reference_points, torch.Tensor):
            reference_points = ttnn.from_torch(
                reference_points, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

        # Handle input defaults
        if value is None:
            value = query if key is None else key
        if key is None:
            key = query
        if identity is None:
            identity = query

        # Add query positional encoding
        if query_pos is not None:
            if isinstance(query_pos, torch.Tensor):
                query_pos = ttnn.from_torch(query_pos, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            query = ttnn.add(query, query_pos)

        if use_signpost:
            signpost(header="TT MS Deformable Attn Module Start")

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

        # Project value and reshape to multi-head format
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, self.params.value_proj.weight, bias=self.params.value_proj.bias)

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            mask = ttnn.unsqueeze(key_padding_mask, -1)  # [bs, num_keys, 1]
            zeros_like_value = ttnn.zeros_like(value)
            value = ttnn.where(mask, zeros_like_value, value)

        value = ttnn.reshape(value, (bs, num_keys, self.num_heads, self.head_dim))

        # Generate sampling offsets
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        sampling_offsets = ttnn.linear(
            query, self.params.sampling_offsets.weight, bias=self.params.sampling_offsets.bias
        )
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_queries, self.num_heads, self.num_levels, self.num_points, 2)
        )

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

        # Handle different reference point formats
        if reference_points.shape[-1] == 2:
            D = reference_points.shape[2]

            # Convert spatial_shapes to ttnn for operations
            spatial_shapes_tt = ttnn.from_torch(spatial_shapes, device=self.device, dtype=ttnn.bfloat16)

            # Create offset normalizer [num_levels, 2] with [width, height] order (following torch reference)
            offset_normalizer = ttnn.stack([spatial_shapes_tt[..., 1], spatial_shapes_tt[..., 0]], dim=-1)

            # Convert reference_points to ttnn if it's not already
            if not isinstance(reference_points, ttnn.Tensor):
                reference_points = ttnn.from_torch(reference_points, device=self.device, dtype=ttnn.bfloat16)

            # Expand offset_normalizer for broadcasting [None, None, None, :, None, :]
            offset_normalizer = ttnn.unsqueeze(offset_normalizer, 0)  # Add batch dimension
            offset_normalizer = ttnn.unsqueeze(offset_normalizer, 0)  # Add query dimension
            offset_normalizer = ttnn.unsqueeze(offset_normalizer, 0)  # Add head dimension
            offset_normalizer = ttnn.unsqueeze(offset_normalizer, -2)  # Add point dimension

            # reference_points[:, :, None, None, None, :, :] - Add head, level and point dimensions
            reference_points_expanded = ttnn.unsqueeze(reference_points, 2)  # Add head dimension
            reference_points_expanded = ttnn.unsqueeze(reference_points_expanded, 3)  # Add level dimension
            reference_points_expanded = ttnn.unsqueeze(reference_points_expanded, 4)  # Add point dimension

            sampling_offsets = ttnn.div(sampling_offsets, offset_normalizer)

            sampling_offsets = ttnn.reshape(
                sampling_offsets, (bs, num_queries, self.num_heads, self.num_levels, self.num_points // D, D, 2)
            )
            sampling_locations = ttnn.add(reference_points_expanded, sampling_offsets)
            sampling_locations = ttnn.reshape(
                sampling_locations, (bs, num_queries, self.num_heads, self.num_levels, self.num_points, 2)
            )

        elif reference_points.shape[-1] == 4:
            raise NotImplementedError("4D reference points not implemented")
        else:
            raise ValueError(f"Reference points must have 2 or 4 dimensions, got {reference_points.shape[-1]}")

        # Apply multi-scale deformable attention
        output = multi_scale_deformable_attn_ttnn(
            value=value,
            value_spatial_shapes=spatial_shapes,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
            device=self.device,
        )

        # Apply output projection
        if hasattr(self.params, "output_proj"):
            output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
            output = ttnn.linear(output, self.params.output_proj.weight, bias=self.params.output_proj.bias)

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
