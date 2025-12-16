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
    ttnn implementation of multi-scale deformable attention core operation.
    Based on the reference implementation but using ttnn operations.

    Args:
        value (ttnn.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (ttnn.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (ttnn.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),
    """
    bs, _, num_heads, head_dim = value.shape
    num_levels = value_spatial_shapes.shape[0]
    num_queries = sampling_locations.shape[1]
    num_points = sampling_locations.shape[4]

    if use_signpost:
        signpost(header="MS Deformable Attn TTNN Start")

    # Split value into a list of tensors for each level using ttnn.split
    value_list = ttnn.split(value, [H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    # Normalize sampling locations to [-1, 1] using ttnn operations
    sampling_grids = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # Extract grid for this level
        grid = sampling_locations[:, :, :, level, :, :]
        level_dims = ttnn.from_torch(torch.tensor([W_, H_]), device=device, dtype=ttnn.float32)
        ttnn.div_(grid, level_dims)
        ttnn.mul_(grid, 2.0)
        ttnn.sub_(grid, 1.0)

        sampling_grids.append(grid)

    # Initialize output tensor
    output = ttnn.zeros(
        [bs, num_queries, num_heads, head_dim], device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
    )

    # Perform sampling and attention for each level
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l = ttnn.to_layout(value_list[level], layout=ttnn.ROW_MAJOR_LAYOUT)

        # Prepare value tensor for grid sampling
        # [bs, len_l, num_heads, head_dim] -> [bs * num_heads, H_, W_, head_dim]
        value_l = ttnn.permute(value_l, (0, 2, 1, 3))  # [bs, num_heads, head_dim, len_l]
        value_l = ttnn.reshape(value_l, (bs * num_heads, H_, W_, head_dim))

        # Prepare sampling grid
        # [bs, num_queries, num_heads, num_points, 2] -> [bs * num_heads, num_queries * num_points, 1, 2]
        grid = ttnn.permute(sampling_grids[level], (0, 2, 1, 3, 4))  # [bs, num_heads, num_queries, num_points, 2]
        grid = ttnn.reshape(grid, (bs * num_heads, num_queries * num_points, 1, 2))

        # Ensure proper layout and dtype for grid_sample
        grid = ttnn.to_layout(grid, layout=ttnn.ROW_MAJOR_LAYOUT)
        logger.debug(f"Level {level}: grid shape {grid.shape}, value_l shape {value_l.shape}")

        # Convert to NHWC format for grid_sample (ttnn requirement)
        # value_l = ttnn.permute(value_l, (0, 2, 3, 1))  # [bs * num_heads, H_, W_, head_dim]

        # Perform bilinear sampling
        sampled = ttnn.grid_sample(value_l, grid)
        logger.debug(f"Sampled shape: {sampled.shape}")

        # Convert back to NCHW and reshape
        sampled = ttnn.permute(sampled, (0, 3, 1, 2))  # [bs * num_heads, head_dim, num_queries * num_points, 1]
        sampled = ttnn.squeeze(sampled, -1)  # [bs * num_heads, head_dim, num_queries * num_points]
        sampled = ttnn.reshape(sampled, (bs, num_heads, head_dim, num_queries, num_points))
        sampled = ttnn.permute(sampled, (0, 3, 1, 4, 2))  # [bs, num_queries, num_heads, num_points, head_dim]

        # Apply attention weights
        attn = attention_weights[:, :, :, level, :]  # [bs, num_queries, num_heads, num_points]
        attn = ttnn.unsqueeze(attn, -1)  # [bs, num_queries, num_heads, num_points, 1]

        # Weighted sum over sampling points
        logger.debug(f"Attn shape: {attn.shape}, Sampled shape: {sampled.shape}")
        level_output = ttnn.mul(sampled, attn)  # Element-wise multiplication
        logger.debug(f"Level output shape before sum: {level_output.shape}")
        level_output = ttnn.sum(level_output, -2)  # Sum over num_points dimension
        logger.debug(f"Level output shape after sum: {level_output.shape}")

        ttnn.add_(output, level_output)

    # Final reshape to combine heads
    output = ttnn.reshape(output, (bs, num_queries, num_heads * head_dim))
    if use_signpost:
        signpost(header="MS Deformable Attn TTNN End")

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

        # # Create mock params if none provided for testing
        # if self.params is None:
        #     self._create_mock_params()
        # else:
        #     # Convert ParameterDict to object with attribute access
        self.params = self._convert_parameterdict_to_object(self.params)

    # TODO: Move to model_preprocessing script
    def _convert_parameterdict_to_object(self, param_dict):
        """Convert ParameterDict to object with attribute access"""
        params_obj = type("Params", (), {})()

        for layer_name, layer_params in param_dict.items():
            layer_obj = type("Layer", (), {})()
            for param_name, param_tensor in layer_params.items():
                setattr(layer_obj, param_name, param_tensor)
            setattr(params_obj, layer_name, layer_obj)

        return params_obj

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
            reference_points: [bs, num_queries, num_levels, 2] Reference points
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

        # Handle batch_first format
        if not self.batch_first:
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))
            identity = ttnn.permute(identity, (1, 0, 2))

        bs, num_queries, _ = query.shape
        bs, num_keys, _ = value.shape

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

        # Deallocate weights after use (ttnn pattern)
        ttnn.deallocate(self.params.attention_weights.weight)
        ttnn.deallocate(self.params.attention_weights.bias)

        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_queries, self.num_heads, self.num_levels * self.num_points)
        )
        attention_weights = ttnn.softmax(attention_weights, dim=-1)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_queries, self.num_heads, self.num_levels, self.num_points)
        )

        # Handle different reference point formats
        if reference_points.shape[-1] == 2:
            # 2D reference points: [bs, num_queries, num_levels, 2]
            # Convert spatial_shapes to ttnn for operations
            spatial_shapes_tt = ttnn.from_torch(spatial_shapes, device=self.device, dtype=ttnn.bfloat16)

            # Create offset normalizer [num_levels, 2] with [width, height] order
            offset_normalizer = ttnn.stack([spatial_shapes_tt[..., 1], spatial_shapes_tt[..., 0]], dim=-1)

            # Convert reference_points to ttnn if it's not already
            if not isinstance(reference_points, ttnn.Tensor):
                reference_points = ttnn.from_torch(reference_points, device=self.device, dtype=ttnn.bfloat16)

            # Expand dimensions for broadcasting
            reference_points = ttnn.unsqueeze(reference_points, 2)  # Add head dimension
            reference_points = ttnn.unsqueeze(reference_points, 4)  # Add point dimension

            offset_normalizer = ttnn.unsqueeze(offset_normalizer, 0)  # Add batch dimension
            offset_normalizer = ttnn.unsqueeze(offset_normalizer, 1)  # Add query dimension
            offset_normalizer = ttnn.unsqueeze(offset_normalizer, 2)  # Add head dimension
            offset_normalizer = ttnn.unsqueeze(offset_normalizer, 4)  # Add point dimension

            # Expand batch dimension to match sampling_offsets for 6D broadcasting limitation
            # ttnn has rank 5 limit for broadcasting, so dimensions beyond rank 5 must match exactly
            offset_normalizer = ttnn.repeat(offset_normalizer, [bs, 1, 1, 1, 1, 1])

            # Compute sampling locations
            normalized_offsets = ttnn.div(sampling_offsets, offset_normalizer)
            sampling_locations = ttnn.add(reference_points, normalized_offsets)

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
            ttnn.deallocate(self.params.output_proj.weight)
            ttnn.deallocate(self.params.output_proj.bias)

        # Add residual connection
        output = ttnn.add(output, identity)

        # Handle batch_first format for output
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        # Convert output back to torch tensor for compatibility
        output = ttnn.to_torch(output)

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
