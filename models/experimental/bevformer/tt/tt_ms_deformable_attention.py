# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
    sampling_grids,
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
        sampling_grids (ttnn.Tensor): The location of sampling points, already
            rescaled to grid_sample's [-1, 1], head-major and in ROW_MAJOR, has
            shape (bs, num_heads, num_queries, num_levels * num_points * 2),
            innermost (point, (x, y)) within a level.
        attention_weights (ttnn.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs, num_queries, num_heads, num_levels, num_points),
        device: TTNN device

    Returns:
        ttnn.Tensor: Attended features with shape (bs, num_queries, embed_dims)
    """
    bs, _, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points = attention_weights.shape
    head_dim = embed_dims // num_heads

    if ENABLE_LOGGING:
        logger.info("MSDA Start")

    # Split value into a list of tensors for each level
    value_list = ttnn.split(value, [H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    # The fused op takes all three inputs ROW_MAJOR, so the head-major permute runs there rather
    # than on a tiled tensor.
    attention_weights = ttnn.to_layout(attention_weights, ttnn.ROW_MAJOR_LAYOUT)
    attention_weights = ttnn.permute(
        attention_weights, (0, 2, 1, 3, 4)
    )  # [bs, num_heads, num_queries, num_levels, num_points]

    # ``ttnn.experimental.multi_scale_deformable_attn`` implements the num_levels == 1 case, so the
    # levels are summed here. The split is exact rather than an approximation: attention_weights is
    # already softmaxed over num_levels * num_points, so the reduction over (level, point) is the
    # plain sum of the per-level reductions.
    output = None
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # [bs, H_*W_, embed_dims] -> [bs, H_, W_, embed_dims]. Heads stay packed in the last
        # dimension and the fused op addresses one by byte offset, so the head-major copy this
        # used to make — 92.6 MB across the levels, at the 64-byte page it produced — is gone.
        # Splitting H_*W_ is a leading-dimension split, so the page is untouched and it is a view.
        value_l_ = ttnn.reshape(value_list[level], (bs, H_, W_, embed_dims))

        # Already head-major, so the level slice is the fused op's grid up to a reshape. The point
        # axis stays folded into the last dimension: a ROW_MAJOR page is the last dimension, so
        # spelling it out would rewrite the grid at a 4-byte page. Folded, the reshape only merges
        # leading dimensions and the page is untouched.
        sampling_grid_l_ = ttnn.reshape(
            sampling_grids[:, :, :, level * num_points * 2 : (level + 1) * num_points * 2],
            (bs * num_heads, num_queries, 1, num_points * 2),
        )  # [N, Q, 1, P*2] = [bs*num_heads, num_queries, 1, num_points*2]

        attn_l_ = ttnn.reshape(attention_weights[:, :, :, level, :], (bs * num_heads, num_queries, num_points))

        # value (bs, H_, W_, embed_dims), grid (N, num_queries, 1, num_points*2), attn
        # (N, num_queries, num_points) -> (N, num_queries, head_dim), with N = bs * num_heads.
        output_l_ = ttnn.experimental.multi_scale_deformable_attn(
            value_l_, sampling_grid_l_, attn_l_, num_heads=num_heads
        )
        output = output_l_ if output is None else ttnn.add(output, output_l_)

    # [bs*num_heads, num_queries, head_dim] -> [bs, num_queries, num_heads*head_dim]. Channels stay
    # head-major, matching the split value was reshaped with.
    output = ttnn.reshape(output, (bs, num_heads, num_queries, head_dim))
    output = ttnn.permute(output, (0, 2, 1, 3))
    output = ttnn.reshape(output, (bs, num_queries, num_heads * head_dim))

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

        # Built on first use and kept: it is derived from spatial_shapes and the head/level/point
        # counts, none of which vary between calls, and rebuilding it uploads from host every time.
        self._offset_normalizer = None

    def _flat_offset_normalizer(self, spatial_shapes):
        """One row scaling the flat offset vector, laid out to match the linear's output order.

        num_heads outermost, then num_levels, num_points, and (x, y) innermost. The axis swap that
        pairs level extents with (x, y) is the reference implementation's, kept as-is.
        """
        if self._offset_normalizer is None:
            level_scale = torch.stack([spatial_shapes[:, 1], spatial_shapes[:, 0]], dim=-1)
            row = (
                level_scale.reshape(self.num_levels, 1, 2)
                .expand(self.num_levels, self.num_points, 2)
                .reshape(1, -1)
                .repeat(1, self.num_heads)
                .to(torch.float32)
            )
            self._offset_normalizer = ttnn.from_torch(
                row, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
        return self._offset_normalizer

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

        # Untilized here because the core attention needs every level ROW_MAJOR anyway. The head
        # split that used to follow is gone: the fused op reads a head out of the embed_dims-wide
        # stick by byte offset, so value keeps its 512-byte page all the way down.
        value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)

        if ENABLE_LOGGING:
            logger.info("MSDA Sampling Offset Generation")

        # Generate sampling offsets
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        sampling_offsets = ttnn.linear(
            query, self.params.sampling_offsets.weight, bias=self.params.sampling_offsets.bias
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

            # The offsets, the reference points and the [0,1] -> [-1,1] rescale are all elementwise,
            # and none of them needs the (num_levels, num_points, 2) axes spelled out. Kept flat as
            # [bs*num_queries, num_heads*num_levels*num_points*2] they stay tile-clean; split out,
            # the trailing (num_points, 2) pads to a full 32x32 tile and every op in this chain
            # carries 128x its own data.
            fan = self.num_heads * self.num_levels * (self.num_points // D)
            width = self.num_heads * self.num_levels * self.num_points * 2

            sampling_offsets = ttnn.div(
                ttnn.reshape(sampling_offsets, (bs * num_queries, width)),
                self._flat_offset_normalizer(spatial_shapes),
            )

            # Each query's (D, 2) reference block applies to every (head, level, point//D) triple,
            # so the flat row is that block repeated ``fan`` times. Built in ROW_MAJOR because the
            # D*2 source is narrower than a tile.
            reference_rows = ttnn.to_layout(reference_points, ttnn.ROW_MAJOR_LAYOUT)
            reference_rows = ttnn.reshape(reference_rows, (bs * num_queries, D * 2))
            reference_rows = ttnn.repeat(reference_rows, ttnn.Shape((1, fan)))
            reference_rows = ttnn.to_layout(reference_rows, ttnn.TILE_LAYOUT)

            # Sampling locations, rescaled to grid_sample's [-1, 1] in the same flat pass.
            sampling_grids = ttnn.add(reference_rows, sampling_offsets)
            sampling_grids = ttnn.sub(ttnn.mul(sampling_grids, 2.0), 1.0)

            # Only the head axis is spelled out, and only because the permute needs it; the level,
            # point and (x, y) axes stay folded into the last dimension all the way into the core,
            # so no op here ever writes a page narrower than num_points * 2.
            #
            # Head-major, because the core attention slices a level per fused call and each slice
            # would otherwise need its own permute. One permute over the whole tensor beats
            # num_levels permutes over slices of it: at these sizes the per-call overhead dominates
            # the bytes moved, so the small ones run two orders of magnitude below bandwidth.
            sampling_grids = ttnn.to_layout(sampling_grids, ttnn.ROW_MAJOR_LAYOUT)
            sampling_grids = ttnn.reshape(
                sampling_grids,
                (bs, num_queries, self.num_heads, self.num_levels * self.num_points * 2),
            )
            sampling_grids = ttnn.permute(sampling_grids, (0, 2, 1, 3))
        else:
            raise ValueError(f"Reference points must have 2 dimensions, got {reference_points.shape[-1]}")

        # Apply multi-scale deformable attention
        output = multi_scale_deformable_attn_ttnn(
            value=value,
            value_spatial_shapes=spatial_shapes,
            sampling_grids=sampling_grids,
            attention_weights=attention_weights,
            device=self.device,
        )

        if ENABLE_LOGGING:
            logger.info("MSDA Core Attention Complete")

        # The core attention returns ROW_MAJOR, and the residual below is tiled, so the conversion
        # belongs outside the projection guard.
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)

        # Apply output projection
        if hasattr(self.params, "output_proj"):
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
