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


def _fused_msda_level(value_level, sampling_grids, attention_weights, level, H, W, shape):
    """Run one pyramid level through the fused `multi_scale_deformable_attn` device op.

    The op fuses grid_sample with the weighted sum over sampling points, so it returns
    `(N, Q, D)` already reduced over `P` — there is no per-level tensor left to stack.

    `value_level` is `(bs, H*W, num_heads, D)`; `sampling_grids` and `attention_weights`
    carry all levels and are sliced here. All three op inputs must be ROW_MAJOR bfloat16
    and INTERLEAVED, which the device op enforces with TT_FATAL rather than converting.
    """
    bs, num_heads, num_queries, num_points, head_dim = shape

    # (bs, H*W, num_heads, D) -> (N, H, W, D). In ROW_MAJOR the trailing reshape is a
    # view, so splitting H*W into H,W after the permute costs nothing.
    value_l = ttnn.permute(value_level, (0, 2, 1, 3))
    value_l = ttnn.to_layout(value_l, layout=ttnn.ROW_MAJOR_LAYOUT)
    value_l = ttnn.reshape(value_l, (bs * num_heads, H, W, head_dim))

    grid = sampling_grids[:, :, :, level]  # (bs, Q, num_heads, P, 2)
    grid = ttnn.permute(grid, (0, 2, 1, 3, 4))
    grid = ttnn.reshape(grid, (bs * num_heads, num_queries * num_points, 1, 2))

    attn = attention_weights[:, :, :, level, :]  # (bs, Q, num_heads, P)
    attn = ttnn.permute(attn, (0, 2, 1, 3))
    attn = ttnn.reshape(attn, (bs * num_heads, num_queries, num_points))
    attn = ttnn.to_layout(attn, layout=ttnn.ROW_MAJOR_LAYOUT)

    if value_l.dtype != ttnn.bfloat16:
        value_l = ttnn.typecast(value_l, ttnn.bfloat16)
    if grid.dtype != ttnn.bfloat16:
        grid = ttnn.typecast(grid, ttnn.bfloat16)
    if attn.dtype != ttnn.bfloat16:
        attn = ttnn.typecast(attn, ttnn.bfloat16)

    return ttnn.experimental.multi_scale_deformable_attn(value_l, grid, attn)  # (N, Q, D)


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
        sampling_grids (ttnn.Tensor): Sampling points already normalized to [-1, 1] and in
            ROW_MAJOR, has shape
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
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_grids.shape

    if ENABLE_LOGGING:
        logger.info("MSDA Start")

    # Split value into a list of tensors for each level
    value_list = ttnn.split(value, [H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    # `attention_weights` is softmaxed jointly over levels and points, so the joint
    # weighted sum decomposes exactly into a sum of per-level weighted sums — each fused
    # call reduces its own level's points and the levels are added. No renormalization.
    shape = (bs, num_heads, num_queries, num_points, head_dim)
    output = None
    if use_signpost:
        signpost(header=f"MSDA Fused Core Start, Q={num_queries} L={num_levels} P={num_points}")
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        level_out = _fused_msda_level(
            value_list[level],
            sampling_grids,
            attention_weights,
            level,
            int(H_),
            int(W_),
            shape,
        )
        output = level_out if output is None else ttnn.add(output, level_out)

    if use_signpost:
        signpost(header="MSDA Fused Core End")

    output = ttnn.reshape(output, (bs, num_heads, num_queries, head_dim))
    output = ttnn.to_layout(output, layout=ttnn.TILE_LAYOUT)
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
        self._folded_sampling_offsets_cache = {}
        self.params = params

        self.head_dim = self.embed_dims // self.num_heads

    def _folded_sampling_offsets(self, spatial_shapes):
        """Pre-scale the `sampling_offsets` Linear by `2 / [W, H]` per level.

        The normalizer is `[W, H]` per level, fixed by the feature-pyramid config, so
        dividing the Linear's output by it is a static per-output-channel scale and folds
        into the weight exactly: `s · (Wx + b) == (Wx + b) / normalizer`. That removes a
        broadcast SFPU divide whose operands tile-pad an extent-2 axis to 32.

        The factor of 2 is the `[0, 1] → [-1, 1]` grid rescale, folded into the same
        constant: `grid == 2·(ref + off) - 1 == (2·ref - 1) + 2·off`. The offset half lives
        here; the reference half is :meth:`_grid_bias`.

        The Linear emits `num_heads * num_levels * num_points * 2` channels ordered
        (head, level, point, xy) with xy innermost, which is what makes the scale vector
        expressible; `preprocess_linear_weight` stores the weight transposed as
        `(in, out)`, so one `(1, out)` row broadcasts over weight and bias alike.
        """
        key = tuple(spatial_shapes.flatten().tolist())
        folded = self._folded_sampling_offsets_cache.get(key)
        if folded is not None:
            return folded

        weight = self.params.sampling_offsets.weight
        out_features = weight.shape[-1]
        expected = self.num_heads * self.num_levels * self.num_points * 2
        assert out_features == expected, f"sampling_offsets width {out_features} != {expected}"
        assert (
            spatial_shapes.shape[0] == self.num_levels
        ), f"spatial_shapes has {spatial_shapes.shape[0]} levels, config says {self.num_levels}"

        scale = torch.ones(self.num_heads, self.num_levels, self.num_points, 2, dtype=torch.float32)
        for level, (h, w) in enumerate(spatial_shapes.tolist()):
            scale[:, level, :, 0] = 2.0 / float(w)
            scale[:, level, :, 1] = 2.0 / float(h)
        scale_tt = ttnn.from_torch(
            scale.reshape(1, out_features), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        bias = self.params.sampling_offsets.bias
        folded = (
            ttnn.mul(weight, scale_tt),
            ttnn.mul(bias, scale_tt) if bias is not None else None,
        )
        ttnn.deallocate(scale_tt)

        self._folded_sampling_offsets_cache[key] = folded
        return folded

    def _grid_bias(self, reference_points, depth_levels):
        """`2·ref - 1`, laid out in the `sampling_offsets` Linear's channel order.

        The other half of the grid fold in :meth:`_folded_sampling_offsets`. The Linear emits
        channels ordered (head, level, point, xy) with the points grouped as
        `(num_points // depth_levels, depth_levels)`, and the reference point broadcasts over
        everything but the innermost `(depth_levels, 2)` block — so the bias is that flat block
        repeated once per (head, level, point-group).

        ROW_MAJOR throughout: the repeat writes the channel axis contiguously, which is what makes
        the caller's split back into `(heads, levels, points, 2)` a view instead of a re-layout.
        """
        bs, num_queries = reference_points.shape[0], reference_points.shape[1]
        block = depth_levels * 2
        groups = self.num_heads * self.num_levels * (self.num_points // depth_levels)

        ref = ttnn.to_layout(reference_points, ttnn.ROW_MAJOR_LAYOUT)
        ref = ttnn.reshape(ref, (bs, num_queries, 1, block))
        ref = ttnn.mul(ref, 2.0)
        ref = ttnn.sub(ref, 1.0)
        ref = ttnn.repeat(ref, ttnn.Shape((1, 1, groups, 1)))
        return ttnn.reshape(ref, (bs, num_queries, groups * block))

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

        # Generate sampling offsets, already normalized and grid-rescaled by the folded Linear.
        #
        # The Linear emits (bs, Q, num_heads*num_levels*num_points*2) — 256 channels, a shape that
        # tiles cleanly. Everything downstream of it is elementwise or a view, so the tensor stays
        # in that shape and goes to ROW_MAJOR here: the split into (heads, levels, points, 2) is
        # then a trailing-axis view, and the extent-2 coordinate axis never lands in a tiled
        # dimension where it would pad 2 -> 32.
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        offsets_weight, offsets_bias = self._folded_sampling_offsets(spatial_shapes)
        sampling_offsets = ttnn.linear(query, offsets_weight, bias=offsets_bias)
        sampling_offsets = ttnn.to_layout(sampling_offsets, ttnn.ROW_MAJOR_LAYOUT)

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

            # One ROW_MAJOR add produces the finished grid: the reference term carries `2·ref - 1`
            # and the Linear already carries `2 / [W, H]`, so no rescale follows.
            sampling_grids = ttnn.add(sampling_offsets, self._grid_bias(reference_points, D))
            sampling_grids = ttnn.reshape(
                sampling_grids, (bs, num_queries, self.num_heads, self.num_levels, self.num_points, 2)
            )
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
