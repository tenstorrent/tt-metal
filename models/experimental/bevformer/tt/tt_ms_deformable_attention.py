# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Multi-Scale Deformable Attention implementation for BEVFormer.

The core attention is a single device op: ``ttnn.experimental.fused_msda_from_offsets``.
It takes the reference points and the raw sampling offsets and does the whole of
MSDA -- sampling-location generation, bilinear sampling, the attention multiply
and the reduction over (levels, points) -- inside one kernel. No intermediate
grid or per-level sampled-value tensor is materialized.
"""

import torch

import ttnn
from models.experimental.bevformer.model_config import DeformableAttentionConfig
from models.experimental.bevformer.tt.tt_common import signpost


def _spatial_shapes_list(spatial_shapes):
    """torch (L, 2) of (H, W) -> the host-side list of int pairs the fused op takes.

    ``spatial_shapes`` is a static property of the feature pyramid, not a device
    tensor: the op needs it for address arithmetic inside the reader, and reading
    it back from device would force a host sync on every call.
    """
    return [(int(h), int(w)) for h, w in spatial_shapes.tolist()]


def multi_scale_deformable_attn_fused(value, reference_points, sampling_offsets, attention_weights, spatial_shapes):
    """Core multi-scale deformable attention, as one device op.

    Computes, for each (b, q, h), with ``r(l, p) = p % R`` (BEVFormer's z-anchor
    grouping, ``reference_mode="pillar"``)::

        out[b, q, h] = sum_l sum_p attention_weights[b, q, h, l, p]
            * bilinear(value[b, level l, h],
                       reference_points[b, q, r(l, p)] + sampling_offsets[b, q, h, l, p] / [W_l, H_l])

    Args:
        value: (bs, num_keys, num_heads, head_dim) or packed (bs, num_keys, num_heads*head_dim) ROW_MAJOR
        reference_points: (bs, num_queries, num_points_in_pillar, 2), normalized [0, 1]
        sampling_offsets: (bs, num_queries, num_heads, num_levels*num_points*2) ROW_MAJOR,
            raw, in feature-map pixel units -- the reader applies the per-level
            ``/ [W_l, H_l]`` normalization itself.
        attention_weights: (bs, num_queries, num_heads, num_levels*num_points) ROW_MAJOR
        spatial_shapes: torch (num_levels, 2) of (H, W)

    Returns:
        (bs, num_queries, num_heads*head_dim) ROW_MAJOR -- the heads are already
        concatenated by the op's writer, so no reshape/permute follows.
    """
    signpost(header="fused_msda Start")

    value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)
    attention_weights = ttnn.to_layout(attention_weights, ttnn.ROW_MAJOR_LAYOUT)
    reference_points = ttnn.to_layout(reference_points, ttnn.ROW_MAJOR_LAYOUT)

    output = ttnn.experimental.fused_msda_from_offsets(
        value,
        reference_points,
        sampling_offsets,
        attention_weights,
        _spatial_shapes_list(spatial_shapes),
        reference_mode="pillar",
    )

    signpost(header="fused_msda End")
    return output


class TTMSDeformableAttention:
    """mmcv's MultiScaleDeformableAttention, batch-first, on the fused MSDA op."""

    def __init__(self, config: DeformableAttentionConfig, device, params=None, *, spatial_shapes, residual=True):
        """
        Args:
            config (DeformableAttentionConfig): Configuration object containing:
                - embed_dims (int): Feature embedding dimensions
                - num_heads (int): Number of attention heads
                - num_levels (int): Number of feature pyramid levels
                - num_points (int): Sampling points per head and level
                - batch_first (bool): Must be True; callers permute sequence-first tensors
            device: TTNN device for tensor operations
            params: Pre-computed TTNN parameters containing linear layer weights and biases.
                Should include: value_proj, sampling_offsets, attention_weights, and output_proj
                unless the caller projects the output itself (the spatial cross-attention does).
            spatial_shapes: Feature-map (H, W) per level. Fixed for the lifetime of the
                module: forward takes no shapes of its own, and they are handed to the
                fused op as a static attribute. Features at a different resolution
                require a new instance.
            residual: Add the query, before ``query_pos``, to the output. BEVFormer's MSDeformableAttention3D, inside
                the spatial cross-attention, has no shortcut.

        Raises:
            ValueError: If the configuration or spatial shapes are invalid.
        """
        if config.embed_dims % config.num_heads != 0:
            raise ValueError(f"embed_dims ({config.embed_dims}) must be divisible by num_heads ({config.num_heads})")

        if not config.batch_first:
            raise ValueError("TTMSDeformableAttention takes batch-first tensors only")

        if spatial_shapes is None:
            raise ValueError("spatial_shapes is required")
        if not isinstance(spatial_shapes, torch.Tensor):
            spatial_shapes = torch.as_tensor(spatial_shapes)
        if spatial_shapes.ndim != 2 or spatial_shapes.shape[1] != 2:
            raise ValueError(f"spatial_shapes must have shape [num_levels, 2], got {tuple(spatial_shapes.shape)}")
        if spatial_shapes.shape[0] != config.num_levels:
            raise ValueError(
                f"spatial_shapes has {spatial_shapes.shape[0]} levels, but config requires {config.num_levels}"
            )
        if spatial_shapes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError(f"spatial_shapes must contain integers, got {spatial_shapes.dtype}")
        if torch.any(spatial_shapes <= 0):
            raise ValueError(f"spatial_shapes dimensions must be positive, got {spatial_shapes.tolist()}")

        self.embed_dims = config.embed_dims
        self.num_heads = config.num_heads
        self.num_levels = config.num_levels
        self.num_points = config.num_points
        self.device = device
        self.params = params
        self.spatial_shapes = spatial_shapes.to(dtype=torch.long).clone()
        self.total_keys = int(self.spatial_shapes.prod(dim=1).sum().item())

        self.head_dim = self.embed_dims // self.num_heads
        self.residual = residual

    def forward(
        self,
        *,
        query,
        value=None,
        query_pos=None,
        reference_points=None,
    ):
        """
        Args:
            query: [bs, num_queries, embed_dims] Query features
            value: [bs, num_keys, embed_dims] Value features (optional, defaults to query)
            query_pos: [bs, num_queries, embed_dims] Query positional encoding
            reference_points: [bs, num_queries, num_points_in_pillar, 2] Reference points

        Returns:
            output: [bs, num_queries, embed_dims]
        """

        if value is None:
            value = query
        identity = query

        if query_pos is not None:
            query = ttnn.add(query, query_pos)

        signpost(header=f"TT MS Deformable Attn Module Start, {query.shape[1]} - {self.total_keys}")

        assert reference_points is not None, "reference_points is required"

        # query is the authority on batch and query count; reference_points only supplies
        # the pillar depth. Reading bs from it would silently mis-shape the sampling.
        bs, num_queries, _ = query.shape
        num_keys = value.shape[1]
        assert reference_points.shape[0] == bs and reference_points.shape[1] == num_queries, (
            f"reference_points {list(reference_points.shape)} does not match " f"query [{bs}, {num_queries}, ...]"
        )
        if reference_points.shape[-1] != 2:
            raise ValueError(f"Reference points must have 2 dimensions, got {reference_points.shape[-1]}")

        assert self.total_keys == num_keys, f"Inconsistent keys: {self.total_keys} != {num_keys}"

        # The fused op takes packed (B, S, H*D), which is what this Linear emits.
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, self.params.value_proj.weight, bias=self.params.value_proj.bias)

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)

        # Generate attention weights. The softmax is over (levels, points) jointly, so
        # (bs, Q, heads, levels*points) is both the natural shape here and exactly the
        # packed layout the op consumes -- it is never split into (levels, points).
        attention_weights = ttnn.linear(
            query, self.params.attention_weights.weight, bias=self.params.attention_weights.bias
        )
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_queries, self.num_heads, self.num_levels * self.num_points)
        )
        attention_weights = ttnn.softmax(attention_weights, dim=-1)

        # Offsets stay in feature-map pixel units; the op forms the sampling location.
        # The Linear's channels are already (head, level, point, xy), the op's
        # packed layout, so the reshape below splits dimensions.
        sampling_offsets = ttnn.linear(
            query, self.params.sampling_offsets.weight, bias=getattr(self.params.sampling_offsets, "bias", None)
        )
        sampling_offsets = ttnn.to_layout(sampling_offsets, ttnn.ROW_MAJOR_LAYOUT)
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_queries, self.num_heads, self.num_levels * self.num_points * 2)
        )

        output = multi_scale_deformable_attn_fused(
            value=value,
            reference_points=reference_points,
            sampling_offsets=sampling_offsets,
            attention_weights=attention_weights,
            spatial_shapes=self.spatial_shapes,
        )

        if hasattr(self.params, "output_proj"):
            output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
            output = ttnn.linear(output, self.params.output_proj.weight, bias=self.params.output_proj.bias)

        if self.residual:
            output = ttnn.add(output, identity)

        signpost(header="TT MS Deformable Attn Module End")

        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        return (
            f"embed_dims={self.embed_dims}, num_heads={self.num_heads}, "
            f"num_levels={self.num_levels}, num_points={self.num_points}"
        )
