# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Multi-Scale Deformable Attention implementation for BEVFormer.

The core attention is a single device op: ``ttnn.experimental.fused_msda_from_offsets``.
It takes the reference points and the raw sampling offsets and does the whole of
MSDA -- sampling-location generation, bilinear sampling, the attention multiply
and the reduction over (levels, points) -- inside one device op. No intermediate
grid or per-level sampled-value tensor is materialized.

Key components:
- multi_scale_deformable_attn_fused: Core attention computation function
- TTMSDeformableAttention: Main attention class with parameter management
"""

from typing import Literal

import ttnn

import torch
from ..model_config import DeformableAttentionConfig
from .tt_common import grid_linear_kwargs, l1_if_fits, signpost

from loguru import logger

# Enable/disable logging output
ENABLE_LOGGING = False

ValueMemory = Literal["auto", "l1", "dram"]
_warned_dram_fallbacks = set()


def _spatial_shapes_list(spatial_shapes):
    """torch (L, 2) of (H, W) -> the host-side list of int pairs the fused op takes.

    ``spatial_shapes`` is a static property of the feature pyramid, not a device
    tensor: the op needs it for address arithmetic inside the reader, and reading
    it back from device would force a host sync on every call.
    """
    return [(int(h), int(w)) for h, w in spatial_shapes.tolist()]


# Starts the warning a DRAM fallback logs, once per shape and process (_warned_dram_fallbacks).
DRAM_FALLBACK_PREFIX = "fused_msda: "


def _warn_dram_fallback(value, attention_weights, spatial_shapes, reason):
    warning_key = (tuple(value.shape), tuple(attention_weights.shape), tuple(spatial_shapes.flatten().tolist()))
    if warning_key not in _warned_dram_fallbacks:
        _warned_dram_fallbacks.add(warning_key)
        logger.warning(f"{DRAM_FALLBACK_PREFIX}{reason}; using interleaved DRAM for value shape {list(value.shape)}")


def _is_allocation_oom(error):
    """Keep the auto retry narrow: kernel and validation failures must remain visible."""
    return "Out of Memory" in str(error)


def grouped_softmax(logits, group_size):
    """Softmax over each group of columns of TILE ``logits`` (..., num_groups * group_size).

    Each group gets its own row and maximum so other groups' logits cannot change its
    distribution. Keep the tile-padded rows in DRAM; only the packed input and output
    use the caller's memory placement.
    """
    grouped = ttnn.reshape(logits, (-1, group_size), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # TODO(#62506): softmax's additive -inf mask cannot clear NaN or +inf in tile padding.
    grouped = ttnn.fill_implicit_tile_padding(grouped, 0.0)
    weights = ttnn.softmax(grouped, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(weights, logits.shape, memory_config=logits.memory_config())


def heads_packed_offsets(rows, num_heads):
    """Row-major sampling offsets ``rows`` (bs, Q, num_heads * L*P*2), a Linear's channels, as the
    fused op reads them: as they are when each head's L*P*2 values fill whole alignment units of
    their buffer (the op's rule for heads-packed offsets), else split into a page per head."""
    per_head = rows.shape[-1] // num_heads
    in_l1 = rows.memory_config().buffer_type == ttnn.BufferType.L1
    alignment = ttnn.get_l1_alignment() if in_l1 else ttnn.get_dram_alignment()
    if (per_head * rows.element_size()) % alignment == 0:
        return rows
    bs, num_queries = rows.shape[0], rows.shape[1]
    return ttnn.reshape(rows, (bs, num_queries, num_heads, per_head))


def multi_scale_deformable_attn_fused(
    value,
    reference_points,
    sampling_offsets,
    attention_weights,
    spatial_shapes,
    num_valid_queries=None,
    *,
    num_valid_queries_tensor=None,
    value_memory: ValueMemory = "auto",
    reference_mode="pillar",
):
    """Core multi-scale deformable attention, as one device op.

    Computes, for each (b, q, h), with ``r(l, p) = p % R`` (BEVFormer's z-anchor
    grouping, ``reference_mode="pillar"``), or ``r(l, p) = l`` (``reference_mode="level"``)::

        out[b, q, h] = sum_l sum_p attention_weights[b, q, h, l, p]
            * bilinear(value[b, level l, h],
                       reference_points[b, q, r(l, p)] + sampling_offsets[b, q, h, l, p] / [W_l, H_l])

    Args:
        value: (bs, num_keys, num_heads, head_dim) or packed (bs, num_keys, num_heads*head_dim)
        reference_points: (bs, num_queries, R, 2), normalized [0, 1]: R points in the pillar, or
            one per level with ``reference_mode="level"``
        sampling_offsets: (bs, num_queries, num_heads, num_levels*num_points*2) ROW_MAJOR,
            raw, in feature-map pixel units. The SFPU applies the per-level scale. Heads-packed
            (bs, num_queries, num_heads*num_levels*num_points*2) when each head's
            num_levels*num_points*2 values fill whole alignment units of their buffer.
        attention_weights: (bs, num_queries, num_heads, num_levels*num_points) ROW_MAJOR
        spatial_shapes: torch (num_levels, 2) of (H, W)
        num_valid_queries: optional per-batch count of rows to compute, as host ints. Rows past a
            count are left unwritten and may hold NaN or Inf, so the caller must drop them with a
            select (``ttnn.where``), not by multiplying with a mask.
        num_valid_queries_tensor: the same counts as a (bs,) uint32 ROW_MAJOR device tensor, read on
            device, so a captured trace follows its contents. Counts above num_queries are
            clamped. Pass at most one of the two.
        value_memory: ``"auto"`` selects L1 only when value, output, and fused-op
            circular buffers fit safely, retries in DRAM if an L1 allocation still
            runs out of memory, and logs one warning per input shape when it falls
            back; ``"dram"`` forces DRAM; ``"l1"`` requires L1 and raises
            ``RuntimeError`` when it does not fit or cannot be allocated.

    Returns:
        (bs, num_queries, num_heads*head_dim) ROW_MAJOR -- the heads are already
        concatenated by the op's writer, so no reshape/permute follows. The output
        lives in the memory ``value_memory`` selected (L1 or interleaved DRAM).
    """
    signpost(header="fused_msda Start")

    if value_memory not in {"auto", "l1", "dram"}:
        raise ValueError(f'value_memory must be "auto", "l1", or "dram", got {value_memory!r}')

    attention_weights = ttnn.to_layout(attention_weights, ttnn.ROW_MAJOR_LAYOUT)
    reference_points = ttnn.to_layout(reference_points, ttnn.ROW_MAJOR_LAYOUT)

    if value_memory == "dram":
        target_memory_config = ttnn.DRAM_MEMORY_CONFIG
    else:
        # TODO: Expose placement on the module config and use explicit L1/DRAM for fixed
        # deployments. This would improve host-side performance by removing the per-call
        # capacity query, while keeping auto placement available for dynamic environments.
        l1_is_safe = ttnn.experimental.can_use_fused_msda_from_offsets_l1_value(
            value,
            reference_points,
            sampling_offsets,
            attention_weights,
            _spatial_shapes_list(spatial_shapes),
            reference_mode=reference_mode,
            device_valid_counts=num_valid_queries_tensor is not None,
        )
        if value_memory == "l1" and not l1_is_safe:
            raise RuntimeError("fused_msda value_memory='l1' is unsafe under the current L1 allocation")
        target_memory_config = ttnn.L1_MEMORY_CONFIG if l1_is_safe else ttnn.DRAM_MEMORY_CONFIG
        if value_memory == "auto" and not l1_is_safe:
            _warn_dram_fallback(value, attention_weights, spatial_shapes, "value and output do not safely fit L1")

    valid_queries = list(num_valid_queries) if num_valid_queries else None

    def run_with_memory_config(memory_config):
        staged_value = value
        owns_staged_value = False
        try:
            if value.layout != ttnn.ROW_MAJOR_LAYOUT:
                staged_value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
                owns_staged_value = True
            elif value.memory_config() != memory_config:
                staged_value = ttnn.to_memory_config(value, memory_config)
                owns_staged_value = True

            return ttnn.experimental.fused_msda_from_offsets(
                staged_value,
                reference_points,
                sampling_offsets,
                attention_weights,
                _spatial_shapes_list(spatial_shapes),
                reference_mode=reference_mode,
                memory_config=memory_config,
                num_valid_queries=valid_queries,
                num_valid_queries_tensor=num_valid_queries_tensor,
            )
        finally:
            if owns_staged_value:
                ttnn.deallocate(staged_value)

    try:
        output = run_with_memory_config(target_memory_config)
    except RuntimeError as error:
        if value_memory != "auto" or target_memory_config != ttnn.L1_MEMORY_CONFIG or not _is_allocation_oom(error):
            raise
        _warn_dram_fallback(
            value,
            attention_weights,
            spatial_shapes,
            "L1 allocation failed after the capacity check",
        )
        output = run_with_memory_config(ttnn.DRAM_MEMORY_CONFIG)

    signpost(header="fused_msda End")
    return output


class TTMSDeformableAttention:
    """
    ttnn implementation of Multi-Scale Deformable Attention.
    Based on the MMCV/BEVFormer approach.
    """

    def __init__(self, config: DeformableAttentionConfig, device, params=None, *, spatial_shapes, residual=True):
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
            spatial_shapes: Feature-map (H, W) per level. Fixed for the lifetime of the
                module: forward takes no shapes of its own, and they are handed to the
                fused op as a static attribute. Features at a different resolution
                require a new instance.
            residual: Add the identity to the output. BEVFormer's MSDeformableAttention3D, inside
                the spatial cross-attention, has no shortcut. Without it and without
                ``output_proj``, the output is the fused op's ROW_MAJOR rows.

        Raises:
            ValueError: If the configuration or spatial shapes are invalid.
        """
        # Validate configuration
        if config.embed_dims % config.num_heads != 0:
            raise ValueError(f"embed_dims ({config.embed_dims}) must be divisible by num_heads ({config.num_heads})")

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

        # Set attributes
        self.embed_dims = config.embed_dims
        self.num_heads = config.num_heads
        self.num_levels = config.num_levels
        self.num_points = config.num_points
        self.batch_first = config.batch_first
        self.device = device
        self.params = params
        self.spatial_shapes = spatial_shapes.to(dtype=torch.long).clone()
        self.total_keys = int(self.spatial_shapes.prod(dim=1).sum().item())

        self.head_dim = self.embed_dims // self.num_heads
        self.residual = residual
        self.linear_kwargs = grid_linear_kwargs(device)

        # The attention weights' softmax over each head's samples:
        # - "tile_row" when they are one tile row (base's SCA, 4 levels x 8 points);
        # - "grouped" over each head's group of the Linear's columns when they are not, but all
        #   heads' samples fill whole tiles (small's and tiny's SCA, 8 x 8, and every decoder,
        #   8 heads x 4 points);
        # - "padded" otherwise: the heads split off in TILE layout, every query's (heads, samples)
        #   padded to 32x32.
        samples = self.num_levels * self.num_points
        if samples == ttnn.TILE_SIZE:
            self.attention_softmax = "tile_row"
        elif (self.num_heads * samples) % ttnn.TILE_SIZE == 0:
            self.attention_softmax = "grouped"
        else:
            self.attention_softmax = "padded"

    def forward(
        self,
        *,
        query,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        num_valid_queries=None,
        num_valid_queries_tensor=None,
        consume_query=False,
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
            num_valid_queries: Optional per-batch count of rows to compute, as host ints. Rows past
                it are left unwritten and may hold NaN or Inf; drop them with ``ttnn.where``.
            num_valid_queries_tensor: The same counts as a (bs,) uint32 device tensor, read on
                device so a captured trace follows them.
            consume_query: Free the query once the offsets and weights are computed, so an L1
                query does not hold the L1 the fused op stages its value in. Requires no residual.
                It is the caller's ``query`` only when no copy was made: batch-first, TILE, and
                no ``query_pos``, as the spatial cross-attention calls it.

        Returns:
            output: [bs, num_queries, embed_dims]
        """

        # Handle input defaults
        assert not (consume_query and self.residual), "consume_query frees query, which the residual may read"
        if value is None:
            value = query
        if identity is None:
            identity = query

        # Add query positional encoding
        if query_pos is not None:
            query = ttnn.add(query, query_pos)

        deformable_attn_region = f"TT MS Deformable Attn Module {query.shape[1]} - {self.total_keys}"
        signpost(header=f"{deformable_attn_region} Start")

        # Handle batch_first format
        if not self.batch_first:
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))
            identity = ttnn.permute(identity, (1, 0, 2))

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

        # Verify spatial shapes consistency
        assert self.total_keys == num_keys, f"Inconsistent keys: {self.total_keys} != {num_keys}"

        if ENABLE_LOGGING:
            logger.info("MSDA Value Projection Start")

        # The fused op takes packed (B, S, H*D), which is what this Linear emits.
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, self.params.value_proj.weight, bias=self.params.value_proj.bias)

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            mask = ttnn.unsqueeze(key_padding_mask, -1)  # [bs, num_keys, 1]
            zeros_like_value = ttnn.zeros_like(value)
            value = ttnn.where(mask, zeros_like_value, value)

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        sampling_offsets, attention_weights = self.sampling_operands(query, consume_query=consume_query)

        output = multi_scale_deformable_attn_fused(
            value=value,
            reference_points=reference_points,
            sampling_offsets=sampling_offsets,
            attention_weights=attention_weights,
            spatial_shapes=self.spatial_shapes,
            num_valid_queries=num_valid_queries,
            num_valid_queries_tensor=num_valid_queries_tensor,
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
        if self.residual:
            output = ttnn.add(output, identity)

        # Handle batch_first format for output
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        signpost(header=f"{deformable_attn_region} End")

        return output

    def sampling_operands(self, query, consume_query=False):
        """The fused op's sampling offsets and attention weights from TILE ``query``
        [bs, num_queries, embed_dims], as ``forward`` hands them to it; ``consume_query`` as there."""
        bs, num_queries, _ = query.shape

        # Placement:
        # - the Linears and packed weights run in L1 when they fit, the weights' first, then the
        #   offsets'. Besides the query, the weights' Linear and softmax hold at most two packed
        #   weight-sized tensors in L1; padded softmax intermediates stay in DRAM. The offsets hold
        #   one offset-sized tensor, as many values as two weight-sized: the budget below;
        # - both results land in DRAM, since the fused op stages its value and output in L1;
        # - they are row-major but for "padded", which the fused wrapper untilizes. Its reshape
        #   is 32x32 / (heads * samples) times the weights, so it goes to DRAM.
        samples = self.num_levels * self.num_points
        memory = l1_if_fits(
            self.device,
            bs * num_queries * (self.embed_dims + 2 * self.num_heads * samples) * query.element_size(),
        )
        DRAM = ttnn.DRAM_MEMORY_CONFIG

        if ENABLE_LOGGING:
            logger.info("MSDA Attention Weight Generation")

        # Generate attention weights. The softmax is over (levels, points) jointly, so
        # (bs, Q, heads, levels*points) is both the natural shape here and exactly the
        # packed layout the op consumes -- it is never split into (levels, points).
        attention_weights = ttnn.linear(
            query,
            self.params.attention_weights.weight,
            bias=self.params.attention_weights.bias,
            memory_config=memory,
            **self.linear_kwargs,
        )
        heads_shape = (bs, num_queries, self.num_heads, samples)
        if self.attention_softmax == "tile_row":
            # One head's samples per tile row: the softmax runs on unpadded tiles, and after the
            # untilize the head split keeps the row width, a view.
            attention_weights = ttnn.reshape(attention_weights, (bs, num_queries * self.num_heads, samples))
            attention_weights = ttnn.softmax(attention_weights, dim=-1)
            attention_weights = ttnn.reshape(
                ttnn.to_layout(attention_weights, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM), heads_shape
            )
        elif self.attention_softmax == "grouped":
            logits = attention_weights
            attention_weights = grouped_softmax(logits, samples)
            ttnn.deallocate(logits)
            attention_weights = ttnn.reshape(
                ttnn.to_layout(attention_weights, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM), heads_shape
            )
        else:
            attention_weights = ttnn.reshape(attention_weights, heads_shape, memory_config=DRAM)
            # TODO(#62506): drop once softmax zeroes tile padding itself. The reshape leaves stale memory in
            # the padding, and softmax's additive -inf pad mask turns NaN or +inf there into NaN rows.
            attention_weights = ttnn.fill_implicit_tile_padding(attention_weights, 0.0)
            attention_weights = ttnn.softmax(attention_weights, dim=-1, memory_config=DRAM)

        if ENABLE_LOGGING:
            logger.info("MSDA Sampling Offset Generation")

        # Offsets stay in feature-map pixel units; the op forms the sampling location.
        # The Linear's channels are already (head, level, point, xy): the op reads each head's
        # offsets from the query's page when they fill whole alignment units of the buffer, else
        # the reshape below gives each head a page.
        sampling_offsets = ttnn.linear(
            query,
            self.params.sampling_offsets.weight,
            bias=getattr(self.params.sampling_offsets, "bias", None),
            memory_config=memory,
            **self.linear_kwargs,
        )
        if consume_query:
            ttnn.deallocate(query)
        rows = ttnn.to_layout(sampling_offsets, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(sampling_offsets)
        sampling_offsets = heads_packed_offsets(rows, self.num_heads)

        return sampling_offsets, attention_weights

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
