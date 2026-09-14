# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Spatial Cross Attention (SCA) module for BEVFormer.

This module implements the spatial cross-attention mechanism using TTNN operations
to enable BEV queries to extract spatial features from regions of interest across
multiple camera views using deformable attention.
"""

import ttnn
import torch
from dataclasses import dataclass
from typing import Optional


from .tt_ms_deformable_attention import TTMSDeformableAttention
from ..config import DeformableAttentionConfig

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

from loguru import logger

# Enable/disable logging output
ENABLE_LOGGING = False


def _batch_offsets(bs: int, num_queries: int) -> torch.Tensor:
    """Convert batch-local query IDs to stacked row IDs."""
    return (torch.arange(bs, dtype=torch.int32) * num_queries).reshape(bs, 1, 1)


def _flat_row_index(row_ids: torch.Tensor, device) -> ttnn.Tensor:
    """Format row IDs for ``ttnn.embedding``."""
    return ttnn.from_torch(
        row_ids.reshape(1, 1, 1, row_ids.numel()),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


@dataclass(frozen=True)
class SCARebatchPlan:
    """Per-frame SCA rebatch plan shared by every encoder layer.

    The plan is valid only for the reference points, visibility mask, tensor shapes, and device used to build it.
    Tensor fields are ``None`` and ``inverse_index`` is empty when ``is_empty`` is true.

    Attributes:
        rebatch_len: Tile-aligned number of query rows processed per camera.
        query_index: Gather indices for query rows.
        reference_points_batched: Rebatched camera reference points with shape
            ``[batch_size * num_cams, rebatch_len, num_depth_levels, 2]``.
        row_mask: Per-camera-row validity with shape ``[batch_size * num_cams, rebatch_len, 1]``.
        inverse_index: One gather index per contribution slot, each selecting the camera row that
            feeds a query. Slots a query does not fill point at a padded row.
        count: Camera-contributor count with shape ``[batch_size, num_queries, 1]``.
        is_empty: Whether the visibility mask contains no valid query-camera pairs.
    """

    rebatch_len: int
    query_index: Optional[ttnn.Tensor]
    reference_points_batched: Optional[ttnn.Tensor]
    row_mask: Optional[ttnn.Tensor]
    inverse_index: tuple[ttnn.Tensor, ...]
    count: Optional[ttnn.Tensor]
    is_empty: bool = False


def build_rebatch_plan(reference_points_cam, bev_mask, device) -> SCARebatchPlan:
    """Build a rebatch plan for one frame's camera projections.

    Args:
        reference_points_cam: Bfloat16 device tensor with shape
            ``[num_cams, batch_size, num_queries, num_depth_levels, 2]``.
        bev_mask: Device tensor with shape
            ``[num_cams, batch_size, num_queries, num_depth_levels]``.
        device: Device on which plan tensors are allocated. It must match the input tensors.

    Returns:
        A plan owned by these frame inputs. If no query-camera pair is valid, ``is_empty`` is true,
        ``rebatch_len`` is zero, and the plan carries no tensors.
    """
    num_cams, bs, num_queries, num_depth_levels = bev_mask.shape

    # max_len sizes tensors, so it must be a Python int. Mask reduction, index construction and
    # the contributor count stay on the host; every gather runs on device.
    valid_per_cam = ttnn.to_torch(bev_mask).sum(-1) > 0  # [num_cams, B, num_queries]
    max_len = int(valid_per_cam.sum(-1).max().item())

    if ENABLE_LOGGING:
        logger.info(f"SCA Valid Queries: {valid_per_cam.sum(-1).flatten().tolist()}")

    if max_len == 0:
        return SCARebatchPlan(
            rebatch_len=0,
            query_index=None,
            reference_points_batched=None,
            row_mask=None,
            inverse_index=(),
            count=None,
            is_empty=True,
        )

    # Tile-align so folding num_cams into the row dim stays a view; unaligned would move data.
    # Costs up to TILE_SIZE padded rows per camera through MSDA, discarded afterwards. Aligning
    # max_len + 1 keeps at least one padded row per camera, which aggregation needs as a zero
    # source for queries that fewer than max_count cameras see.
    rebatch_len = ((max_len // ttnn.TILE_SIZE) + 1) * ttnn.TILE_SIZE

    # Compact each camera to the queries it sees, so MSDA runs on rebatch_len rows, not all of
    # them. IDs are batch-local; num_queries marks padding, clamped away because embedding
    # rejects out-of-range indices.
    # TODO: vectorize this index construction off the host
    compact_ids = torch.full((bs, num_cams, rebatch_len), num_queries, dtype=torch.int32)

    # Invert the compaction: for every query, the rows of queries_output that contribute to it.
    # Row (j, i, p) of the compacted output is flat row (j * num_cams + i) * rebatch_len + p.
    # Unfilled slots read the last row of the first camera, which is always padding.
    max_count = int(valid_per_cam.sum(0).max().item())
    inverse_ids = torch.full((max_count, bs, num_queries), rebatch_len - 1, dtype=torch.int32)
    next_slot = torch.zeros((bs, num_queries), dtype=torch.int64)
    for j in range(bs):
        for i in range(num_cams):
            valid_indices = torch.nonzero(valid_per_cam[i, j], as_tuple=False).squeeze(-1)
            num_valid = valid_indices.numel()
            compact_ids[j, i, :num_valid] = valid_indices.to(torch.int32)
            base = (j * num_cams + i) * rebatch_len
            inverse_ids[next_slot[j, valid_indices], j, valid_indices] = base + torch.arange(
                num_valid, dtype=torch.int32
            )
            next_slot[j, valid_indices] += 1
    gather_ids = compact_ids.clamp(max=num_queries - 1)

    reference_points_cam = ttnn.clamp(reference_points_cam, -10.0, 10.0)
    assert (
        reference_points_cam.dtype == ttnn.bfloat16
    ), f"SCA rebatch gathers reference_points_cam, so it must be bfloat16, got {reference_points_cam.dtype}."

    # Reference points are camera-major, so their row IDs include camera and batch offsets.
    ref_rows = ttnn.reshape(
        ttnn.to_layout(reference_points_cam, ttnn.ROW_MAJOR_LAYOUT),
        (1, 1, num_cams * bs * num_queries, num_depth_levels * 2),
    )
    ref_index = _flat_row_index(
        gather_ids
        + _batch_offsets(bs, num_queries)
        + torch.arange(num_cams, dtype=torch.int32).reshape(1, num_cams, 1) * (bs * num_queries),
        device,
    )
    # Split the gathered row back into MSDA's [.., rebatch_len, num_depth_levels, 2], the last
    # dim being each depth level's (x, y) camera coordinate. Changing the row width from
    # num_depth_levels * 2 to 2 requires a data-moving reshape before conversion to TILE layout.
    # TODO: profile whether tilizing first is cheaper.
    reference_points_batched = ttnn.to_layout(
        ttnn.reshape(
            ttnn.embedding(ref_index, ref_rows),
            (bs * num_cams, rebatch_len, num_depth_levels, 2),
        ),
        ttnn.TILE_LAYOUT,
    )

    row_mask = torch.arange(rebatch_len) < valid_per_cam.sum(-1).t().reshape(bs * num_cams, 1)
    row_mask = ttnn.from_torch(
        row_mask.to(torch.bfloat16).unsqueeze(-1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    count = torch.clamp(valid_per_cam.permute(1, 2, 0).sum(-1), min=1.0)
    count = ttnn.unsqueeze(ttnn.from_torch(count, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), -1)

    return SCARebatchPlan(
        rebatch_len=rebatch_len,
        query_index=_flat_row_index(gather_ids + _batch_offsets(bs, num_queries), device),
        reference_points_batched=reference_points_batched,
        row_mask=row_mask,
        inverse_index=tuple(_flat_row_index(inverse_ids[slot], device) for slot in range(max_count)),
        count=count,
    )


class TTSpatialCrossAttention:
    """
    TTNN Spatial Cross Attention module for BEVFormer.

    This attention mechanism allows BEV queries to extract spatial features
    from regions of interest across camera views using deformable attention.
    Each BEV query can attend to multiple camera features at different scales
    and locations.

    Note: This module expects pre-projected reference points and validity masks.
    Point sampling/projection from 3D to camera coordinates should be handled
    by the encoder before calling this attention module.

    Args:
        device: TTNN device for computation
        params: Parameter dict containing weights and biases
        embed_dims (int): The embedding dimension.
        num_cams (int): Number of cameras.
        batch_first (bool): Whether the first dimension of input is batch_size.
        deformable_attention (dict): Config for MSDeformableAttention.
        spatial_shapes: Multi-scale feature shapes [num_levels, 2]
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        device,
        params,
        embed_dims: int = 256,
        num_cams: int = 6,
        batch_first: bool = True,
        deformable_attention: Optional[dict] = None,
        *,
        spatial_shapes,
        **kwargs,
    ):
        self.device = device
        self.params = params
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.batch_first = batch_first

        if deformable_attention is None:
            deformable_attention = dict(embed_dims=embed_dims, num_levels=4, num_points=4, num_heads=8)

        deform_config = DeformableAttentionConfig(
            embed_dims=deformable_attention.get("embed_dims", embed_dims),
            num_heads=deformable_attention.get("num_heads", 8),
            num_levels=deformable_attention.get("num_levels", 4),
            num_points=deformable_attention.get("num_points", 4),
            batch_first=batch_first,
        )

        # Its own namespace, not the SCA's: both own an ``output_proj`` and both
        # apply it, so sharing one namespace makes the inner attention project
        # with the SCA's matrix and the SCA apply that matrix a second time.
        self.deformable_attention = TTMSDeformableAttention(
            deform_config, device, params.deformable_attention, spatial_shapes=spatial_shapes
        )

    def forward(
        self,
        query,
        reference_points_cam,
        bev_mask,
        value,
        residual=None,
        query_pos=None,
        key_padding_mask=None,
        level_start_index=None,
        rebatch_plan=None,
        **kwargs,
    ):
        """
        Forward pass of TTNN Spatial Cross Attention.

        Args:
            query: Bfloat16 BEV queries [B, num_queries, embed_dims].
            reference_points_cam: Bfloat16 camera projected reference points [num_cams, B, num_queries, D, 2].
            bev_mask: Valid mask for camera projections [num_cams, B, num_queries, D].
            value: Multi-camera features [B, num_cams, H*W, embed_dims], batch-first; the
                encoder permutes once per forward. Attention here takes no key: deformable
                attention never scores query against one. Two Linears on query predict where to
                sample and with what weight, replacing Q*K^T, so ``value`` is the only feature
                tensor the device path needs.
            residual: Residual connection input.
            query_pos: Query positional encoding.
            key_padding_mask: Key padding mask.
            level_start_index: Start index of each level.
            rebatch_plan: Prebuilt :class:`SCARebatchPlan`. Shared by every encoder layer; built here if absent.
            **kwargs: Additional arguments.

        Returns:
            Output features [B, num_queries, embed_dims].
        """
        if use_signpost:
            signpost(header="TTNN SCA Forward Start")

        assert value is not None, "value is required"

        # Handle input defaults
        if residual is None:
            inp_residual = ttnn.clone(query)
        else:
            inp_residual = residual

        # Add query positional encoding
        if query_pos is not None:
            query = ttnn.add(query, query_pos)

        if ENABLE_LOGGING:
            logger.info("SCA Tensor Conversion Complete")

        bs, num_queries, _ = query.shape
        # Extract number of depth levels for 3D point sampling
        # Each BEV query samples points at multiple Z-coordinates (depth levels) in 3D space
        num_depth_levels = reference_points_cam.shape[3]

        # Validate sampling points divisibility to prevent runtime errors in deformable attention
        assert self.deformable_attention.num_points % num_depth_levels == 0, (
            f"num_points ({self.deformable_attention.num_points}) must be divisible by depth levels ({num_depth_levels}). "
            f"This is required for proper reshaping in deformable attention. Consider adjusting num_points in config."
        )

        # Every encoder layer in a forward shares one plan; building it here is the standalone path.
        if rebatch_plan is None:
            rebatch_plan = build_rebatch_plan(reference_points_cam, bev_mask, self.device)

        if rebatch_plan.is_empty:
            if ENABLE_LOGGING:
                logger.warning("No valid points found in SCA, returning residual")
            return inp_residual

        rebatch_len = rebatch_plan.rebatch_len

        if ENABLE_LOGGING:
            logger.info("SCA Rebatching Start")

        assert query.dtype == ttnn.bfloat16, f"SCA rebatch gathers query, so it must be bfloat16, got {query.dtype}."

        # Fold batch and camera into row IDs to gather all valid queries in one embedding call.
        # Reference points come from the plan — they do not depend on query.
        query_rows = ttnn.reshape(
            ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT), (1, 1, bs * num_queries, self.embed_dims)
        )
        queries_batched = ttnn.reshape(
            ttnn.embedding(rebatch_plan.query_index, query_rows, layout=ttnn.TILE_LAYOUT),
            (bs * self.num_cams, rebatch_len, self.embed_dims),
        )
        reference_points_batched = rebatch_plan.reference_points_batched

        if ENABLE_LOGGING:
            logger.info("SCA Rebatching Complete")

        # A camera-first tensor has the same volume, so the reshape folds it into the wrong
        # rows instead of failing. We need to prevent this by asserting the shape is correct.
        assert value.shape[0] == bs and value.shape[1] == self.num_cams, (
            f"value {list(value.shape)} is not batch-first " f"[{bs}, {self.num_cams}, L, {self.embed_dims}]"
        )
        L = value.shape[2]
        value_reshaped = ttnn.reshape(value, (bs * self.num_cams, L, self.embed_dims))

        if ENABLE_LOGGING:
            logger.info("SCA Calling Deformable Attention")
        queries_output = self.deformable_attention(
            query=queries_batched,
            value=value_reshaped,
            reference_points=reference_points_batched,
            level_start_index=level_start_index,
            **kwargs,
        )

        if isinstance(queries_output, torch.Tensor):
            queries_output = ttnn.from_torch(
                queries_output, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

        if ENABLE_LOGGING:
            logger.info("SCA Deformable Attention Complete")

        if ENABLE_LOGGING:
            logger.info("SCA Feature Aggregation Start")

        # Accumulate camera features by query ID. The plan inverts the compaction, so this is a
        # gather per contribution slot rather than a scatter: ttnn.scatter only writes along the
        # last dimension and transposes all three operands to get there.
        #
        # Zeroing the padded rows first makes the row every unfilled slot points at the neutral
        # element of the sum. Their values are finite because the plan clamps reference points.
        contribution_rows = ttnn.reshape(
            ttnn.to_layout(ttnn.multiply(queries_output, rebatch_plan.row_mask), ttnn.ROW_MAJOR_LAYOUT),
            (1, 1, bs * self.num_cams * rebatch_len, self.embed_dims),
        )
        slots = None
        for index in rebatch_plan.inverse_index:
            gathered = ttnn.to_layout(
                ttnn.reshape(ttnn.embedding(index, contribution_rows), (bs, num_queries, self.embed_dims)),
                ttnn.TILE_LAYOUT,
            )
            slots = gathered if slots is None else ttnn.add(slots, gathered)

        if ENABLE_LOGGING:
            logger.info("SCA Feature Aggregation Complete")

        slots = ttnn.div(slots, rebatch_plan.count)

        # Output projection
        if hasattr(self.params, "output_proj") and self.params.output_proj is not None:
            slots = ttnn.to_layout(slots, ttnn.TILE_LAYOUT)
            slots = ttnn.linear(slots, self.params.output_proj.weight, bias=self.params.output_proj.bias)

        if ENABLE_LOGGING:
            logger.info("SCA Adding Residual")

        # Residual connection
        output = ttnn.add(slots, inp_residual)

        if use_signpost:
            signpost(header="TTNN SCA Forward End")

        return output

    def __call__(self, *args, **kwargs):
        """Make the class callable"""
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        """String representation for debugging"""
        return f"embed_dims={self.embed_dims}, num_cams={self.num_cams}, " f"batch_first={self.batch_first}"
