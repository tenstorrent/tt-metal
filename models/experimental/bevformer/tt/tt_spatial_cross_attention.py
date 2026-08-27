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
    """Offset that turns a per-batch-item query id into an id over ``bs`` stacked query blocks."""
    return (torch.arange(bs, dtype=torch.int32) * num_queries).reshape(bs, 1, 1)


def _flat_row_index(row_ids: torch.Tensor, device) -> ttnn.Tensor:
    """Row ids as the index ``ttnn.embedding`` wants: flat, uint32, leading dims 1."""
    return ttnn.from_torch(
        row_ids.reshape(1, 1, 1, row_ids.numel()),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


@dataclass(frozen=True)
class SCARebatchPlan:
    """Everything the rebatch needs that does not depend on ``query``.

    ``bev_mask`` and ``reference_points_cam`` come from the camera projection, so they change per
    frame but are identical for every encoder layer within one. The whole reference-point rebatch
    falls out of them without touching ``query`` at all, and so does the mask readback that decides
    ``rebatch_len``. Building this once per forward is what keeps the readback off the layer loop.
    """

    rebatch_len: int
    query_index: ttnn.Tensor
    reference_points_batched: ttnn.Tensor
    scatter_index: ttnn.Tensor
    count: ttnn.Tensor
    is_empty: bool = False


def build_rebatch_plan(reference_points_cam, bev_mask, embed_dims: int, device) -> SCARebatchPlan:
    """Resolve the per-frame rebatch geometry. See :class:`SCARebatchPlan`."""
    num_cams, bs, num_queries, num_depth_levels = bev_mask.shape

    # ``rebatch_len`` sizes the rebatched tensors, so it must be a Python int; that is what forces
    # this host readback of the mask.
    valid_per_cam = ttnn.to_torch(bev_mask).sum(-1) > 0  # [num_cams, B, num_queries]
    max_len = int(valid_per_cam.sum(-1).max().item())

    if ENABLE_LOGGING:
        logger.info(f"SCA Valid Queries: {valid_per_cam.sum(-1).flatten().tolist()}")

    if max_len == 0:
        return SCARebatchPlan(0, None, None, None, None, is_empty=True)

    # Rounded up so that merging and splitting (num_cams, rebatch_len) stays a view on a tiled
    # tensor rather than a re-layout. The extra slots cost deformable-attention compute and nothing
    # else.
    #
    # Note this is the max over (camera, batch item) pairs, where the torch reference maxes over
    # cameras only, collapsing the batch into the count. For bs > 1 the reference allocates the
    # larger tensor; the extra rows there are zero-padded and never scattered back, so both produce
    # the same result from different shapes.
    rebatch_len = ((max_len + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE

    # Query ids of the valid queries per (batch, camera), padded to rebatch_len.
    #
    # Two invariants the consumers depend on:
    #   * ids are **local** to one batch item, i.e. in [0, num_queries). Each gather adds the offset
    #     for its own flattening; the scatter adds none, because it scatters along the query axis
    #     and so is already per-batch-item.
    #   * padded slots hold ``num_queries``, one past the end. That is a real row id for neither
    #     gather, so they are clamped to 0 there and produce a discarded result, and it is the
    #     sentinel row for the scatter, which is dropped. Nothing keys correctness on a padded
    #     slot's value.
    query_ids = torch.full((bs, num_cams, rebatch_len), num_queries, dtype=torch.int32)
    for j in range(bs):
        for i in range(num_cams):
            valid_indices = torch.nonzero(valid_per_cam[i, j], as_tuple=False).squeeze(-1)
            query_ids[j, i, : valid_indices.numel()] = valid_indices.to(torch.int32)
    gather_ids = query_ids.clamp(max=num_queries - 1)

    reference_points_cam = ttnn.clamp(reference_points_cam, -10.0, 10.0)
    assert (
        reference_points_cam.dtype == ttnn.bfloat16
    ), f"SCA rebatch requires bfloat16 reference points, got {reference_points_cam.dtype}."

    # query is [bs, num_queries, ...] but reference_points_cam is camera-major, so its row id folds
    # in the camera as well as the batch item. Reshaped before tilizing: splitting the last
    # dimension 8 -> (4, 2) on a tiled tensor is a re-layout, and the intermediate pads those 8
    # columns to a full tile on the way.
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
    reference_points_batched = ttnn.to_layout(
        ttnn.reshape(
            ttnn.embedding(ref_index, ref_rows),
            (bs * num_cams, rebatch_len, num_depth_levels, 2),
        ),
        ttnn.TILE_LAYOUT,
    )

    # Uploaded as one id per row and widened on device; materializing the embed_dims copies on host
    # would put a multi-megabyte index on the bus.
    scatter_index = ttnn.repeat(
        ttnn.from_torch(
            query_ids.reshape(bs, num_cams * rebatch_len, 1),
            device=device,
            dtype=_index_dtype(num_queries + 1),
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        ttnn.Shape((1, 1, embed_dims)),
    )

    # How many cameras contributed to each query, for the averaging that follows the scatter.
    count = torch.clamp(valid_per_cam.permute(1, 2, 0).sum(-1), min=1.0)
    count = ttnn.unsqueeze(ttnn.from_torch(count, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), -1)

    return SCARebatchPlan(
        rebatch_len=rebatch_len,
        query_index=_flat_row_index(gather_ids + _batch_offsets(bs, num_queries), device),
        reference_points_batched=reference_points_batched,
        scatter_index=scatter_index,
        count=count,
    )


def _index_dtype(num_rows: int):
    """Narrowest index dtype ``ttnn.scatter_add`` accepts that still addresses ``num_rows``.

    The scatter index is widened across embed_dims on device, so it is the largest tensor in the
    aggregation — the width choice is worth more than it looks.
    """
    return ttnn.uint16 if num_rows <= 0xFFFF else ttnn.uint32


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
        self.deformable_attention = TTMSDeformableAttention(deform_config, device, params.deformable_attention)

    def forward(
        self,
        query,
        reference_points_cam,
        bev_mask,
        key=None,
        value=None,
        residual=None,
        query_pos=None,
        key_padding_mask=None,
        spatial_shapes=None,
        level_start_index=None,
        rebatch_plan=None,
        **kwargs,
    ):
        """
        Forward pass of TTNN Spatial Cross Attention.

        Args:
            query: BEV queries [B, num_queries, embed_dims].
            reference_points_cam: Camera projected reference points [num_cams, B, num_queries, D, 2].
            bev_mask: Valid mask for camera projections [num_cams, B, num_queries, D].
            key: Multi-camera features [num_cams, H*W, B, embed_dims].
            value: Same as key.
            residual: Residual connection input.
            query_pos: Query positional encoding.
            key_padding_mask: Key padding mask.
            spatial_shapes: Spatial shapes of multi-scale features.
            level_start_index: Start index of each level.
            rebatch_plan: Prebuilt :class:`SCARebatchPlan`. Every encoder layer in a forward shares
                one; built here when absent, which is the standalone path.
            **kwargs: Additional arguments.

        Returns:
            Output features [B, num_queries, embed_dims].
        """
        if use_signpost:
            signpost(header="TTNN SCA Forward Start")

        # Handle input defaults
        if key is None:
            key = ttnn.clone(query)
        if value is None:
            value = ttnn.clone(key)
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
            rebatch_plan = build_rebatch_plan(reference_points_cam, bev_mask, self.embed_dims, self.device)

        if rebatch_plan.is_empty:
            # No valid points, return original query
            if ENABLE_LOGGING:
                logger.warning("No valid points found in SCA, returning residual")
            return inp_residual

        rebatch_len = rebatch_plan.rebatch_len

        if ENABLE_LOGGING:
            logger.info("SCA Rebatching Start")

        # ttnn.embedding takes its table in bfloat16 only, and ttnn.scatter_add requires the
        # accumulator and the source to agree, so the whole rebatch path is pinned to bfloat16.
        # Asserted here because the failure is otherwise a TT_FATAL from inside the device op.
        assert query.dtype == ttnn.bfloat16, f"SCA rebatch requires a bfloat16 query, got {query.dtype}."

        # Create compact rebatched tensors to eliminate invalid query-camera pairs
        # Instead of processing all [bs, num_queries] for each camera (many of which are invalid),
        # we create compact tensors [bs, num_cams, rebatch_len] containing only valid queries per
        # camera. This significantly reduces computation in the subsequent deformable attention.
        #
        # The rebatch is a row gather, and ``embedding`` takes one index per output row rather than
        # one per element. Folding every leading dimension into the row id lets it run as a single
        # call over all batch items and cameras; per-camera calls would need a concat to stack them,
        # and concat at these shapes costs more than the gather it serves.
        #
        # Only the queries are gathered here. The reference points come from the plan — they are the
        # same for every layer, because nothing about them depends on ``query``.
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

        _, L, _, _ = key.shape

        # Validate spatial shapes consistency to prevent incorrect sampling locations
        if spatial_shapes is not None:
            if isinstance(spatial_shapes, ttnn.Tensor):
                spatial_shapes_torch = ttnn.to_torch(spatial_shapes)
            else:
                spatial_shapes_torch = spatial_shapes
            expected_L = spatial_shapes_torch.prod(dim=1).sum().item()
            assert expected_L == L, (
                f"Spatial shapes mismatch: spatial_shapes total ({expected_L}) != key spatial dimension ({L}). "
                f"spatial_shapes: {spatial_shapes_torch.tolist()}, key.shape: {key.shape}"
            )

        # [num_cams, L, bs, embed_dims] -> [bs * num_cams, L, embed_dims]
        key_reshaped = ttnn.permute(key, (2, 0, 1, 3))  # [bs, num_cams, L, embed_dims]
        key_reshaped = ttnn.reshape(key_reshaped, (bs * self.num_cams, L, self.embed_dims))
        value_reshaped = ttnn.permute(value, (2, 0, 1, 3))  # [bs, num_cams, L, embed_dims]
        value_reshaped = ttnn.reshape(value_reshaped, (bs * self.num_cams, L, self.embed_dims))

        if ENABLE_LOGGING:
            logger.info("SCA Calling Deformable Attention")
        queries_output = self.deformable_attention(
            query=queries_batched,
            key=key_reshaped,
            value=value_reshaped,
            reference_points=reference_points_batched,
            spatial_shapes=spatial_shapes,
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

        # Aggregate features back to original query positions
        # We need to reverse the rebatching: from compact [bs, num_cams, rebatch_len] back to
        # [bs, num_queries]. Each query accumulates features from all cameras where it has valid
        # projections, which is what scatter_add does with repeated indices — so all cameras go in
        # one call rather than one per camera.
        #
        # The scatter target carries one row past num_queries. Padded slots address that sentinel
        # row and it is sliced off, so whatever deformable attention produced for them is discarded
        # without their value having to be zero.
        scatter_src = ttnn.to_layout(
            ttnn.reshape(queries_output, (bs, self.num_cams * rebatch_len, self.embed_dims)),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        slots = ttnn.scatter_add(
            ttnn.zeros(
                (bs, num_queries + 1, self.embed_dims),
                device=self.device,
                dtype=queries_output.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            dim=1,
            index=rebatch_plan.scatter_index,
            src=scatter_src,
        )
        slots = ttnn.to_layout(ttnn.slice(slots, (0, 0, 0), (bs, num_queries, self.embed_dims)), ttnn.TILE_LAYOUT)

        if ENABLE_LOGGING:
            logger.info("SCA Feature Aggregation Complete")

        # Normalize accumulated features by the number of contributing cameras
        # This gives us the average feature across all valid camera views for each query
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
