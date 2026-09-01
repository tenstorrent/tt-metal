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


def _round_up_to_tile(n: int) -> int:
    return ((n + 31) // 32) * 32


def _fold_cameras_into_batch(tensor, bs: int, num_cams: int, seq_len: int, embed_dims: int):
    """``[num_cams, seq_len, bs, embed_dims]`` -> ``[bs * num_cams, seq_len, embed_dims]``.

    Done in ROW_MAJOR. ``bs`` arrives second-to-last, so a tiled input pads it to a full tile — 32x
    the buffer at bs=1, on the largest tensor in the layer. The folded shape is tile-clean, so
    converting after the fold is the cheap direction; converting before it is not.
    """
    folded = ttnn.permute(ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT), (2, 0, 1, 3))
    folded = ttnn.reshape(folded, (bs * num_cams, seq_len, embed_dims))
    return ttnn.to_layout(folded, ttnn.TILE_LAYOUT)


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

        # ``embedding_bw`` reads its weight argument for one number, the row count, and never
        # touches the data. Kept per shape so the allocation happens once instead of per call.
        self._slot_tables = {}

    def _slot_table(self, bs: int, slot_stride: int, dtype):
        key = (bs, slot_stride, dtype)
        if key not in self._slot_tables:
            self._slot_tables[key] = ttnn.zeros(
                (1, 1, bs * slot_stride, self.embed_dims),
                device=self.device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
            )
        return self._slot_tables[key]

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
        **kwargs,
    ):
        """
        Forward pass of TTNN Spatial Cross Attention.

        Args:
            query: BEV queries [B, num_queries, embed_dims].
            reference_points_cam: Camera projected reference points, ROW_MAJOR,
                [num_cams, B, num_queries, D * 2].
            bev_mask: Valid mask for camera projections [num_cams, B, num_queries, D].
            key: Multi-camera features [num_cams, H*W, B, embed_dims].
            value: Same as key.
            residual: Residual connection input.
            query_pos: Query positional encoding.
            key_padding_mask: Key padding mask.
            spatial_shapes: Spatial shapes of multi-scale features.
            level_start_index: Start index of each level.
            **kwargs: Additional arguments.

        Returns:
            Output features [B, num_queries, embed_dims].
        """
        if use_signpost:
            signpost(header="TTNN SCA Forward Start")

        # Clamped here rather than in point sampling, which returns the points as computed. On the
        # flat form the trailing dims are (num_queries, num_points * 2), so this pays 4x of tile
        # padding where the spelled-out (num_points, 2) tail paid 128x.
        reference_points_cam = ttnn.clamp(reference_points_cam, -10.0, 10.0)

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
        num_depth_levels = reference_points_cam.shape[3] // 2

        # Validate sampling points divisibility to prevent runtime errors in deformable attention
        assert self.deformable_attention.num_points % num_depth_levels == 0, (
            f"num_points ({self.deformable_attention.num_points}) must be divisible by depth levels ({num_depth_levels}). "
            f"This is required for proper reshaping in deformable attention. Consider adjusting num_points in config."
        )

        # Find valid queries for each camera
        # Many BEV queries don't have valid projections to all cameras (due to occlusion, field of view, etc.)
        #
        # ``max_len`` sizes the rebatched tensors, so it must be a Python int; that is what forces
        # this host readback of the mask. The rebatch and scatter-back themselves run on device,
        # driven by the small index tensors derived here.
        bev_mask_torch = ttnn.to_torch(bev_mask)

        valid_per_cam = bev_mask_torch.sum(-1) > 0  # [num_cams, B, num_queries]
        max_len = int(valid_per_cam.sum(-1).max().item())

        if ENABLE_LOGGING:
            logger.info(f"SCA Valid Queries: {valid_per_cam.sum(-1).flatten().tolist()}")

        if max_len == 0:
            # No valid points, return original query
            if ENABLE_LOGGING:
                logger.warning("No valid points found in SCA, returning residual")
            return inp_residual

        # Rounded up so that merging and splitting (num_cams, rebatch_len) below stays a view on a
        # tiled tensor rather than a re-layout. The extra slots cost deformable-attention compute
        # and nothing else.
        #
        # Note this is the max over (camera, batch item) pairs, where the torch reference maxes over
        # cameras only, collapsing the batch into the count. For bs > 1 the reference allocates the
        # larger tensor; the extra rows there are zero-padded and never scattered back, so both
        # produce the same result from different shapes.
        rebatch_len = ((max_len + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE

        if ENABLE_LOGGING:
            logger.info("SCA Valid Query Detection Complete")

        if ENABLE_LOGGING:
            logger.info("SCA Rebatching Start")

        # Query ids of the valid queries per (batch, camera), padded to rebatch_len.
        #
        # Two invariants the consumers below depend on:
        #   * ids are **local** to one batch item, i.e. in [0, num_queries). Each gather adds the
        #     offset for its own flattening; the scatter adds none, because it scatters along the
        #     query axis and so is already per-batch-item.
        #   * padded slots hold ``num_queries``, one past the end. That is a real row id for
        #     neither gather, so they are clamped to 0 there and produce a discarded result, and it
        #     is the sentinel row for the scatter, which is dropped. Nothing keys correctness on a
        #     padded slot's value.
        query_ids = torch.full((bs, self.num_cams, rebatch_len), num_queries, dtype=torch.int32)
        for j in range(bs):
            for i in range(self.num_cams):
                valid_indices = torch.nonzero(valid_per_cam[i, j], as_tuple=False).squeeze(-1)
                query_ids[j, i, : valid_indices.numel()] = valid_indices.to(torch.int32)
        gather_ids = query_ids.clamp(max=num_queries - 1)

        # ttnn.embedding takes its table in bfloat16 only, and ttnn.scatter_add requires the
        # accumulator and the source to agree, so the whole rebatch path is pinned to bfloat16.
        # Asserted here because the failure is otherwise a TT_FATAL from inside the device op.
        assert query.dtype == ttnn.bfloat16 and reference_points_cam.dtype == ttnn.bfloat16, (
            f"SCA rebatch requires bfloat16 inputs, got query={query.dtype} and "
            f"reference_points_cam={reference_points_cam.dtype}."
        )

        # Create compact rebatched tensors to eliminate invalid query-camera pairs
        # Instead of processing all [bs, num_queries] for each camera (many of which are invalid),
        # we create compact tensors [bs, num_cams, rebatch_len] containing only valid queries per
        # camera. This significantly reduces computation in the subsequent deformable attention.
        #
        # Both rebatches are row gathers, and ``embedding`` takes one index per output row rather
        # than one per element. Folding every leading dimension into the row id lets each run as a
        # single call over all batch items and cameras; per-camera calls would need a concat to
        # stack them, and concat at these shapes costs more than the gather it serves.
        query_rows = ttnn.reshape(
            ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT), (1, 1, bs * num_queries, self.embed_dims)
        )
        query_index = _flat_row_index(gather_ids + _batch_offsets(bs, num_queries), self.device)
        queries_batched = ttnn.reshape(
            ttnn.embedding(query_index, query_rows, layout=ttnn.TILE_LAYOUT),
            (bs * self.num_cams, rebatch_len, self.embed_dims),
        )

        # query is [bs, num_queries, ...] but reference_points_cam is camera-major, so its row id
        # folds in the camera as well as the batch item.
        # Already ROW_MAJOR and flat, so this only merges leading dimensions.
        ref_rows = ttnn.reshape(reference_points_cam, (1, 1, self.num_cams * bs * num_queries, num_depth_levels * 2))
        ref_index = _flat_row_index(
            gather_ids
            + _batch_offsets(bs, num_queries)
            + torch.arange(self.num_cams, dtype=torch.int32).reshape(1, self.num_cams, 1) * (bs * num_queries),
            self.device,
        )
        # Reshaped before tilizing: splitting the last dimension 8 -> (4, 2) on a tiled tensor is a
        # re-layout, and the intermediate pads those 8 columns to a full tile on the way.
        reference_points_batched = ttnn.to_layout(
            ttnn.reshape(
                ttnn.embedding(ref_index, ref_rows),
                (bs * self.num_cams, rebatch_len, num_depth_levels, 2),
            ),
            ttnn.TILE_LAYOUT,
        )

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

        # Callers routinely pass one tensor as both key and value; the fold is the most expensive
        # reshape in the layer, so it is not worth doing twice for the same buffer.
        key_reshaped = _fold_cameras_into_batch(key, bs, self.num_cams, L, self.embed_dims)
        value_reshaped = (
            key_reshaped if value is key else _fold_cameras_into_batch(value, bs, self.num_cams, L, self.embed_dims)
        )

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
        # ``embedding_bw`` is the row-wise scatter-add: it accumulates one grad row per index, which
        # is what this is. ``scatter_add`` takes an index per *element*, so it needs the row id
        # widened across embed_dims, and it transposes input, index and source to put the scatter
        # axis last and transposes the result back — four permutes and a repeat that this does not
        # need. It is the same asymmetry that made ``embedding`` beat ``gather`` on the rebatch.
        slot_stride = _round_up_to_tile(num_queries + 1)
        scatter_grad = ttnn.reshape(queries_output, (1, 1, bs * self.num_cams * rebatch_len, self.embed_dims))
        scatter_index = _flat_row_index(
            query_ids + (torch.arange(bs, dtype=torch.int32) * slot_stride).reshape(bs, 1, 1), self.device
        )
        slots = ttnn.embedding_bw(scatter_index, self._slot_table(bs, slot_stride, queries_output.dtype), scatter_grad)
        # The table is padded to a tile boundary per batch item so this split lands on one.
        slots = ttnn.slice(
            ttnn.reshape(slots, (bs, slot_stride, self.embed_dims)),
            (0, 0, 0),
            (bs, num_queries, self.embed_dims),
        )

        if ENABLE_LOGGING:
            logger.info("SCA Feature Aggregation Complete")

        # Count how many cameras contributed valid features for each query
        # Since queries accumulate features from multiple cameras, we need to normalize by the number of contributors
        count = valid_per_cam.permute(1, 2, 0).sum(-1)  # Sum across cameras: [B, num_queries]
        count = torch.clamp(count, min=1.0)

        count_ttnn = ttnn.from_torch(count, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        count_expanded = ttnn.unsqueeze(count_ttnn, -1)  # [bs, num_queries, 1]

        # Normalize accumulated features by the number of contributing cameras
        # This gives us the average feature across all valid camera views for each query
        slots = ttnn.div(slots, count_expanded)

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
