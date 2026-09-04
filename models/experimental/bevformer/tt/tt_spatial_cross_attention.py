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


def _index_dtype(num_rows: int):
    """Use the narrowest scatter index that covers all rows."""
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

        self.deformable_attention = TTMSDeformableAttention(deform_config, device, params)

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
            reference_points_cam: Camera projected reference points [num_cams, B, num_queries, D, 2].
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

        # Clamp reference points between -10 and 10 to avoid NaNs
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
        num_depth_levels = reference_points_cam.shape[3]

        # Validate sampling points divisibility to prevent runtime errors in deformable attention
        assert self.deformable_attention.num_points % num_depth_levels == 0, (
            f"num_points ({self.deformable_attention.num_points}) must be divisible by depth levels ({num_depth_levels}). "
            f"This is required for proper reshaping in deformable attention. Consider adjusting num_points in config."
        )

        # Find valid queries for each camera
        # Many BEV queries don't have valid projections to all cameras (due to occlusion, field of view, etc.)
        # max_len must be a Python int, so only mask processing remains on the host.
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

        # Tile alignment keeps camera-axis merges as views. Padding is discarded after attention.
        rebatch_len = ((max_len + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE

        if ENABLE_LOGGING:
            logger.info("SCA Valid Query Detection Complete")

        if ENABLE_LOGGING:
            logger.info("SCA Rebatching Start")

        # IDs stay batch-local. num_queries marks padding: gathers clamp it, while scatter writes
        # it to the extra row discarded below.
        query_ids = torch.full((bs, self.num_cams, rebatch_len), num_queries, dtype=torch.int32)
        for j in range(bs):
            for i in range(self.num_cams):
                valid_indices = torch.nonzero(valid_per_cam[i, j], as_tuple=False).squeeze(-1)
                query_ids[j, i, : valid_indices.numel()] = valid_indices.to(torch.int32)
        gather_ids = query_ids.clamp(max=num_queries - 1)

        # embedding requires bfloat16; scatter_add must use the same dtype.
        assert query.dtype == ttnn.bfloat16 and reference_points_cam.dtype == ttnn.bfloat16, (
            f"SCA rebatch requires bfloat16 inputs, got query={query.dtype} and "
            f"reference_points_cam={reference_points_cam.dtype}."
        )

        # Fold batch and camera into row IDs to gather all valid queries in one embedding call.
        query_rows = ttnn.reshape(
            ttnn.to_layout(query, ttnn.ROW_MAJOR_LAYOUT), (1, 1, bs * num_queries, self.embed_dims)
        )
        query_index = _flat_row_index(gather_ids + _batch_offsets(bs, num_queries), self.device)
        queries_batched = ttnn.reshape(
            ttnn.embedding(query_index, query_rows, layout=ttnn.TILE_LAYOUT),
            (bs * self.num_cams, rebatch_len, self.embed_dims),
        )

        # Reference points are camera-major, so their row IDs include camera and batch offsets.
        ref_rows = ttnn.reshape(
            ttnn.to_layout(reference_points_cam, ttnn.ROW_MAJOR_LAYOUT),
            (1, 1, self.num_cams * bs * num_queries, num_depth_levels * 2),
        )
        ref_index = _flat_row_index(
            gather_ids
            + _batch_offsets(bs, num_queries)
            + torch.arange(self.num_cams, dtype=torch.int32).reshape(1, self.num_cams, 1) * (bs * num_queries),
            self.device,
        )
        # Split the last dimension before tilizing to avoid a re-layout.
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

        # Accumulate all camera features by query ID. Padding lands in an extra row removed below.
        scatter_src = ttnn.to_layout(
            ttnn.reshape(queries_output, (bs, self.num_cams * rebatch_len, self.embed_dims)),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        # Expand row IDs on device to avoid transferring a full-width index.
        scatter_index = ttnn.repeat(
            ttnn.from_torch(
                query_ids.reshape(bs, self.num_cams * rebatch_len, 1),
                device=self.device,
                dtype=_index_dtype(num_queries + 1),
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            ttnn.Shape((1, 1, self.embed_dims)),
        )
        slots = ttnn.scatter_add(
            ttnn.zeros(
                (bs, num_queries + 1, self.embed_dims),
                device=self.device,
                dtype=queries_output.dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            dim=1,
            index=scatter_index,
            src=scatter_src,
        )
        slots = ttnn.to_layout(ttnn.slice(slots, (0, 0, 0), (bs, num_queries, self.embed_dims)), ttnn.TILE_LAYOUT)

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
