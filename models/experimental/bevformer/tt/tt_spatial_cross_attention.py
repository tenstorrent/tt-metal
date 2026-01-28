# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
        bev_mask_torch = ttnn.to_torch(bev_mask)

        indexes = []
        for i, mask_per_img in enumerate(bev_mask_torch):
            index_query_per_img = mask_per_img.sum(-1) > 0  # [B, num_queries]
            indexes.append(index_query_per_img)

        if ENABLE_LOGGING:
            logger.info(f"SCA Valid Queries: {[index.sum().item() for index in indexes]}")

        max_len = max([index.sum().max().item() for index in indexes])

        indexes_ttnn = []
        for index_torch in indexes:
            index_ttnn = ttnn.from_torch(index_torch, device=self.device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT)
            indexes_ttnn.append(index_ttnn)
        indexes = indexes_ttnn  # Replace with TTNN versions

        if max_len == 0:
            # No valid points, return original query
            if ENABLE_LOGGING:
                logger.warning("No valid points found in SCA, returning residual")
            return inp_residual

        if ENABLE_LOGGING:
            logger.info("SCA Valid Query Detection Complete")

        if ENABLE_LOGGING:
            logger.info("SCA Rebatching Start")

        # Create compact rebatched tensors to eliminate invalid query-camera pairs
        # Instead of processing all [bs, num_queries] for each camera (many of which are invalid),
        # we create compact tensors [bs, num_cams, max_len] containing only valid queries per camera
        # This significantly reduces computation in the subsequent deformable attention
        queries_rebatch = ttnn.zeros(
            (bs, self.num_cams, max_len, self.embed_dims),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        reference_points_rebatch = ttnn.zeros(
            (bs, self.num_cams, max_len, num_depth_levels, 2),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        # Fill rebatched tensors with valid queries per camera
        # TODO: Currently done on CPU, to be modified once TTNN supports required indexing ops
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):  # For each camera
                index_torch = ttnn.to_torch(index_query_per_img[j])

                # Find indices of valid queries for this camera
                valid_indices_torch = torch.nonzero(index_torch, as_tuple=False).squeeze(-1)

                if len(valid_indices_torch) > 0:
                    # Limit to max_len to ensure consistent tensor sizes across cameras
                    num_valid = min(len(valid_indices_torch), max_len)

                    query_torch = ttnn.to_torch(query)
                    ref_points_torch = ttnn.to_torch(reference_points_cam)

                    queries_rebatch_torch = ttnn.to_torch(queries_rebatch)
                    ref_rebatch_torch = ttnn.to_torch(reference_points_rebatch)

                    # Pack valid queries and their reference points into compact tensors
                    # Original query[j, valid_indices] -> rebatched[j, camera_i, 0:num_valid]
                    queries_rebatch_torch[j, i, :num_valid] = query_torch[j, valid_indices_torch[:num_valid]]
                    ref_rebatch_torch[j, i, :num_valid] = ref_points_torch[i, j, valid_indices_torch[:num_valid]]

                    queries_rebatch = ttnn.from_torch(
                        queries_rebatch_torch, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                    )
                    reference_points_rebatch = ttnn.from_torch(
                        ref_rebatch_torch, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                    )

        if ENABLE_LOGGING:
            logger.info("SCA Rebatching Complete")

        slots = ttnn.zeros_like(query)

        num_cams, L, bs_key, embed_dims_key = key.shape

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

        # [bs, num_cams, max_len, embed_dims] -> [bs * num_cams, max_len, embed_dims]
        queries_batched = ttnn.reshape(queries_rebatch, (bs * self.num_cams, max_len, self.embed_dims))

        # MSDA expects: [bs, num_queries, num_points_in_pillar, 2] where num_points_in_pillar = depth levels
        # [bs, num_cams, max_len, num_depth_levels, 2] -> [bs * num_cams, max_len, num_depth_levels, 2]
        reference_points_batched = ttnn.reshape(
            reference_points_rebatch, (bs * self.num_cams, max_len, num_depth_levels, 2)
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

        # Reshape deformable attention output back to per-camera format
        queries_output = ttnn.reshape(queries_output, (bs, self.num_cams, max_len, self.embed_dims))

        # Aggregate features back to original query positions
        # We need to reverse the rebatching: from compact [bs, num_cams, max_len] back to [bs, num_queries]
        # Each query accumulates features from all cameras where it has valid projections
        # TODO: Currently done on CPU, to be modified once TTNN supports required indexing ops
        slots_torch = ttnn.to_torch(slots)
        queries_output_torch = ttnn.to_torch(queries_output)

        for j in range(bs):  # For each batch item
            for i, index_query_per_img in enumerate(indexes):  # For each camera
                index_torch = ttnn.to_torch(index_query_per_img[j])  # Valid queries mask for this batch-camera pair
                valid_indices = torch.nonzero(index_torch, as_tuple=False).squeeze(-1)

                if len(valid_indices) > 0:
                    num_valid = min(len(valid_indices), max_len)
                    # Accumulate features: rebatched[j, camera_i, 0:num_valid] -> original[j, valid_indices]
                    # Each query gets contributions from all cameras where it's valid (multi-view aggregation)
                    slots_torch[j, valid_indices[:num_valid]] += queries_output_torch[j, i, :num_valid]

        slots = ttnn.from_torch(slots_torch, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        if ENABLE_LOGGING:
            logger.info("SCA Feature Aggregation Complete")

        # Count how many cameras contributed valid features for each query
        # Since queries accumulate features from multiple cameras, we need to normalize by the number of contributors
        count = bev_mask_torch.sum(-1) > 0  # Check validity per camera: [num_cams, B, num_queries]
        count = count.permute(1, 2, 0).sum(-1)  # Sum across cameras: [B, num_queries]
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
