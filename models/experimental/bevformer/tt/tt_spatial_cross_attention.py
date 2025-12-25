# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
        dropout (float): Dropout rate.
        batch_first (bool): Whether the first dimension of input is batch_size.
        deformable_attention (dict): Config for MSDeformableAttention3D.
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        device,
        params,
        embed_dims: int = 256,
        num_cams: int = 6,
        dropout: float = 0.1,
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
            num_levels=4,
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
            query: BEV queries [B, num_query, embed_dims].
            reference_points_cam: Camera projected reference points [num_cams, B, num_query, D, 2].
            bev_mask: Valid mask for camera projections [num_cams, B, num_query, D].
            key: Multi-camera features [num_cams, H*W, B, embed_dims].
            value: Same as key.
            residual: Residual connection input.
            query_pos: Query positional encoding.
            key_padding_mask: Key padding mask.
            spatial_shapes: Spatial shapes of multi-scale features.
            level_start_index: Start index of each level.
            **kwargs: Additional arguments.

        Returns:
            Output features [B, num_query, embed_dims].
        """
        if use_signpost:
            signpost(header="TTNN SCA Forward Start")

        # Convert torch tensors to ttnn tensors if needed
        if isinstance(query, torch.Tensor):
            query = ttnn.from_torch(query, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if key is not None and isinstance(key, torch.Tensor):
            key = ttnn.from_torch(key, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if value is not None and isinstance(value, torch.Tensor):
            value = ttnn.from_torch(value, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if residual is not None and isinstance(residual, torch.Tensor):
            residual = ttnn.from_torch(residual, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if reference_points_cam is not None and isinstance(reference_points_cam, torch.Tensor):
            reference_points_cam = ttnn.from_torch(
                reference_points_cam, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
        if bev_mask is not None and isinstance(bev_mask, torch.Tensor):
            bev_mask = ttnn.from_torch(bev_mask, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Handle input defaults
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            inp_residual = query
        else:
            inp_residual = residual

        # Add query positional encoding
        if query_pos is not None:
            if isinstance(query_pos, torch.Tensor):
                query_pos = ttnn.from_torch(query_pos, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            query = ttnn.add(query, query_pos)

        bs, num_query, _ = query.shape
        # Extract number of depth levels for 3D point sampling
        # Each BEV query samples points at multiple Z-coordinates (depth levels) in 3D space
        num_depth_levels = reference_points_cam.shape[3]

        # Find valid queries for each camera (EXACT torch logic)
        # Convert to torch to ensure exact same computation as reference
        bev_mask_torch = ttnn.to_torch(bev_mask)

        indexes = []
        for i, mask_per_img in enumerate(bev_mask_torch):
            index_query_per_img = mask_per_img.sum(-1) > 0  # [B, num_query]
            indexes.append(index_query_per_img)

        max_len = max([index.sum().max().item() for index in indexes])

        indexes_ttnn = []
        for index_torch in indexes:
            index_ttnn = ttnn.from_torch(index_torch, device=self.device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT)
            indexes_ttnn.append(index_ttnn)
        indexes = indexes_ttnn  # Replace with TTNN versions

        if max_len == 0:
            # No valid points, return original query
            logger.warning("No valid points found in SCA, returning residual")
            return inp_residual

        # Create rebatched tensors
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

        # Fill rebatched tensors with valid queries
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                index_torch = ttnn.to_torch(index_query_per_img[j])  # [num_query]

                valid_indices_torch = torch.nonzero(index_torch, as_tuple=False).squeeze(-1)

                if len(valid_indices_torch) > 0:
                    num_valid = min(len(valid_indices_torch), max_len)

                    query_torch = ttnn.to_torch(query)
                    ref_points_torch = ttnn.to_torch(reference_points_cam)

                    queries_rebatch_torch = ttnn.to_torch(queries_rebatch)
                    ref_rebatch_torch = ttnn.to_torch(reference_points_rebatch)

                    queries_rebatch_torch[j, i, :num_valid] = query_torch[j, valid_indices_torch[:num_valid]]
                    ref_rebatch_torch[j, i, :num_valid] = ref_points_torch[i, j, valid_indices_torch[:num_valid]]

                    queries_rebatch = ttnn.from_torch(
                        queries_rebatch_torch, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                    )
                    reference_points_rebatch = ttnn.from_torch(
                        ref_rebatch_torch, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                    )

        slots = ttnn.zeros_like(query)

        num_cams, L, bs_key, embed_dims_key = key.shape

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

        # Reshape output back to [bs, num_cams, max_len, embed_dims]
        queries_output = ttnn.reshape(queries_output, (bs, self.num_cams, max_len, self.embed_dims))

        # Aggregate features back to original query positions
        slots_torch = ttnn.to_torch(slots)
        queries_output_torch = ttnn.to_torch(queries_output)

        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                index_torch = ttnn.to_torch(index_query_per_img[j])  # [num_query]
                valid_indices = torch.nonzero(index_torch, as_tuple=False).squeeze(-1)

                if len(valid_indices) > 0:
                    num_valid = min(len(valid_indices), max_len)
                    slots_torch[j, valid_indices[:num_valid]] += queries_output_torch[j, i, :num_valid]

        slots = ttnn.from_torch(slots_torch, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Count valid queries per camera
        count = bev_mask_torch.sum(-1) > 0  # [num_cams, B, num_query]
        count = count.permute(1, 2, 0).sum(-1)  # [B, num_query]
        count = torch.clamp(count, min=1.0)

        count_ttnn = ttnn.from_torch(count, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        count_expanded = ttnn.unsqueeze(count_ttnn, -1)  # [bs, num_query, 1]

        # Average by dividing by count (exact torch logic: slots = slots / count[..., None])
        slots = ttnn.div(slots, count_expanded)

        # Output projection
        if hasattr(self.params, "output_proj") and self.params.output_proj is not None:
            slots = ttnn.to_layout(slots, ttnn.TILE_LAYOUT)
            slots = ttnn.linear(slots, self.params.output_proj.weight, bias=self.params.output_proj.bias)

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
