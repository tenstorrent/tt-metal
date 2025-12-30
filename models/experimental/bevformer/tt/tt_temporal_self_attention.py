# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Temporal Self Attention (TSA) module for BEVFormer.

This module implements the temporal self-attention mechanism using TTNN operations
to enable BEV features to model temporal dependencies across different timesteps.
It uses deformable attention to aggregate information from current and
historical BEV features.

Based on the reference PyTorch implementation but optimized for TTNN.
"""

import ttnn
import torch
import warnings

# Handle imports for both relative and absolute import contexts
try:
    from .tt_ms_deformable_attention import TTMSDeformableAttention
    from ..config import DeformableAttentionConfig
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    bevformer_path = Path(__file__).parent.parent
    sys.path.insert(0, str(bevformer_path))
    from tt_ms_deformable_attention import TTMSDeformableAttention
    from config import DeformableAttentionConfig

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class TTTemporalSelfAttention:
    """
    TTNN Temporal Self Attention module for BEVFormer.

    This attention mechanism models temporal dependencies by allowing BEV queries
    to attend to both current and historical BEV features using deformable attention.
    It's designed to handle temporal relationships for object tracking and motion
    understanding in autonomous driving scenarios.

    Args:
        device: TTNN device for computation
        params: Parameter dict containing weights and biases
        embed_dims (int): The embedding dimension.
        num_heads (int): Number of attention heads.
        num_levels (int): Number of feature levels.
        num_points (int): Number of sampling points in deformable attention.
        num_bev_queue (int): Number of BEV timesteps (typically 2: current + history).
        batch_first (bool): Whether the first dimension of input is batch_size.
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        device,
        params,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
        num_bev_queue: int = 2,
        batch_first: bool = True,
        **kwargs,
    ):
        self.device = device
        self.params = params

        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")

        dim_per_head = embed_dims // num_heads
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.batch_first = batch_first

        # Check if dim_per_head is power of 2 for efficiency
        if not self._is_power_of_2(dim_per_head):
            warnings.warn(
                "For optimal performance with TTNN, embed_dims should be set "
                "so that dimension of each attention head is a power of 2"
            )

        # Initialize TTNN MSDeformableAttention using existing implementation
        deform_config = DeformableAttentionConfig(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            batch_first=batch_first,
        )

        # Initialize TTNN MSDeformableAttention using existing implementation
        # Pass params directly since TSA doesn't have its own parameters beyond deformable attention
        self.deformable_attention = TTMSDeformableAttention(
            deform_config, device, self.params  # Pass params directly to deformable attention
        )

    def forward(
        self,
        query,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        prev_bev=None,
        **kwargs,
    ):
        """
        Forward pass of TTNN Temporal Self Attention.

        Args:
            query: Current BEV queries [B, num_query, embed_dims].
            value: Temporal BEV features [B*num_bev_queue, num_query, embed_dims].
                If None, will be constructed from current query and prev_bev.
            identity: Identity connection input.
            query_pos: Query positional encoding.
            key_padding_mask: Key padding mask.
            reference_points: Reference points for deformable attention.
            spatial_shapes: Spatial shapes of BEV features.
            level_start_index: Start index of each level.
            prev_bev: Previous BEV features [B, num_query, embed_dims].
            **kwargs: Additional arguments.

        Returns:
            Output features [B, num_query, embed_dims].
        """
        if use_signpost:
            signpost(header="TTNN TSA Forward Start")

        # Convert torch tensors to ttnn tensors if needed
        if isinstance(query, torch.Tensor):
            query = ttnn.from_torch(query, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if prev_bev is not None and isinstance(prev_bev, torch.Tensor):
            prev_bev = ttnn.from_torch(prev_bev, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if identity is not None and isinstance(identity, torch.Tensor):
            identity = ttnn.from_torch(identity, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        if reference_points is not None and isinstance(reference_points, torch.Tensor):
            reference_points = ttnn.from_torch(
                reference_points, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

        # Handle defaults
        if identity is None:
            identity = query

        # Add query positional encoding
        if query_pos is not None:
            if isinstance(query_pos, torch.Tensor):
                query_pos = ttnn.from_torch(query_pos, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            query = ttnn.add(query, query_pos)

        bs, num_query, _ = query.shape

        # Create temporal value features exactly
        if value is None:
            # For simplified version, just use query as value (no temporal information)
            value = query
        else:
            # Value already provided in the expected format
            if isinstance(value, torch.Tensor):
                value = ttnn.from_torch(value, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Use reference points as-is for simplified version
        ref_points = reference_points

        # Prepare level start index if not provided
        if level_start_index is None:
            level_start_index = torch.tensor([0], dtype=torch.long)

        # Convert level_start_index to ttnn if needed
        if isinstance(level_start_index, torch.Tensor):
            level_start_index = ttnn.from_torch(
                level_start_index, device=self.device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        # Apply deformable attention with integrated temporal processing
        output = self.deformable_attention(
            query=query,
            value=value,
            reference_points=ref_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )

        # Convert back to ttnn tensor if needed
        if isinstance(output, torch.Tensor):
            output = ttnn.from_torch(output, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Residual connection
        output = ttnn.add(output, identity)

        if use_signpost:
            signpost(header="TTNN TSA Forward End")

        return output

    @staticmethod
    def _is_power_of_2(n: int) -> bool:
        """Check if a number is a power of 2."""
        return (n != 0) and (n & (n - 1) == 0)

    def __call__(self, *args, **kwargs):
        """Make the class callable"""
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        """String representation for debugging"""
        return (
            f"embed_dims={self.embed_dims}, num_heads={self.num_heads}, "
            f"num_levels={self.num_levels}, num_points={self.num_points}, "
            f"num_bev_queue={self.num_bev_queue}, batch_first={self.batch_first}"
        )
