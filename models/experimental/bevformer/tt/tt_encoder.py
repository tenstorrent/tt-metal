# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN BEVFormer Encoder implementation.

This module implements the main BEVFormer encoder using TTNN operations that
combines spatial cross-attention and temporal self-attention to learn Bird's-Eye-View
representations from multi-camera images. The encoder processes BEV queries through
multiple transformer layers that enable both spatial feature extraction from camera
views and temporal modeling.

Based on the PyTorch reference implementation but optimized for TTNN operations.
"""

import ttnn
import torch
from typing import Optional, List, Dict, Any

# Handle imports for both relative and absolute import contexts
try:
    from .tt_spatial_cross_attention import TTSpatialCrossAttention
    from .tt_temporal_self_attention import TTTemporalSelfAttention
    from .tt_point_sampling_3d_2d import point_sampling_3d_to_2d_ttnn
    from ..reference.point_sampling_3d_2d import generate_reference_points
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    bevformer_path = Path(__file__).parent.parent
    sys.path.insert(0, str(bevformer_path))
    from tt_spatial_cross_attention import TTSpatialCrossAttention
    from tt_temporal_self_attention import TTTemporalSelfAttention
    from tt_point_sampling_3d_2d import point_sampling_3d_to_2d_ttnn
    from reference.point_sampling_3d_2d import generate_reference_points

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

from loguru import logger


class TTBEVFormerLayer:
    """
    TTNN implementation of a single BEVFormer transformer layer.

    This layer combines spatial cross-attention and temporal self-attention
    to process BEV queries using TTNN operations. Each layer performs:
    1. Temporal self-attention (optional) - models temporal dependencies
    2. Spatial cross-attention - extracts features from camera views
    3. Feed-forward network - additional processing
    4. Layer normalization after each component

    Args:
        device: TTNN device for computation
        params: Parameter dict containing weights and biases
        embed_dims (int): Feature embedding dimensions
        num_heads (int): Number of attention heads
        num_levels (int): Number of feature pyramid levels
        num_points (int): Number of sampling points in deformable attention
        num_cams (int): Number of cameras
        use_temporal_self_attention (bool): Whether to use temporal self-attention
        use_spatial_cross_attention (bool): Whether to use spatial cross-attention
        feedforward_channels (int): FFN intermediate channel size
        batch_first (bool): Whether batch dimension is first
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        device,
        params,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        num_cams: int = 6,
        use_temporal_self_attention: bool = True,
        use_spatial_cross_attention: bool = True,
        feedforward_channels: int = 1024,
        batch_first: bool = True,
        **kwargs,
    ):
        self.device = device
        self.params = params
        self.embed_dims = embed_dims
        self.use_temporal_self_attention = use_temporal_self_attention
        self.use_spatial_cross_attention = use_spatial_cross_attention
        self.batch_first = batch_first
        self.feedforward_channels = feedforward_channels

        # Temporal Self-Attention
        if use_temporal_self_attention and hasattr(params, "temporal_self_attention"):
            self.temporal_self_attention = TTTemporalSelfAttention(
                device=device,
                params=params.temporal_self_attention,
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=1,  # Single level for temporal attention
                num_points=num_points,
                batch_first=batch_first,
                **kwargs,
            )

        # Spatial Cross-Attention
        if use_spatial_cross_attention and hasattr(params, "spatial_cross_attention"):
            deform_config = {
                "embed_dims": embed_dims,
                "num_heads": num_heads,
                "num_levels": num_levels,
                "num_points": num_points,
            }
            self.spatial_cross_attention = TTSpatialCrossAttention(
                device=device,
                params=params.spatial_cross_attention,
                embed_dims=embed_dims,
                num_cams=num_cams,
                batch_first=batch_first,
                deformable_attention=deform_config,
                **kwargs,
            )

    def forward(
        self,
        bev_query,
        key=None,
        value=None,
        bev_pos=None,
        spatial_shapes=None,
        bev_shape=None,
        level_start_index=None,
        valid_ratios=None,
        prev_bev=None,
        shift=None,
        reference_points_3d=None,
        reference_points_cam=None,
        bev_mask=None,
        **kwargs,
    ):
        """
        Forward pass for a single TTBEVFormerLayer.

        Args:
            bev_query: Current BEV query features [B, num_query, embed_dims]
            key: Multi-camera features [num_cams, H*W, B, embed_dims]
            value: Same as key
            bev_pos: BEV positional encoding [B, num_query, embed_dims]
            spatial_shapes: Spatial shapes of multi-scale features [num_levels, 2]
            bev_shape: BEV grid shape [1, 2] containing [bev_h, bev_w]
            level_start_index: Start index of each level [num_levels]
            valid_ratios: Valid ratios for each level [B, num_levels, 2]
            prev_bev: Previous timestep BEV features [B, num_query, embed_dims]
            shift: Camera shift information for temporal alignment
            reference_points_3d: 3D reference points [B, num_query, D, 3]
            reference_points_cam: Camera reference points [num_cams, B, num_query, D, 2]
            bev_mask: Validity mask for camera projections [num_cams, B, num_query, D]

        Returns:
            Updated BEV features [B, num_query, embed_dims]
        """
        if use_signpost:
            signpost(header="TTNN BEVFormerLayer Forward Start")

        # Convert torch tensors to ttnn tensors if needed
        if isinstance(bev_query, torch.Tensor):
            bev_query = ttnn.from_torch(bev_query, device=self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

        # Temporal Self-Attention
        if isinstance(prev_bev, torch.Tensor):
            prev_bev = ttnn.from_torch(prev_bev, device=self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

        if isinstance(reference_points_3d, torch.Tensor):
            # Extract 2D coordinates from the first depth level of 3D reference points
            # reference_points_3d shape: [bs, num_query, num_points, 3]
            bev_reference_points = reference_points_3d[:, :, 0, :2].unsqueeze(2)  # [bs, num_query, 1, 2]

            # Convert to TTNN tensor if needed
            if hasattr(ttnn, "Device") and isinstance(self.device, ttnn.Device):
                bev_reference_points = ttnn.from_torch(
                    bev_reference_points, device=self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
                )
        else:
            # reference_points_3d is already a TTNN tensor
            # Convert to torch, do the operations, then convert back
            torch_ref_points = ttnn.to_torch(reference_points_3d)
            bev_reference_points = torch_ref_points[:, :, 0, :2].unsqueeze(2)  # [bs, num_query, 1, 2]
            bev_reference_points = ttnn.from_torch(
                bev_reference_points, device=self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
            )

        # Combine current BEV with history for temporal modeling
        temp_query = self.temporal_self_attention(
            query=bev_query,
            value=prev_bev,  # Use previous BEV as value for temporal context
            query_pos=bev_pos,
            reference_points=bev_reference_points,
            spatial_shapes=bev_shape,  # Use bev_shape for temporal attention like reference implementation
            level_start_index=level_start_index,
            **kwargs,
        )

        # Layer normalization (norm1)
        temp_query = ttnn.layer_norm(temp_query, weight=self.params.norm1.weight, bias=self.params.norm1.bias)
        bev_query = temp_query

        # Spatial Cross-Attention
        if value is None:
            value = key

        spatial_query = self.spatial_cross_attention(
            query=bev_query,
            key=key,
            value=value,
            residual=bev_query,
            query_pos=bev_pos,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs,
        )

        # Layer normalization (norm2)
        spatial_query = ttnn.layer_norm(spatial_query, weight=self.params.norm2.weight, bias=self.params.norm2.bias)
        bev_query = spatial_query

        # Feed-Forward Network
        ffn_output = self._forward_ffn(bev_query)

        # Layer normalization and residual connection (norm3)
        ffn_output_with_residual = ttnn.add(bev_query, ffn_output)
        bev_query = ttnn.layer_norm(
            ffn_output_with_residual, weight=self.params.norm3.weight, bias=self.params.norm3.bias
        )

        if use_signpost:
            signpost(header="TTNN BEVFormerLayer Forward End")

        return bev_query

    def _forward_ffn(self, x):
        """
        Forward pass through the feed-forward network using TTNN operations.

        Args:
            x: Input tensor [B, num_query, embed_dims]

        Returns:
            Output tensor [B, num_query, embed_dims]
        """
        # Check if FFN parameters exist - mixed object/dict structure
        if not (
            hasattr(self.params, "ffn") and hasattr(self.params.ffn, "linear1") and hasattr(self.params.ffn, "linear2")
        ):
            logger.warning("FFN parameters not found, skipping FFN forward pass")
            return ttnn.zeros_like(x)

        # First linear layer: [embed_dims] -> [feedforward_channels]
        # FFN has mixed structure: params.ffn.linear1 is a dict, not an object
        x = ttnn.linear(x, self.params.ffn.linear1["weight"], bias=self.params.ffn.linear1["bias"])

        # ReLU activation
        x = ttnn.relu(x)

        # Second linear layer: [feedforward_channels] -> [embed_dims]
        x = ttnn.linear(x, self.params.ffn.linear2["weight"], bias=self.params.ffn.linear2["bias"])

        return x

    def __call__(self, *args, **kwargs):
        """Make the class callable"""
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        """String representation for debugging"""
        return (
            f"embed_dims={self.embed_dims}, "
            f"use_temporal_self_attention={self.use_temporal_self_attention}, "
            f"use_spatial_cross_attention={self.use_spatial_cross_attention}, "
            f"feedforward_channels={self.feedforward_channels}, "
            f"batch_first={self.batch_first}"
        )


class TTBEVFormerEncoder:
    """
    TTNN implementation of BEVFormer Encoder.

    The encoder transforms multi-camera features into unified BEV representations
    using spatiotemporal transformers implemented with TTNN operations. It consists
    of multiple TTBEVFormerLayer modules that perform spatial cross-attention and
    temporal self-attention.

    Args:
        device: TTNN device for computation
        params: Parameter dict containing all layer weights and biases
        num_layers (int): Number of transformer layers
        embed_dims (int): Feature embedding dimensions
        num_heads (int): Number of attention heads
        num_levels (int): Number of feature pyramid levels
        num_points (int): Number of sampling points in deformable attention
        num_cams (int): Number of cameras
        pc_range (List[float]): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        num_points_in_pillar (int): Number of points sampled in each pillar
        return_intermediate (bool): Whether to return intermediate layer outputs
        dataset (str): Dataset type for specific configurations
        feedforward_channels (int): FFN intermediate channel size
        batch_first (bool): Whether batch dimension is first
        z_cfg (Dict[str, Any]): Z-axis configuration for point sampling
        **kwargs: Additional arguments
    """

    def __init__(
        self,
        device,
        params,
        num_layers: int = 6,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        num_cams: int = 6,
        pc_range: List[float] = None,
        num_points_in_pillar: int = 4,
        return_intermediate: bool = False,
        dataset: str = "nuScenes",
        feedforward_channels: int = 1024,
        batch_first: bool = True,
        z_cfg: Dict[str, Any] = None,
        **kwargs,
    ):
        self.device = device
        self.params = params

        if pc_range is None:
            pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        if z_cfg is None:
            z_cfg = {
                "num_points": num_points_in_pillar,
                "start": pc_range[2],  # z_min
                "end": pc_range[5],  # z_max
            }

        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_cams = num_cams
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.return_intermediate = return_intermediate
        self.batch_first = batch_first
        self.z_cfg = z_cfg
        self.feedforward_channels = feedforward_channels

        # Build transformer layers
        self.layers = []
        for layer_idx in range(num_layers):
            layer_params = getattr(params, f"layer_{layer_idx}", None)
            if layer_params is None:
                logger.warning(f"Parameters for layer {layer_idx} not found")
                continue

            layer = TTBEVFormerLayer(
                device=device,
                params=layer_params,
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                num_cams=num_cams,
                feedforward_channels=feedforward_channels,
                batch_first=batch_first,
                **kwargs,
            )
            self.layers.append(layer)

    def forward(
        self,
        bev_query,
        key=None,
        value=None,
        bev_h: int = 30,
        bev_w: int = 30,
        bev_pos=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        prev_bev=None,
        shift=None,
        img_metas: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        """
        Forward pass of TTNN BEVFormer encoder.

        Args:
            bev_query: Initial BEV query features [B, num_query, embed_dims]
            key: Multi-camera features [num_cams, H*W, B, embed_dims]
            value: Same as key (optional)
            bev_h: BEV grid height
            bev_w: BEV grid width
            bev_pos: BEV positional encoding [B, num_query, embed_dims]
            spatial_shapes: Multi-scale feature shapes [num_levels, 2]
            level_start_index: Start indices for each level [num_levels]
            valid_ratios: Valid ratios for each level [B, num_levels, 2]
            prev_bev: Previous timestep BEV features [B, num_query, embed_dims]
            shift: Camera shift for temporal alignment
            img_metas: Camera metadata for point sampling

        Returns:
            Final BEV features [B, num_query, embed_dims]
            If return_intermediate=True, returns list of intermediate features.
        """
        if use_signpost:
            signpost(header="TTNN BEVFormerEncoder Forward Start")

        # Convert torch tensors to ttnn tensors if needed
        if isinstance(bev_query, torch.Tensor):
            bev_query = ttnn.from_torch(bev_query, device=self.device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

        output = bev_query
        intermediate = []

        # Get batch size and number of queries for reference point generation
        bs, num_query, _ = bev_query.shape

        # Generate reference points for spatial cross-attention
        reference_points_cam = None
        bev_mask = None

        if img_metas is not None:
            reference_points_3d = generate_reference_points(
                bev_h=bev_h,
                bev_w=bev_w,
                z_cfg=self.z_cfg,
                batch_size=bs,
                device="cpu",  # Generate on CPU first
                dtype=torch.float32,
            )

            # Extract lidar2img matrices from img_metas
            if "lidar2img" in img_metas[0]:
                # Create lidar2img tensor from metadata
                lidar2img_list = []
                for meta in img_metas:
                    if isinstance(meta["lidar2img"], torch.Tensor):
                        lidar2img_list.append(meta["lidar2img"])
                    else:
                        lidar2img_list.append(torch.tensor(meta["lidar2img"], dtype=torch.float32))

                lidar2img = torch.stack(lidar2img_list)
                # Shape: [batch, num_cams, 4, 4] - keep batch dimension for compatibility
                lidar2img = lidar2img.cpu()
            else:
                # Construct lidar2img from intrinsics and extrinsics if available
                camera_intrinsics = torch.tensor(img_metas[0]["camera_intrinsics"], dtype=torch.float32)
                camera_extrinsics = torch.tensor(img_metas[0]["camera_extrinsics"], dtype=torch.float32)
                lidar2img = camera_intrinsics @ camera_extrinsics  # [num_cams, 4, 4]
            # Project to camera coordinates using PyTorch function (more stable)
            reference_points_cam, bev_mask = point_sampling_3d_to_2d_ttnn(
                reference_points=reference_points_3d,
                pc_range=self.pc_range,
                lidar2img=lidar2img,
                img_metas=img_metas,
                device=self.device,
            )

        # Process through transformer layers
        for lid, layer in enumerate(self.layers):
            # Create bev_shape tensor like the reference implementation
            bev_shape = torch.tensor([[bev_h, bev_w]])

            output = layer(
                bev_query=output,
                key=key,
                value=value,
                bev_pos=bev_pos,
                spatial_shapes=spatial_shapes,
                bev_shape=bev_shape,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                prev_bev=prev_bev,
                shift=shift,
                reference_points_3d=reference_points_3d,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                **kwargs,
            )

            if self.return_intermediate:
                intermediate.append(output)

        if use_signpost:
            signpost(header="TTNN BEVFormerEncoder Forward End")

        if self.return_intermediate:
            return intermediate

        return output

    def __call__(self, *args, **kwargs):
        """Make the class callable"""
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"num_layers={self.num_layers}, embed_dims={self.embed_dims}, "
            f"num_heads={self.num_heads}, num_levels={self.num_levels}, "
            f"num_points={self.num_points}, num_cams={self.num_cams}, "
            f"pc_range={self.pc_range}, num_points_in_pillar={self.num_points_in_pillar}, "
            f"feedforward_channels={self.feedforward_channels}, z_cfg={self.z_cfg}"
        )
