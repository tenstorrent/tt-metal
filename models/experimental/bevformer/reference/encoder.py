# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BEVFormer Encoder implementation in PyTorch.

This module implements the main BEVFormer encoder that combines spatial cross-attention
and temporal self-attention to learn Bird's-Eye-View representations from multi-camera
images. The encoder processes BEV queries through multiple transformer layers that
enable both spatial feature extraction from camera views and temporal modeling.

Based on the original BEVFormer implementation:
https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/encoder.py
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any

from .spatial_cross_attention import SpatialCrossAttention
from .temporal_self_attention import TemporalSelfAttention
from .point_sampling_3d_2d import generate_reference_points, point_sampling_3d_to_2d


class BEVFormerLayer(nn.Module):
    """
    Single transformer layer for BEVFormer encoder.

    This layer combines spatial cross-attention and temporal self-attention
    to process BEV queries. Each layer performs:
    1. Temporal self-attention (optional) - models temporal dependencies
    2. Spatial cross-attention - extracts features from camera views
    3. Feed-forward network (optional) - additional processing
    """

    def __init__(
        self,
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
        super(BEVFormerLayer, self).__init__()

        self.embed_dims = embed_dims
        self.use_temporal_self_attention = use_temporal_self_attention
        self.use_spatial_cross_attention = use_spatial_cross_attention
        self.batch_first = batch_first

        # Temporal Self-Attention
        if use_temporal_self_attention:
            self.temporal_self_attention = TemporalSelfAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=1,  # Single level for temporal attention
                num_points=num_points,
                batch_first=batch_first,
                **kwargs,
            )
            self.norm1 = nn.LayerNorm(embed_dims)

        # Spatial Cross-Attention
        if use_spatial_cross_attention:
            deform_config = {
                "embed_dims": embed_dims,
                "num_heads": num_heads,
                "num_levels": num_levels,
                "num_points": num_points,
            }
            self.spatial_cross_attention = SpatialCrossAttention(
                embed_dims=embed_dims,
                num_cams=num_cams,
                batch_first=batch_first,
                deformable_attention=deform_config,
                **kwargs,
            )
            self.norm2 = nn.LayerNorm(embed_dims)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward_channels, embed_dims),
        )
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(
        self,
        bev_query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        bev_pos: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        bev_shape: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        valid_ratios: Optional[torch.Tensor] = None,
        prev_bev: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
        reference_points_3d: Optional[torch.Tensor] = None,
        reference_points_cam: Optional[torch.Tensor] = None,
        bev_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for a single BEVFormer layer.

        Args:
            bev_query: Current BEV query features [B, num_query, embed_dims]
            key: Multi-camera features [num_cams, H*W, B, embed_dims]
            value: Same as key
            bev_pos: BEV positional encoding [B, num_query, embed_dims]
            spatial_shapes: Spatial shapes of multi-scale features [num_levels, 2]
            level_start_index: Start index of each level [num_levels]
            valid_ratios: Valid ratios for each level [B, num_levels, 2]
            prev_bev: Previous timestep BEV features [B, num_query, embed_dims]
            shift: Camera shift information for temporal alignment
            reference_points_3d: 3D reference points [num_query, D, 3]
            reference_points_cam: Camera reference points [num_cams, B, num_query, D, 2]
            bev_mask: Validity mask for camera projections [num_cams, B, num_query, D]

        Returns:
            torch.Tensor: Updated BEV features [B, num_query, embed_dims]
        """

        # Temporal Self-Attention
        temp_query = bev_query
        temp_query = self.temporal_self_attention(
            query=temp_query,
            value=prev_bev,  # Use previous BEV as value for temporal context
            query_pos=bev_pos,
            reference_points=reference_points_3d[:, :, 0, :2].unsqueeze(2),
            spatial_shapes=bev_shape,
            **kwargs,
        )
        bev_query = self.norm1(temp_query)

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
        bev_query = self.norm2(spatial_query)

        # Feed-Forward Network
        ffn_output = self.ffn(bev_query)
        bev_query = self.norm3(bev_query + ffn_output)

        return bev_query


class BEVFormerEncoder(nn.Module):
    """
    BEVFormer Encoder implementation.

    The encoder transforms multi-camera features into unified BEV representations
    using spatiotemporal transformers. It consists of multiple BEVFormerLayer modules
    that perform spatial cross-attention and temporal self-attention.

    Args:
        num_layers (int): Number of transformer layers.
        embed_dims (int): Feature embedding dimensions.
        num_heads (int): Number of attention heads.
        num_levels (int): Number of feature pyramid levels.
        num_points (int): Number of sampling points in deformable attention.
        num_cams (int): Number of cameras.
        pc_range (List[float]): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        num_points_in_pillar (int): Number of points sampled in each pillar.
        return_intermediate (bool): Whether to return intermediate layer outputs.
        dataset (str): Dataset type for specific configurations.
    """

    def __init__(
        self,
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
        super(BEVFormerEncoder, self).__init__()

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

        # Build transformer layers
        self.layers = nn.ModuleList(
            [
                BEVFormerLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_levels=num_levels,
                    num_points=num_points,
                    num_cams=num_cams,
                    feedforward_channels=feedforward_channels,
                    batch_first=batch_first,
                    use_temporal_self_attention=True,
                    use_spatial_cross_attention=True,
                    **kwargs,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        bev_query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        bev_h: int = 30,
        bev_w: int = 30,
        bev_pos: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        valid_ratios: Optional[torch.Tensor] = None,
        prev_bev: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
        img_metas: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of BEVFormer encoder.

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
            torch.Tensor: Final BEV features [B, num_query, embed_dims]
                If return_intermediate=True, returns list of intermediate features.
        """
        output = bev_query
        intermediate = []

        bs, num_query, _ = bev_query.shape

        # Generate reference points for spatial cross-attention
        if img_metas is not None:
            # Generate 3D reference points using existing function
            reference_points_3d = generate_reference_points(
                bev_h=bev_h,
                bev_w=bev_w,
                z_cfg=self.z_cfg,
                batch_size=bs,
                device=bev_query.device,
                dtype=bev_query.dtype,
            )

            # Extract lidar2img matrices from img_metas
            # Assuming img_metas contains lidar2img or we can construct it from camera params
            if "lidar2img" in img_metas[0]:
                lidar2img = torch.stack(
                    [
                        torch.tensor(meta["lidar2img"], dtype=torch.float32, device=bev_query.device)
                        for meta in img_metas
                    ]
                )
            else:
                # Construct lidar2img from intrinsics and extrinsics if available
                camera_intrinsics = torch.tensor(
                    img_metas[0]["camera_intrinsics"], dtype=torch.float32, device=bev_query.device
                )
                camera_extrinsics = torch.tensor(
                    img_metas[0]["camera_extrinsics"], dtype=torch.float32, device=bev_query.device
                )
                lidar2img = camera_intrinsics @ camera_extrinsics  # [num_cams, 4, 4]

            # Project to camera coordinates using existing function
            reference_points_cam, bev_mask = point_sampling_3d_to_2d(
                reference_points=reference_points_3d, pc_range=self.pc_range, lidar2img=lidar2img, img_metas=img_metas
            )
        else:
            reference_points_cam = None
            bev_mask = None

        # Process through transformer layers
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query=output,
                key=key,
                value=value,
                bev_pos=bev_pos,
                spatial_shapes=spatial_shapes,
                bev_shape=torch.tensor([[bev_h, bev_w]], device=bev_query.device),
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

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"num_layers={self.num_layers}, embed_dims={self.embed_dims}, "
            f"num_heads={self.num_heads}, num_levels={self.num_levels}, "
            f"num_points={self.num_points}, num_cams={self.num_cams}, "
            f"pc_range={self.pc_range}, num_points_in_pillar={self.num_points_in_pillar}, "
            f"z_cfg={self.z_cfg}"
        )
