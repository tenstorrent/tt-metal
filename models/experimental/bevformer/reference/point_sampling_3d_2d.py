# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
3D to 2D point sampling and transformation functions for BEVFormer.

This module implements the core point sampling logic that projects 3D reference points
to 2D camera coordinates. This is a crucial component for spatial cross-attention
where BEV queries need to sample features from specific locations in camera views.

Based on the original BEVFormer implementation:
https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/encoder.py
"""

import torch
from typing import Tuple, List, Optional


def generate_reference_points(
    bev_h: int,
    bev_w: int,
    z_cfg: dict,
    batch_size: int = 1,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate 3D reference points for BEV queries.

    Args:
        bev_h (int): Height of the BEV grid.
        bev_w (int): Width of the BEV grid.
        z_cfg (dict): Z dimension configuration with keys:
            - 'num_points': Number of points along Z-axis
            - 'start': Start height (e.g., -5.0 meters)
            - 'end': End height (e.g., 3.0 meters)
        batch_size (int): Batch size.
        device (torch.device): Device to place tensors on.
        dtype (torch.dtype): Data type for tensors.

    Returns:
        torch.Tensor: Reference points [bs, bev_h*bev_w, num_points, 3] in normalized coordinates [0, 1].
    """
    if device is None:
        device = torch.device("cpu")

    num_points_z = z_cfg["num_points"]
    height_z = z_cfg["end"] - z_cfg["start"]

    # Generate normalized grid coordinates [0, 1] for x and y
    xs = torch.linspace(0.5, (bev_w - 0.5), bev_w, device=device, dtype=dtype) / bev_w
    ys = torch.linspace(0.5, (bev_h - 0.5), bev_h, device=device, dtype=dtype) / bev_h

    # Generate normalized z coordinates [0, 1]
    zs = torch.linspace(0.5, (height_z - 0.5), num_points_z, device=device, dtype=dtype) / height_z

    # Create meshgrid for all dimensions
    ref_y, ref_x = torch.meshgrid(ys, xs, indexing="ij")  # [bev_h, bev_w]
    ref_y = ref_y.flatten()  # [bev_h*bev_w]
    ref_x = ref_x.flatten()  # [bev_h*bev_w]

    # Expand to include z dimension
    num_queries = bev_h * bev_w
    ref_y = ref_y.unsqueeze(1).expand(num_queries, num_points_z)  # [bev_h*bev_w, num_points_z]
    ref_x = ref_x.unsqueeze(1).expand(num_queries, num_points_z)  # [bev_h*bev_w, num_points_z]
    ref_z = zs.unsqueeze(0).expand(num_queries, num_points_z)  # [bev_h*bev_w, num_points_z]

    # Stack coordinates
    ref_3d = torch.stack((ref_x, ref_y, ref_z), dim=-1)  # [bev_h*bev_w, num_points_z, 3]

    # Add batch dimension
    ref_3d = ref_3d.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [bs, bev_h*bev_w, num_points_z, 3]

    return ref_3d


def point_sampling_3d_to_2d(
    reference_points: torch.Tensor,
    pc_range: List[float],
    lidar2img: torch.Tensor,
    img_metas: Optional[List[dict]] = None,
    img_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D reference points to 2D camera coordinates with batch support.

    This function implements the core 3D to 2D transformation used in BEVFormer's
    spatial cross-attention mechanism. It projects 3D world coordinates to 2D image
    coordinates using camera transformation matrices. Now supports batch dimension.

    Args:
        reference_points (torch.Tensor): 3D reference points in normalized coordinates [0, 1].
            Shape: [bs, num_queries, num_points, 3] where last dim is (x, y, z).
        pc_range (List[float]): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        lidar2img (torch.Tensor): Camera transformation matrices.
            Shape: [bs, num_cams, 4, 4] - transforms from LiDAR coordinate to image coordinate.
        img_metas (Optional[List[dict]]): Image metadata containing image shapes.
        img_shape (Optional[Tuple[int, int]]): Image shape (height, width). Used if img_metas is None.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - reference_points_cam: Projected 2D points [num_cams, bs, num_queries, num_points, 2].
            - bev_mask: Validity mask [num_cams, bs, num_queries, num_points] indicating valid projections.
    """

    # Clone to avoid modifying input
    reference_points = reference_points.clone()

    # Extract point cloud range
    x_min, y_min, z_min = pc_range[0], pc_range[1], pc_range[2]
    x_max, y_max, z_max = pc_range[3], pc_range[4], pc_range[5]

    # Scale normalized coordinates [0, 1] to actual world coordinates
    reference_points[..., 0:1] = reference_points[..., 0:1] * (x_max - x_min) + x_min  # X
    reference_points[..., 1:2] = reference_points[..., 1:2] * (y_max - y_min) + y_min  # Y
    reference_points[..., 2:3] = reference_points[..., 2:3] * (z_max - z_min) + z_min  # Z

    # Convert to homogeneous coordinates by adding ones
    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), dim=-1
    )  # [bs, num_queries, num_points, 4]

    # [num_points, bs, num_queries, 4]
    reference_points = reference_points.permute(2, 0, 1, 3)
    D, B, num_queries = reference_points.size()[:3]
    num_cams = lidar2img.shape[1]  # lidar2img is [bs, num_cams, 4, 4]
    # Reshape for batch matrix multiplication
    # reference_points: [D, B, 1, num_queries, 4] -> [D, B, num_cams, num_queries, 4]
    reference_points = reference_points.view(D, B, 1, num_queries, 4).repeat(1, 1, num_cams, 1, 1).unsqueeze(-1)

    # lidar2img: [1, B, num_cams, 1, 4, 4] -> [D, B, num_cams, num_queries, 4, 4]
    lidar2img = lidar2img.view(1, B, num_cams, 1, 4, 4).repeat(D, 1, 1, num_queries, 1, 1)

    # Apply transformation: [D, B, num_cams, num_queries, 4, 4] @ [D, B, num_cams, num_queries, 4, 1]
    reference_points_cam = torch.matmul(lidar2img.to(torch.float32), reference_points.to(torch.float32)).squeeze(-1)

    # Extract depth for validity check
    depth = reference_points_cam[..., 2:3]  # [D, B, num_cams, num_queries, 1]

    # Create validity mask (points must be in front of camera)
    bev_mask = depth > eps

    # Perspective division to get image coordinates
    # Avoid division by zero using maximum
    depth_safe = torch.maximum(depth, torch.ones_like(depth) * eps)
    reference_points_cam = reference_points_cam[..., 0:2] / depth_safe

    # Determine image shape
    if img_metas is not None and len(img_metas) > 0:
        if "img_shape" in img_metas[0]:
            img_h, img_w = img_metas[0]["img_shape"][0][:2]  # Assume all cameras have same shape
        else:
            raise ValueError("img_shape not found in img_metas")
    elif img_shape is not None:
        img_h, img_w = img_shape
    else:
        raise ValueError("Either img_metas or img_shape must be provided")

    # Normalize by image dimensions to get coordinates in [0, 1] range
    reference_points_cam[..., 0] /= img_w  # Normalize x by width
    reference_points_cam[..., 1] /= img_h  # Normalize y by height

    # Check if points are within image bounds [0, 1]
    valid_x = (reference_points_cam[..., 0:1] >= 0.0) & (reference_points_cam[..., 0:1] <= 1.0)
    valid_y = (reference_points_cam[..., 1:2] >= 0.0) & (reference_points_cam[..., 1:2] <= 1.0)
    valid_bounds = valid_x & valid_y

    # Combine depth and bounds validity
    bev_mask = bev_mask & valid_x & valid_y

    # Handle any NaN values
    bev_mask = torch.nan_to_num(bev_mask)
    # Permute to get expected output format: [num_cams, bs, num_queries, num_points, 2/1]
    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)  # [num_cams, B, num_queries, D, 2]
    bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)  # [num_cams, B, num_queries, D]

    return reference_points_cam, bev_mask
