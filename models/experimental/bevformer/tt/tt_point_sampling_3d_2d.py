# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN 3D to 2D point sampling and transformation functions for BEVFormer.

This module implements the core point sampling logic using TTNN operations
to project 3D reference points to 2D camera coordinates. This is a crucial
component for spatial cross-attention where BEV queries need to sample
features from specific locations in camera views.

Based on the reference PyTorch implementation but optimized for TTNN.
"""

import ttnn
import torch
from typing import Tuple, List, Optional

from ..reference.point_sampling_3d_2d import generate_reference_points as torch_generate_reference_points

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


# Get default dtype with fallback
try:
    DEFAULT_DTYPE = ttnn.float32
except AttributeError:
    try:
        DEFAULT_DTYPE = ttnn.DataType.float32
    except AttributeError:
        # Ultimate fallback
        DEFAULT_DTYPE = None


def generate_reference_points_ttnn(
    bev_h: int,
    bev_w: int,
    z_cfg: dict,
    device,
    dtype=None,
    batch_size: int = 1,
):
    """
    Generate 3D reference points for BEV queries using TTNN operations with batch support.

    Args:
        bev_h (int): Height of the BEV grid.
        bev_w (int): Width of the BEV grid.
        z_cfg (dict): Z dimension configuration with keys:
            - 'num_points': Number of points along Z-axis
            - 'start': Start height (e.g., -5.0 meters)
            - 'end': End height (e.g., 3.0 meters)
        device: TTNN device to place tensors on.
        dtype: TTNN data type for tensors.
        batch_size (int): Batch size.

    Returns:
        Reference points [bs, bev_h*bev_w, num_points, 3] in normalized coordinates [0, 1].
    """
    if use_signpost:
        signpost(header="TTNN Generate Reference Points Start")

    # TODO: Calculate during initialization, and pass as parameters
    torch_ref_points = torch_generate_reference_points(
        bev_h=bev_h,
        bev_w=bev_w,
        z_cfg=z_cfg,
        batch_size=batch_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Convert to TTNN tensor
    ttnn_ref_points = ttnn.from_torch(torch_ref_points, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    if use_signpost:
        signpost(header="TTNN Generate Reference Points End")

    return ttnn_ref_points


def point_sampling_3d_to_2d_ttnn(
    reference_points,
    pc_range: List[float],
    lidar2img,
    img_metas: Optional[List[dict]] = None,
    img_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-5,
    device=None,
):
    """
    Project 3D reference points to 2D camera coordinates using TTNN operations with batch support.

    This function implements the core 3D to 2D transformation used in BEVFormer's
    spatial cross-attention mechanism using TTNN for optimized execution.

    Args:
        reference_points: 3D reference points in normalized coordinates [0, 1].
            Shape: [bs, num_queries, num_points, 3] where last dim is (x, y, z).
            Can be ttnn.Tensor or torch.Tensor.
        pc_range (List[float]): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        lidar2img: Camera transformation matrices.
            Shape: [bs, num_cams, 4, 4] - transforms from LiDAR coordinate to image coordinate.
            Can be ttnn.Tensor or torch.Tensor.
        img_metas (Optional[List[dict]]): Image metadata containing image shapes.
        img_shape (Optional[Tuple[int, int]]): Image shape (height, width). Used if img_metas is None.
        eps (float): Small epsilon to avoid division by zero.
        device: TTNN device for computation.

    Returns:
        Tuple[ttnn.Tensor, ttnn.Tensor]:
            - reference_points_cam: Projected 2D points [num_cams, bs, num_queries, num_points, 2].
            - bev_mask: Validity mask [num_cams, bs, num_queries, num_points] indicating valid projections.
    """
    if use_signpost:
        signpost(header="TTNN Point Sampling 3D to 2D Start")

    # Convert inputs to TTNN tensors if needed
    if isinstance(reference_points, torch.Tensor):
        reference_points = ttnn.from_torch(
            reference_points, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    if isinstance(lidar2img, torch.Tensor):
        lidar2img = ttnn.from_torch(lidar2img, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Get tensor shapes
    batch_size, num_queries, num_points, _ = reference_points.shape
    num_cams = lidar2img.shape[1]

    # Extract point cloud range values
    x_min, y_min, z_min = pc_range[0], pc_range[1], pc_range[2]
    x_max, y_max, z_max = pc_range[3], pc_range[4], pc_range[5]

    # Scale normalized coordinates [0, 1] to actual world coordinates
    reference_points_x = reference_points[:, :, :, 0:1]
    reference_points_y = reference_points[:, :, :, 1:2]
    reference_points_z = reference_points[:, :, :, 2:3]

    reference_points_x = ttnn.add(ttnn.mul(reference_points_x, x_max - x_min), x_min)
    reference_points_y = ttnn.add(ttnn.mul(reference_points_y, y_max - y_min), y_min)
    reference_points_z = ttnn.add(ttnn.mul(reference_points_z, z_max - z_min), z_min)

    ones = ttnn.ones_like(reference_points_x, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    reference_points = ttnn.concat([reference_points_x, reference_points_y, reference_points_z, ones], dim=-1)
    reference_points = ttnn.permute(reference_points, (0, 2, 1, 3))  # [bs, num_points, num_queries, 4]

    reference_points_cam_list = []

    for cam_idx in range(num_cams):
        cam_lidar2img = lidar2img[:, cam_idx, :, :]  # [4, 4] for current camera
        ref_points_flat = ttnn.reshape(reference_points, (batch_size, num_points * num_queries, 4))  # [B, D*Q, 4]

        # [B, D*Q, 4] @ [B, 4, 4]
        cam_lidar2img_T = ttnn.transpose(cam_lidar2img, -2, -1)  # Last two dims transpose
        points_cam_flat = ttnn.matmul(ref_points_flat, cam_lidar2img_T)  # [B, D*Q, 4]

        points_cam = ttnn.reshape(points_cam_flat, (batch_size, num_points, 1, num_queries, 4))  # [B, D, 1, Q, 4]
        points_cam = ttnn.permute(points_cam, (1, 0, 2, 3, 4))  # [D, B, 1, Q, 4]
        reference_points_cam_list.append(points_cam)

    reference_points_cam = ttnn.concat(reference_points_cam_list, dim=2)  # [D, B, num_cam, Q, 4]

    depth = reference_points_cam[..., 2:3]  # [D, B, num_cams, Q, 1]
    bev_mask = depth > eps

    depth_safe = ttnn.maximum(
        depth, ttnn.mul(ttnn.ones_like(depth, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), eps)
    )
    reference_points_cam_xy = ttnn.div(reference_points_cam[..., 0:2], depth_safe)

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

    reference_points_cam_x = ttnn.div(reference_points_cam_xy[..., 0:1], img_w)
    reference_points_cam_y = ttnn.div(reference_points_cam_xy[..., 1:2], img_h)
    reference_points_cam = ttnn.concat(
        [reference_points_cam_x, reference_points_cam_y], dim=-1
    )  # [D, B, num_cams, Q, 2]

    valid_x = ttnn.logical_and((reference_points_cam[..., 0:1] >= 0.0), (reference_points_cam[..., 0:1] <= 1.0))
    valid_y = ttnn.logical_and((reference_points_cam[..., 1:2] >= 0.0), (reference_points_cam[..., 1:2] <= 1.0))
    valid_bounds = ttnn.logical_and(valid_x, valid_y)

    bev_mask = ttnn.logical_and(bev_mask, valid_x)
    bev_mask = ttnn.logical_and(bev_mask, valid_y)

    nan_mask = ttnn.isnan(bev_mask)
    bev_mask = ttnn.logical_or(
        ttnn.logical_and(nan_mask, ttnn.zeros_like(bev_mask)), ttnn.logical_and(ttnn.logical_not(nan_mask), bev_mask)
    )

    reference_points_cam = ttnn.permute(reference_points_cam, (2, 1, 3, 0, 4))  # [num_cams, B, Q, D, 2]
    bev_mask = ttnn.squeeze(ttnn.permute(bev_mask, (2, 1, 3, 0, 4)), -1)  # [num_cams, B, Q, D]

    if use_signpost:
        signpost(header="TTNN Point Sampling 3D to 2D End")

    return reference_points_cam, bev_mask
