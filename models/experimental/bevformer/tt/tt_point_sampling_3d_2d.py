# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
from typing import List, NamedTuple, Optional, Tuple

from ..reference.point_sampling_3d_2d import generate_reference_points as torch_generate_reference_points
from .tt_common import signpost


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

    signpost(header="TTNN Generate Reference Points End")

    return ttnn_ref_points


class WorldPoints(NamedTuple):
    """A reference grid in world coordinates, laid out for :func:`point_sampling_3d_to_2d_ttnn`.

    Each (query, pillar point) appears twice in a row, for its camera x and then its y, so one
    pass of multiply-adds projects both, already interleaved as ``reference_points_cam`` holds
    them, and a row is dense: :func:`projection_layout`'s queries per row.

    Attributes:
        x, y, z: ``[bs, 1, rows, width]`` float32 TILE.
        pairs: ``[width, width // 2]`` bfloat16 TILE, 1 where a column belongs to an (x, y) pair:
            a matmul by it counts each pair's valid coordinates.
        num_queries, num_points: The grid's queries and pillar points.
    """

    x: ttnn.Tensor
    y: ttnn.Tensor
    z: ttnn.Tensor
    pairs: ttnn.Tensor
    num_queries: int
    num_points: int


def projection_layout(num_queries: int, num_points: int) -> Tuple[int, int]:
    """``(rows, width)`` of :class:`WorldPoints`: each row holds the largest number of queries up
    to a tile's width that divides ``num_queries``, each query its ``num_points`` (x, y) pairs."""
    per_row = max(n for n in range(1, ttnn.TILE_SIZE + 1) if num_queries % n == 0)
    return num_queries // per_row, 2 * num_points * per_row


def world_points_ttnn(reference_points: torch.Tensor, pc_range: List[float], device) -> WorldPoints:
    """Torch ``reference_points`` ``[bs, num_queries, num_points, 3]`` in [0, 1], scaled to
    ``pc_range`` on the host and uploaded as :class:`WorldPoints`."""
    bs, num_queries, num_points, _ = reference_points.shape
    rows, width = projection_layout(num_queries, num_points)
    low = torch.tensor(pc_range[:3], dtype=torch.float32)
    high = torch.tensor(pc_range[3:6], dtype=torch.float32)
    world = reference_points.to(torch.float32) * (high - low) + low
    doubled = world.unsqueeze(3).expand(bs, num_queries, num_points, 2, 3).reshape(bs, 1, rows, width, 3)
    pairs = torch.zeros(width, width // 2)
    pairs[torch.arange(width), torch.arange(width) // 2] = 1.0

    def upload(tensor, dtype):
        return ttnn.from_torch(tensor.contiguous(), device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    return WorldPoints(
        *(upload(doubled[..., axis], ttnn.float32) for axis in range(3)),
        pairs=upload(pairs, ttnn.bfloat16),
        num_queries=num_queries,
        num_points=num_points,
    )


def image_size(img_metas: Optional[List[dict]] = None, img_shape: Optional[Tuple[int, int]] = None):
    """The images' ``(height, width)``: the first camera's ``img_shape`` in ``img_metas``, all
    cameras having one, else ``img_shape``."""
    if img_metas is not None and len(img_metas) > 0:
        if "img_shape" not in img_metas[0]:
            raise ValueError("img_shape not found in img_metas")
        img_h, img_w = img_metas[0]["img_shape"][0][:2]
        return img_h, img_w
    if img_shape is not None:
        return img_shape
    raise ValueError("Either img_metas or img_shape must be provided")


def camera_rows(lidar2img: torch.Tensor, img_size: Tuple[int, int], num_queries: int, num_points: int):
    """Host ``[bs, num_cams, 9, width]`` float32 rows that project :class:`WorldPoints` of that
    grid through ``lidar2img`` ``[bs, num_cams, 4, 4]``: rows 0-3 weigh x, y, z and 1 into the
    camera's x at even columns and its y at odd ones, rows 4-7 into its depth, and row 8 holds the
    image's width at even columns and height at odd ones, ``img_size`` being ``(height, width)``."""
    _, width = projection_layout(num_queries, num_points)
    lidar2img = lidar2img.to(torch.float32)
    bs, num_cams = lidar2img.shape[:2]
    planar = lidar2img[:, :, :2, :].transpose(-1, -2).repeat(1, 1, 1, width // 2)
    depth = lidar2img[:, :, 2:3, :].transpose(-1, -2).expand(bs, num_cams, 4, width)
    img_h, img_w = img_size
    size = torch.tensor([float(img_w), float(img_h)]).repeat(width // 2).expand(bs, num_cams, 1, width)
    return torch.cat([planar, depth, size], dim=2).contiguous()


def point_sampling_3d_to_2d_ttnn(
    reference_points,
    pc_range: List[float],
    lidar2img,
    img_metas: Optional[List[dict]] = None,
    img_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-5,
    device=None,
    world_points: Optional[WorldPoints] = None,
    projection=None,
    output_dtype=ttnn.float32,
):
    """
    Project 3D reference points to 2D camera coordinates using TTNN operations with batch support.

    This function implements the core 3D to 2D transformation used in BEVFormer's
    spatial cross-attention mechanism using TTNN for optimized execution.

    Every camera projects every point at once in :class:`WorldPoints`' dense layout, whose rows
    are the outputs' rows: a point's camera x and y come out interleaved, and a matmul pairs their
    flags into its mask. The projection is multiply-adds in float32: a matmul truncates float32 to
    tf32 and moves the points by about a pixel, and bfloat16 by several.

    Args:
        reference_points: 3D reference points in normalized coordinates [0, 1].
            Shape: [bs, num_queries, num_points, 3] where last dim is (x, y, z).
            Can be ttnn.Tensor or torch.Tensor; ignored with ``world_points``.
        pc_range (List[float]): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        lidar2img: Camera transformation matrices.
            Shape: [bs, num_cams, 4, 4] - transforms from LiDAR coordinate to image coordinate.
            Can be ttnn.Tensor or torch.Tensor; ignored with ``projection``.
        img_metas (Optional[List[dict]]): Image metadata containing image shapes.
        img_shape (Optional[Tuple[int, int]]): Image shape (height, width). Used if img_metas is None.
        eps (float): Small epsilon to avoid division by zero.
        device: TTNN device for computation.
        world_points: The grid from :func:`world_points_ttnn`, which a fixed grid uploads once.
        projection: :func:`camera_rows` on device, float32 TILE, which a frame's inputs can carry;
            ``lidar2img`` and the image size are then ignored.
        output_dtype: ``reference_points_cam``'s dtype.

    Returns:
        Tuple[ttnn.Tensor, ttnn.Tensor], ROW_MAJOR:
            - reference_points_cam: Projected 2D points [num_cams, bs, num_queries, num_points, 2].
            - bev_mask: bfloat16 validity mask [num_cams, bs, num_queries, num_points] indicating
              valid projections.
    """
    signpost(header="TTNN Point Sampling 3D to 2D Start")
    signpost(header="Point Sampling Tensor Setup Start")

    if world_points is None:
        if not isinstance(reference_points, torch.Tensor):
            reference_points = ttnn.to_torch(reference_points)
        world_points = world_points_ttnn(reference_points, pc_range, device)
    if projection is None:
        if not isinstance(lidar2img, torch.Tensor):
            lidar2img = ttnn.to_torch(lidar2img)
        rows = camera_rows(
            lidar2img, image_size(img_metas, img_shape), world_points.num_queries, world_points.num_points
        )
        projection = ttnn.from_torch(rows, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

    signpost(header="Point Sampling Tensor Setup End")
    signpost(header="Point Sampling Projection Start")

    batch_size, num_cams = projection.shape[0], projection.shape[1]
    weights = [projection[:, :, row : row + 1, :] for row in range(9)]

    def project(x_weight, y_weight, z_weight, offset):
        # [bs, num_cams, rows, width]: the world points broadcast over the cameras, each camera's
        # weights over the rows.
        return ttnn.add(
            ttnn.add(ttnn.mul(world_points.x, x_weight), ttnn.mul(world_points.y, y_weight)),
            ttnn.add(ttnn.mul(world_points.z, z_weight), offset),
        )

    planar = project(*weights[0:4])
    depth = project(*weights[4:8])
    valid = ttnn.gt(depth, eps)
    # Normalized image coordinates; the open bounds are upstream's, and a NaN fails them.
    points = ttnn.div(ttnn.div(planar, ttnn.maximum(depth, eps)), weights[8])
    valid = ttnn.logical_and(valid, ttnn.logical_and(ttnn.gt(points, 0.0), ttnn.lt(points, 1.0)))
    # A point is valid where both its coordinates are: the matmul sums each pair's flags, exactly
    # in bfloat16.
    pair_counts = ttnn.matmul(ttnn.typecast(valid, ttnn.bfloat16), world_points.pairs)
    bev_mask = ttnn.gt(pair_counts, 1.5)
    if output_dtype != points.dtype:
        points = ttnn.typecast(points, output_dtype)

    signpost(header="Point Sampling Projection End")
    signpost(header="Point Sampling Camera-major Layout Start")

    def camera_major(tensor, shape):
        """[bs, num_cams, rows, w] -> ROW_MAJOR [num_cams, bs, *shape]; the rows hold the queries in
        order, so past the camera-major permute it is a view."""
        if batch_size > 1:
            tensor = ttnn.permute(tensor, (1, 0, 2, 3))
        else:
            tensor = ttnn.reshape(tensor, (num_cams, 1, tensor.shape[2], tensor.shape[3]))
        return ttnn.reshape(ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT), (num_cams, batch_size, *shape))

    num_queries, num_points = world_points.num_queries, world_points.num_points
    reference_points_cam = camera_major(points, (num_queries, num_points, 2))
    bev_mask = camera_major(bev_mask, (num_queries, num_points))

    signpost(header="Point Sampling Camera-major Layout End")
    signpost(header="TTNN Point Sampling 3D to 2D End")

    return reference_points_cam, bev_mask
