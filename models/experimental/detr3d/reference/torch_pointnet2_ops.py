# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch implementations of PointNet2 operations.

This module provides pure PyTorch implementations of the CUDA operations
from the third_party/pointnet2 extension. These implementations are equivalent
to the CUDA code but can run on CPU or GPU without requiring compilation.

The implementations closely follow the CUDA code in:
- sampling_gpu.cu
- ball_query_gpu.cu
- group_points_gpu.cu
- interpolate_gpu.cu
"""

import torch
import torch.nn as nn
from typing import Optional


# ============================================================================
# Sampling Operations (from sampling_gpu.cu)
# ============================================================================


def furthest_point_sampling(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Furthest Point Sampling (FPS) algorithm.

    Iteratively selects the point that is farthest from all previously selected points.
    This is the PyTorch equivalent of furthest_point_sampling_kernel in sampling_gpu.cu.

    Args:
        xyz: (B, N, 3) tensor of point coordinates
        npoint: int, number of points to sample

    Returns:
        idx: (B, npoint) tensor of sampled point indices

    Algorithm (from CUDA code):
        1. Initialize with first point (index 0)
        2. For each iteration j from 1 to npoint:
           - Compute distance from last selected point to all points
           - Update minimum distance to any sampled point: temp[k] = min(d, temp[k])
           - Select point with maximum minimum distance
           - Filter out points with magnitude <= 1e-3
    """
    device = xyz.device
    B, N, _ = xyz.shape

    # Initialize output indices
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)

    # Initialize distance array (temp in CUDA code)
    # Start with large values (infinity)
    temp = torch.ones(B, N, device=device) * 1e10

    # Start with first point (index 0)
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    # Compute magnitude to filter out points (mag <= 1e-3 in CUDA code)
    mag = torch.sum(xyz**2, dim=-1)  # (B, N)

    for i in range(npoint):
        idx[:, i] = farthest

        if i < npoint - 1:
            # Get coordinates of current centroid (old in CUDA code)
            centroid = xyz[batch_indices, farthest, :]  # (B, 3)

            # Compute squared distance from centroid to all points
            # d = (x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2
            dist = torch.sum((xyz - centroid.unsqueeze(1)) ** 2, dim=-1)  # (B, N)

            # Filter out points with magnitude <= 1e-3
            # Set their distances to -infinity so they won't be selected
            mask = mag <= 1e-3
            dist = torch.where(mask, torch.tensor(-1e10, device=device), dist)

            # Update minimum distances: temp[k] = min(d, temp[k])
            temp = torch.minimum(temp, dist)

            # Select the farthest point (maximum of minimum distances)
            farthest = torch.argmax(temp, dim=-1)

    return idx


def gather_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather points/features based on indices.

    This is the PyTorch equivalent of gather_points_kernel in sampling_gpu.cu.

    Args:
        points: (B, C, N) tensor of features/points
        idx: (B, M) tensor of indices to gather

    Returns:
        out: (B, C, M) tensor of gathered features

    CUDA code logic:
        out[(i * c + l) * m + j] = points[(i * c + l) * n + idx[i * m + j]]
        where i is batch, l is channel, j is point index
    """
    B, C, N = points.shape
    M = idx.shape[1]

    # Ensure idx is long type for indexing
    idx = idx.long()

    # Expand idx to match all channels
    idx_expanded = idx.unsqueeze(1).expand(B, C, M)  # (B, C, M)

    # Gather operation
    out = torch.gather(points, 2, idx_expanded)  # (B, C, M)

    return out


# ============================================================================
# Ball Query Operations (from ball_query_gpu.cu)
# ============================================================================


def query_ball_point(new_xyz: torch.Tensor, xyz: torch.Tensor, radius: float, nsample: int) -> torch.Tensor:
    """
    Find all points within a ball of given radius.

    This is the PyTorch equivalent of query_ball_point_kernel in ball_query_gpu.cu.

    Args:
        new_xyz: (B, M, 3) query positions (centroids)
        xyz: (B, N, 3) input point cloud
        radius: float, radius of ball query
        nsample: int, maximum number of points to sample in each ball

    Returns:
        idx: (B, M, nsample) tensor of indices

    CUDA code logic:
        - For each query point in new_xyz:
            - Find all points in xyz within radius
            - Store up to nsample indices
            - If fewer points found, fill remaining with first found point
    """
    device = new_xyz.device
    B, M, _ = new_xyz.shape
    _, N, _ = xyz.shape

    # Compute pairwise squared distances
    # diff: (B, M, N, 3)
    diff = new_xyz.unsqueeze(2) - xyz.unsqueeze(1)
    dist2 = torch.sum(diff**2, dim=-1)  # (B, M, N)

    # Square the radius for comparison
    radius2 = radius * radius

    # Create mask for points within radius
    mask = dist2 < radius2  # (B, M, N)

    # Initialize output with zeros
    idx = torch.zeros(B, M, nsample, dtype=torch.long, device=device)

    # For each batch and query point, collect indices
    for b in range(B):
        for m in range(M):
            # Find indices of points within radius
            within_radius = torch.where(mask[b, m])[0]

            if len(within_radius) > 0:
                # Take first nsample points (or all if fewer)
                sampled = within_radius[:nsample]

                # Fill idx, repeating first point if needed
                first_idx = sampled[0]
                for k in range(nsample):
                    if k < len(sampled):
                        idx[b, m, k] = sampled[k]
                    else:
                        idx[b, m, k] = first_idx
            else:
                # If no points within radius, fill with 0
                idx[b, m, :] = 0

    return idx


def query_ball_point_vectorized(new_xyz: torch.Tensor, xyz: torch.Tensor, radius: float, nsample: int) -> torch.Tensor:
    """
    Vectorized version of query_ball_point for better performance.

    Args:
        new_xyz: (B, M, 3) query positions
        xyz: (B, N, 3) input point cloud
        radius: float, radius of ball query
        nsample: int, maximum number of points to sample

    Returns:
        idx: (B, M, nsample) tensor of indices
    """
    device = new_xyz.device
    B, M, _ = new_xyz.shape
    _, N, _ = xyz.shape

    # Compute pairwise squared distances
    diff = new_xyz.unsqueeze(2) - xyz.unsqueeze(1)  # (B, M, N, 3)
    dist2 = torch.sum(diff**2, dim=-1)  # (B, M, N)

    # Square the radius
    radius2 = radius * radius

    # Create mask for points within radius
    mask = dist2 < radius2  # (B, M, N)

    # Create indices tensor
    arange_n = torch.arange(N, device=device).view(1, 1, N).expand(B, M, N)

    # Use a large value for points outside radius
    arange_n_masked = torch.where(mask, arange_n, torch.full_like(arange_n, N + 1))

    # Sort to bring valid indices to the front
    sorted_indices, _ = torch.sort(arange_n_masked, dim=2)

    # Take first nsample indices
    first_nsample = sorted_indices[:, :, :nsample]

    # Handle case where fewer than nsample points are found
    # Replace invalid indices (N+1) with first valid index
    invalid_mask = first_nsample >= N
    first_valid = first_nsample[:, :, 0:1].expand_as(first_nsample)
    first_nsample = torch.where(invalid_mask, first_valid, first_nsample)

    return first_nsample


# ============================================================================
# Group Points Operations (from group_points_gpu.cu)
# ============================================================================


def group_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Group features based on indices.

    This is the PyTorch equivalent of group_points_kernel in group_points_gpu.cu.

    Args:
        points: (B, C, N) tensor of features
        idx: (B, npoint, nsample) tensor of indices

    Returns:
        out: (B, C, npoint, nsample) tensor of grouped features

    CUDA code logic:
        out[(l * npoints + j) * nsample + k] = points[l * n + idx[j * nsample + k]]
        where l is channel, j is grouped point, k is sample within group
    """
    B, C, N = points.shape
    _, npoint, nsample = idx.shape

    # Ensure idx is long type
    idx = idx.long()

    # Reshape points for gathering: (B, C, N)
    # Expand idx to (B, C, npoint, nsample)
    idx_expanded = idx.unsqueeze(1).expand(B, C, npoint, nsample)

    # Gather operation
    # For each (b, c, i, j), get points[b, c, idx[b, i, j]]
    out = torch.gather(points.unsqueeze(3).expand(B, C, N, nsample), 2, idx_expanded)

    return out


def group_points_efficient(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    More efficient implementation of group_points using direct indexing.

    Args:
        points: (B, C, N) tensor of features
        idx: (B, npoint, nsample) tensor of indices

    Returns:
        out: (B, C, npoint, nsample) tensor of grouped features
    """
    B, C, N = points.shape
    _, npoint, nsample = idx.shape

    idx = idx.long()

    # Create batch indices
    batch_indices = torch.arange(B, device=points.device).view(B, 1, 1, 1)
    batch_indices = batch_indices.expand(B, C, npoint, nsample)

    # Create channel indices
    channel_indices = torch.arange(C, device=points.device).view(1, C, 1, 1)
    channel_indices = channel_indices.expand(B, C, npoint, nsample)

    # Expand idx for all channels
    idx_expanded = idx.unsqueeze(1).expand(B, C, npoint, nsample)

    # Use advanced indexing (more memory efficient)
    # Flatten points: (B, C, N)
    # Create indices to gather
    out = torch.zeros(B, C, npoint, nsample, dtype=points.dtype, device=points.device)

    for b in range(B):
        for c in range(C):
            out[b, c] = points[b, c, idx[b]]

    return out


# ============================================================================
# PyTorch Module Wrappers (Compatible with model_utils.py structure)
# ============================================================================


class FurthestPointSampling(nn.Module):
    """
    Module wrapper for Furthest Point Sampling.
    Compatible with the reference implementation in model_utils.py.
    """

    def __init__(self):
        super().__init__()

    def forward(self, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) input points
            npoint: int, number of points to sample

        Returns:
            idx: (B, npoint) sampled indices
        """
        return furthest_point_sampling(xyz, npoint)


class GatherOperation(nn.Module):
    """
    Module wrapper for gathering points/features.
    Compatible with the reference implementation in model_utils.py.
    """

    def __init__(self):
        super().__init__()

    def forward(self, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, N) tensor
            idx: (B, M) or (B, M, K) tensor of indices

        Returns:
            gathered: (B, C, M) or (B, C, M, K) tensor
        """
        if idx.ndim == 2:
            return gather_points(features, idx)
        elif idx.ndim == 3:
            # For 3D idx (B, M, K), need to use group_points
            return group_points(features, idx)
        else:
            raise ValueError(f"Unsupported idx dimension: {idx.ndim}")


class BallQuery(nn.Module):
    """
    Module wrapper for ball query operation.
    Compatible with the reference implementation in model_utils.py.
    """

    def __init__(self, radius: float, nsample: int):
        super().__init__()
        self.radius = radius
        self.nsample = nsample

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) input point cloud
            new_xyz: (B, M, 3) query positions

        Returns:
            idx: (B, M, nsample) tensor of indices
        """
        return query_ball_point_vectorized(new_xyz, xyz, self.radius, self.nsample)


class GroupingOperation(nn.Module):
    """
    Module wrapper for grouping operation.
    Compatible with the reference implementation in model_utils.py.
    """

    def __init__(self):
        super().__init__()

    def forward(self, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, N) tensor
            idx: (B, npoint, nsample) tensor of indices

        Returns:
            grouped: (B, C, npoint, nsample) tensor
        """
        return group_points(features, idx)


# ============================================================================
# PyTorch Module Wrappers (Compatible with model_utils.py structure)
# ============================================================================


class QueryAndGroup(nn.Module):
    """
    Complete QueryAndGroup module combining ball query and grouping.
    This matches the interface in model_utils.py.
    """

    def __init__(
        self,
        radius: float,
        nsample: int,
        use_xyz: bool = True,
        ret_grouped_xyz: bool = False,
        normalize_xyz: bool = False,
        sample_uniformly: bool = False,
        ret_unique_cnt: bool = False,
    ):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt

        if self.ret_unique_cnt:
            assert self.sample_uniformly

        self.ball_query = BallQuery(radius=radius, nsample=nsample)
        self.grouping_operation = GroupingOperation()

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor, features: Optional[torch.Tensor] = None):
        """
        Args:
            xyz: (B, N, 3) xyz coordinates of features
            new_xyz: (B, npoint, 3) centroids
            features: (B, C, N) descriptors of features

        Returns:
            new_features: (B, 3 + C, npoint, nsample) tensor
        """
        # Ball query to find neighbors
        idx = self.ball_query(xyz, new_xyz)  # (B, npoint, nsample)

        # Sample uniformly if requested
        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    if num_unique < self.nsample:
                        sample_ind = torch.randint(
                            0, num_unique, (self.nsample - num_unique,), dtype=torch.long, device=idx.device
                        )
                        all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                        idx[i_batch, i_region, :] = all_ind

        # Group xyz coordinates
        xyz_trans = xyz.transpose(1, 2).contiguous()  # (B, 3, N)
        grouped_xyz = self.grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)

        # Normalize to local coordinates
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if self.normalize_xyz:
            grouped_xyz /= self.radius

        # Group features
        if features is not None:
            grouped_features = self.grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have no features and not use xyz as a feature!"
            new_features = grouped_xyz

        # Return results
        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)

        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)
