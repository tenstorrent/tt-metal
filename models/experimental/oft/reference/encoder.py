# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from collections import namedtuple

ObjectData = namedtuple("ObjectData", ["classname", "position", "dimensions", "angle", "score"])


def gaussian_kernel(sigma=1.0, trunc=2.0, dtype=torch.float32):
    width = round(trunc * sigma)
    x = torch.arange(-width, width + 1, dtype=dtype) / sigma
    kernel1d = torch.exp(-0.5 * torch.square(x))
    kernel2d = kernel1d.view(1, -1) * kernel1d.view(-1, 1)

    return kernel2d / kernel2d.sum()


class ObjectEncoder(nn.Module):
    def __init__(
        self,
        dtype,
        classnames=["Car"],
        pos_std=[0.5, 0.36, 0.5],
        log_dim_mean=[[0.42, 0.48, 1.35]],
        log_dim_std=[[0.085, 0.067, 0.115]],
        sigma=1.0,
        nms_thresh=0.05,
    ):
        super().__init__()
        self.classnames = classnames
        self.nclass = len(classnames)
        self.pos_std = torch.tensor(pos_std, dtype=dtype)
        self.log_dim_mean = torch.tensor(log_dim_mean, dtype=dtype)
        self.log_dim_std = torch.tensor(log_dim_std, dtype=dtype)

        self.sigma = sigma
        self.nms_thresh = nms_thresh

        num_class = len(classnames)
        kernel = gaussian_kernel(sigma, dtype=dtype)
        kernel = kernel.expand(num_class, num_class, -1, -1)
        padding = int((kernel.size(2) - 1) / 2)
        self.nms_conv = nn.Conv2d(
            num_class, num_class, kernel_size=kernel.size(2), stride=1, padding=padding, bias=False, dtype=dtype
        )
        # Initialize the conv weights with the gaussian kernel
        with torch.no_grad():
            self.nms_conv.weight.copy_(kernel)

    def decode(self, heatmaps, pos_offsets, dim_offsets, ang_offsets, grid):
        # Apply NMS to find positive heatmap locations
        peaks, max_inds, scores, classids, smoothed, mp = self._decode_heatmaps(heatmaps)
        positions = self._decode_positions(pos_offsets, peaks, grid)
        dimensions = self._decode_dimensions(dim_offsets, peaks)
        angles = self._decode_angles(ang_offsets, peaks)
        return [scores, classids, positions, dimensions, angles], [peaks, max_inds, smoothed, mp]

    def create_objects(self, scores, classids, positions, dimensions, angles):
        """Separate method to create ObjectData list from tensors"""
        objects = []
        for score, cid, pos, dim, ang in zip(scores, classids, positions, dimensions, angles):
            objects.append(ObjectData(self.classnames[cid], pos, dim, ang, score))
        return objects

    def _decode_heatmaps(self, heatmaps):
        peaks, max_inds, smoothed, mp = self._non_maximum_suppression(heatmaps)
        scores = heatmaps[peaks]
        classids = torch.nonzero(peaks)[:, 0]
        return peaks, max_inds, scores, classids, smoothed, mp

    def _decode_positions(self, pos_offsets, peaks, grid):
        # Compute the center of each grid cell
        centers = (grid[1:, 1:] + grid[:-1, :-1]) / 2.0

        # Un-normalize grid offsets
        positions = pos_offsets.permute(0, 2, 3, 1) * self.pos_std.to(grid) + centers
        return positions[peaks]

    def _decode_dimensions(self, dim_offsets, peaks):
        dim_offsets = dim_offsets.permute(0, 2, 3, 1)
        dimensions = torch.exp(dim_offsets * self.log_dim_std.to(dim_offsets) + self.log_dim_mean.to(dim_offsets))
        return dimensions[peaks]

    def _decode_angles(self, angle_offsets, peaks):
        cos, sin = torch.unbind(angle_offsets, 1)
        return torch.atan2(sin, cos)[peaks]

    def forward(self, heatmaps, pos_offsets, dim_offsets, ang_offsets, grid):
        """
        Forward pass that calls decode() to convert network outputs to object detections.

        Args:
            heatmaps: Class heatmaps with shape [num_classes, H, W]
            pos_offsets: Position offsets with shape [3, H, W]
            dim_offsets: Dimension offsets with shape [3, H, W]
            ang_offsets: Angle offsets with shape [2, H, W]
            grid: Grid coordinates

        Returns:
            List of ObjectData containing detected objects
        """
        [scores, classids, positions, dimensions, angles], _ = self.decode(
            heatmaps, pos_offsets, dim_offsets, ang_offsets, grid
        )

        objects = self.create_objects(scores, classids, positions, dimensions, angles)
        return objects

    def _non_maximum_suppression(self, heatmaps, thresh=0.05, max_peaks=50):
        # Smooth with a Gaussian kernel
        # num_class = heatmaps.size(0)
        # kernel = gaussian_kernel(sigma, dtype=heatmaps.dtype)
        # kernel = kernel.to(heatmaps)
        # kernel = kernel.expand(num_class, num_class, -1, -1)
        # smoothed = F.conv2d(heatmaps[None], kernel, padding=int((kernel.size(2) - 1) / 2))
        smoothed = self.nms_conv(heatmaps[None])
        # Max pool over the heatmaps
        mp, max_inds = F.max_pool2d(smoothed, 3, stride=1, padding=1, return_indices=True)
        max_inds = max_inds.squeeze(0)

        _, height, width = heatmaps.size()
        flat_inds = torch.arange(height * width).type_as(max_inds).view(height, width)
        peaks = flat_inds == max_inds

        peaks = peaks & (heatmaps > self.nms_thresh)
        if peaks.long().sum() > max_peaks:
            scores = heatmaps[peaks]
            scores, _ = torch.sort(scores, descending=True)
            peaks = peaks & (heatmaps > scores[max_peaks - 1])

        logger.debug(f"ref_peaks {peaks.long().sum()}")
        return peaks, max_inds, smoothed, mp
