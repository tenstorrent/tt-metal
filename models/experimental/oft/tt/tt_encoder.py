# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from loguru import logger

from collections import namedtuple

from models.experimental.oft.tt.common import Conv

ObjectData = namedtuple("ObjectData", ["classname", "position", "dimensions", "angle", "score"])


def gaussian_kernel(sigma=1.0, trunc=2.0):
    width = round(trunc * sigma)
    x = torch.arange(-width, width + 1).float() / sigma
    kernel1d = torch.exp(-0.5 * x**2)
    kernel2d = kernel1d.view(1, -1) * kernel1d.view(-1, 1)

    return kernel2d / kernel2d.sum()


class TTObjectEncoder:
    def __init__(
        self,
        device,
        parameters,
        classnames=["Car"],
        pos_std=[0.5, 0.36, 0.5],
        log_dim_mean=[[0.42, 0.48, 1.35]],
        log_dim_std=[[0.085, 0.067, 0.115]],
        sigma=1.0,
        nms_thresh=0.05,
    ):
        self.classnames = classnames
        self.nclass = len(classnames)
        self.pos_std = torch.tensor(pos_std)
        self.log_dim_mean = torch.tensor(log_dim_mean)
        self.log_dim_std = torch.tensor(log_dim_std)

        self.pos_std_ttnn = ttnn.from_torch(
            self.pos_std,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=device,
        )
        self.log_dim_mean_ttnn = ttnn.from_torch(
            self.log_dim_mean,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=device,
        )
        self.log_dim_std_ttnn = ttnn.from_torch(
            self.log_dim_std,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=device,
        )

        self.sigma = sigma
        self.nms_thresh = nms_thresh  # is there a typo in refernece code? nms_tresh is passed but heatmaps is called with default value 0.05
        self.nms_conv = Conv(
            parameters.nms_conv,
            parameters.layer_args,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
        )
        self.max_peaks = 50
        self.thresh = 0.05

    def decode(self, device, heatmaps, pos_offsets, dim_offsets, ang_offsets, grid):
        positions = self._decode_positions(device, pos_offsets, grid)
        dimensions = self._decode_dimensions(device, dim_offsets)
        angles = self._decode_angles(device, ang_offsets)
        peaks_torch, max_inds, scores, smoothed, mp = self._decode_heatmaps(device, heatmaps)
        # fallback to torch
        classids = torch.nonzero(peaks_torch)[:, 0]

        scores_torch = ttnn.to_torch(scores, dtype=torch.float32)
        positions_torch = ttnn.to_torch(positions, dtype=torch.float32)
        dimensions_torch = ttnn.to_torch(dimensions, dtype=torch.float32)
        angles_torch = ttnn.to_torch(angles, dtype=torch.float32)

        scores_torch = scores_torch[peaks_torch]
        positions_torch = positions_torch[peaks_torch]
        dimensions_torch = dimensions_torch[peaks_torch]
        angles_torch = angles_torch[peaks_torch]

        return (
            [scores_torch, classids, positions_torch, dimensions_torch, angles_torch],
            [peaks_torch, max_inds, smoothed, mp],
            ("scores", "classids", "positions", "dimensions", "angles"),
            ("peaks", "max_inds", "smoothed", "mp"),
        )
        # THIS SHOULD BE ADDED BACK

        objects = list()
        for score, cid, pos, dim, ang in zip(scores_torch, classids, positions_torch, dimensions_torch, angles_torch):
            objects.append(ObjectData(self.classnames[cid], pos, dim, ang, score))

        return [
            objects,
            peaks_torch,
            peaks_torch,
            max_inds,
            scores_torch,
            classids,
            positions_torch,
            dimensions_torch,
            angles_torch,
            smoothed,
            mp,
        ], ("objects", "peaks", "max_inds", "scores", "classids", "positions", "dimensions", "angles", "smoothed", "mp")

    def _decode_heatmaps(self, device, heatmaps):
        peaks, max_inds, smoothed, mp = self._non_maximum_suppression(device, heatmaps)
        scores = heatmaps
        # classids = torch.nonzero(peaks)[:, 0] #moved to level above
        return peaks[0], max_inds, scores, smoothed, mp

    def _decode_positions(self, device, pos_offsets, grid):
        # Compute the center of each grid cell
        # perhaps could be moved to init block
        centers = grid[1:, 1:] + grid[:-1, :-1]
        centers = ttnn.div(centers, 2.0)
        # Un-normalize grid offsets
        pos_offsets = ttnn.permute(pos_offsets, (0, 2, 3, 1))
        positions = pos_offsets * self.pos_std_ttnn + centers
        return positions

    def _decode_dimensions(self, device, dim_offsets):
        dim_offsets = ttnn.permute(dim_offsets, (0, 2, 3, 1))
        coef = dim_offsets * self.log_dim_std_ttnn + self.log_dim_mean_ttnn
        dimensions = ttnn.exp(coef)
        return dimensions

    def _decode_angles(self, device, angle_offsets):
        cos = angle_offsets[:, 0, :, :]
        sin = angle_offsets[:, 1, :, :]
        atan2 = ttnn.atan2(sin, cos)
        return atan2

    def _non_maximum_suppression(self, device, heatmaps):
        heatmaps_4d = ttnn.unsqueeze(heatmaps, 0)
        n, c, h, w = heatmaps_4d.shape
        heatmaps_4d = ttnn.permute(heatmaps_4d, (0, 2, 3, 1))  # NHWC for conv/maxpool

        smoothed, out_h, out_w = self.nms_conv(device, heatmaps_4d)
        torch_smoothed = ttnn.to_torch(smoothed, dtype=torch.float32)

        mp, indices = ttnn.max_pool2d(
            input_tensor=smoothed,
            batch_size=n,
            input_h=h,
            input_w=w,
            channels=c,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            dilation=[1, 1],
            applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED if not smoothed.is_sharded() else None,
            ceil_mode=False,
            in_place_halo=False,
            deallocate_input=False,
            reallocate_halo_output=True,
            return_indices=True,
        )
        torch_mp = ttnn.to_torch(mp, dtype=torch.float32).permute(0, 3, 1, 2)

        # TODO(mbezulj): figure out ttnn way to handle this

        # fallback to torch to calculate max_peaks
        max_inds = ttnn.to_torch(indices, dtype=torch.int64).permute(0, 3, 1, 2).view(n, c, h, w)
        heatmaps_torch = ttnn.to_torch(heatmaps, dtype=torch.float32)
        # indices = indices.squeeze(0)
        # Find the pixels which correspond to the maximum indices
        _, height, width = heatmaps_torch.size()
        flat_inds = torch.arange(height * width).type_as(max_inds).view(height, width)
        peaks = (flat_inds == max_inds) & (heatmaps_torch > self.nms_thresh)

        # Keep only the top N peaks
        if peaks.long().sum() > self.max_peaks:
            scores = heatmaps_torch[peaks]
            scores, _ = torch.sort(scores, descending=True)
            peaks = peaks & (heatmaps_torch > scores[self.max_peaks - 1])

        logger.debug(f"tt_peaks {peaks.long().sum()}")
        return peaks, max_inds, torch_smoothed, torch_mp

    def create_objects(self, scores, classids, positions, dimensions, angles):
        """Separate method to create ObjectData list from tensors"""
        objects = []
        for score, cid, pos, dim, ang in zip(scores, classids, positions, dimensions, angles):
            objects.append(ObjectData(self.classnames[cid], pos, dim, ang, score))
        return objects
