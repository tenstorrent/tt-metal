# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


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
        topk_scores, topk_inds, smoothed, mp = self._decode_heatmaps(device, heatmaps)
        # fallback to torch

        scores_torch = ttnn.to_torch(topk_scores, dtype=torch.float32).flatten()
        inds_torch = ttnn.to_torch(topk_inds, dtype=torch.int32).flatten()
        positions_torch = ttnn.to_torch(positions, dtype=torch.float32)
        dimensions_torch = ttnn.to_torch(dimensions, dtype=torch.float32)
        angles_torch = ttnn.to_torch(angles, dtype=torch.float32)
        classids = torch.zeros(
            50, dtype=torch.int32
        )  # TODO add support for multiple classes by making classids an output of nms

        positions_torch = positions_torch[0, inds_torch // 159, inds_torch % 159, :]
        dimensions_torch = dimensions_torch[0, inds_torch // 159, inds_torch % 159, :]
        angles_torch = angles_torch[0, inds_torch // 159, inds_torch % 159]

        smoothed_torch = ttnn.to_torch(smoothed, dtype=torch.float32)
        mp_torch = ttnn.to_torch(mp, dtype=torch.float32)
        score_map = scores_torch > 0

        return (
            [
                scores_torch[score_map],
                classids[score_map],
                positions_torch[score_map],
                dimensions_torch[score_map],
                angles_torch[score_map],
            ],
            [smoothed_torch, mp_torch],
            ("objects", "scores", "classids", "positions", "dimensions", "angles"),
            ("smoothed", "maxpool"),
        )

    def _decode_heatmaps(self, device, heatmaps):
        topk_scores, topk_inds, smoothed, mp = self._non_maximum_suppression(device, heatmaps)
        # classids = torch.nonzero(peaks)[:, 0] #moved to level above
        return topk_scores, topk_inds, smoothed, mp

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

        mp = ttnn.max_pool2d(
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
            output_layout=ttnn.TILE_LAYOUT,
            ceil_mode=False,
            deallocate_input=False,
            reallocate_halo_output=True,
            return_indices=False,
        )
        heatmaps_flat = ttnn.reshape(heatmaps, (1, 1, -1, 1))
        heatmaps_flat = heatmaps_flat * ttnn.eq(smoothed, mp)
        topk_scores, topk_inds = ttnn.topk(heatmaps_flat, 50, dim=2, sorted=False)
        topk_scores = ttnn.threshold(topk_scores, self.nms_thresh, -1.0)

        return topk_scores, topk_inds, smoothed, mp

    def create_objects(self, scores, classids, positions, dimensions, angles):
        """Separate method to create ObjectData list from tensors"""
        objects = []
        for score, cid, pos, dim, ang in zip(scores, classids, positions, dimensions, angles):
            objects.append(ObjectData(self.classnames[cid], pos, dim, ang, score))

        return objects
