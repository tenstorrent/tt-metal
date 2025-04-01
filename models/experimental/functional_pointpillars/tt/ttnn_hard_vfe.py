# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor
from models.experimental.functional_pointpillars.tt.ttnn_point_pillars_utils import get_paddings_indicator
import ttnn


class TtVFELayer:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_cfg: Optional[dict] = dict(type="BN1d", eps=1e-3, momentum=0.01),
        max_out: Optional[bool] = True,
        cat_max: Optional[bool] = True,
        parameters=None,
        device=None,
    ):
        self.device = device
        self.parameters = parameters
        self.cat_max = cat_max
        self.max_out = max_out
        # self.units = int(out_channels / 2)

        self.norm = parameters[
            "norm"
        ]  # nn.BatchNorm1d(out_channels, eps=norm_cfg["eps"], momentum=norm_cfg["momentum"])

        self.linear = ttnn.linear

    def __call__(self, inputs):
        device = self.device
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]

        x = self.linear(inputs, self.parameters["linear"]["weight"], memory_config=ttnn.L1_MEMORY_CONFIG)

        x = ttnn.from_torch(
            self.norm(ttnn.to_torch(ttnn.permute(x, (0, 2, 1)))),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        x = ttnn.permute(x, (0, 2, 1))
        pointwise = ttnn.relu(x)
        # [K, T, units]
        if self.max_out:
            aggregated = ttnn.max(pointwise, dim=1)
        else:
            # this is for fusion layer
            return pointwise

        if not self.cat_max:
            return ttnn.squeeze(aggregated, dim=1)
        else:
            # [K, 1, units]
            aggregated = ttnn.to_torch(aggregated)

            repeated = aggregated.repeat(1, voxel_count, 1)
            repeated = ttnn.from_torch(repeated, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
            concatenated = ttnn.concat([pointwise, repeated], dim=2)
            # [K, T, 2 * units]
            return concatenated


class TtHardVFE:
    def __init__(
        self,
        in_channels: int = 4,
        feat_channels: list = [],
        with_distance: bool = False,
        with_cluster_center: bool = False,
        with_voxel_center: bool = False,
        voxel_size: Tuple[float] = (0.2, 0.2, 4),
        point_cloud_range: Tuple[float] = (0, -40, -3, 70.4, 40, 1),
        norm_cfg: dict = dict(type="BN1d", eps=1e-3, momentum=0.01),
        mode: str = "max",
        fusion_layer: dict = None,
        return_point_feats: bool = False,
        parameters=None,
        device=None,
    ):
        self.device = device
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats

        # Need pillar (voxel) size and x/y offset to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            # TODO: pass norm_cfg to VFE
            # norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            if i == (len(feat_channels) - 2):
                cat_max = False
                max_out = True
                if fusion_layer:
                    max_out = False
            else:
                max_out = True
                cat_max = True
            vfe_layers.append(
                TtVFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    max_out=max_out,
                    cat_max=cat_max,
                    parameters=parameters["vfe_layers"][i],
                    device=device,
                )
            )
        self.vfe_layers = vfe_layers
        self.num_vfe = len(vfe_layers)

        self.fusion_layer = None
        if fusion_layer is not None:  # fusion_layer is None
            self.fusion_layer = MODELS.build(fusion_layer)

    def __call__(
        self,
        features: Tensor,
        num_points: Tensor,
        coors: Tensor,
        img_feats: Optional[Sequence[Tensor]] = None,
        img_metas: Optional[dict] = None,
    ) -> tuple:
        device = self.device
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            num_points_bf16 = ttnn.from_device(num_points)
            num_points_bf16 = ttnn.to_dtype(num_points_bf16, dtype=ttnn.bfloat16)
            num_points_bf16 = ttnn.to_device(num_points_bf16, device=device)
            points_mean = ttnn.div(ttnn.sum(features[:, :, :3], dim=1), ttnn.reshape(num_points_bf16, (-1, 1, 1)))
            # TODO: maybe also do cluster for reflectivity
            f_cluster = ttnn.sub(features[:, :, :3], points_mean)
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = ttnn.zeros((features.shape[0], features.shape[1], 3), device=device)
            f_center = ttnn.to_torch(f_center)
            f_center[:, :, 0] = ttnn.to_torch(features)[:, :, 0] - (
                ttnn.to_torch(coors)[:, 3].unsqueeze(1) * self.vx + self.x_offset
            )
            f_center[:, :, 1] = ttnn.to_torch(features)[:, :, 1] - (
                ttnn.to_torch(coors)[:, 2].unsqueeze(1) * self.vy + self.y_offset
            )
            f_center[:, :, 2] = ttnn.to_torch(features)[:, :, 2] - (
                ttnn.to_torch(coors)[:, 1].unsqueeze(1) * self.vz + self.z_offset
            )
            f_center = ttnn.from_torch(f_center, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        # Combine together feature decorations
        voxel_feats = ttnn.concat(features_ls, dim=-1)
        voxel_count = voxel_feats.shape[1]

        mask = get_paddings_indicator(ttnn.to_torch(num_points), voxel_count, axis=0)
        mask = ttnn.from_torch(mask, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32)
        mask = ttnn.from_device(mask)
        mask = ttnn.to_dtype(mask, dtype=ttnn.bfloat16)
        mask = ttnn.to_device(mask, device=device)
        voxel_feats = ttnn.multiply(voxel_feats, ttnn.unsqueeze(mask, -1))

        for i, vfe in enumerate(self.vfe_layers):
            voxel_feats = vfe(voxel_feats)

        if self.fusion_layer is not None and img_feats is not None:
            voxel_feats = self.fusion_with_mask(features, mask, voxel_feats, coors, img_feats, img_metas)

        return voxel_feats

    def fusion_with_mask(
        self,
        features: Tensor,
        mask: Tensor,
        voxel_feats: Tensor,
        coors: Tensor,
        img_feats: Sequence[Tensor],
        img_metas: Sequence[dict],
    ) -> Tensor:
        # the features is consist of a batch of points
        batch_size = coors[-1, 0] + 1
        points = []
        for i in range(batch_size):
            single_mask = coors[:, 0] == i
            points.append(features[single_mask][mask[single_mask]])

        point_feats = voxel_feats[mask]
        point_feats = self.fusion_layer(img_feats, points, point_feats, img_metas)

        voxel_canvas = voxel_feats.new_zeros(size=(voxel_feats.size(0), voxel_feats.size(1), point_feats.size(-1)))
        voxel_canvas[mask] = point_feats
        out = torch.max(voxel_canvas, dim=1)[0]

        return out
