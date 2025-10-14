# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.detr3d.ttnn.shared_mlp import TtnnSharedMLP
from models.experimental.detr3d.reference.model_utils import (
    QueryAndGroup,
    GatherOperation,
    FurthestPointSampling,
)
from typing import List


class TtnnBallQuery(LightweightModule):
    def __init__(
        self,
        device,
        radius,
        nsample,
    ):
        super().__init__()
        self.device = device
        self.radius = radius
        self.nsample = nsample

    def forward(self, xyz, new_xyz):
        # Get shapes
        b, m, _ = new_xyz.shape
        _, n, _ = xyz.shape
        radius2 = self.radius * self.radius

        # Compute pairwise distances
        # new_xyz: (b, m, 3) -> (b, m, 1, 3)
        new_xyz_expanded = ttnn.unsqueeze(new_xyz, 2)
        # xyz: (b, n, 3) -> (b, 1, n, 3)
        xyz_expanded = ttnn.unsqueeze(xyz, 1)

        # Compute difference: (b, m, n, 3)
        diff = ttnn.subtract(new_xyz_expanded, xyz_expanded)

        # Compute squared distances: (b, m, n)
        diff_squared = ttnn.multiply(diff, diff)
        dist2 = ttnn.sum(diff_squared, dim=3)

        # Create mask for points within radius
        radius2_tensor = ttnn.full((b, m, n), radius2, dtype=ttnn.float32, device=self.device, layout=ttnn.TILE_LAYOUT)
        mask = ttnn.lt(dist2, radius2_tensor)

        # Create index tensor
        arange_n = ttnn.arange(0, n, dtype=ttnn.int32, device=self.device)
        arange_n = ttnn.reshape(arange_n, (1, 1, n))
        arange_n = ttnn.expand(arange_n, (b, m, n))

        # Apply mask - set invalid indices to n+1
        invalid_value = ttnn.full((b, m, n), n + 1, dtype=ttnn.int32, device=self.device, layout=ttnn.TILE_LAYOUT)
        arange_n_masked = ttnn.where(mask, arange_n, invalid_value)

        # Sort to get closest indices first
        sorted_indices = ttnn.sort(arange_n_masked, dim=2)
        first_nsample = sorted_indices[:, :, : self.nsample]

        # Handle invalid indices by replacing with first valid index
        invalid_mask = ttnn.eq(first_nsample, invalid_value[:, :, : self.nsample])
        first_valid = ttnn.unsqueeze(first_nsample[:, :, 0], 2)
        first_valid = ttnn.expand(first_valid, first_nsample.shape)
        result = ttnn.where(invalid_mask, first_valid, first_nsample)

        return result


class TtnnGatherOperation(LightweightModule):
    def __init__(self):
        super().__init__()

    def forward(self, points, idx):
        B, C, N = points.shape
        M = idx.shape[1]
        # idx = ttnn.to_layout(idx, ttnn.TILE_LAYOUT)
        # idx = ttnn.typecast(idx, ttnn.uint32)
        idx_expand = ttnn.unsqueeze(idx, 1)
        idx_expand = ttnn.expand(idx_expand, (B, C, M))
        points = ttnn.to_layout(points, ttnn.TILE_LAYOUT)
        idx_expand = ttnn.to_layout(idx_expand, ttnn.TILE_LAYOUT)
        output = ttnn.gather(points, 2, idx_expand)

        return output


class TtnnGroupingOperation(LightweightModule):
    def __init__(self):
        super().__init__()

    def forward(self, points, idx):
        B, C, N = points.shape
        _, npoint, nsample = idx.shape

        idx = idx.to(ttnn.int32)
        points_flat = ttnn.reshape(points, (B * C, N))
        idx_flat = ttnn.reshape(idx, (B, npoint * nsample))

        idx_expand = ttnn.unsqueeze(idx_flat, dim=1)
        idx_expand = ttnn.expand(idx_expand, (-1, C, -1))
        idx_expand = ttnn.reshape(idx_expand, (B * C, npoint * nsample))

        out_flat = ttnn.gather(points_flat, 1, idx_expand)
        output = ttnn.reshape(out_flat, (B, C, npoint, nsample))
        return output


class TtnnQueryAndGroup(LightweightModule):
    def __init__(
        self,
        device,
        radius,
        nsample,
        use_xyz=True,
        ret_grouped_xyz=False,
        normalize_xyz=False,
        sample_uniformly=False,
        ret_unique_cnt=False,
    ):
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        self.ball_query = TtnnBallQuery(device, self.radius, self.nsample)
        self.grouping_operation = TtnnGroupingOperation()
        if self.ret_unique_cnt:
            assert self.sample_uniformly

    def forward(self, xyz, new_xyz, features):
        idx = self.ball_query(xyz, new_xyz)
        xyz_trans = ttnn.permute(xyz, (1, 2))

        grouped_xyz = self.grouping_operation(xyz_trans, idx)
        new_xyz_trans = ttnn.permute(new_xyz, (1, 2))
        new_xyz_trans = ttnn.unsqueeze(new_xyz_trans, dim=-1)
        grouped_xyz -= new_xyz_trans
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = self.grouping_operation(features, idx)
            if self.use_xyz:
                new_features = ttnn.concat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        ret.append(grouped_xyz)
        return tuple(ret)


class TtnnFurthestPointSampling(LightweightModule):
    def __init__(self):
        super().__init__()

    def forward(self, points: ttnn.Tensor, n_samples: int, device):
        B, N, _ = points.shape

        # Initialize centroids tensor
        centroids = ttnn.zeros((B, n_samples), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Initialize distance tensor with large values
        distance = ttnn.full((B, N), fill_value=1e10, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        # Initialize farthest indices
        farthest = ttnn.zeros((B,), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

        # Create batch indices
        batch_indices = ttnn.arange(0, B, dtype=ttnn.int32, device=device)
        batch_indices = ttnn.reshape(batch_indices, (B, 1))

        centroid_list = []
        for i in range(n_samples):
            # Create index tensor with proper shape for scatter
            # The index needs to have shape (B, 1) to match the src tensor shape
            index_tensor = ttnn.full((B, 1), fill_value=i, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

            # Reshape farthest to match expected src shape for scatter
            farthest_reshaped = ttnn.reshape(farthest, (B, 1))
            farthest_reshaped = ttnn.to_layout(farthest_reshaped, ttnn.TILE_LAYOUT)
            farthest_reshaped = ttnn.typecast(farthest_reshaped, ttnn.bfloat16)

            # Store current farthest points as centroids
            centroids = ttnn.scatter(centroids, dim=1, index=index_tensor, src=farthest_reshaped)

            # Get current centroid coordinates using gather
            # Create index tensor with shape (B, 1, 3) where each slice along dim=2 has the same index
            farthest_indices = ttnn.unsqueeze(farthest, -1)
            # Expand to (B, 1, 3) by repeating the index for each coordinate
            farthest_indices = ttnn.repeat(farthest_indices, ttnn.Shape((B, 1, 3)))
            farthest_indices = ttnn.pad(farthest_indices, [(0, 0), (0, 0), (0, 29)], 0)
            farthest_indices = ttnn.to_layout(farthest_indices, ttnn.TILE_LAYOUT)
            farthest_indices = ttnn.typecast(farthest_indices, ttnn.uint32)
            points_padded = ttnn.pad(points, [(0, 0), (0, 0), (0, 29)], 0)
            points_padded = ttnn.to_layout(points_padded, ttnn.TILE_LAYOUT)
            centroid = ttnn.gather(points_padded, dim=1, index=farthest_indices)
            centroid = centroid[:, :, :3]

            centroid_list.append(farthest)

            # Calculate squared distances to current centroid
            diff = points - centroid
            diff = ttnn.to_layout(diff, ttnn.TILE_LAYOUT)
            dist = ttnn.sum(ttnn.pow(diff, 2), dim=2)

            # Update minimum distances
            dist = ttnn.to_layout(dist, ttnn.ROW_MAJOR_LAYOUT)
            distance = ttnn.minimum(distance, dist)

            # Find farthest point for next iteration
            farthest = ttnn.argmax(distance, dim=1)

        centroids = ttnn.typecast(centroids, ttnn.int32)

        return centroids


class TtnnPointnetSAModuleVotes(LightweightModule):
    def __init__(
        self,
        mlp: List[int],
        npoint: int = None,
        radius: float = None,
        nsample: int = None,
        use_xyz: bool = True,
        pooling: str = "max",
        normalize_xyz: bool = False,  # noramlize local XYZ with radius
        sample_uniformly: bool = False,
        ret_unique_cnt: bool = False,
        module=None,
        parameters=None,
        device=None,
    ):
        super().__init__()
        self.device = device
        self.parameters = parameters
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt
        self.sample_uniformly = sample_uniformly

        if npoint is not None:
            self.grouper = QueryAndGroup(
                radius,
                nsample,
                use_xyz=use_xyz,
                ret_grouped_xyz=True,
                normalize_xyz=normalize_xyz,
                sample_uniformly=sample_uniformly,
                ret_unique_cnt=ret_unique_cnt,
            )
        else:
            raise NotImplementedError("Not supported currently")

        mlp_spec = mlp
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3

        self.mlp_module = TtnnSharedMLP(module.mlp_module, parameters, device)
        self.gather_operation = GatherOperation()
        self.furthest_point_sample = FurthestPointSampling()

    def forward(self, xyz, features=None, inds=None):
        if not isinstance(xyz, torch.Tensor):
            xyz = ttnn.to_torch(xyz)
        if not isinstance(features, torch.Tensor) and features is not None:
            features = ttnn.to_torch(features)
        if not isinstance(inds, torch.Tensor) and inds is not None:
            inds = ttnn.to_torch(inds)

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = self.furthest_point_sample(xyz, self.npoint)
        else:
            assert inds.shape[1] == self.npoint
        new_xyz = (
            self.gather_operation(xyz_flipped, inds).transpose(1, 2).contiguous() if self.npoint is not None else None
        )

        unique_cnt = None
        if not self.ret_unique_cnt:
            grouped_features, grouped_xyz = self.grouper(xyz, new_xyz, features)  # (B, C, npoint, nsample)
        else:
            grouped_features, grouped_xyz, unique_cnt = self.grouper(
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample), (B,3,npoint,nsample), (B,npoint)

        if inds is not None:
            inds = ttnn.from_torch(inds, dtype=ttnn.bfloat16, device=self.device)
        grouped_features = torch.permute(grouped_features, (0, 2, 3, 1))
        grouped_features = ttnn.from_torch(grouped_features, dtype=ttnn.bfloat16, device=self.device)
        grouped_xyz = ttnn.from_torch(grouped_xyz, dtype=ttnn.bfloat16, device=self.device)
        if unique_cnt is not None:
            unique_cnt = ttnn.from_torch(unique_cnt, dtype=ttnn.bfloat16, device=self.device)

        new_features = self.mlp_module(grouped_features)  # (B, mlp[-1], npoint, nsample)

        if self.pooling == "max":
            if new_features.shape[1] == 2048:
                partial_maxpool_out = []
                num_maxpool_slice = 2
                slice_h = new_features.shape[-3] // num_maxpool_slice
                B, H, W, C = (new_features.shape[-4], slice_h, new_features.shape[-2], new_features.shape[-1])
                for slice in range(num_maxpool_slice):
                    slice_input = new_features[:, slice_h * slice : slice_h * (slice + 1), :, :]
                    slice_input = ttnn.reallocate(slice_input)
                    slice_input = ttnn.reshape(slice_input, (1, 1, B * H * W, C))
                    partial_maxpool_out.append(
                        ttnn.max_pool2d(
                            input_tensor=slice_input,
                            batch_size=B,
                            input_h=slice_h,
                            input_w=W,
                            channels=C,
                            kernel_size=[1, W],
                            stride=[1, W],
                            padding=[0, 0],
                            dilation=[1, 1],
                            applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                    )

                for i in range(len(partial_maxpool_out)):
                    partial_maxpool_out[i] = ttnn.reshape(partial_maxpool_out[i], (B, H, 1, C))

                new_features = ttnn.concat((partial_maxpool_out), dim=1)
            else:
                new_features = ttnn.to_torch(new_features)
                new_features = torch.permute(new_features, (0, 3, 1, 2))
                new_features = torch.nn.functional.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
                new_features = torch.permute(new_features, (0, 2, 3, 1))
                new_features = ttnn.from_torch(new_features, dtype=ttnn.bfloat16, device=self.device)
        else:
            raise NotImplementedError("Currently only Maxpool is supported")
        new_features = ttnn.permute(new_features, (0, 3, 1, 2))
        new_features = ttnn.squeeze(new_features, -1)  # (B, mlp[-1], npoint)

        if not self.ret_unique_cnt:
            return new_xyz, new_features, inds
        else:
            return (
                new_xyz,
                new_features,
                inds,
                unique_cnt,
            )
