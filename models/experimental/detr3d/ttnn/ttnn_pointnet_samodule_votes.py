# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.detr3d.ttnn.ttnn_shared_mlp import TtnnSharedMLP
from models.experimental.detr3d.reference.model_utils import (
    QueryAndGroup,
    FurthestPointSampling,
)


def _fallback_furthestpointsampling(
    points,
    num_samples,
    device,
):
    furthestpointsampling = FurthestPointSampling()
    points_torch = ttnn.to_torch(points)
    num_samples,
    centroids = furthestpointsampling(points_torch, num_samples)
    return ttnn.from_torch(centroids, dtype=ttnn.uint32, device=device)


def _fallback_queryandgroup(xyz, new_xyz, features, device, QnG):
    xyz_torch = ttnn.to_torch(xyz)
    new_xyz_torch, features_torch = None, None
    if new_xyz is not None:
        new_xyz_torch = ttnn.to_torch(new_xyz)
    if features is not None:
        features_torch = ttnn.to_torch(features)

    grouped_features_torch, grouped_xyz_torch = QnG(xyz_torch, new_xyz_torch, features_torch)
    grouped_features = ttnn.from_torch(
        grouped_features_torch,
        dtype=ttnn.bfloat16,
        device=device,
    )
    grouped_xyz = ttnn.from_torch(
        grouped_xyz_torch,
        dtype=ttnn.bfloat16,
        device=device,
    )

    return grouped_features, grouped_xyz


class TtnnBallQuery:
    def __init__(
        self,
        device,
        radius,
        nsample,
    ):
        self.device = device
        self.radius = radius
        self.nsample = nsample

    def __call__(self, xyz, new_xyz):
        b, m, _ = new_xyz.shape
        _, n, _ = xyz.shape
        device = new_xyz.device
        radius2 = self.radius * self.radius

        unsqueezed_new_xyz = ttnn.unsqueeze(new_xyz, dim=2)
        unsqueezed_xyz = ttnn.unsqueeze(xyz, dim=1)
        diff = unsqueezed_new_xyz - unsqueezed_xyz
        diff = ttnn.pow(diff, 2)
        dist2 = ttnn.sum(diff, dim=3)
        mask = dist2 < radius2

        idx = ttnn.zeros((b, m, self.nsample), dtype=ttnn.int32, device=self.device)

        arange_n = ttnn.arange(n, device=self.device)
        arange_n = ttnn.reshape(arange_n, (1, 1, n))
        arange_n = ttnn.expand(arange_n, (b, m, n))

        arange_n_like = ttnn.full_like(arange_n, n + 1)
        arange_n_masked = ttnn.where(mask, arange_n, arange_n_like)
        sorted_indices, _ = ttnn.sort(arange_n_masked, dim=2)
        first_nsample = sorted_indices[:, :, : self.nsample]

        invalid_mask = first_nsample == (n + 1)
        first_valid = first_nsample[:, :, 0]
        first_valid = ttnn.unsqueeze(first_valid, dim=2)
        first_valid = ttnn.expand(first_valid, first_nsample.shape)
        first_nsample[invalid_mask] = first_valid[invalid_mask]

        return first_nsample.to(ttnn.int32)


class TtnnGatherOperation:
    def __call__(self, points, idx):
        B, C, N = points.shape
        M = idx.shape[1]
        # idx = idx.to(ttnn.int32)
        idx_expand = ttnn.unsqueeze(idx, 1)
        idx_expand = ttnn.expand(idx_expand, (B, C, M))
        points = ttnn.to_layout(points, ttnn.TILE_LAYOUT)
        idx_expand = ttnn.to_layout(idx_expand, ttnn.TILE_LAYOUT)
        output = ttnn.gather(points, 2, idx_expand)

        return output


class TtnnGroupingOperation:
    def __call__(self, points, idx):
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


class TtnnQueryAndGroup:
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
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        self.ball_query = TtnnBallQuery(device, self.radius, self.nsample)
        self.grouping_operation = TtnnGroupingOperation()
        if self.ret_unique_cnt:
            assert self.sample_uniformly

    def __call__(self, xyz, new_xyz, features):
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


class TtnnFurthestPointSampling:
    def __call__(self, points, n_samples, device):
        B, N, _ = points.shape
        centroids = ttnn.zeros((B, n_samples), dtype=ttnn.int32, device=device)
        distance = ttnn.ones(shape=[B, N], dtype=points.dtype, device=device) * 1e10
        farthest = ttnn.zeros((B,), dtype=ttnn.int32, device=device)
        batch_indices = ttnn.arange(B, dtype=ttnn.int32, device=device)

        centroids = []
        for i in range(n_samples):
            # centroids[:, i] = farthest
            centroids.append(farthest)
            # centroid = points[batch_indices, farthest, :]
            centroid = []
            for b in range(B):
                batch_center = points[b, farthest[b], :]
                centroid.append(batch_center)
            centroid = ttnn.concat(centroid)
            centroid = ttnn.reshape(centroid, (B, 1, 3))
            dist = ttnn.sum((points - centroid) ** 2, dim=2)
            distance = ttnn.min(distance, dist)
            farthest = ttnn.max(distance, dim=1)[1]

        # centroids = ttnn.concat(centroids)

        return ttnn.concat(centroids)


class TtnnPointnetSAModuleVotes:
    def __init__(
        self,
        mlp,
        npoint,
        radius,
        nsample,
        bn,
        use_xyz,
        pooling,
        sigma,
        normalize_xyz,
        sample_uniformly,
        ret_unique_cnt,
        module,
        parameters,
        device,
    ):
        self.device = device
        self.parameters = parameters
        params = {k: v for k, v in locals().items() if k != "self"}
        print("ttnn PointnetSAModuleVotes init is called with params:")
        for k, v in params.items():
            print(f"  {k}: {v}")

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius / 2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt

        # self.grouper = TtnnQueryAndGroup(
        #     device,
        #     radius,
        #     nsample,
        #     use_xyz=use_xyz,
        #     ret_grouped_xyz=True,
        #     normalize_xyz=normalize_xyz,
        #     sample_uniformly=sample_uniformly,
        #     ret_unique_cnt=ret_unique_cnt,
        # )
        self.grouper = self.QnG_init(
            radius,
            nsample,
            use_xyz=use_xyz,
            ret_grouped_xyz=True,
            normalize_xyz=normalize_xyz,
            sample_uniformly=sample_uniformly,
            ret_unique_cnt=ret_unique_cnt,
        )
        mlp_spec = mlp
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3

        self.mlp_module = TtnnSharedMLP(module.mlp_module, parameters, device)
        self.gather_operation = TtnnGatherOperation()
        self.furthest_point_sample = TtnnFurthestPointSampling()

    def QnG_init(
        self,
        radius,
        nsample,
        use_xyz,
        ret_grouped_xyz,
        normalize_xyz,
        sample_uniformly,
        ret_unique_cnt,
    ):
        return QueryAndGroup(
            radius,
            nsample,
            use_xyz=use_xyz,
            ret_grouped_xyz=ret_grouped_xyz,
            normalize_xyz=normalize_xyz,
            sample_uniformly=sample_uniformly,
            ret_unique_cnt=ret_unique_cnt,
        )

    def __call__(self, xyz, features=None, inds=None):
        xyz_flipped = ttnn.transpose(xyz, 1, 2)
        if inds is None:
            inds = _fallback_furthestpointsampling(xyz, self.npoint, self.device)
            # inds = self.furthest_point_sample(xyz, self.npoint, self.device)
        else:
            assert inds.shape[1] == self.npoint

        new_xyz = None
        if self.npoint is not None:
            new_xyz = self.gather_operation(xyz_flipped, inds)
            new_xyz = ttnn.transpose(new_xyz, 1, 2)

        if not self.ret_unique_cnt:
            grouped_features, grouped_xyz = _fallback_queryandgroup(
                xyz, new_xyz, features, device=self.device, QnG=self.grouper
            )
            # grouped_features, grouped_xyz = self.grouper(xyz, new_xyz, features)  # (B, C, npoint, nsample)
        else:
            grouped_features, grouped_xyz, unique_cnt = self.grouper(
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample), (B,3,npoint,nsample), (B,npoint)
        # print("input to shared mlpforward ", grouped_features.shape)
        grouped_features = ttnn.permute(grouped_features, (0, 2, 3, 1))
        new_features = self.mlp_module(grouped_features)  # (B, mlp[-1], npoint, nsample)

        if self.pooling == "max":
            partial_maxpool_out = []
            num_maxpool_slice = 2
            new_features = ttnn.reshape(new_features, (1, 2048, 64, 256))
            slice_h = 2048 // num_maxpool_slice
            B, H, W, C = (1, slice_h, 64, 256)
            for slice in range(num_maxpool_slice):
                slice_input = new_features[:, slice_h * slice : slice_h * (slice + 1), :, :]
                slice_input = ttnn.reallocate(slice_input)
                slice_input = ttnn.reshape(slice_input, (1, 1, B * H * W, C))
                partial_maxpool_out.append(
                    ttnn.max_pool2d(
                        input_tensor=slice_input,
                        batch_size=1,
                        input_h=slice_h,
                        input_w=64,
                        channels=256,
                        kernel_size=[1, 64],
                        stride=[1, 64],
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
            )
