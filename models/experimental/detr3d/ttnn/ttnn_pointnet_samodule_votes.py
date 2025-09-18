# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.detr3d.ttnn.ttnn_shared_mlp import TtnnSharedMLP


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
        centroids = ttnn.zeros((B, n_samples), device=device)
        distance = ttnn.ones(shape=[B, N], dtype=points.dtype, device=device) * 1e10


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

        self.grouper = TtnnQueryAndGroup(
            device,
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
        self.gather_operation = TtnnGroupingOperation()
        self.furthest_point_sample = TtnnFurthestPointSampling()
