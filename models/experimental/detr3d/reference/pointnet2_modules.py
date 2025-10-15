# Copyright (c) Facebook, Inc. and its affiliates.
# TODO: Check copyright
""" Pointnet2 layers.
Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch
Extended with the following:
1. Uniform sampling in each local region (sample_uniformly)
2. Return sampled points indices to support votenet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from models.experimental.detr3d.reference.model_utils import (
    # QueryAndGroup,
    GroupAll,
    # GatherOperation,
    # FurthestPointSampling,
    Conv2d,
)
from models.experimental.detr3d.reference.torch_pointnet2_ops import (
    QueryAndGroup,
    GatherOperation,
    FurthestPointSampling,
)


class SharedMLP(nn.Sequential):
    def __init__(
        self,
        args: List[int],
        *,
        bn: bool = False,
        activation=nn.ReLU(inplace=True),
        preact: bool = False,
        first: bool = False,
        name: str = "",
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + "layer{}".format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation if (not first or not preact or (i != 0)) else None,
                    preact=preact,
                ),
            )


class PointnetSAModuleVotes(nn.Module):
    """Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes"""

    def __init__(
        self,
        *,
        mlp: List[int],
        npoint: int = None,
        radius: float = None,
        nsample: int = None,
        bn: bool = True,
        use_xyz: bool = True,
        pooling: str = "max",
        sigma: float = None,  # for RBF pooling
        normalize_xyz: bool = False,  # noramlize local XYZ with radius
        sample_uniformly: bool = False,
        ret_unique_cnt: bool = False,
    ):
        super().__init__()
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
            self.grouper = GroupAll(use_xyz, ret_grouped_xyz=True)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3
        self.mlp_module = SharedMLP(mlp_spec, bn=bn)
        self.gather_operation = GatherOperation()
        self.furthest_point_sample = FurthestPointSampling()

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor = None, inds: torch.Tensor = None
    ) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = self.furthest_point_sample(xyz, self.npoint)
        else:
            assert inds.shape[1] == self.npoint
        new_xyz = (
            self.gather_operation(xyz_flipped, inds).transpose(1, 2).contiguous() if self.npoint is not None else None
        )

        if not self.ret_unique_cnt:
            grouped_features, grouped_xyz = self.grouper(xyz, new_xyz, features)  # (B, C, npoint, nsample)
        else:
            grouped_features, grouped_xyz, unique_cnt = self.grouper(
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample), (B,3,npoint,nsample), (B,npoint)

        new_features = self.mlp_module(grouped_features)  # (B, mlp[-1], npoint, nsample)
        if self.pooling == "max":
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        elif self.pooling == "avg":
            new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        elif self.pooling == "rbf":
            # Use radial basis function kernel for weighted sum of features (normalized by nsample and sigma)
            # Ref: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
            rbf = torch.exp(
                -1 * grouped_xyz.pow(2).sum(1, keepdim=False) / (self.sigma**2) / 2
            )  # (B, npoint, nsample)
            new_features = torch.sum(new_features * rbf.unsqueeze(1), -1, keepdim=True) / float(
                self.nsample
            )  # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

        if not self.ret_unique_cnt:
            return new_xyz, new_features, inds
        else:
            return new_xyz, new_features, inds, unique_cnt
