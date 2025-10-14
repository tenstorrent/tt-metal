# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from models.experimental.detr3d.reference.model_3detr import PointnetSAModuleVotes as ref_model
from models.experimental.detr3d.source.detr3d.third_party.pointnet2.pointnet2_modules import (
    PointnetSAModuleVotes as org_model,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "mlp, npoint, radius, nsample, bn, use_xyz, pooling, sigma, normalize_xyz, sample_uniformly, ret_unique_cnt,xyz_shape,features_shape,inds_shape",
    [
        (
            [0, 64, 128, 256],  # mlp
            2048,  # npoint
            0.2,  # radius
            64,  # nsample
            True,  # bn
            True,  # use_xyz
            "max",  # pooling
            None,  # sigma
            True,  # normalize_xyz
            False,  # sample_uniformly
            False,  # ret_unique_cnt
            (1, 20000, 3),  # xyz
            None,  # features
            None,  # inds
        ),
        (
            [256, 256, 256, 256],  # mlp
            1024,  # npoint
            0.4,  # radius
            32,  # nsample
            True,  # bn
            True,  # use_xyz
            "max",  # pooling
            None,  # sigma
            True,  # normalize_xyz
            False,  # sample_uniformly
            False,  # ret_unique_cnt
            (1, 2048, 3),  # xyz
            (1, 256, 2048),  # features
            None,  # inds
        ),
    ],
)
def test_pointnet_samodule_votes(
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
    xyz_shape,
    features_shape,
    inds_shape,
):
    org_module = org_model(
        mlp=mlp[:],
        npoint=npoint,
        radius=radius,
        nsample=nsample,
        bn=bn,
        use_xyz=use_xyz,
        pooling=pooling,
        sigma=sigma,
        normalize_xyz=normalize_xyz,
        sample_uniformly=sample_uniformly,
        ret_unique_cnt=ret_unique_cnt,
    ).to(torch.bfloat16)
    org_module.eval()
    print("model is", org_module)
    ref_module = ref_model(
        mlp=mlp[:],
        npoint=npoint,
        radius=radius,
        nsample=nsample,
        bn=bn,
        use_xyz=use_xyz,
        pooling=pooling,
        sigma=sigma,
        normalize_xyz=normalize_xyz,
        sample_uniformly=sample_uniformly,
        ret_unique_cnt=ret_unique_cnt,
    ).to(torch.bfloat16)
    ref_module.eval()
    ref_module.load_state_dict(org_module.state_dict())
    if xyz_shape is not None:
        xyz = torch.randn(xyz_shape, dtype=torch.bfloat16)
    else:
        xyz = None

    if features_shape is not None:
        features = torch.randn(features_shape, dtype=torch.bfloat16)
    else:
        features = None

    if inds_shape is not None:
        inds = torch.randn(inds_shape, dtype=torch.bfloat16)
    else:
        inds = None
    org_out = org_module(xyz=xyz, features=features, inds=inds)
    ref_out = ref_module(xyz=xyz, features=features, inds=inds)
    print(assert_with_pcc(org_out[0], ref_out[0], 1.0))
