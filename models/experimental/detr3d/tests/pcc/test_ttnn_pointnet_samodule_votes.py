# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from models.experimental.detr3d.reference.detr3d_model import PointnetSAModuleVotes
from models.experimental.detr3d.ttnn.ttnn_pointnet_samodule_votes import TtnnPointnetSAModuleVotes
from models.experimental.detr3d.tests.pcc.test_ttnn_shared_mlp import (
    custom_preprocessor_whole_model as custom_preprocessor_shared_mlp,
)
from ttnn.model_preprocessing import preprocess_model_parameters


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
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
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
    device,
):
    torch_model = PointnetSAModuleVotes(
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
    weights_path = "models/experimental/detr3d/sunrgbd_masked_ep720.pth"
    state_dict = torch.load(weights_path)["model"]

    pointnet_state_dict = {k: v for k, v in state_dict.items() if (k.startswith("pre_encoder"))}
    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in pointnet_state_dict.items()]

    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.eval()

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

    torch_output = torch_model(xyz=xyz, features=features, inds=inds)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model.mlp_module,
        custom_preprocessor=custom_preprocessor_shared_mlp,
        device=device,
    )

    ttnn_model = TtnnPointnetSAModuleVotes(
        mlp[:],
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
        torch_model,
        parameters,
        device,
    )
