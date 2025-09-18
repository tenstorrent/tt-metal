# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
<<<<<<< HEAD
import ttnn

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, comp_allclose
from models.experimental.detr3d.common import load_torch_model_state
from models.experimental.detr3d.reference.model_3detr import PointnetSAModuleVotes
from models.experimental.detr3d.ttnn.pointnet_samodule_votes import TtnnPointnetSAModuleVotes
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor


@pytest.mark.parametrize(
    "mlp, npoint, radius, nsample, normalize_xyz, xyz_shape,features_shape, weight_key_prefix",
=======
from models.experimental.detr3d.reference.detr3d_model import PointnetSAModuleVotes
from models.experimental.detr3d.ttnn.ttnn_pointnet_samodule_votes import TtnnPointnetSAModuleVotes
from models.experimental.detr3d.tests.pcc.test_ttnn_shared_mlp import (
    custom_preprocessor_whole_model as custom_preprocessor_shared_mlp,
)
from ttnn.model_preprocessing import preprocess_model_parameters


@pytest.mark.parametrize(
    "mlp, npoint, radius, nsample, bn, use_xyz, pooling, sigma, normalize_xyz, sample_uniformly, ret_unique_cnt,xyz_shape,features_shape,inds_shape",
>>>>>>> 3a871f2607 (added 3detr files from venkatesh/DETR3D_implementation)
    [
        (
            [0, 64, 128, 256],  # mlp
            2048,  # npoint
            0.2,  # radius
            64,  # nsample
<<<<<<< HEAD
            True,  # normalize_xyz
            (1, 20000, 3),  # xyz
            None,  # features
            "pre_encoder",
        ),
        (
            [256, 256, 256, 256],  # mlp
            1024,  # npoint
            0.4,  # radius
            32,  # nsample
            True,  # normalize_xyz
            (1, 2048, 3),  # xyz
            (1, 256, 2048),  # features
            "encoder.interim_downsampling",
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
=======
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
>>>>>>> 3a871f2607 (added 3detr files from venkatesh/DETR3D_implementation)
def test_pointnet_samodule_votes(
    mlp,
    npoint,
    radius,
    nsample,
<<<<<<< HEAD
    normalize_xyz,
    xyz_shape,
    features_shape,
    weight_key_prefix,
    device,
):
    torch_model = PointnetSAModuleVotes(
        radius=radius,
        nsample=nsample,
        npoint=npoint,
        mlp=mlp[:],
        normalize_xyz=normalize_xyz,
    )
    load_torch_model_state(torch_model, weight_key_prefix)

    xyz = torch.randn(xyz_shape)
    if features_shape is not None:
        features = torch.randn(features_shape)
    else:
        features = None
    ref_out = torch_model(xyz=xyz, features=features, inds=None)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model.mlp_module.to(torch.bfloat16),
        custom_preprocessor=create_custom_mesh_preprocessor(),
=======
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
>>>>>>> 3a871f2607 (added 3detr files from venkatesh/DETR3D_implementation)
        device=device,
    )

    ttnn_model = TtnnPointnetSAModuleVotes(
<<<<<<< HEAD
        radius=radius,
        nsample=nsample,
        npoint=npoint,
        mlp=mlp[:],
        normalize_xyz=normalize_xyz,
        parameters=parameters,
        device=device,
    )

    ttnn_features = None
    if features is not None:
        ttnn_features = ttnn.from_torch(
            features,
            dtype=ttnn.bfloat16,
            device=device,
        )
    tt_output = ttnn_model(xyz=xyz, features=ttnn_features, inds=None)

    ttnn_torch_out = []
    all_passing = True
    for idx, (tt_out, torch_out) in enumerate(zip(tt_output, ref_out)):
        if not isinstance(tt_out, torch.Tensor):
            tt_out = ttnn.to_torch(tt_out)
            tt_out = torch.reshape(tt_out, torch_out.shape)
        ttnn_torch_out.append(tt_out)

        passing, pcc_message = comp_pcc(torch_out, tt_out, 0.99)
        logger.info(f"Output {idx} PCC: {pcc_message}")
        logger.info(comp_allclose(torch_out, tt_out))

        if passing:
            logger.info(f"Output {idx} Test Passed!")
        else:
            logger.warning(f"Output {idx} Test Failed!")
            all_passing = False

    logger.info(f"Weight prefix: {weight_key_prefix}, npoint: {npoint}, radius: {radius}")
    assert all_passing, "One or more outputs failed PCC check with threshold 0.99"
=======
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
>>>>>>> 3a871f2607 (added 3detr files from venkatesh/DETR3D_implementation)
