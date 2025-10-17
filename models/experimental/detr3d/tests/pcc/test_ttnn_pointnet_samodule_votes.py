# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from loguru import logger
from models.common.utility_functions import comp_pcc, comp_allclose
from models.experimental.detr3d.reference.model_3detr import PointnetSAModuleVotes
from models.experimental.detr3d.ttnn.pointnet_samodule_votes import TtnnPointnetSAModuleVotes
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.detr3d.ttnn.custom_preprocessing import create_custom_mesh_preprocessor
from models.experimental.detr3d.common import load_torch_model_state


@pytest.mark.parametrize(
    "mlp, npoint, radius, nsample, bn, use_xyz, pooling, sigma, normalize_xyz, sample_uniformly, ret_unique_cnt,xyz_shape,features_shape, weight_key_prefix",
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
            "pre_encoder",
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
            "encoder.interim_downsampling",
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
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
    weight_key_prefix,
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
        device=device,
    )

    ttnn_model = TtnnPointnetSAModuleVotes(
        mlp[:],
        npoint,
        radius,
        nsample,
        use_xyz,
        pooling,
        normalize_xyz,
        sample_uniformly,
        ret_unique_cnt,
        torch_model,
        parameters,
        device,
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
