# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import run_for_wormhole_b0
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.pointpillars.common import load_torch_model, POINTPILLARS_L1_SMALL_SIZE

from models.experimental.pointpillars.tt.ttnn_hard_vfe import TtHardVFE
from models.experimental.pointpillars.tt.model_preprocessing import create_custom_preprocessor


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        True,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": POINTPILLARS_L1_SMALL_SIZE}], indirect=True)
@run_for_wormhole_b0()
def test_ttnn_hard_vfe(device, model_location_generator, use_pretrained_weight, reset_seeds):
    reference_model = load_torch_model(model_location_generator)

    reference_model = reference_model.pts_voxel_encoder

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device)
    )

    features = torch.load(f"models/experimental/pointpillars/inputs_weights/features.pt")
    num_points = torch.load("models/experimental/pointpillars/inputs_weights/num_points.pt")
    coors = torch.load("models/experimental/pointpillars/inputs_weights/coors.pt")
    img_feats = None
    img_metas = None  # It's not none, using none as we are not using this variable inside the hardvfe

    ttnn_features = ttnn.from_torch(features, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_num_points = ttnn.from_torch(num_points, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)
    ttnn_coors = ttnn.from_torch(coors, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)

    reference_output = reference_model(
        features=features, num_points=num_points, coors=coors, img_feats=img_feats, img_metas=img_metas
    )

    ttnn_model = TtHardVFE(
        in_channels=4,
        feat_channels=[64, 64],
        with_distance=False,
        voxel_size=[0.25, 0.25, 8],
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=[-50, -50, -5, 50, 50, 3],
        norm_cfg={"type": "BN1d", "eps": 0.001, "momentum": 0.01},
        parameters=parameters,
        device=device,
    )

    ttnn_output = ttnn_model(
        features=ttnn_features, num_points=ttnn_num_points, coors=ttnn_coors, img_feats=img_feats, img_metas=img_metas
    )

    passing, pcc = assert_with_pcc(reference_output, ttnn.to_torch(ttnn_output), 0.989)
    logger.info(f"Passing: {passing}, PCC: {pcc}")
