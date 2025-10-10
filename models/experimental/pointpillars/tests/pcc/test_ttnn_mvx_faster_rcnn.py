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
from models.experimental.pointpillars.tt.ttnn_mvx_faster_rcnn import TtMVXFasterRCNN
from models.experimental.pointpillars.tt.ttnn_point_pillars_utils import TtLiDARInstance3DBoxes
from models.experimental.pointpillars.tt.model_preprocessing import create_custom_preprocessor


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        True,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": POINTPILLARS_L1_SMALL_SIZE}], indirect=True)
@run_for_wormhole_b0()
def test_ttnn_mvx_faster_rcnn(device, model_location_generator, use_pretrained_weight, reset_seeds):
    reference_model = load_torch_model(model_location_generator)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device)
    )

    batch_inputs_dict = torch.load("models/experimental/pointpillars/inputs_weights/batch_inputs_dict_orig.pth")
    batch_data_samples_modified = torch.load(
        "models/experimental/pointpillars/inputs_weights/batch_data_samples_orig.pth", weights_only=False
    )
    reference_output = reference_model(
        batch_inputs_dict=batch_inputs_dict, batch_data_samples=batch_data_samples_modified
    )

    ttnn_batch_inputs_dict = batch_inputs_dict.copy()
    ttnn_batch_inputs_dict["points"][0] = ttnn.from_torch(
        ttnn_batch_inputs_dict["points"][0], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    ttnn_batch_inputs_dict["voxels"]["num_points"] = ttnn.from_torch(
        ttnn_batch_inputs_dict["voxels"]["num_points"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32, device=device
    )
    ttnn_batch_inputs_dict["voxels"]["voxel_centers"] = ttnn.from_torch(
        ttnn_batch_inputs_dict["voxels"]["voxel_centers"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    ttnn_batch_inputs_dict["voxels"]["voxels"] = ttnn.from_torch(
        ttnn_batch_inputs_dict["voxels"]["voxels"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    ttnn_batch_inputs_dict["voxels"]["coors"] = ttnn.from_torch(
        ttnn_batch_inputs_dict["voxels"]["coors"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32, device=device
    )
    ttnn_batch_data_samples_modified = batch_data_samples_modified.copy()
    # ttnn_batch_data_samples_modified[0].box_type_3d = TtLiDARInstance3DBoxes
    ttnn_batch_data_samples_modified[0].metainfo["box_type_3d"] = TtLiDARInstance3DBoxes

    ttnn_model = TtMVXFasterRCNN(
        pts_voxel_encoder=True,
        pts_middle_encoder=True,
        pts_backbone=True,
        pts_neck=True,
        pts_bbox_head=True,
        train_cfg=None,
        parameters=parameters,
        device=device,
    )
    ttnn_output = ttnn_model(
        batch_inputs_dict=ttnn_batch_inputs_dict, batch_data_samples=ttnn_batch_data_samples_modified
    )

    for i in range(len(ttnn_output)):
        for j in range(len(ttnn_output[i])):
            ttnn_output[i][j] = ttnn.to_torch(ttnn_output[i][j])
            ttnn_output[i][j] = ttnn_output[i][j].permute(0, 3, 1, 2)
            ttnn_output[i][j] = ttnn_output[i][j].to(dtype=torch.float)

    for i in range(len(ttnn_output)):
        for j in range(len(ttnn_output[i])):
            output_temp = ttnn_output[i][j]
            passing, pcc = assert_with_pcc(reference_output[i][j], output_temp, 0.95)
            logger.info(f"Passing: {passing}, PCC: {pcc}")
