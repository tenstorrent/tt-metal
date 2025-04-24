# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import run_for_wormhole_b0
from models.experimental.functional_pointpillars.reference.mvx_faster_rcnn import MVXFasterRCNN
from models.experimental.functional_pointpillars.tt.ttnn_point_pillars_scatter import TtPointPillarsScatter


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        True,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@run_for_wormhole_b0()
def test_ttnn_point_scatter(device, use_pretrained_weight, reset_seeds):
    reference_model = MVXFasterRCNN(
        pts_voxel_encoder=True,
        pts_middle_encoder=True,
        pts_backbone=True,
        pts_neck=True,
        pts_bbox_head=True,
        train_cfg=None,
    )
    if use_pretrained_weight == True:
        state_dict = torch.load("hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth")["state_dict"]
        reference_model.load_state_dict(state_dict)
    reference_model.eval()
    reference_model = reference_model.pts_middle_encoder

    voxel_features = torch.load(
        "models/experimental/functional_pointpillars/reference/voxel_features_point_pillars_scatter.pt"
    )  # torch.randn(4352, 64)
    coors = torch.load(
        "models/experimental/functional_pointpillars/reference/coors_point_pillars_scatter.pt"
    )  # torch.randn(4352, 4)
    batch_size = torch.tensor(1, dtype=torch.int32)

    ttnn_voxel_features = ttnn.from_torch(voxel_features, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_batch_size = ttnn.from_torch(batch_size, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)
    ttnn_coors = ttnn.from_torch(coors, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)

    reference_output = reference_model(voxel_features=voxel_features, coors=coors, batch_size=batch_size)

    ttnn_model = TtPointPillarsScatter(in_channels=64, output_shape=[400, 400], device=device)

    ttnn_output = ttnn_model(voxel_features=ttnn_voxel_features, coors=ttnn_coors, batch_size=1)

    passing, pcc = assert_with_pcc(reference_output, ttnn.to_torch(ttnn_output), 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")
