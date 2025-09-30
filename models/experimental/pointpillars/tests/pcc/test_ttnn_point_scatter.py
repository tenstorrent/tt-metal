# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.pointpillars.common import load_torch_model, POINTPILLARS_L1_SMALL_SIZE
from models.experimental.pointpillars.tt.ttnn_point_pillars_scatter import TtPointPillarsScatter


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        True,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": POINTPILLARS_L1_SMALL_SIZE}], indirect=True)
@run_for_wormhole_b0()
def test_ttnn_point_scatter(device, model_location_generator, use_pretrained_weight, reset_seeds):
    reference_model = load_torch_model(model_location_generator)
    reference_model = reference_model.pts_middle_encoder

    voxel_features = torch.rand(6522, 64).to(dtype=torch.float32)

    coors = torch.rand((6522, 4)).to(dtype=torch.int32)

    batch_size = torch.tensor(1, dtype=torch.int32)

    ttnn_voxel_features = ttnn.from_torch(voxel_features, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_batch_size = ttnn.from_torch(batch_size, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)
    ttnn_coors = ttnn.from_torch(coors, device=device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)  # bfloat16

    reference_output = reference_model(voxel_features=voxel_features, coors=coors, batch_size=batch_size)

    ttnn_model = TtPointPillarsScatter(in_channels=64, output_shape=[400, 400], device=device)

    ttnn_output = ttnn_model(voxel_features=ttnn_voxel_features, coors=ttnn_coors, batch_size=1)

    passing, pcc = assert_with_pcc(reference_output, ttnn.to_torch(ttnn_output), 0.67)
    logger.info(f"Passing: {passing}, PCC: {pcc}")  # PCC - 0.6725618478186925
