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
from models.experimental.pointpillars.tt.ttnn_anchor3d_head import TtAnchor3DHead
from models.experimental.pointpillars.tt.model_preprocessing import create_custom_preprocessor


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        True,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": POINTPILLARS_L1_SMALL_SIZE}], indirect=True)
@run_for_wormhole_b0()
def test_ttnn_anchor3d_head(device, model_location_generator, use_pretrained_weight, reset_seeds):
    reference_model = load_torch_model(model_location_generator)
    reference_model = reference_model.pts_bbox_head

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device)
    )

    x = [torch.randn(1, 256, 200, 200), torch.randn(1, 256, 100, 100), torch.randn(1, 256, 50, 50)]

    ttnn_x = x[:]
    for i in range(len(ttnn_x)):
        input_0 = ttnn_x[i].permute(0, 2, 3, 1)
        ttnn_x[i] = ttnn.from_torch(input_0, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    reference_output = reference_model(x=x)

    ttnn_model = TtAnchor3DHead(
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator={
            "type": "AlignedAnchor3DRangeGenerator",
            "ranges": [[-50, -50, -1.8, 50, 50, -1.8]],
            "scales": [1, 2, 4],
            "sizes": [[2.5981, 0.866, 1.0], [1.7321, 0.5774, 1.0], [1.0, 1.0, 1.0], [0.4, 0.4, 1]],
            "custom_values": [0, 0],
            "rotations": [0, 1.57],
            "reshape_out": True,
        },
        assign_per_class=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,
        bbox_coder={"type": "DeltaXYZWLHRBBoxCoder", "code_size": 9},
        test_cfg={
            "pts": {
                "use_rotate_nms": True,
                "nms_across_levels": False,
                "nms_pre": 1000,
                "nms_thr": 0.2,
                "score_thr": 0.05,
                "min_bbox_size": 0,
                "max_num": 500,
            }
        },
        parameters=parameters,
        device=device,
    )

    ttnn_output = ttnn_model(x=ttnn_x)

    for i in range(len(ttnn_output)):
        for j in range(len(ttnn_output[i])):
            torch_output = ttnn.to_torch(ttnn_output[i][j])
            torch_output = torch_output.reshape(
                reference_output[i][j].shape[0],
                reference_output[i][j].shape[2],
                reference_output[i][j].shape[3],
                reference_output[i][j].shape[1],
            )
            passing, pcc = assert_with_pcc(reference_output[i][j], torch_output.permute(0, 3, 1, 2), 0.99)
            logger.info(f"Passing: {passing}, PCC: {pcc}")
