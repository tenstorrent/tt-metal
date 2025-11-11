# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, comp_allclose

from models.experimental.efficientdetd0.reference.modules import Regressor
from models.experimental.efficientdetd0.tt.regressor import Regressor as TTRegressor
from models.experimental.efficientdetd0.tt.custom_preprocessor import (
    create_custom_mesh_preprocessor,
    infer_torch_module_args,
)
from models.experimental.efficientdetd0.common import load_torch_model_state
from models.demos.utils.common_demo_utils import get_mesh_mappers


torch.manual_seed(0)


@pytest.mark.parametrize(
    "features, box_class_repeats, num_anchors",
    [
        (
            (
                # N C H W
                torch.randn([1, 64, 64, 64]),
                torch.randn([1, 64, 32, 32]),
                torch.randn([1, 64, 16, 16]),
                torch.randn([1, 64, 8, 8]),
                torch.randn([1, 64, 4, 4]),
            ),  # features
            3,  # box_class_repeats
            9,  # num_anchors
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_regressor(
    features,
    box_class_repeats,
    num_anchors,
    device,
):
    PCC_THRESHOLD = 0.99
    pyramid_levels = len(features)
    torch_model = Regressor(
        in_channels=features[0].shape[1],
        num_anchors=num_anchors,
        num_layers=box_class_repeats,
        pyramid_levels=pyramid_levels,
    ).eval()
    load_torch_model_state(torch_model, "regressor")

    torch_out = torch_model(features)
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )
    module_args = infer_torch_module_args(model=torch_model, input=features, layer_type=torch.nn.Conv2d)

    ttnn_model = TTRegressor(
        device,
        parameters,
        module_args,
        num_layers=box_class_repeats,
        pyramid_levels=pyramid_levels,
    )
    ttnn_features = [
        ttnn.from_torch(
            x.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for x in features
    ]
    ttnn_out = ttnn_model(ttnn_features)
    ttnn_out = ttnn.to_torch(ttnn_out)

    passing, pcc_message = comp_pcc(torch_out, ttnn_out, PCC_THRESHOLD)
    logger.info(f"Output PCC: {pcc_message}")
    logger.info(comp_allclose(torch_out, ttnn_out))

    if passing:
        logger.info("Regressor Test Passed!")
    else:
        logger.warning("Regressor Test Failed!")

    assert passing, f"PCC value is lower than {PCC_THRESHOLD}. Check implementation! {pcc_message}"
