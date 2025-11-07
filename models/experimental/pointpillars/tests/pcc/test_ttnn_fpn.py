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
from models.experimental.pointpillars.tt.ttnn_fpn import TtFPN
from models.experimental.pointpillars.tt.model_preprocessing import create_custom_preprocessor


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        True,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": POINTPILLARS_L1_SMALL_SIZE}], indirect=True)
@run_for_wormhole_b0()
def test_ttnn_fpn(device, model_location_generator, use_pretrained_weight, reset_seeds):
    reference_model = load_torch_model(model_location_generator)
    reference_model = reference_model.pts_neck

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device)
    )

    inputs = [torch.randn(1, 64, 200, 200), torch.randn(1, 128, 100, 100), torch.randn(1, 256, 50, 50)]

    ttnn_inputs = inputs[:]
    for i in range(len(ttnn_inputs)):
        input_0 = ttnn_inputs[i].permute(0, 2, 3, 1)
        ttnn_inputs[i] = ttnn.from_torch(
            input_0, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )

    reference_output = reference_model(inputs=inputs)

    ttnn_model = TtFPN(
        in_channels=[64, 128, 256],
        start_level=0,
        num_outs=3,
        out_channels=256,
        parameters=parameters,
        device=device,
    )

    ttnn_output = ttnn_model(inputs=ttnn_inputs)

    for i in range(len(ttnn_output)):
        torch_output = ttnn.to_torch(ttnn_output[i])
        torch_output = torch_output.reshape(
            reference_output[i].shape[0],
            reference_output[i].shape[2],
            reference_output[i].shape[3],
            reference_output[i].shape[1],
        )
        passing, pcc = assert_with_pcc(reference_output[i], torch_output.permute(0, 3, 1, 2), 0.99)
        logger.info(f"Passing: {passing}, PCC: {pcc}")
