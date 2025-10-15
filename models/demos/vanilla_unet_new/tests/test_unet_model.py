# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.vanilla_unet_new.tt.common import (
    VANILLA_UNET_L1_SMALL_SIZE,
    VANILLA_UNET_PCC_WH,
    create_unet_preprocessor,
    load_reference_model,
)
from models.demos.vanilla_unet_new.tt.config import create_unet_configs_from_parameters
from models.demos.vanilla_unet_new.tt.model import create_unet_from_configs
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": VANILLA_UNET_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch, input_channels, input_height, input_width", [(1, 3, 480, 640)])
def test_vanilla_unet_model(
    batch, input_channels, input_height, input_width, device, reset_seeds, model_location_generator
):
    reference_model = load_reference_model(model_location_generator)

    torch_input_tensor = torch.randn(batch, input_channels, input_height, input_width)
    torch_output_tensor = reference_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_unet_preprocessor(device), device=None
    )
    configs = create_unet_configs_from_parameters(
        parameters=parameters, input_height=input_height, input_width=input_width, batch_size=batch
    )
    model = create_unet_from_configs(configs, device)

    ttnn_input_host = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1),  # NCHW -> NHWC
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn_output = model(ttnn_input_host)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)  # NHWC -> NCHW
    ttnn_output = ttnn_output.reshape(torch_output_tensor.shape)

    assert ttnn_output.shape == torch_output_tensor.shape
    pcc_passed, pcc_message = assert_with_pcc(torch_output_tensor, ttnn_output, pcc=VANILLA_UNET_PCC_WH)
    logger.info(f"PCC check was successful ({pcc_message:.5f} > {VANILLA_UNET_PCC_WH:.5f})")
