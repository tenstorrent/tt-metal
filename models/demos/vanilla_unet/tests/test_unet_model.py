# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.demos.vanilla_unet.tt.common import (
    VANILLA_UNET_L1_SMALL_SIZE,
    VANILLA_UNET_PCC_WH,
    create_unet_preprocessor,
    load_reference_model,
)
from models.demos.vanilla_unet.tt.config import create_unet_configs_from_parameters
from models.demos.vanilla_unet.tt.model import create_unet_from_configs
from models.experimental.functional_unet.tt.model_preprocessing import create_unet_input_tensors
from tests.ttnn.utils_for_testing import assert_with_pcc


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": VANILLA_UNET_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch, input_channels, input_height, input_width", [(1, 3, 480, 640)])
def test_vanilla_unet_model(
    batch, input_channels, input_height, input_width, device, reset_seeds, model_location_generator
):
    device.disable_and_clear_program_cache()

    reference_model = load_reference_model(model_location_generator)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_unet_preprocessor(device), device=None
    )
    configs = create_unet_configs_from_parameters(
        parameters=parameters, input_height=input_height, input_width=input_width, batch_size=batch
    )

    torch_input_tensor, ttnn_input_tensor = create_unet_input_tensors(
        batch=batch,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        groups=1,
        channel_order="first",
        pad=False,
        fold=True,
        device=device,
        memory_config=configs.l1_input_memory_config,
    )
    torch_output_tensor = reference_model(torch_input_tensor)

    model = create_unet_from_configs(configs, device)
    ttnn_output_tensor = model(ttnn_input_tensor)
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor).reshape(torch_output_tensor.shape)

    assert ttnn_output_tensor.shape == torch_output_tensor.shape, "Expected output tensor shapes to match"
    pcc_passed, pcc_message = assert_with_pcc(torch_output_tensor, ttnn_output_tensor, pcc=VANILLA_UNET_PCC_WH)
    logger.info(f"PCC check was successful ({pcc_message:.5f} > {VANILLA_UNET_PCC_WH:.5f})")
