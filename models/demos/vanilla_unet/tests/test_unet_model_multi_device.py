# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": VANILLA_UNET_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch, input_channels, input_height, input_width", [(1, 3, 480, 640)])
def test_vanilla_unet_model(
    batch, input_channels, input_height, input_width, mesh_device, reset_seeds, model_location_generator
):
    reference_model = load_reference_model(model_location_generator)

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    num_devices = len(mesh_device.get_device_ids())
    total_batch = num_devices * batch
    logger.info(f"Using {num_devices} devices for this test")

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_unet_preprocessor(mesh_device), device=None
    )
    configs = create_unet_configs_from_parameters(
        parameters=parameters, input_height=input_height, input_width=input_width, batch_size=batch
    )

    torch_input_tensor, ttnn_input_tensor = create_unet_input_tensors(
        batch=total_batch,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        groups=1,
        channel_order="first",
        pad=False,
        fold=True,
        device=mesh_device,
        memory_config=configs.l1_input_memory_config,
        mesh_mapper=inputs_mesh_mapper,
    )
    logger.info(f"Created reference input tensors: {list(torch_input_tensor.shape)}")
    logger.info(
        f"Created multi-device input tensors: shape={list(ttnn_input_tensor.shape)} on devices={mesh_device.get_device_ids()}"
    )

    torch_output_tensor = reference_model(torch_input_tensor)

    model = create_unet_from_configs(configs, mesh_device)
    ttnn_output_tensor = model(ttnn_input_tensor)
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor, mesh_composer=output_mesh_composer).reshape(
        torch_output_tensor.shape
    )

    assert ttnn_output_tensor.shape == torch_output_tensor.shape, "Expected output tensor shapes to match"
    pcc_passed, pcc_message = assert_with_pcc(torch_output_tensor, ttnn_output_tensor, pcc=VANILLA_UNET_PCC_WH)
    logger.info(f"PCC check was successful ({pcc_message:.5f} > {VANILLA_UNET_PCC_WH:.5f})")
