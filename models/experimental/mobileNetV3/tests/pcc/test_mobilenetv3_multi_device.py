# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import run_for_wormhole_b0
from models.experimental.mobileNetV3.tests.pcc.common import inverted_residual_setting, last_channel
from models.experimental.mobileNetV3.tt.custom_preprocessor import create_custom_preprocessor
from models.experimental.mobileNetV3.tt.ttnn_mobileNetV3 import ttnn_MobileNetV3
from tests.ttnn.utils_for_testing import assert_with_pcc
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

MOBILENETV3_L1_SMALL_SIZE = 24576
MOBILENETV3_PCC_WH = 0.99


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": MOBILENETV3_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch, input_channels, input_height, input_width", [(1, 3, 224, 224)])
def test_mobilenetv3_multi_device(
    batch, input_channels, input_height, input_width, mesh_device, reset_seeds, model_location_generator
):
    # Load reference PyTorch model
    reference_model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    reference_model.eval()

    # Set up mesh mappers for multi-device
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    num_devices = len(mesh_device.get_device_ids())
    total_batch = num_devices * batch
    logger.info(f"Using {num_devices} devices for this test")

    # Preprocess model parameters with mesh mapper for weight replication
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_preprocessor(mesh_mapper=weights_mesh_mapper),
        device=None,
    )

    # Create input tensor (NCHW format for PyTorch)
    torch_input_tensor = torch.randn(total_batch, input_channels, input_height, input_width)
    logger.info(f"Created reference input tensors: {list(torch_input_tensor.shape)}")

    # Run PyTorch reference model
    with torch.no_grad():
        torch_output_tensor = reference_model(torch_input_tensor)

    # Convert input to TTNN format (NHWC)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=inputs_mesh_mapper,
    )
    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG)
    logger.info(
        f"Created multi-device input tensors: shape={list(ttnn_input_tensor.shape)} on devices={mesh_device.get_device_ids()}"
    )

    # Create TTNN model
    ttnn_model = ttnn_MobileNetV3(
        inverted_residual_setting=inverted_residual_setting,
        last_channel=last_channel,
        parameters=parameters,
        device=mesh_device,
        input_height=input_height,
        input_width=input_width,
    )

    # Run TTNN model on mesh device
    ttnn_output_tensor = ttnn_model(mesh_device, ttnn_input_tensor)

    # Convert output back to torch and compose from mesh
    ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor, mesh_composer=output_mesh_composer).reshape(
        torch_output_tensor.shape
    )

    # Validate output shapes match
    assert (
        ttnn_output_tensor.shape == torch_output_tensor.shape
    ), f"Expected output tensor shapes to match. Got TTNN: {ttnn_output_tensor.shape}, PyTorch: {torch_output_tensor.shape}"

    # Validate PCC
    pcc_passed, pcc_message = assert_with_pcc(torch_output_tensor, ttnn_output_tensor, pcc=MOBILENETV3_PCC_WH)
    logger.info(f"PCC check was successful ({pcc_message:.5f} > {MOBILENETV3_PCC_WH:.5f})")
