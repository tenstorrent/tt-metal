# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from loguru import logger

from models.experimental.functional_unet.tt.model_preprocessing import (
    create_unet_input_tensors,
    create_unet_model_parameters,
)
from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn
from models.experimental.functional_unet.tests.common import (
    verify_with_pcc,
    is_n300_with_eth_dispatch_cores,
    is_t3k_with_eth_dispatch_cores,
    UNET_FULL_MODEL_PCC,
    UNET_L1_SMALL_REGION_SIZE,
)


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("groups", [4])
@pytest.mark.parametrize("device_params", [{"l1_small_size": UNET_L1_SMALL_REGION_SIZE}], indirect=True)
def test_unet_multi_device_model(batch, groups, mesh_device, reset_seeds):
    if not is_n300_with_eth_dispatch_cores(mesh_device) and not is_t3k_with_eth_dispatch_cores(mesh_device):
        pytest.skip("Test is only valid for N300 or T3000")

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    torch_input, ttnn_input = create_unet_input_tensors(batch, groups)
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=mesh_device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device=mesh_device, mesh_mapper=weights_mesh_mapper)

    num_devices = len(mesh_device.get_device_ids())
    total_batch = num_devices * batch
    logger.info(f"Using {num_devices} devices for this test")
    torch_input, ttnn_input = create_unet_input_tensors(
        total_batch,
        groups,
        channel_order="first",
        pad=False,
        fold=True,
        device=mesh_device,
        memory_config=unet_shallow_ttnn.UNet.input_sharded_memory_config,
        mesh_mapper=inputs_mesh_mapper,
    )
    logger.info(f"Created reference input tensors: {list(torch_input.shape)}")
    logger.info(
        f"Created multi-device input tensors: shape={list(ttnn_input.shape)} on devices={mesh_device.get_device_ids()}"
    )

    torch_output_tensor = model(torch_input)
    output_tensor = ttnn_model(ttnn_input, move_input_tensor_to_device=False)

    B, C, H, W = torch_output_tensor.shape
    ttnn_output_tensor = ttnn.to_torch(output_tensor, mesh_composer=output_mesh_composer).reshape(B, C, H, W)
    verify_with_pcc(torch_output_tensor, ttnn_output_tensor, UNET_FULL_MODEL_PCC)
