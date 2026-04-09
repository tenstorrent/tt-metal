# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from loguru import logger

from ttnn.device import is_wormhole_b0

from models.experimental.functional_unet.tt.model_preprocessing import (
    create_unet_input_tensors,
    create_unet_model_parameters,
)
from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn
from models.experimental.functional_unet.tests.common import (
    verify_with_pcc,
    UNET_FULL_MODEL_PCC,
    UNET_FULL_MODEL_PCC_BH,
    UNET_L1_SMALL_REGION_SIZE,
)


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("groups", [4])
@pytest.mark.parametrize("device_params", [{"l1_small_size": UNET_L1_SMALL_REGION_SIZE}], indirect=True)
def test_unet_model(batch, groups, mesh_device, reset_seeds):
    num_devices = mesh_device.get_num_devices()

    if num_devices == 1 and not is_wormhole_b0(mesh_device) and (
        mesh_device.compute_with_storage_grid_size().x * mesh_device.compute_with_storage_grid_size().y != 110
        and mesh_device.compute_with_storage_grid_size().x * mesh_device.compute_with_storage_grid_size().y != 130
    ):
        pytest.skip(f"Shallow UNet only supports 110 or 130 cores on BH (was {mesh_device.compute_with_storage_grid_size()})")

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    total_batch = num_devices * batch
    logger.info(f"Using {num_devices} device(s) for this test")

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

    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)
    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=mesh_device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device=mesh_device, mesh_mapper=weights_mesh_mapper)

    torch_output_tensor = model(torch_input)
    output_tensor = ttnn_model(ttnn_input, move_input_tensor_to_device=False, deallocate_input_activation=True).cpu()

    B, C, H, W = torch_output_tensor.shape
    ttnn_output_tensor = ttnn.to_torch(output_tensor, mesh_composer=output_mesh_composer).reshape(B, C, H, W)
    verify_with_pcc(
        torch_output_tensor,
        ttnn_output_tensor,
        UNET_FULL_MODEL_PCC if is_wormhole_b0(mesh_device) else UNET_FULL_MODEL_PCC_BH,
    )
