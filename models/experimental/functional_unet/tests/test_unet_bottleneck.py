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
    is_n300_with_eth_dispatch_cores,
    check_pcc_conv,
    UNET_L1_SMALL_REGION_SIZE,
)


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("groups", [4])
@pytest.mark.parametrize("device_params", [{"l1_small_size": UNET_L1_SMALL_REGION_SIZE}], indirect=True)
def test_unet_bottleneck(batch: int, groups: int, device: ttnn.Device, reset_seeds):
    torch_input, ttnn_input = create_unet_input_tensors(batch, groups)
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    torch_input, ttnn_input = create_unet_input_tensors(
        batch, groups, input_channels=32, input_height=66, input_width=10
    )
    logger.info(f"Created reference input tensors: {list(torch_input.shape)}")
    logger.info(f"Created input tensors: shape={list(ttnn_input.shape)}")

    torch_output = model.bottleneck(torch_input)

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_output = ttnn_model.bottleneck(ttnn_input)

    check_pcc_conv(torch_output, ttnn_output, pcc=0.999)


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("groups", [4])
@pytest.mark.parametrize("device_params", [{"l1_small_size": UNET_L1_SMALL_REGION_SIZE}], indirect=True)
def test_unet_bottleneck_multi_device(batch: int, groups: int, mesh_device: ttnn.MeshDevice, reset_seeds):
    if not is_n300_with_eth_dispatch_cores(mesh_device):
        pytest.skip("Test is only valid for N300")

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    torch_input, ttnn_input = create_unet_input_tensors(batch, groups)
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=mesh_device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, mesh_device, mesh_mapper=weights_mesh_mapper)

    num_devices = len(mesh_device.get_device_ids())
    torch_input, ttnn_input = create_unet_input_tensors(
        num_devices * batch,
        groups,
        input_channels=32,
        input_height=66,
        input_width=10,
        mesh_mapper=inputs_mesh_mapper,
    )
    logger.info(f"Created reference input tensors: {list(torch_input.shape)}")
    logger.info(
        f"Created multi-device input tensors: shape={list(ttnn_input.shape)} on devices={mesh_device.get_device_ids()}"
    )
    torch_output = model.bottleneck(torch_input)

    ttnn_input = ttnn_input.to(mesh_device)
    ttnn_output = ttnn_model.bottleneck(ttnn_input)

    assert ttnn_output.device().get_num_devices() == 2, "Expected output tensor to be sharded across 2 devices"
    check_pcc_conv(torch_output, ttnn_output, mesh_composer=output_mesh_composer, pcc=0.999)
