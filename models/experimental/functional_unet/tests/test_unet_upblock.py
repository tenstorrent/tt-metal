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
    check_pcc_conv,
    is_n300_with_eth_dispatch_cores,
    UNET_L1_SMALL_REGION_SIZE,
)


@pytest.mark.skip("Reshape in upblock crashes in these tests (#20182)")
@pytest.mark.parametrize("batch, groups", [(1, 4)])
@pytest.mark.parametrize(
    "block_name, input_channels, input_height, input_width, residual_channels",
    [
        ("upblock1", 64, 66, 10, 32),
        ("upblock2", 32, 132, 20, 32),
        ("upblock3", 32, 264, 40, 16),
        ("upblock4", 16, 528, 80, 16),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": UNET_L1_SMALL_REGION_SIZE}], indirect=True)
def test_unet_upblock(
    batch: int,
    groups: int,
    block_name: str,
    input_channels: int,
    input_height: int,
    input_width: int,
    residual_channels: int,
    device: ttnn.Device,
    reset_seeds,
):
    torch_input, ttnn_input = create_unet_input_tensors(batch, groups)
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    torch_input, ttnn_input = create_unet_input_tensors(
        batch,
        groups,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
    )
    torch_residual, ttnn_residual = create_unet_input_tensors(
        batch,
        groups,
        input_channels=residual_channels,
        input_height=input_height * 2,
        input_width=input_width * 2,
    )
    torch_output = getattr(model, block_name)(torch_input, torch_residual)

    ttnn_input, ttnn_residual = ttnn.to_device(ttnn_input, device), ttnn.to_device(ttnn_residual, device)
    ttnn_output = getattr(ttnn_model, block_name)(ttnn_input, ttnn_residual)

    check_pcc_conv(torch_output, ttnn_output, pcc=0.999)


@pytest.mark.skip("Reshape in upblock crashes in these tests (#20182)")
@pytest.mark.parametrize("batch, groups", [(1, 4)])
@pytest.mark.parametrize(
    "block_name, input_channels, input_height, input_width, residual_channels",
    [
        ("upblock1", 64, 66, 10, 32),
        ("upblock2", 32, 132, 20, 32),
        ("upblock3", 32, 264, 40, 16),
        ("upblock4", 16, 528, 80, 16),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_upblock_multi_device(
    batch,
    groups,
    block_name,
    input_channels,
    input_height,
    input_width,
    residual_channels,
    mesh_device,
    reset_seeds,
):
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
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        mesh_mapper=inputs_mesh_mapper,
    )
    logger.info(f"Created reference input tensors: {list(torch_input.shape)}")
    logger.info(
        f"Created multi-device input tensors: shape={list(ttnn_input.shape)} on devices={mesh_device.get_device_ids()}"
    )
    torch_residual, ttnn_residual = create_unet_input_tensors(
        num_devices * batch,
        groups,
        input_channels=residual_channels,
        input_height=input_height * 2,
        input_width=input_width * 2,
        mesh_mapper=inputs_mesh_mapper,
    )
    logger.info(f"Created reference residual input tensors: {list(torch_residual.shape)}")
    logger.info(
        f"Created multi-device residual input tensors: shape={list(ttnn_residual.shape)} on devices={mesh_device.get_device_ids()}"
    )
    torch_output = getattr(model, block_name)(torch_input, torch_residual)

    ttnn_input, ttnn_residual = ttnn_input.to(mesh_device), ttnn_residual.to(mesh_device)
    ttnn_output = getattr(ttnn_model, block_name)(ttnn_input, ttnn_residual)

    assert ttnn_output.device().get_num_devices() == 2, "Expected output tensor to be sharded across 2 devices"
    check_pcc_conv(torch_output, ttnn_output, pcc=0.999, mesh_composer=output_mesh_composer)
