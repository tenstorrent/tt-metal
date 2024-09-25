# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_unet.tt.model_preprocessing import (
    create_unet_input_tensors,
    create_unet_model_parameters,
)
from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn
from models.experimental.functional_unet.tests.common import (
    check_pcc_conv,
    is_n300_with_eth_dispatch_cores,
)


@pytest.mark.parametrize("batch, groups", [(2, 1)])
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
def test_unet_upblock(
    batch,
    groups,
    block_name,
    input_channels,
    input_height,
    input_width,
    residual_channels,
    device,
    reset_seeds,
):
    torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=False)
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    torch_input, ttnn_input = create_unet_input_tensors(
        device,
        batch,
        groups,
        pad_input=False,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
    )
    torch_residual, ttnn_residual = create_unet_input_tensors(
        device,
        batch,
        groups,
        pad_input=False,
        input_channels=residual_channels,
        input_height=input_height * 2,
        input_width=input_width * 2,
    )
    torch_output = getattr(model, block_name)(torch_input, torch_residual)

    ttnn_input, ttnn_residual = ttnn_input.to(device), ttnn_residual.to(device)
    ttnn_output = getattr(ttnn_model, block_name)(ttnn_input, ttnn_residual)

    check_pcc_conv(torch_output, ttnn_output, pcc=0.998)


@pytest.mark.parametrize("batch, groups", [(2, 1)])
@pytest.mark.parametrize(
    "block_name, input_channels, input_height, input_width, residual_channels",
    [
        ("upblock1", 64, 66, 10, 32),
        ("upblock2", 32, 132, 20, 32),
        ("upblock3", 32, 264, 40, 16),
        ("upblock4", 16, 528, 80, 16),
    ],
)
@pytest.mark.parametrize("enable_async_mode", (True,), indirect=True)
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
    enable_async_mode,
):
    if not is_n300_with_eth_dispatch_cores(mesh_device):
        pytest.skip("Test is only valid for N300")

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    torch_input, ttnn_input = create_unet_input_tensors(mesh_device, batch, groups, pad_input=False)
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=mesh_device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, mesh_device, mesh_mapper=weights_mesh_mapper)

    num_devices = len(mesh_device.get_device_ids())
    torch_input, ttnn_input = create_unet_input_tensors(
        mesh_device,
        num_devices * batch,
        groups,
        pad_input=False,
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
        mesh_device,
        num_devices * batch,
        groups,
        pad_input=False,
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

    assert len(ttnn_output.devices()) == 2, "Expected output tensor to be sharded across 2 devices"
    check_pcc_conv(torch_output, ttnn_output, mesh_composer=output_mesh_composer, pcc=0.998)
