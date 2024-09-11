# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
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
@pytest.mark.parametrize("device_params", [{"l1_small_size": 3 * 32768}], indirect=True)
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
    use_program_cache
):
    torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=False)
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    logger.info(f"Run compilation run")
    output_tensor = ttnn_model(ttnn_input).cpu()
    output_tensor = ttnn_model(ttnn_input).cpu()
    logger.info(f"Done running compilztion")

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
    # torch_output = getattr(model, block_name)(torch_input, torch_residual)

    logger.info(f"RUNNING TORCH UPBLOCK STEP BY STEP")
    u1 = model.u1(torch_input)
    conc4 = torch.cat([u1, torch_residual], dim=1)
    c8 = model.c8(conc4)
    b8 = model.b8(c8)
    r8 = model.r8(b8)
    c8_2 = model.c8_2(r8)
    b8_2 = model.b8_2(c8_2)
    r8_2 = model.r8_2(b8_2)
    # c8_3 = model.c8_3(r8_2)
    # b8_3 = model.b8_3(c8_3)
    # r8_3 = model.r8_3(b8_3)
    torch_output = r8_2

    # ttnn_output = block(ttnn_input, ttnn_residual)

    logger.info(f"RUNNING TTNN UPBLOCK STEP BY STEP")
    x, residual = ttnn_input.to(device), ttnn_residual.to(device)
    block = getattr(ttnn_model, block_name)
    residual = ttnn.to_layout(residual, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
    x = block.upsample(x)
    y = unet_shallow_ttnn.unet_concat([x, residual], dim=-1)
    ttnn.deallocate(x)
    ttnn.deallocate(residual)
    y = block.conv1(y)
    breakpoint()
    y = block.conv2(y)
    # y = block.conv3(y)

    check_pcc_conv(torch_output, y, pcc=0.999)


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
def test_unet_upblock_multi_device(
    batch, groups, block_name, input_channels, input_height, input_width, residual_channels, mesh_device, reset_seeds
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
