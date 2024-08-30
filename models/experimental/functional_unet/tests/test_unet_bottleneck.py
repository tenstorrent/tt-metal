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
from models.experimental.functional_unet.tests.common import is_n300_with_eth_dispatch_cores, check_pcc_conv


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_bottleneck(batch, groups, device, reset_seeds):
    torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=False)
    model = unet_shallow_torch.UNet.from_random_weights(groups=1)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

    torch_input, ttnn_input = create_unet_input_tensors(
        device, batch, groups, pad_input=True, input_channels=32, input_height=66, input_width=10
    )
    torch_output = model.bottleneck(torch_input)

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_output = ttnn_model.bottleneck(ttnn_input)

    check_pcc_conv(torch_output, ttnn_output, pcc=0.999)


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_unet_bottleneck_multi_device(batch, groups, device_mesh, reset_seeds):
    if not is_n300_with_eth_dispatch_cores(device_mesh):
        pytest.skip("Test is only valid for N300")

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(device_mesh, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(device_mesh)
    output_mesh_composer = ttnn.ConcatMeshToTensor(device_mesh, dim=0)

    torch_input, ttnn_input = create_unet_input_tensors(device_mesh, batch, groups, pad_input=False)
    model = unet_shallow_torch.UNet.from_random_weights(groups=1)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device_mesh)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device_mesh, mesh_mapper=weights_mesh_mapper)

    num_devices = len(device_mesh.get_device_ids())
    torch_input, ttnn_input = create_unet_input_tensors(
        device_mesh,
        num_devices * batch,
        groups,
        pad_input=True,
        input_channels=32,
        input_height=66,
        input_width=10,
        mesh_mapper=inputs_mesh_mapper,
    )
    logger.info(f"Created reference input tensors: {list(torch_input.shape)}")
    logger.info(
        f"Created multi-device input tensors: shape={list(ttnn_input.shape)} on devices={device_mesh.get_device_ids()}"
    )
    torch_output = model.bottleneck(torch_input)

    ttnn_input = ttnn_input.to(device_mesh)
    ttnn_output = ttnn_model.bottleneck(ttnn_input)

    assert len(ttnn_output.devices()) == 2, "Expected output tensor to be sharded across 2 devices"
    check_pcc_conv(torch_output, ttnn_output, mesh_composer=output_mesh_composer, pcc=0.999)
