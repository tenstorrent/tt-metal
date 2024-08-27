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
)


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 64768}], indirect=True)
def test_unet_multi_device_model(batch, groups, device_mesh, use_program_cache, reset_seeds):
    if not is_n300_with_eth_dispatch_cores(device_mesh):
        pytest.skip("Test is only valid for N300")

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(device_mesh, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(device_mesh)
    output_mesh_composer = ttnn.ConcatMeshToTensor(device_mesh, dim=0)

    torch_input, ttnn_input = create_unet_input_tensors(device_mesh, batch, groups, pad_input=True)
    model = unet_shallow_torch.UNet.from_random_weights(groups=groups)

    parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device_mesh)
    ttnn_model = unet_shallow_ttnn.UNet(parameters, device=device_mesh, mesh_mapper=weights_mesh_mapper)

    num_devices = len(device_mesh.get_device_ids())
    torch_input, ttnn_input = create_unet_input_tensors(
        device_mesh, num_devices * batch, groups, pad_input=True, mesh_mapper=inputs_mesh_mapper
    )
    logger.info(f"Created reference input tensors: {list(torch_input.shape)}")
    logger.info(
        f"Created multi-device input tensors: shape={list(ttnn_input.shape)} on devices={device_mesh.get_device_ids()}"
    )

    torch_output_tensor = model(torch_input)
    output_tensor = ttnn_model(ttnn_input, list(torch_input.shape))

    check_pcc_conv(torch_output_tensor, output_tensor, mesh_composer=output_mesh_composer, pcc=0.99)
