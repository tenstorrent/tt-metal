# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_unet.tt.model_preprocessing import (
    create_unet_input_tensors,
    create_unet_model_parameters,
)
from models.experimental.functional_unet.tt import unet_shallow_torch
from models.experimental.functional_unet.tt import unet_shallow_ttnn


@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 64768, "num_command_queues": 2}], indirect=True)
@pytest.mark.parametrize("version", ["n300"])
def test_unet_model(batch, groups, device_mesh, version, use_program_cache):
    if version == "n150":
        # torch_input, ttnn_input = create_unet_input_tensors(device, batch, groups, pad_input=True)
        model = unet_shallow_torch.UNet.from_random_weights(groups=1)

        # parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device)
        # ttnn_model = unet_shallow_ttnn.UNet(parameters, device)

        # torch_output_tensor = model(torch_input)
        # output_tensor = ttnn_model(ttnn_input, list(torch_input.shape))

        # B, C, H, W = torch_output_tensor.shape
        # ttnn_tensor = ttnn.to_torch(output_tensor).reshape(B, H, W, -1)[:, :, :, :C].permute(0, 3, 1, 2)
        # assert_with_pcc(torch_output_tensor, ttnn_tensor, 0.985)
    elif version == "n300":
        num_devices = 1 if isinstance(device_mesh, ttnn.Device) else device_mesh.get_num_devices()

        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device_mesh, dim=0)
        weights_mesh_mapper = ttnn.ReplicateTensorToMesh(device_mesh)
        output_mesh_composer = ttnn.ConcatMeshToTensor(device_mesh, dim=0)

        torch_input, ttnn_input = create_unet_input_tensors(
            device_mesh, batch * num_devices, groups, pad_input=True, mesh_mapper=inputs_mesh_mapper
        )

        model = unet_shallow_torch.UNet.from_random_weights(groups=1)

        parameters = create_unet_model_parameters(model, torch_input, groups=groups, device=device_mesh)
        ttnn_model = unet_shallow_ttnn.UNet(parameters, batch, device_mesh, weights_mesh_mapper=weights_mesh_mapper)
        torch_output_tensor = model(torch_input)
        output_tensor = ttnn_model(ttnn_input, [2, 4, 1056, 160])

        B, C, H, W = torch_output_tensor.shape
        ttnn_tensor = ttnn.to_torch(output_tensor, device=device_mesh, mesh_composer=output_mesh_composer)
        ttnn_tensor = ttnn_tensor.reshape(B, H, W, -1)[:, :, :, :C].permute(0, 3, 1, 2)
        assert_with_pcc(torch_output_tensor, ttnn_tensor, 0.985)
