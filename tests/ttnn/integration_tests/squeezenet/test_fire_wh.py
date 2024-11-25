# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch
from models.demos.wormhole.squeezenet.tt.tt_squeezenet import tt_Fire
from torchvision import models
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import is_wormhole_b0, skip_for_grayskull
import torch.nn as nn
from ttnn.model_preprocessing import preprocess_model_parameters


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        parameters["bias"] = ttnn.from_torch(
            model.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
    return parameters


@pytest.mark.parametrize(
    "batch_size, input_height, input_width, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes, features_block",
    [
        (16, 54, 54, 96, 16, 64, 64, 3),
        (16, 54, 54, 128, 16, 64, 64, 4),
        (16, 54, 54, 128, 32, 128, 128, 5),
        (16, 27, 27, 256, 32, 128, 128, 7),
        (16, 27, 27, 256, 48, 192, 192, 8),
        (16, 27, 27, 384, 48, 192, 192, 9),
        (16, 27, 27, 384, 64, 256, 256, 10),
        (16, 13, 13, 512, 64, 256, 256, 12),
    ],
)
@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_fire(
    mesh_device,
    batch_size,
    input_height,
    input_width,
    inplanes,
    squeeze_planes,
    expand1x1_planes,
    expand3x3_planes,
    features_block,
):
    inputs_mesh_mapper = None
    weights_mesh_mapper = None
    output_mesh_composer = None
    torch_squeezenet = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    state_dict = torch_squeezenet.state_dict()
    torch.manual_seed(42)
    torch_input = torch.randn([batch_size, input_height, input_width, inplanes])
    torch_input_for_premodel = torch.permute(torch_input, (0, 3, 1, 2))
    if is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        with ttnn.distribute(weights_mesh_mapper):
            parameters = preprocess_model_parameters(
                initialize_model=lambda: torch_squeezenet.features[features_block],
                custom_preprocessor=custom_preprocessor,
                device=None,
            )
    else:
        print("2 devices are not detected")

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=inputs_mesh_mapper
    )
    tt_out = tt_Fire(
        inplanes,
        squeeze_planes,
        expand1x1_planes,
        expand3x3_planes,
        input_tensor=tt_input,
        parameters=parameters,
        mesh_device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
        batch_size_tt=batch_size // 2,
        state_dict=state_dict,
    )

    tt_out_in_torch = ttnn.to_torch(tt_out, mesh_composer=output_mesh_composer).permute(0, 3, 1, 2)

    torch_model = torch_squeezenet.features[features_block]
    torch_out = torch_model(torch_input_for_premodel)
    assert_with_pcc(torch_out, tt_out_in_torch, pcc=0.99)
