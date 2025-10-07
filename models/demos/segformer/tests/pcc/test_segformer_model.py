# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from ttnn.model_preprocessing import ParameterDict, ParameterList, preprocess_model_parameters

import ttnn
from models.common.utility_functions import skip_for_grayskull
from models.demos.segformer.common import load_config, load_torch_model
from models.demos.segformer.reference.segformer_model import SegformerModelReference
from models.demos.segformer.tests.pcc.test_segformer_encoder import (
    create_custom_mesh_preprocessor as create_customer_preprocessor_encoder,
)
from models.demos.segformer.tt.ttnn_segformer_model import TtSegformerModel
from models.demos.utils.common_demo_utils import get_mesh_mappers
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(model, name, mesh_mapper)

    def custom_preprocessor(model, name, mesh_mapper=None):
        parameters = {}
        parameters["encoder"] = {}
        if isinstance(model, SegformerModelReference):
            encoder_prepocessor = create_customer_preprocessor_encoder(mesh_mapper)
            parameters["encoder"] = encoder_prepocessor(model.encoder, None, None, None)

        return parameters

    return custom_mesh_preprocessor


def move_to_device(object, device):
    if isinstance(object, ParameterDict):
        for name, value in list(object.items()):
            if name in ["sr", "proj", "dwconv"]:
                continue
            object[name] = move_to_device(value, device)
        return object
    elif isinstance(object, ParameterList):
        for index, element in enumerate(object):
            object[index] = move_to_device(element, device)
        return object
    elif isinstance(object, ttnn.Tensor):
        return ttnn.to_device(object, device)
    else:
        return object


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch_size, num_channels, height, width",
    [
        (1, 3, 512, 512),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_model(batch_size, num_channels, height, width, device, model_location_generator):
    torch_input_tensor = torch.randn(batch_size, num_channels, height, width)

    config = load_config("configs/segformer_semantic_config.json")
    reference_model = SegformerModelReference(config)
    target_prefix = f""
    reference_model = load_torch_model(
        reference_model, target_prefix, module="semantic_sub", model_location_generator=model_location_generator
    )

    torch_output = reference_model(torch_input_tensor)
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=None,
    )
    parameters = move_to_device(parameters, device)

    ttnn_model = TtSegformerModel(config, parameters)

    torch_input_tensor_permuted = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor_permuted,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    CONV2D_MIN_CHANNEL_SIZE = 8
    # adjust padding if necessary
    if num_channels < CONV2D_MIN_CHANNEL_SIZE:
        ttnn_input_tensor = ttnn.pad(
            ttnn_input_tensor, [batch_size, height, width, CONV2D_MIN_CHANNEL_SIZE], [0, 0, 0, 0], 0
        )
    elif num_channels > CONV2D_MIN_CHANNEL_SIZE and num_channels % 32 != 0:
        ttnn_input_tensor = ttnn.pad(
            ttnn_input_tensor, [batch_size, height, width, (num_channels + 31) // 32 * 32], [0, 0, 0, 0], 0
        )

    ttnn_input_tensor = ttnn.to_device(ttnn_input_tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    ttnn_output = ttnn_model(
        device,
        ttnn_input_tensor,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        parameters=parameters,
    )
    ttnn_final_output = ttnn.to_torch(ttnn_output[0])
    torch_final_output = torch.permute(torch_output.last_hidden_state, (0, 2, 3, 1))

    assert_with_pcc(torch_final_output, ttnn_final_output, pcc=0.929)
