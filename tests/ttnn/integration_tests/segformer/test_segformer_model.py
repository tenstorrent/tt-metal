# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters, ParameterDict, ParameterList
from tests.ttnn.integration_tests.segformer.test_segformer_encoder import (
    create_custom_preprocessor as create_customer_preprocessor_encoder,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

from transformers import SegformerModel
import pytest
from models.experimental.functional_segformer.tt.ttnn_segformer_model import (
    TtSegformerModel,
)
from models.experimental.functional_segformer.reference.segformer_model import SegformerModelReference
from models.utility_functions import skip_for_grayskull


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        parameters["encoder"] = {}
        if isinstance(model, SegformerModelReference):
            encoder_prepocessor = create_customer_preprocessor_encoder(device)
            parameters["encoder"] = encoder_prepocessor(model.encoder, None, None)

        return parameters

    return custom_preprocessor


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
def test_segformer_model(
    batch_size,
    num_channels,
    height,
    width,
    device,
    reset_seeds,
):
    torch_input_tensor = torch.randn(batch_size, num_channels, height, width)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    torch_model = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    config = torch_model.config

    torch_model = torch_model
    reference_model = SegformerModelReference(config)
    state_dict = torch_model.state_dict()

    new_state_dict = {}
    keys = [name for name, parameter in reference_model.state_dict().items()]
    values = [parameter for name, parameter in state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    torch_output = reference_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=None
    )
    parameters = move_to_device(parameters, device)

    ttnn_model = TtSegformerModel(config, parameters)

    ttnn_output = ttnn_model(
        ttnn_input_tensor,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        parameters=parameters,
    )
    ttnn_final_output = ttnn.to_torch(ttnn_output[0])

    assert_with_pcc(torch_output[0], ttnn_final_output, pcc=0.885)
