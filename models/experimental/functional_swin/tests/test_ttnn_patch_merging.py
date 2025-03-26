# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import models
import pytest
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_layernorm_parameter,
)
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_swin.reference.patchmerging_v2 import PatchMergingV2
from models.experimental.functional_swin.tt.tt_patchmerging_v2 import TtPatchMergingV2


def create_custom_preprocessor(device):
    def custom_preprocessor(torch_model, name, ttnn_module_args):
        parameters = {}
        if isinstance(torch_model, PatchMergingV2):
            parameters["reduction"] = {}
            parameters["reduction"]["weight"] = preprocess_linear_weight(
                torch_model.reduction.weight, dtype=ttnn.bfloat16
            )
            parameters["norm"] = {}
            parameters["norm"]["weight"] = preprocess_layernorm_parameter(torch_model.norm.weight, dtype=ttnn.bfloat16)
            parameters["norm"]["bias"] = preprocess_layernorm_parameter(torch_model.norm.bias, dtype=ttnn.bfloat16)
        return parameters

    return custom_preprocessor


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "dim,i, input_shape",
    [
        (96, 2, [1, 128, 128, 96]),
        (192, 4, [1, 64, 64, 192]),
        (384, 6, [1, 32, 32, 384]),
    ],
)
def test_patchmerging_v2(device, dim, i, input_shape, reset_seeds):
    model = models.swin_v2_s(weights="IMAGENET1K_V1")
    state_dict = state_dict = model.state_dict()
    patchmerging_state_dict = {k: v for k, v in state_dict.items() if (k.startswith(f"features.{i}."))}

    if not patchmerging_state_dict:
        raise ValueError("No parameters found in resblock_state_dict")

    torch_model = PatchMergingV2(dim)
    for layer in torch_model.children():
        print(layer)

    torch_input_tensor = torch.rand(input_shape)

    new_state_dict = {}
    new_torch_state_dic = {}
    for k, v in patchmerging_state_dict.items():
        new_state_dict[k] = patchmerging_state_dict[k]
        new_torch_state_dic[k.replace(f"features.{i}.", "")] = patchmerging_state_dict[k]

    torch_model.load_state_dict(new_torch_state_dic)
    torch_model.eval()

    # Input tensor for testing
    torch_output_tensor = torch_model(torch_input_tensor)

    # Preprocess the model parameters for TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(device), device=device
    )

    # Convert the model to TTNN
    ttnn_model = TtPatchMergingV2(device, parameters, dim)

    # Convert input tensor to TTNN format
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Apply TTNN model
    output_tensor = ttnn_model(input_tensor)

    # Convert output tensor back to Torch format
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
