# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.swin_s.reference.patchmerging import PatchMerging
from models.experimental.swin_s.tt.tt_patchmerging import TtPatchMerging
from models.experimental.swin_s.common import load_torch_model, SWIN_S_L1_SMALL_SIZE
from models.experimental.swin_s.tt.common import (
    preprocess_layernorm_parameter,
    preprocess_linear_weight,
)
from models.demos.utils.common_demo_utils import get_mesh_mappers


def custom_preprocessor(torch_model, name, mesh_mapper=None):
    parameters = {}
    if isinstance(torch_model, PatchMerging):
        parameters["reduction"] = {}
        parameters["reduction"]["weight"] = preprocess_linear_weight(
            torch_model.reduction.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        parameters["norm"] = {}
        parameters["norm"]["weight"] = preprocess_layernorm_parameter(
            torch_model.norm.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
        parameters["norm"]["bias"] = preprocess_layernorm_parameter(
            torch_model.norm.bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
        )
    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name):
        return custom_preprocessor(model, name, mesh_mapper)

    return custom_mesh_preprocessor


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": SWIN_S_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    [
        1,
    ],
)
@pytest.mark.parametrize(
    "dim,seq_len,i",
    [
        (96, 128, 2),
        (192, 64, 4),
        (384, 32, 6),
    ],
)
def test_patchmerging(device, batch_size, dim, seq_len, i, reset_seeds, model_location_generator):
    torch_model = PatchMerging(dim)

    torch_model = load_torch_model(
        torch_model=torch_model, i=i, module="patchmerging", model_location_generator=model_location_generator
    )

    # Input tensor for testing
    torch_input_tensor = torch.randn(batch_size, seq_len, seq_len, dim)  # Sample input tensor
    torch_output_tensor = torch_model(torch_input_tensor)
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    # Preprocess the model parameters for TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )

    # Convert the model to TTNN
    ttnn_model = TtPatchMerging(device, parameters, dim)

    # Convert input tensor to TTNN format
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Apply TTNN model
    output_tensor = ttnn_model(input_tensor)

    # Convert output tensor back to Torch format
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99)
