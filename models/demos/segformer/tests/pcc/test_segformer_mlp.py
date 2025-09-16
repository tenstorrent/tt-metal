# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import skip_for_grayskull
from models.demos.segformer.common import load_config, load_torch_model
from models.demos.segformer.reference.segformer_mlp import SegformerMLP
from models.demos.segformer.tt.common import preprocess_linear_bias, preprocess_linear_weight
from models.demos.segformer.tt.ttnn_segformer_mlp import TtSegformerMLP
from models.demos.utils.common_demo_utils import get_mesh_mappers
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(model, name, mesh_mapper)

    def custom_preprocessor(model, name, mesh_mapper=None):
        parameters = {}
        if isinstance(model, SegformerMLP):
            parameters["proj"] = {}
            parameters["proj"]["weight"] = preprocess_linear_weight(
                model.proj.weight, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )
            parameters["proj"]["bias"] = preprocess_linear_bias(
                model.proj.bias, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )

        return parameters

    return custom_mesh_preprocessor


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "input_dim, mlp_id, batch_size, height, width,",
    [
        (32, 0, 1, 128, 128),
        (64, 1, 1, 64, 64),
        (160, 2, 1, 32, 32),
        (256, 3, 1, 16, 16),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_mlp(
    device,
    input_dim,
    batch_size,
    height,
    width,
    mlp_id,
    model_location_generator,
):
    torch_input_tensor = torch.randn(batch_size, input_dim, height, width)

    torch_input_tensor_folded = torch.reshape(torch_input_tensor, (batch_size, input_dim, height * width))
    torch_input_tensor_folded = torch.permute(torch_input_tensor_folded, (0, 2, 1))

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor_folded,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    config = load_config("configs/segformer_semantic_config.json")
    reference_model = SegformerMLP(config, input_dim)
    target_prefix = f"decode_head.linear_c.{mlp_id}"
    reference_model = load_torch_model(
        reference_model, target_prefix, module="semantic_sub", model_location_generator=model_location_generator
    )

    torch_output = reference_model(torch_input_tensor)
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )

    ttnn_model = TtSegformerMLP()
    ttnn_output = ttnn_model(device, ttnn_input_tensor, parameters=parameters)

    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
