# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.segformer.common import load_config, load_torch_model
from models.demos.segformer.reference.segformer_selfoutput import SegformerSelfOutput
from models.demos.segformer.tt.common import preprocess_linear_bias, preprocess_linear_weight
from models.demos.segformer.tt.ttnn_segformer_selfoutput import TtSegformerSelfOutput
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(model, name, mesh_mapper)

    def custom_preprocessor(model, name, mesh_mapper=None):
        parameters = {}
        if isinstance(model, SegformerSelfOutput):
            parameters["dense"] = {}
            parameters["dense"]["weight"] = preprocess_linear_weight(
                model.dense.weight, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )
            parameters["dense"]["bias"] = preprocess_linear_bias(
                model.dense.bias, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )

        return parameters

    return custom_mesh_preprocessor


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size, block_i, self_output_i",
    [
        (1, 16384, 32, 0, 0),
        (1, 16384, 32, 0, 1),
        (1, 4096, 64, 1, 0),
        (1, 4096, 64, 1, 1),
        (1, 1024, 160, 2, 0),
        (1, 1024, 160, 2, 1),
        (1, 256, 256, 3, 0),
        (1, 256, 256, 3, 1),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_selfoutput(
    device, block_i, self_output_i, batch_size, seq_len, hidden_size, model_location_generator
):
    torch_input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    config = load_config("configs/segformer_semantic_config.json")
    reference_model = SegformerSelfOutput(config=config, hidden_size=hidden_size)
    target_prefix = f"encoder.block.{block_i}.{self_output_i}.attention.output"
    reference_model = load_torch_model(
        reference_model, target_prefix, module="semantic_sub", model_location_generator=model_location_generator
    )

    torch_output = reference_model(torch_input_tensor, None)
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )
    ttnn_model = TtSegformerSelfOutput()

    ttnn_output = ttnn_model(device, ttnn_input_tensor, parameters=parameters)
    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
