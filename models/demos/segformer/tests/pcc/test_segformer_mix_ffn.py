# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import skip_for_grayskull
from models.demos.segformer.common import load_config, load_torch_model
from models.demos.segformer.reference.segformer_mixffn import SegformerMixFFN
from models.demos.segformer.tests.pcc.test_segformer_dwconv import (
    create_custom_mesh_preprocessor as create_custom_preprocessor_dwconv,
)
from models.demos.segformer.tt.common import preprocess_linear_bias, preprocess_linear_weight
from models.demos.segformer.tt.ttnn_segformer_mix_ffn import TtSegformerMixFFN
from models.demos.utils.common_demo_utils import get_mesh_mappers
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(model, name, mesh_mapper)

    def custom_preprocessor(model, name, mesh_mapper=None):
        parameters = {}
        if isinstance(model, SegformerMixFFN):
            parameters["dense1"] = {}
            parameters["dense1"]["weight"] = preprocess_linear_weight(
                model.dense1.weight, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )
            parameters["dense1"]["bias"] = preprocess_linear_bias(
                model.dense1.bias, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )

            parameters["dwconv"] = {}
            dwconv_preprocessor = create_custom_preprocessor_dwconv(mesh_mapper=mesh_mapper)
            parameters["dwconv"] = dwconv_preprocessor(model.dwconv, None, None, None)

            parameters["dense2"] = {}
            parameters["dense2"]["weight"] = preprocess_linear_weight(
                model.dense2.weight, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )
            parameters["dense2"]["bias"] = preprocess_linear_bias(
                model.dense2.bias, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )

        return parameters

    return custom_mesh_preprocessor


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "in_features, hidden_features, out_features, batch_size, seq_len, height, width, block_i, mixffn_i",
    [
        (32, 128, 32, 1, 16384, 128, 128, 0, 0),
        (32, 128, 32, 1, 16384, 128, 128, 0, 1),
        (64, 256, 64, 1, 4096, 64, 64, 1, 0),
        (64, 256, 64, 1, 4096, 64, 64, 1, 1),
        (160, 640, 160, 1, 1024, 32, 32, 2, 0),
        (160, 640, 160, 1, 1024, 32, 32, 2, 1),
        (256, 1024, 256, 1, 256, 16, 16, 3, 0),
        (256, 1024, 256, 1, 256, 16, 16, 3, 1),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_mix_ffn(
    device,
    in_features,
    hidden_features,
    out_features,
    batch_size,
    seq_len,
    height,
    width,
    block_i,
    mixffn_i,
    model_location_generator,
):
    torch_input_tensor = torch.randn(batch_size, seq_len, in_features)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    config = load_config("configs/segformer_semantic_config.json")
    reference_model = SegformerMixFFN(
        config=config, in_features=in_features, hidden_features=hidden_features, out_features=out_features
    )
    target_prefix = f"encoder.block.{block_i}.{mixffn_i}.mlp"
    reference_model = load_torch_model(
        reference_model, target_prefix, module="semantic_sub", model_location_generator=model_location_generator
    )

    torch_output = reference_model(torch_input_tensor, height, width)
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=None,
    )
    parameters["dense1"]["weight"] = ttnn.to_device(parameters["dense1"]["weight"], device=device)
    parameters["dense1"]["bias"] = ttnn.to_device(parameters["dense1"]["bias"], device=device)
    parameters["dense2"]["weight"] = ttnn.to_device(parameters["dense2"]["weight"], device=device)
    parameters["dense2"]["bias"] = ttnn.to_device(parameters["dense2"]["bias"], device=device)

    ttnn_model = TtSegformerMixFFN(parameters, hidden_features)

    ttnn_output = ttnn_model(device, ttnn_input_tensor, height=height, width=width, parameters=parameters)
    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)[0]

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
