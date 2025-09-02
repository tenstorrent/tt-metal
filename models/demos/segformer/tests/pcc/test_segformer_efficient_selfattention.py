# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.segformer.common import load_config, load_torch_model
from models.demos.segformer.reference.segformer_efficient_selfattention import SegformerEfficientSelfAttention
from models.demos.segformer.tt.common import (
    preprocess_layernorm_parameter,
    preprocess_linear_bias,
    preprocess_linear_weight,
)
from models.demos.segformer.tt.ttnn_segformer_efficient_selfattention import TtSegformerEfficientSelfAttention
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(model, name, mesh_mapper)

    def custom_preprocessor(model, name, mesh_mapper=None):
        parameters = {}
        if isinstance(model, SegformerEfficientSelfAttention):
            parameters["query"] = {}
            parameters["query"]["weight"] = preprocess_linear_weight(
                model.query.weight, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )
            parameters["query"]["bias"] = preprocess_linear_bias(
                model.query.bias, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )

            parameters["key"] = {}
            parameters["key"]["weight"] = preprocess_linear_weight(
                model.key.weight, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )
            parameters["key"]["bias"] = preprocess_linear_bias(
                model.key.bias, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )

            parameters["value"] = {}
            parameters["value"]["weight"] = preprocess_linear_weight(
                model.value.weight, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )
            parameters["value"]["bias"] = preprocess_linear_bias(
                model.value.bias, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
            )

            if model.sr_ratio > 1:
                parameters["sr"] = {}

                parameters["sr"]["weight"] = ttnn.from_torch(
                    model.sr.weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                )
                parameters["sr"]["bias"] = ttnn.from_torch(
                    torch.reshape(model.sr.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper
                )

                parameters["layer_norm"] = {}
                parameters["layer_norm"]["weight"] = preprocess_layernorm_parameter(
                    model.layer_norm.weight, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
                )
                parameters["layer_norm"]["bias"] = preprocess_layernorm_parameter(
                    model.layer_norm.bias, dtype=ttnn.bfloat8_b, mesh_mapper=mesh_mapper
                )

        return parameters

    return custom_mesh_preprocessor


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size, height, width, num_attention_heads, sequence_reduction_ratio, block_i, efficient_self_attention_i",
    [
        (1, 16384, 32, 128, 128, 1, 8, 0, 0),
        (1, 16384, 32, 128, 128, 1, 8, 0, 1),
        (1, 4096, 64, 64, 64, 2, 4, 1, 0),
        (1, 4096, 64, 64, 64, 2, 4, 1, 1),
        (1, 1024, 160, 32, 32, 5, 2, 2, 0),
        (1, 1024, 160, 32, 32, 5, 2, 2, 1),
        (1, 256, 256, 16, 16, 8, 1, 3, 0),
        (1, 256, 256, 16, 16, 8, 1, 3, 1),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_efficient_selfattention(
    device,
    batch_size,
    seq_len,
    hidden_size,
    height,
    width,
    num_attention_heads,
    sequence_reduction_ratio,
    block_i,
    efficient_self_attention_i,
    model_location_generator,
):
    torch_input_tensor = torch.randn(batch_size, 1, seq_len, hidden_size)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat8_b,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    config = load_config("configs/segformer_semantic_config.json")
    reference_model = SegformerEfficientSelfAttention(
        config=config,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        sequence_reduction_ratio=sequence_reduction_ratio,
    )
    target_prefix = f"encoder.block.{block_i}.{efficient_self_attention_i}.attention."
    reference_model = load_torch_model(
        reference_model, target_prefix, module="semantic_sub", model_location_generator=model_location_generator
    )

    torch_input_tensor = torch.reshape(torch_input_tensor, (batch_size, seq_len, hidden_size))
    torch_output = reference_model(torch_input_tensor, height, width)
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )
    if "sr" in parameters:
        parameters["sr"]["weight"] = ttnn.from_device(parameters["sr"]["weight"])
        parameters["sr"]["bias"] = ttnn.from_device(parameters["sr"]["bias"])

    ttnn_model = TtSegformerEfficientSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        parameters=parameters,
        sequence_reduction_ratio=reference_model.sr_ratio,
    )
    ttnn_output = ttnn_model(device, ttnn_input_tensor, height, width, parameters=parameters)
    ttnn_final_output = ttnn.to_torch(ttnn_output[0])
    if len(ttnn_final_output.shape) == 4:
        ttnn_final_output = ttnn_final_output[0]
    assert_with_pcc(torch_output[0], ttnn_final_output, pcc=0.977)
