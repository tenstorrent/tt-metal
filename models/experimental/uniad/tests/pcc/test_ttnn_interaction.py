# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

import ttnn
import pytest

from models.experimental.uniad.reference.motion_transformer_decoder import (
    MapInteraction,
    TrackAgentInteraction,
    IntentionInteraction,
)
from models.experimental.uniad.tt.ttnn_interaction import (
    TtMapInteraction,
    TtTrackAgentInteraction,
    TtIntentionInteraction,
)

from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.uniad.common import load_torch_model


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, MapInteraction) or isinstance(model, TrackAgentInteraction):
        parameters["interaction_transformer"] = {}
        child = model.interaction_transformer
        if isinstance(child, nn.TransformerDecoderLayer):
            parameters_tmp = {}

            parameters_tmp["self_attn"] = {}
            parameters_tmp["self_attn"]["in_proj_weight"] = ttnn.from_torch(
                child.self_attn.in_proj_weight, dtype=ttnn.bfloat16
            )
            parameters_tmp["self_attn"]["in_proj_bias"] = ttnn.from_torch(
                child.self_attn.in_proj_bias, dtype=ttnn.bfloat16
            )
            parameters_tmp["self_attn"]["out_proj"] = {}
            parameters_tmp["self_attn"]["out_proj"]["weight"] = preprocess_linear_weight(
                child.self_attn.out_proj.weight, dtype=ttnn.bfloat16
            )
            parameters_tmp["self_attn"]["out_proj"]["bias"] = preprocess_linear_bias(
                child.self_attn.out_proj.bias, dtype=ttnn.bfloat16
            )

            parameters_tmp["multihead_attn"] = {}
            parameters_tmp["multihead_attn"]["in_proj_weight"] = ttnn.from_torch(
                child.multihead_attn.in_proj_weight, dtype=ttnn.bfloat16
            )
            parameters_tmp["multihead_attn"]["in_proj_bias"] = ttnn.from_torch(
                child.multihead_attn.in_proj_bias, dtype=ttnn.bfloat16
            )
            parameters_tmp["multihead_attn"]["out_proj"] = {}
            parameters_tmp["multihead_attn"]["out_proj"]["weight"] = preprocess_linear_weight(
                child.multihead_attn.out_proj.weight, dtype=ttnn.bfloat16
            )
            parameters_tmp["multihead_attn"]["out_proj"]["bias"] = preprocess_linear_bias(
                child.multihead_attn.out_proj.bias, dtype=ttnn.bfloat16
            )

            parameters_tmp["linear1"] = {}
            parameters_tmp["linear1"]["weight"] = preprocess_linear_weight(child.linear1.weight, dtype=ttnn.bfloat16)
            parameters_tmp["linear1"]["bias"] = preprocess_linear_bias(child.linear1.bias, dtype=ttnn.bfloat16)

            parameters_tmp["linear2"] = {}
            parameters_tmp["linear2"]["weight"] = preprocess_linear_weight(child.linear2.weight, dtype=ttnn.bfloat16)
            parameters_tmp["linear2"]["bias"] = preprocess_linear_bias(child.linear2.bias, dtype=ttnn.bfloat16)

            parameters_tmp["norm1"] = {}
            parameters_tmp["norm1"]["weight"] = preprocess_layernorm_parameter(child.norm1.weight, dtype=ttnn.bfloat16)
            parameters_tmp["norm1"]["bias"] = preprocess_layernorm_parameter(child.norm1.bias, dtype=ttnn.bfloat16)

            parameters_tmp["norm2"] = {}
            parameters_tmp["norm2"]["weight"] = preprocess_layernorm_parameter(child.norm2.weight, dtype=ttnn.bfloat16)
            parameters_tmp["norm2"]["bias"] = preprocess_layernorm_parameter(child.norm2.bias, dtype=ttnn.bfloat16)

            parameters_tmp["norm3"] = {}
            parameters_tmp["norm3"]["weight"] = preprocess_layernorm_parameter(child.norm3.weight, dtype=ttnn.bfloat16)
            parameters_tmp["norm3"]["bias"] = preprocess_layernorm_parameter(child.norm3.bias, dtype=ttnn.bfloat16)

        parameters["interaction_transformer"] = parameters_tmp

    if isinstance(model, IntentionInteraction):
        parameters["interaction_transformer"] = {}
        child = model.interaction_transformer
        if isinstance(child, nn.TransformerEncoderLayer):
            parameters_tmp = {}
            parameters_tmp["self_attn"] = {}
            parameters_tmp["self_attn"]["in_proj_weight"] = ttnn.from_torch(
                child.self_attn.in_proj_weight, dtype=ttnn.bfloat16
            )
            parameters_tmp["self_attn"]["in_proj_bias"] = ttnn.from_torch(
                child.self_attn.in_proj_bias, dtype=ttnn.bfloat16
            )
            parameters_tmp["self_attn"]["out_proj"] = {}
            parameters_tmp["self_attn"]["out_proj"]["weight"] = preprocess_linear_weight(
                child.self_attn.out_proj.weight, dtype=ttnn.bfloat16
            )
            parameters_tmp["self_attn"]["out_proj"]["bias"] = preprocess_linear_bias(
                child.self_attn.out_proj.bias, dtype=ttnn.bfloat16
            )

            parameters_tmp["linear1"] = {}
            parameters_tmp["linear1"]["weight"] = preprocess_linear_weight(child.linear1.weight, dtype=ttnn.bfloat16)
            parameters_tmp["linear1"]["bias"] = preprocess_linear_bias(child.linear1.bias, dtype=ttnn.bfloat16)

            parameters_tmp["linear2"] = {}
            parameters_tmp["linear2"]["weight"] = preprocess_linear_weight(child.linear2.weight, dtype=ttnn.bfloat16)
            parameters_tmp["linear2"]["bias"] = preprocess_linear_bias(child.linear2.bias, dtype=ttnn.bfloat16)

            parameters_tmp["norm1"] = {}
            parameters_tmp["norm1"]["weight"] = preprocess_layernorm_parameter(child.norm1.weight, dtype=ttnn.bfloat16)
            parameters_tmp["norm1"]["bias"] = preprocess_layernorm_parameter(child.norm1.bias, dtype=ttnn.bfloat16)

            parameters_tmp["norm2"] = {}
            parameters_tmp["norm2"]["weight"] = preprocess_layernorm_parameter(child.norm2.weight, dtype=ttnn.bfloat16)
            parameters_tmp["norm2"]["bias"] = preprocess_layernorm_parameter(child.norm2.bias, dtype=ttnn.bfloat16)

        parameters["interaction_transformer"] = parameters_tmp

    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_MapInteraction(device, reset_seeds, model_location_generator):
    reference_model = MapInteraction(embed_dims=256, num_heads=8, batch_first=True)

    reference_model = load_torch_model(
        torch_model=reference_model,
        layer="motion_head.motionformer.map_interaction_layers.0",
        model_location_generator=model_location_generator,
    )

    query = torch.randn(1, 1, 6, 256)
    key = torch.randn(1, 300, 256)
    query_pos = torch.randn(1, 1, 6, 256)
    key_pos = torch.randn(1, 300, 256)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    torch_output = reference_model(query, key, query_pos, key_pos)

    ttnn_query = ttnn.from_torch(query, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn_key = ttnn.from_torch(key, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn_query_pos = ttnn.from_torch(query_pos, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn_key_pos = ttnn.from_torch(key_pos, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    ttnn_model = TtMapInteraction(parameters=parameters, device=device, embed_dims=256, num_heads=8, batch_first=True)

    ttnn_output = ttnn_model(query=ttnn_query, key=ttnn_key, query_pos=ttnn_query_pos, key_pos=ttnn_key_pos)

    assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output), pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_TrackAgentInteraction(device, reset_seeds, model_location_generator):
    reference_model = TrackAgentInteraction(embed_dims=256, num_heads=8, batch_first=True)

    reference_model = load_torch_model(
        torch_model=reference_model,
        layer="motion_head.motionformer.track_agent_interaction_layers.0",
        model_location_generator=model_location_generator,
    )

    query = torch.randn(1, 1, 6, 256)
    key = torch.randn(1, 1, 256)
    query_pos = torch.randn(1, 1, 6, 256)
    key_pos = torch.randn(1, 1, 256)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    torch_output = reference_model(query, key, query_pos, key_pos)

    ttnn_query = ttnn.from_torch(query, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn_key = ttnn.from_torch(key, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn_query_pos = ttnn.from_torch(query_pos, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn_key_pos = ttnn.from_torch(key_pos, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    ttnn_model = TtTrackAgentInteraction(
        parameters=parameters, device=device, embed_dims=256, num_heads=8, batch_first=True
    )

    ttnn_output = ttnn_model(query=ttnn_query, key=ttnn_key, query_pos=ttnn_query_pos, key_pos=ttnn_key_pos)

    assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output), pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_IntentionInteraction(device, reset_seeds, model_location_generator):
    reference_model = IntentionInteraction(embed_dims=256, num_heads=8, batch_first=True)
    reference_model = load_torch_model(
        torch_model=reference_model,
        layer="motion_head.motionformer.intention_interaction_layers",
        model_location_generator=model_location_generator,
    )

    query = torch.randn(1, 1, 6, 256)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    torch_output = reference_model(query)

    ttnn_query = ttnn.from_torch(query, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    ttnn_model = TtIntentionInteraction(
        parameters=parameters, device=device, embed_dims=256, num_heads=8, batch_first=True
    )

    ttnn_output = ttnn_model(query=ttnn_query)

    assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output), pcc=0.99)
