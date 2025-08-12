# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.vadv2.reference import temporal_self_attention
from models.experimental.vadv2.tt import tt_temporal_self_attention
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.vadv2.reference.resnet import ResNet
from models.experimental.vadv2.reference.temporal_self_attention import TemporalSelfAttention
from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from models.experimental.vadv2.common import load_torch_model


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, TemporalSelfAttention):
        parameters["temporal_self_attention"] = {}
        parameters["temporal_self_attention"]["sampling_offsets"] = {}
        parameters["temporal_self_attention"]["sampling_offsets"]["weight"] = preprocess_linear_weight(
            model.sampling_offsets.weight, dtype=ttnn.bfloat16
        )
        parameters["temporal_self_attention"]["sampling_offsets"]["bias"] = preprocess_linear_bias(
            model.sampling_offsets.bias, dtype=ttnn.bfloat16
        )
        parameters["temporal_self_attention"]["attention_weights"] = {}
        parameters["temporal_self_attention"]["attention_weights"]["weight"] = preprocess_linear_weight(
            model.attention_weights.weight, dtype=ttnn.bfloat16
        )
        parameters["temporal_self_attention"]["attention_weights"]["bias"] = preprocess_linear_bias(
            model.attention_weights.bias, dtype=ttnn.bfloat16
        )
        parameters["temporal_self_attention"]["value_proj"] = {}
        parameters["temporal_self_attention"]["value_proj"]["weight"] = preprocess_linear_weight(
            model.value_proj.weight, dtype=ttnn.bfloat16
        )
        parameters["temporal_self_attention"]["value_proj"]["bias"] = preprocess_linear_bias(
            model.value_proj.bias, dtype=ttnn.bfloat16
        )
        parameters["temporal_self_attention"]["output_proj"] = {}
        parameters["temporal_self_attention"]["output_proj"]["weight"] = preprocess_linear_weight(
            model.output_proj.weight, dtype=ttnn.bfloat16
        )
        parameters["temporal_self_attention"]["output_proj"]["bias"] = preprocess_linear_bias(
            model.output_proj.bias, dtype=ttnn.bfloat16
        )

    return parameters


def create_vadv2_model_parameters_tsa(model: ResNet, input_tensor, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model,
        run_model=lambda model: model(
            input_tensor[0],
            query_pos=input_tensor[1],
            reference_points=input_tensor[2],
            spatial_shapes=input_tensor[3],
            level_start_index=input_tensor[4],
        ),
        device=None,
    )
    assert parameters is not None
    for key in parameters.conv_args.keys():
        parameters.conv_args[key].module = getattr(model, key)
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_vadv2_tsa(
    device,
    reset_seeds,
    model_location_generator,
):
    torch_model = temporal_self_attention.TemporalSelfAttention(embed_dims=256, num_levels=1)
    torch_model = load_torch_model(
        torch_model=torch_model,
        layer="pts_bbox_head.transformer.encoder.layers.0.attentions.0.",
        model_location_generator=model_location_generator,
    )

    query = torch.randn(1, 10000, 256)
    query_pos = torch.randn(1, 10000, 256)
    reference_points = torch.randn(2, 10000, 1, 2)
    spatial_shapes = torch.tensor([[100, 100]])
    level_start_index = torch.tensor([0])

    torch_output = torch_model(
        query,
        query_pos=query_pos,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )

    parameter = create_vadv2_model_parameters_tsa(
        torch_model, [query, query_pos, reference_points, spatial_shapes, level_start_index], device
    )

    ttnn_model = tt_temporal_self_attention.TtTemporalSelfAttention(
        params=parameter.temporal_self_attention, device=device, embed_dims=256, num_levels=1
    )

    query = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    query_pos = ttnn.from_torch(query_pos, device=device, dtype=ttnn.bfloat16)
    reference_points = ttnn.from_torch(reference_points, device=device, dtype=ttnn.bfloat16)
    spatial_shapes = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.bfloat16)
    level_start_index = ttnn.from_torch(level_start_index, device=device, dtype=ttnn.bfloat16)

    ttnn_output = ttnn_model(
        query,
        query_pos=query_pos,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(pcc_message)
