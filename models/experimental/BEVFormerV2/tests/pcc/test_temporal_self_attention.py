# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.BEVFormerV2.reference.temporal_self_attention import TemporalSelfAttention
from models.experimental.BEVFormerV2.tt.ttnn_temporal_self_attention import TtTemporalSelfAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from models.experimental.BEVFormerV2.common import download_bevformerv2_weights


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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_temporal_self_attention(
    device,
    reset_seeds,
    model_location_generator,
):
    torch_model = TemporalSelfAttention(embed_dims=256, num_levels=1)

    weights_path = download_bevformerv2_weights()
    checkpoint = torch.load(weights_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    tsa_state = {}
    for key, value in state_dict.items():
        if "encoder.layers.0.attentions.0" in key:
            new_key = key.replace("pts_bbox_head.transformer.encoder.layers.0.attentions.0.", "")
            tsa_state[new_key] = value

    torch_model.load_state_dict(tsa_state, strict=False)
    torch_model.eval()

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

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    ttnn_model = TtTemporalSelfAttention(
        params=parameters.temporal_self_attention, device=device, embed_dims=256, num_levels=1
    )

    query = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    query_pos = ttnn.from_torch(query_pos, device=device, dtype=ttnn.bfloat16)
    reference_points = ttnn.from_torch(reference_points, device=device, dtype=ttnn.bfloat16)
    spatial_shapes = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.uint32)
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
