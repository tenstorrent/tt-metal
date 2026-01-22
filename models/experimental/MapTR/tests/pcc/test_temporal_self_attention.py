# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.MapTR.reference.bevformer import (
    TemporalSelfAttention,
)
from models.experimental.MapTR.resources.download_chkpoint import ensure_checkpoint_downloaded, MAPTR_WEIGHTS_PATH
from models.experimental.MapTR.tt.ttnn_temporal_self_attention import TtTemporalSelfAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)

TEMPORAL_SELF_ATTN_LAYER = "pts_bbox_head.transformer.encoder.layers.0.attentions.0."


def load_maptr_temporal_self_attention_weights(weights_path: str = MAPTR_WEIGHTS_PATH):
    ensure_checkpoint_downloaded(weights_path)

    checkpoint = torch.load(weights_path, map_location="cpu")
    full_state_dict = checkpoint.get("state_dict", checkpoint)

    tsa_weights = {}
    for key, value in full_state_dict.items():
        if key.startswith(TEMPORAL_SELF_ATTN_LAYER):
            relative_key = key[len(TEMPORAL_SELF_ATTN_LAYER) :]
            tsa_weights[relative_key] = value

    logger.info(f"Loaded {len(tsa_weights)} weight tensors for temporal self attention")
    return tsa_weights


def load_torch_model_maptr(torch_model: TemporalSelfAttention, weights_path: str = MAPTR_WEIGHTS_PATH):
    tsa_weights = load_maptr_temporal_self_attention_weights(weights_path)

    # Map the checkpoint keys to model keys
    model_state_dict = torch_model.state_dict()
    new_state_dict = {}

    for model_key in model_state_dict.keys():
        if model_key in tsa_weights:
            new_state_dict[model_key] = tsa_weights[model_key]
        else:
            logger.warning(f"Weight not found in checkpoint for: {model_key}")
            new_state_dict[model_key] = model_state_dict[model_key]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    return torch_model


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, TemporalSelfAttention):
        parameters["temporal_self_attention"] = {
            "sampling_offsets": {
                "weight": preprocess_linear_weight(model.sampling_offsets.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.sampling_offsets.bias, dtype=ttnn.bfloat16),
            },
            "attention_weights": {
                "weight": preprocess_linear_weight(model.attention_weights.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.attention_weights.bias, dtype=ttnn.bfloat16),
            },
            "value_proj": {
                "weight": preprocess_linear_weight(model.value_proj.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.value_proj.bias, dtype=ttnn.bfloat16),
            },
            "output_proj": {
                "weight": preprocess_linear_weight(model.output_proj.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.output_proj.bias, dtype=ttnn.bfloat16),
            },
        }

    return parameters


def create_maptr_model_parameters_tsa(model: TemporalSelfAttention, input_tensor, device=None):
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
def test_maptr_temporal_self_attention(device, reset_seeds):
    embed_dims = 256
    num_levels = 1

    torch_model = TemporalSelfAttention(embed_dims=embed_dims, num_levels=num_levels)
    torch_model = load_torch_model_maptr(torch_model)

    num_query = 10000

    query = torch.randn(1, num_query, embed_dims)
    query_pos = torch.randn(1, num_query, embed_dims)
    reference_points = torch.randn(2, num_query, 1, 2)
    spatial_shapes = torch.tensor([[100, 100]])
    level_start_index = torch.tensor([0])

    torch_output = torch_model(
        query,
        query_pos=query_pos,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )

    parameter = create_maptr_model_parameters_tsa(
        torch_model, [query, query_pos, reference_points, spatial_shapes, level_start_index], device
    )

    tt_model = TtTemporalSelfAttention(
        params=parameter.temporal_self_attention, device=device, embed_dims=embed_dims, num_levels=num_levels
    )

    query_tt = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    query_pos_tt = ttnn.from_torch(query_pos, device=device, dtype=ttnn.bfloat16)
    reference_points_tt = ttnn.from_torch(reference_points, device=device, dtype=ttnn.bfloat16)
    spatial_shapes_tt = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.bfloat16)
    level_start_index_tt = ttnn.from_torch(level_start_index, device=device, dtype=ttnn.bfloat16)

    tt_output = tt_model(
        query_tt,
        query_pos=query_pos_tt,
        reference_points=reference_points_tt,
        spatial_shapes=spatial_shapes_tt,
        level_start_index=level_start_index_tt,
    )

    ttnn_output = ttnn.to_torch(tt_output)
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.99)
    assert pcc_passed, pcc_message
