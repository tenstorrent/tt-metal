# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.MapTR.reference.bevformer import (
    CustomMSDeformableAttention,
)
from models.experimental.MapTR.resources.download_chkpoint import ensure_checkpoint_downloaded, MAPTR_WEIGHTS_PATH
from models.experimental.MapTR.tt.ttnn_custom_defrmble_attention import TtCustomMSDeformableAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)

# Layer prefix for CustomMSDeformableAttention in decoder layer 0
# MapTR uses: pts_bbox_head.transformer.decoder.layers.0.attentions.1
# attentions.0 = MultiheadAttention (self-attention)
# attentions.1 = CustomMSDeformableAttention (cross-attention)
CUSTOM_MS_DEFORMABLE_ATTN_LAYER = "pts_bbox_head.transformer.decoder.layers.0.attentions.1."


def load_maptr_custom_ms_deformable_attention_weights(weights_path: str = MAPTR_WEIGHTS_PATH):
    ensure_checkpoint_downloaded(weights_path)

    checkpoint = torch.load(weights_path, map_location="cpu")
    full_state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    attn_weights = {}
    for key, value in full_state_dict.items():
        if key.startswith(CUSTOM_MS_DEFORMABLE_ATTN_LAYER):
            relative_key = key[len(CUSTOM_MS_DEFORMABLE_ATTN_LAYER) :]
            attn_weights[relative_key] = value

    logger.info(f"Loaded {len(attn_weights)} weight tensors for CustomMSDeformableAttention")
    return attn_weights


def load_torch_model_maptr(torch_model: CustomMSDeformableAttention, weights_path: str = MAPTR_WEIGHTS_PATH):
    attn_weights = load_maptr_custom_ms_deformable_attention_weights(weights_path)

    # Map the checkpoint keys to model keys
    model_state_dict = torch_model.state_dict()
    new_state_dict = {}

    for model_key in model_state_dict.keys():
        if model_key in attn_weights:
            new_state_dict[model_key] = attn_weights[model_key]
        else:
            logger.warning(f"Weight not found in checkpoint for: {model_key}")
            new_state_dict[model_key] = model_state_dict[model_key]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    return torch_model


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, CustomMSDeformableAttention):
        parameters["custom_ms_deformable_attention"] = {}
        parameters["custom_ms_deformable_attention"]["sampling_offsets"] = {}
        parameters["custom_ms_deformable_attention"]["sampling_offsets"]["weight"] = preprocess_linear_weight(
            model.sampling_offsets.weight, dtype=ttnn.bfloat16
        )
        parameters["custom_ms_deformable_attention"]["sampling_offsets"]["bias"] = preprocess_linear_bias(
            model.sampling_offsets.bias, dtype=ttnn.bfloat16
        )
        parameters["custom_ms_deformable_attention"]["attention_weights"] = {}
        parameters["custom_ms_deformable_attention"]["attention_weights"]["weight"] = preprocess_linear_weight(
            model.attention_weights.weight, dtype=ttnn.bfloat16
        )
        parameters["custom_ms_deformable_attention"]["attention_weights"]["bias"] = preprocess_linear_bias(
            model.attention_weights.bias, dtype=ttnn.bfloat16
        )
        parameters["custom_ms_deformable_attention"]["value_proj"] = {}
        parameters["custom_ms_deformable_attention"]["value_proj"]["weight"] = preprocess_linear_weight(
            model.value_proj.weight, dtype=ttnn.bfloat16
        )
        parameters["custom_ms_deformable_attention"]["value_proj"]["bias"] = preprocess_linear_bias(
            model.value_proj.bias, dtype=ttnn.bfloat16
        )
        parameters["custom_ms_deformable_attention"]["output_proj"] = {}
        parameters["custom_ms_deformable_attention"]["output_proj"]["weight"] = preprocess_linear_weight(
            model.output_proj.weight, dtype=ttnn.bfloat16
        )
        parameters["custom_ms_deformable_attention"]["output_proj"]["bias"] = preprocess_linear_bias(
            model.output_proj.bias, dtype=ttnn.bfloat16
        )

    return parameters


def create_maptr_model_parameters_attn(model: CustomMSDeformableAttention, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_custom_ms_deformable_attention(device, reset_seeds):
    embed_dims = 256
    num_heads = 8
    num_levels = 1
    num_points = 4
    batch_first = False
    dropout = 0.1

    torch_model = CustomMSDeformableAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        batch_first=batch_first,
        dropout=dropout,
    )
    torch_model = load_torch_model_maptr(torch_model)

    batch_size = 1
    num_query = 200
    spatial_h, spatial_w = 100, 100
    num_value = spatial_h * spatial_w

    query = torch.randn(num_query, batch_size, embed_dims)
    value = torch.randn(num_value, batch_size, embed_dims)
    identity = query.clone()
    query_pos = torch.randn(num_query, batch_size, embed_dims)
    reference_points = torch.rand(batch_size, num_query, num_levels, 2)
    spatial_shapes = torch.tensor([[spatial_h, spatial_w]], dtype=torch.long)
    level_start_index = torch.tensor([0], dtype=torch.long)

    torch_output = torch_model(
        query=query,
        value=value,
        identity=identity,
        query_pos=query_pos,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )

    parameter = create_maptr_model_parameters_attn(torch_model, device=device)
    tt_model = TtCustomMSDeformableAttention(
        params=parameter.custom_ms_deformable_attention,
        device=device,
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        batch_first=batch_first,
        dropout=dropout,
    )

    query_tt = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    value_tt = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16)
    identity_tt = ttnn.from_torch(identity, device=device, dtype=ttnn.bfloat16)
    query_pos_tt = ttnn.from_torch(query_pos, device=device, dtype=ttnn.bfloat16)
    reference_points_tt = ttnn.from_torch(reference_points, device=device, dtype=ttnn.bfloat16)
    spatial_shapes_tt = ttnn.from_torch(spatial_shapes, device=device, dtype=ttnn.bfloat16)
    level_start_index_tt = ttnn.from_torch(level_start_index, device=device, dtype=ttnn.bfloat16)

    tt_output = tt_model(
        query=query_tt,
        value=value_tt,
        identity=identity_tt,
        query_pos=query_pos_tt,
        reference_points=reference_points_tt,
        spatial_shapes=spatial_shapes_tt,
        level_start_index=level_start_index_tt,
    )

    ttnn_output = ttnn.to_torch(tt_output).float()
    assert torch_output.shape == ttnn_output.shape, f"Shape mismatch: {torch_output.shape} vs {ttnn_output.shape}"
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output.float(), 0.99)
    assert pcc_passed, pcc_message
