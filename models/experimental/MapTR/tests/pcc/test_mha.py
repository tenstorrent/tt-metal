# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from models.experimental.MapTR.reference.dependency import MultiheadAttention
from models.experimental.MapTR.resources.download_chkpoint import ensure_checkpoint_downloaded, MAPTR_WEIGHTS_PATH
from models.experimental.MapTR.tt.ttnn_mha import TtMultiheadAttention
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)

MHA_LAYER = "pts_bbox_head.transformer.decoder.layers.0.attentions.0.attn."


def load_maptr_mha_weights(weights_path: str = MAPTR_WEIGHTS_PATH):
    ensure_checkpoint_downloaded(weights_path)

    checkpoint = torch.load(weights_path, map_location="cpu")
    full_state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    mha_weights = {}
    for key, value in full_state_dict.items():
        if key.startswith(MHA_LAYER):
            relative_key = key[len(MHA_LAYER) :]
            mha_weights[relative_key] = value

    logger.info(f"Loaded {len(mha_weights)} weight tensors for MultiheadAttention")
    return mha_weights


def load_torch_model_maptr(torch_model: MultiheadAttention, weights_path: str = MAPTR_WEIGHTS_PATH):
    mha_weights = load_maptr_mha_weights(weights_path)
    model_state_dict = torch_model.state_dict()
    new_state_dict = {}

    key_mapping = {
        "attn.in_proj_weight": "in_proj_weight",
        "attn.in_proj_bias": "in_proj_bias",
        "attn.out_proj.weight": "out_proj.weight",
        "attn.out_proj.bias": "out_proj.bias",
    }

    for model_key in model_state_dict.keys():
        checkpoint_key = key_mapping.get(model_key)
        if checkpoint_key and checkpoint_key in mha_weights:
            new_state_dict[model_key] = mha_weights[checkpoint_key]
        else:
            logger.warning(f"Weight not found: {model_key}")
            new_state_dict[model_key] = model_state_dict[model_key]

    torch_model.load_state_dict(new_state_dict, strict=False)
    torch_model.eval()
    return torch_model


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, MultiheadAttention):
        parameters["multihead_attention"] = {
            "in_proj": {
                "weight": preprocess_linear_weight(model.attn.in_proj_weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.attn.in_proj_bias, dtype=ttnn.bfloat16)
                if model.attn.in_proj_bias is not None
                else None,
            },
            "out_proj": {
                "weight": preprocess_linear_weight(model.attn.out_proj.weight, dtype=ttnn.bfloat16),
                "bias": preprocess_linear_bias(model.attn.out_proj.bias, dtype=ttnn.bfloat16)
                if model.attn.out_proj.bias is not None
                else None,
            },
        }

    return parameters


def create_maptr_model_parameters_mha(model: MultiheadAttention, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_multihead_attention(device, reset_seeds):
    embed_dims = 256
    num_heads = 8
    batch_first = False
    dropout = 0.1

    torch_model = MultiheadAttention(
        embed_dims=embed_dims, num_heads=num_heads, dropout=dropout, batch_first=batch_first
    )
    torch_model = load_torch_model_maptr(torch_model)

    batch_size = 1
    num_query = 200
    num_key = 200

    query = torch.randn(num_query, batch_size, embed_dims)
    key = torch.randn(num_key, batch_size, embed_dims)
    value = torch.randn(num_key, batch_size, embed_dims)
    identity = query.clone()
    query_pos = torch.randn(num_query, batch_size, embed_dims)
    key_pos = torch.randn(num_key, batch_size, embed_dims)

    torch_output = torch_model(
        query=query, key=key, value=value, identity=identity, query_pos=query_pos, key_pos=key_pos
    )

    parameter = create_maptr_model_parameters_mha(torch_model, device=device)
    tt_model = TtMultiheadAttention(
        params=parameter.multihead_attention,
        device=device,
        embed_dims=embed_dims,
        num_heads=num_heads,
        batch_first=batch_first,
    )

    query_tt = ttnn.from_torch(query, device=device, dtype=ttnn.bfloat16)
    key_tt = ttnn.from_torch(key, device=device, dtype=ttnn.bfloat16)
    value_tt = ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16)
    identity_tt = ttnn.from_torch(identity, device=device, dtype=ttnn.bfloat16)
    query_pos_tt = ttnn.from_torch(query_pos, device=device, dtype=ttnn.bfloat16)
    key_pos_tt = ttnn.from_torch(key_pos, device=device, dtype=ttnn.bfloat16)

    tt_output = tt_model(
        query=query_tt, key=key_tt, value=value_tt, identity=identity_tt, query_pos=query_pos_tt, key_pos=key_pos_tt
    )

    ttnn_output = ttnn.to_torch(tt_output).float()
    assert torch_output.shape == ttnn_output.shape, f"Shape mismatch: {torch_output.shape} vs {ttnn_output.shape}"
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output.float(), 0.99)
    assert pcc_passed, pcc_message
