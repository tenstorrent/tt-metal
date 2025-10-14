# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from models.experimental.detr3d.reference.model_3detr import TransformerDecoderLayer as ref_model
from models.experimental.detr3d.source.detr3d.models.transformer import TransformerDecoderLayer as org_model
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "d_model, nhead, dim_feedforward, dropout, dropout_attn,activation, normalize_before, norm_fn_name,tgt_shape, memory_shape, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,pos_shape, query_pos_shape, return_attn_weights",
    [
        (
            256,
            4,
            256,
            0.1,
            None,
            "relu",
            True,
            "ln",
            (128, 1, 256),  # tgt
            (1024, 1, 256),  # memory
            None,  # tgt_mask
            None,  # memory_mask
            None,  # tgt_key_padding_mask
            None,  # memory_key_padding_mask
            (1024, 1, 256),  # pos
            (128, 1, 256),  # query_pos
            False,  # return_attn_weights
        )
    ],
)
def test_transformer_decoder_layer(
    d_model,
    nhead,
    dim_feedforward,
    dropout,
    dropout_attn,
    activation,
    normalize_before,
    norm_fn_name,
    tgt_shape,
    memory_shape,
    tgt_mask,
    memory_mask,
    tgt_key_padding_mask,
    memory_key_padding_mask,
    pos_shape,
    query_pos_shape,
    return_attn_weights,
):
    org_module = org_model(
        d_model, nhead, dim_feedforward, dropout, dropout_attn, activation, normalize_before, norm_fn_name
    ).to(torch.bfloat16)
    org_module.eval()
    ref_module = ref_model(
        d_model, nhead, dim_feedforward, dropout, dropout_attn, activation, normalize_before, norm_fn_name
    ).to(torch.bfloat16)
    ref_module.eval()
    ref_module.load_state_dict(org_module.state_dict())
    tgt = torch.randn(tgt_shape, dtype=torch.bfloat16)
    memory = torch.randn(memory_shape, dtype=torch.bfloat16)
    pos = torch.randn(pos_shape, dtype=torch.bfloat16)
    query_pos = torch.randn(query_pos_shape, dtype=torch.bfloat16)
    org_out = org_module(
        tgt=tgt,
        memory=memory,
        tgt_mask=tgt_mask,
        memory_mask=memory_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
        pos=pos,
        query_pos=query_pos,
        return_attn_weights=return_attn_weights,
    )
    ref_out = ref_module(
        tgt=tgt,
        memory=memory,
        tgt_mask=tgt_mask,
        memory_mask=memory_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
        pos=pos,
        query_pos=query_pos,
        return_attn_weights=return_attn_weights,
    )
    assert_with_pcc(org_out[0], ref_out[0], 1.0)
