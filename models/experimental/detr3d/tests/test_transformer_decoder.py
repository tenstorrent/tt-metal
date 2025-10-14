# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from models.experimental.detr3d.reference.model_3detr import (
    TransformerDecoder as ref_model,
    TransformerDecoderLayer as ref_layer,
)
from models.experimental.detr3d.source.detr3d.models.transformer import (
    TransformerDecoder as org_model,
    TransformerDecoderLayer as org_layer,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "d_model, nhead, dim_feedforward, dropout, dropout_attn,activation, normalize_before",
    [
        (
            256,
            4,
            256,
            0.1,
            None,
            "relu",
            True,
        )
    ],
)
@pytest.mark.parametrize(
    "num_layers, norm_fn_name, return_intermediate, weight_init_name,tgt_shape, memory_shape, pos_shape, query_pos_shape,tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,transpose_swap, return_attn_weights",
    [
        (
            8,
            "ln",
            True,
            "xavier_uniform",
            (128, 1, 256),  # tgt
            (1024, 1, 256),  # memory
            (1024, 1, 256),  # pos
            (128, 1, 256),  # query_pos
            None,
            None,
            None,
            None,  # masks
            False,
            False,  # flags
        )
    ],
)
def test_transformer_decoder(
    num_layers,
    norm_fn_name,
    return_intermediate,
    weight_init_name,
    tgt_shape,
    memory_shape,
    pos_shape,
    query_pos_shape,
    tgt_mask,
    memory_mask,
    tgt_key_padding_mask,
    memory_key_padding_mask,
    transpose_swap,
    return_attn_weights,
    d_model,
    nhead,
    dim_feedforward,
    dropout,
    dropout_attn,
    activation,
    normalize_before,
):
    org_sub_module = org_layer(
        d_model, nhead, dim_feedforward, dropout, dropout_attn, activation, normalize_before, norm_fn_name
    ).to(torch.bfloat16)
    org_sub_module.eval()
    ref_sub_module = ref_layer(
        d_model, nhead, dim_feedforward, dropout, dropout_attn, activation, normalize_before, norm_fn_name
    ).to(torch.bfloat16)
    ref_sub_module.eval()
    org_module = org_model(org_sub_module, num_layers, norm_fn_name, return_intermediate, weight_init_name).to(
        torch.bfloat16
    )
    org_module.eval()
    ref_module = ref_model(ref_sub_module, num_layers, norm_fn_name, return_intermediate, weight_init_name).to(
        torch.bfloat16
    )
    ref_module.eval()
    ref_module.load_state_dict(org_module.state_dict())
    tgt = torch.randn(tgt_shape, dtype=torch.bfloat16)
    memory = torch.randn(memory_shape, dtype=torch.bfloat16)
    pos = torch.randn(pos_shape, dtype=torch.bfloat16)
    query_pos = torch.randn(query_pos_shape, dtype=torch.bfloat16)
    org_output, org_attns = org_module(
        tgt=tgt,
        memory=memory,
        tgt_mask=tgt_mask,
        memory_mask=memory_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
        pos=pos,
        query_pos=query_pos,
        transpose_swap=transpose_swap,
        return_attn_weights=return_attn_weights,
    )
    ref_output, ref_attns = ref_module(
        tgt=tgt,
        memory=memory,
        tgt_mask=tgt_mask,
        memory_mask=memory_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
        pos=pos,
        query_pos=query_pos,
        transpose_swap=transpose_swap,
        return_attn_weights=return_attn_weights,
    )
    assert_with_pcc(org_output, ref_output, 1.0)
    for i, (a, b) in enumerate(zip(org_attns, ref_attns)):
        assert_with_pcc(a, b, 1.0)
