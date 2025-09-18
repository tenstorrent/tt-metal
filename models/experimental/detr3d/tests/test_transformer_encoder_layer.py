# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from models.experimental.detr3d.source.detr3d.models.transformer import TransformerEncoderLayer as org_model
from models.experimental.detr3d.reference.detr3d_model import TransformerEncoderLayer as ref_model
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "d_model, nhead, dim_feedforward, dropout, dropout_attn, activation, normalize_before, norm_name, use_ffn, ffn_use_bias, src_shape, src_mask_shape, src_key_padding_mask, pos,return_attn_weights",
    [
        (
            256,  # d_model
            4,  # nhead
            128,  # dim_feedforward
            0.0,  # dropout
            None,  # dropout_attn
            "relu",  # activation
            True,  # normalize_before
            "ln",  # norm_name
            True,  # use_ffn
            True,  # ffn_use_bias
            (2048, 1, 256),  # src_shape
            (4, 2048, 2048),  # src_mask_shape
            None,  # src_key_padding_mask
            None,  # pos
            False,  # return_attn_weights
        ),
        (
            256,  # d_model
            4,  # nhead
            128,  # dim_feedforward
            0.0,  # dropout
            None,  # dropout_attn
            "relu",  # activation
            True,  # normalize_before
            "ln",  # norm_name
            True,  # use_ffn
            True,  # ffn_use_bias
            (1024, 1, 256),  # src_shape
            (4, 1024, 1024),  # src_mask_shape
            None,  # src_key_padding_mask
            None,  # pos
            False,  # return_attn_weights
        ),
    ],
)
def test_transformer_encoder_layer(
    d_model,
    nhead,
    dim_feedforward,
    dropout,
    dropout_attn,
    activation,
    normalize_before,
    norm_name,
    use_ffn,
    ffn_use_bias,
    src_shape,
    src_mask_shape,
    src_key_padding_mask,
    pos,
    return_attn_weights,
):
    org_module = org_model(
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        dropout_attn,
        activation,
        normalize_before,
        norm_name,
        use_ffn,
        ffn_use_bias,
    ).to(torch.bfloat16)
    org_module.eval()
    ref_module = ref_model(
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        dropout_attn,
        activation,
        normalize_before,
        norm_name,
        use_ffn,
        ffn_use_bias,
    ).to(torch.bfloat16)
    ref_module.eval()
    ref_module.load_state_dict(org_module.state_dict())
    src = torch.randn(src_shape, dtype=torch.bfloat16)
    src_mask = torch.randn(src_mask_shape, dtype=torch.bfloat16)
    org_out = org_module(src, src_mask, src_key_padding_mask, pos, return_attn_weights)
    ref_out = ref_module(src, src_mask, src_key_padding_mask, pos, return_attn_weights)
    assert_with_pcc(org_out, ref_out, 1.0)
