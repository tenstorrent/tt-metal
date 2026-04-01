# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for RF-DETR Medium transformer decoder.
Tests self-attention and FFN components.
"""

import pytest
import torch

import ttnn

from models.experimental.rfdetr_medium.common import (
    RFDETR_MEDIUM_L1_SMALL_SIZE,
    HIDDEN_DIM,
    NUM_QUERIES,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": RFDETR_MEDIUM_L1_SMALL_SIZE}],
    indirect=True,
)
def test_decoder_self_attention(device, torch_model):
    """
    Test decoder self-attention against PyTorch nn.MultiheadAttention.
    """
    from models.experimental.rfdetr_medium.tt.model_preprocessing import load_decoder_weights
    from models.experimental.rfdetr_medium.tt.tt_decoder import decoder_self_attention

    decoder_params = load_decoder_weights(torch_model, device)

    torch.manual_seed(42)
    tgt = torch.randn(1, NUM_QUERIES, HIDDEN_DIM)
    query_pos = torch.randn(1, NUM_QUERIES, HIDDEN_DIM)

    # PyTorch reference
    decoder = torch_model.transformer.decoder
    layer = decoder.layers[0]
    with torch.no_grad():
        q = k = tgt + query_pos
        v = tgt
        tgt2 = layer.self_attn(q, k, v, need_weights=False)[0]
        ref_output = layer.norm1(tgt + tgt2)

    # TTNN
    tgt_tt = ttnn.from_torch(tgt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    query_pos_tt = ttnn.from_torch(query_pos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = decoder_self_attention(tgt_tt, query_pos_tt, decoder_params["layers"][0])
    tt_output_torch = ttnn.to_torch(tt_output).float()

    assert_with_pcc(ref_output, tt_output_torch, pcc=0.97)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": RFDETR_MEDIUM_L1_SMALL_SIZE}],
    indirect=True,
)
def test_decoder_ffn(device, torch_model):
    """
    Test decoder FFN: Linear(256→2048) → ReLU → Linear(2048→256).
    """
    from models.experimental.rfdetr_medium.tt.model_preprocessing import load_decoder_weights
    from models.experimental.rfdetr_medium.tt.tt_decoder import decoder_ffn

    decoder_params = load_decoder_weights(torch_model, device)

    torch.manual_seed(42)
    tgt = torch.randn(1, NUM_QUERIES, HIDDEN_DIM)

    # PyTorch reference
    layer = torch_model.transformer.decoder.layers[0]
    with torch.no_grad():
        tgt2 = layer.linear2(torch.relu(layer.linear1(tgt)))
        ref_output = layer.norm3(tgt + tgt2)

    # TTNN
    tgt_tt = ttnn.from_torch(tgt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = decoder_ffn(tgt_tt, decoder_params["layers"][0])
    tt_output_torch = ttnn.to_torch(tt_output).float()

    assert_with_pcc(ref_output, tt_output_torch, pcc=0.99)
