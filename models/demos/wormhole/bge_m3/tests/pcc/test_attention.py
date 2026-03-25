# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.wormhole.bge_m3.reference.hf_reference import MultiHeadAttention
from models.demos.wormhole.bge_m3.tests.test_utils import (
    SEQUENCE_LENGTHS,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)
from models.demos.wormhole.bge_m3.tt.attention import BgeM3Attention

HIDDEN_SIZE = 1024
NUM_HEADS = 16
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
BATCH_SIZE = 1


@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS, ids=[f"S{seq_len}" for seq_len in SEQUENCE_LENGTHS])
def test_attention_vs_pytorch(device, seq_len):
    require_single_device(device)
    torch.manual_seed(42)

    reference_layer = MultiHeadAttention(
        hidden_size=HIDDEN_SIZE,
        n_heads=NUM_HEADS,
        drop_prob=0.0,
    ).eval()
    with torch.no_grad():
        reference_layer.qkv_proj.weight.copy_(torch.randn_like(reference_layer.qkv_proj.weight) * 0.02)
        reference_layer.out_proj.weight.copy_(torch.randn_like(reference_layer.out_proj.weight) * 0.02)

    tt_model = BgeM3Attention(
        wqkv=make_lazy_weight(
            reference_layer.qkv_proj.weight.detach().clone().transpose(-1, -2).contiguous(),
            device,
            layout=ttnn.TILE_LAYOUT,
        ),
        wo_weight=make_lazy_weight(
            reference_layer.out_proj.weight.detach().clone().transpose(-1, -2).contiguous(),
            device,
            layout=ttnn.TILE_LAYOUT,
        ),
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        bqkv=None,
        wo_bias=None,
    )
    x = torch.randn((BATCH_SIZE, 1, seq_len, HIDDEN_SIZE), dtype=torch.float32)

    tt_output = tt_model.forward(to_ttnn_tensor(x, device))
    tt_output_torch = to_torch(tt_output, expected_shape=(BATCH_SIZE, 1, seq_len, HIDDEN_SIZE))

    reference_output = reference_layer(x.squeeze(1)).unsqueeze(1).to(torch.float32)
    assert_pcc(reference_output, tt_output_torch, 0.99)
