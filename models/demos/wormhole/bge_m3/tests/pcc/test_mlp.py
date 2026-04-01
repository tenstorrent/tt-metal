# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.wormhole.bge_m3.reference.hf_reference import PositionwiseFeedForward
from models.demos.wormhole.bge_m3.tests.test_utils import (
    SEQUENCE_LENGTHS,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)
from models.demos.wormhole.bge_m3.tt.mlp import BgeM3MLP

HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 4096
BATCH_SIZE = 1


@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS, ids=[f"S{seq_len}" for seq_len in SEQUENCE_LENGTHS])
def test_mlp_vs_pytorch(device, seq_len):
    require_single_device(device)
    torch.manual_seed(42)

    reference_layer = PositionwiseFeedForward(
        hidden_size=HIDDEN_SIZE,
        mlp_size=INTERMEDIATE_SIZE,
        drop_prob=0.0,
    ).eval()
    with torch.no_grad():
        reference_layer.proj1.weight.copy_(torch.randn_like(reference_layer.proj1.weight) * 0.02)
        reference_layer.proj1.bias.copy_(torch.randn_like(reference_layer.proj1.bias) * 0.01)
        reference_layer.proj2.weight.copy_(torch.randn_like(reference_layer.proj2.weight) * 0.02)
        reference_layer.proj2.bias.copy_(torch.randn_like(reference_layer.proj2.bias) * 0.01)

    x = torch.randn((BATCH_SIZE, 1, seq_len, HIDDEN_SIZE), dtype=torch.float32)

    tt_model = BgeM3MLP(
        wi_weight=make_lazy_weight(
            reference_layer.proj1.weight.detach().clone().transpose(-1, -2).contiguous(),
            device,
            layout=ttnn.TILE_LAYOUT,
        ),
        wo_weight=make_lazy_weight(
            reference_layer.proj2.weight.detach().clone().transpose(-1, -2).contiguous(),
            device,
            layout=ttnn.TILE_LAYOUT,
        ),
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        wi_bias=make_lazy_weight(
            reference_layer.proj1.bias.detach().clone().reshape(1, -1).contiguous(),
            device,
            layout=ttnn.TILE_LAYOUT,
        ),
        wo_bias=make_lazy_weight(
            reference_layer.proj2.bias.detach().clone().reshape(1, -1).contiguous(),
            device,
            layout=ttnn.TILE_LAYOUT,
        ),
        activation="gelu",
    )

    tt_output = tt_model.forward(to_ttnn_tensor(x, device))
    tt_output_torch = to_torch(tt_output, expected_shape=(BATCH_SIZE, 1, seq_len, HIDDEN_SIZE))

    reference_output = reference_layer(x.squeeze(1)).unsqueeze(1).to(torch.float32)
    assert_pcc(reference_output, tt_output_torch, 0.999)
