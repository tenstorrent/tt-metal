# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.wormhole.bge_m3.tests.test_utils import (
    SEQUENCE_LENGTHS,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)
from models.demos.wormhole.bge_m3.tt.norm import LayerNorm1D

HIDDEN_SIZE = 1024
BATCH_SIZE = 1
EPS = 1e-5


@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS, ids=[f"S{seq_len}" for seq_len in SEQUENCE_LENGTHS])
def test_layernorm_vs_pytorch(device, seq_len):
    require_single_device(device)
    torch.manual_seed(42)

    reference_layer = torch.nn.LayerNorm(HIDDEN_SIZE, eps=EPS)
    with torch.no_grad():
        reference_layer.weight.copy_(torch.randn_like(reference_layer.weight))
        reference_layer.bias.copy_(torch.randn_like(reference_layer.bias))

    x = torch.randn((BATCH_SIZE, 1, seq_len, HIDDEN_SIZE), dtype=torch.float32)

    tt_model = LayerNorm1D(
        weight=make_lazy_weight(
            reference_layer.weight.detach().clone(),
            device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        bias=make_lazy_weight(
            reference_layer.bias.detach().clone(),
            device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        eps=EPS,
    )

    tt_output = tt_model.forward(to_ttnn_tensor(x, device))
    tt_output_torch = to_torch(tt_output, expected_shape=(BATCH_SIZE, 1, seq_len, HIDDEN_SIZE))

    reference_output = reference_layer(x.squeeze(1)).unsqueeze(1).to(torch.float32)
    assert_pcc(reference_output, tt_output_torch, 0.999)
