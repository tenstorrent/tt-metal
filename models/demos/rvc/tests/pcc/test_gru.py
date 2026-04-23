# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.tt_impl.gru import GRU as TTGRU
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_gru(device):
    torch.manual_seed(0)

    batch_size = 1
    sequence_length = 64
    input_size = 32
    hidden_size = 16
    num_layers = 1
    bidirectional = True

    torch_gru = torch.nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=True,
        bidirectional=bidirectional,
    ).eval()

    torch_input = torch.randn(batch_size, sequence_length, input_size, dtype=torch.float32)
    with torch.no_grad():
        torch_output, torch_hidden = torch_gru(torch_input)

    tt_gru = TTGRU(
        device=device,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=True,
        bidirectional=bidirectional,
        dtype=ttnn.bfloat16,
    )

    state_dict = {f"proj.gru.{k}": v for k, v in torch_gru.state_dict().items()}
    tt_gru.load_state_dict(state_dict=state_dict, key="gru", module_prefix="proj.")

    tt_input = ttnn.from_torch(
        torch_input.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    tt_output, tt_hidden = tt_gru(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32)
    tt_hidden_torch = ttnn.to_torch(tt_hidden).to(torch.float32)

    assert tuple(tt_output_torch.shape) == tuple(torch_output.shape)
    assert tuple(tt_hidden_torch.shape) == tuple(torch_hidden.shape)
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.99)
    assert_with_pcc(torch_hidden, tt_hidden_torch, pcc=0.99)
