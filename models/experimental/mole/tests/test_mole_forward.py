# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.mole.reference.config import MoLEConfig
from models.experimental.mole.reference.mole import MixtureOfLinearExperts
from models.experimental.mole.tt.mole import TtMoLE
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "base_model_type,batch_size,seq_len,pred_len,input_dim,num_experts,expected_pcc,expected_router_pcc,individual",
    [
        ("dlinear", 2, 96, 24, 7, 4, 0.99, 0.99, False),
        ("dlinear", 2, 96, 24, 7, 8, 0.99, 0.99, False),
        ("dlinear", 2, 96, 24, 7, 16, 0.99, 0.99, False),
        ("rlinear", 2, 96, 24, 7, 4, 0.99, 0.99, False),
        ("rlinear", 2, 96, 24, 7, 8, 0.99, 0.99, False),
        ("rlinear", 2, 96, 24, 7, 16, 0.99, 0.99, False),
        ("rlinear", 2, 96, 24, 7, 4, 0.99, 0.99, True),
        ("rmlp", 2, 96, 24, 7, 4, 0.99, 0.99, False),
        ("rmlp", 2, 96, 24, 7, 8, 0.99, 0.99, False),
        ("rmlp", 2, 96, 24, 7, 16, 0.99, 0.99, False),
        ("rmlp", 2, 96, 24, 7, 4, 0.99, 0.99, True),
    ],
)
def test_mole_forward(
    device,
    base_model_type,
    batch_size,
    seq_len,
    pred_len,
    input_dim,
    num_experts,
    expected_pcc,
    expected_router_pcc,
    individual,
):
    torch.manual_seed(0)

    config = MoLEConfig(
        seq_len=seq_len,
        pred_len=pred_len,
        input_dim=input_dim,
        base_model_type=base_model_type,
        num_experts=num_experts,
        individual=individual,
    )
    reference_model = MixtureOfLinearExperts(config).eval()

    torch_input = torch.randn(batch_size, seq_len, input_dim)
    torch_input_mark = torch.randn(batch_size, seq_len, 4)
    torch_output, torch_router_weights = reference_model(torch_input, torch_input_mark)

    tt_model = TtMoLE(config, reference_model=reference_model, device=device)
    tt_output, tt_router_output = tt_model.forward_from_torch_input(
        torch_input, input_marks=torch_input_mark, device=device
    )
    tt_output = ttnn.to_torch(tt_output).squeeze(0)
    tt_router_output = ttnn.to_torch(tt_router_output).squeeze(0)

    assert_with_pcc(torch_router_weights, tt_router_output, pcc=expected_router_pcc)
    assert_with_pcc(torch_output, tt_output, pcc=expected_pcc)
