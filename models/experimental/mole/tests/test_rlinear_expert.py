# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.mole.reference.config import MoLEConfig
from models.experimental.mole.reference.rlinear import RLinearExpert
from models.experimental.mole.tt.rlinear import TtRLinearExpert
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size,seq_len,pred_len,input_dim,expected_pcc",
    [
        (4, 96, 24, 7, 0.995),
    ],
)
def test_rlinear_expert(device, batch_size, seq_len, pred_len, input_dim, expected_pcc):
    torch.manual_seed(0)

    config = MoLEConfig(
        seq_len=seq_len,
        pred_len=pred_len,
        input_dim=input_dim,
        base_model_type="rlinear",
    )
    reference_model = RLinearExpert(config).eval()

    torch_input = torch.randn(batch_size, seq_len, input_dim)
    torch_input_mark = torch.randn(batch_size, seq_len, 4)
    torch_output = reference_model(torch_input, torch_input_mark)

    tt_model = TtRLinearExpert(config, reference_model=reference_model)
    tt_output = tt_model.forward_from_torch_input(torch_input, input_marks=torch_input_mark, device=device)
    tt_output = ttnn.to_torch(tt_output).squeeze(0)

    assert_with_pcc(torch_output, tt_output, pcc=expected_pcc)
