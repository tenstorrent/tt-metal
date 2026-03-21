# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.mole.reference.config import MoLEConfig
from models.experimental.mole.reference.mole import MixtureOfLinearExperts
from models.experimental.mole.tt.mole import TtMoLE
from models.experimental.mole.utils.datasets import create_real_dataset_loaders, resolve_tslib_dataset_path
from tests.ttnn.utils_for_testing import assert_with_pcc


def _load_real_eval_batch(*, dataset_name: str, batch_size: int, seq_len: int, pred_len: int):
    dataset_path = resolve_tslib_dataset_path(dataset_name, auto_download=True)

    loaders, input_dim = create_real_dataset_loaders(
        dataset_name,
        dataset_path,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size,
        eval_batch_size=batch_size,
    )
    inputs, _, input_marks, _ = next(iter(loaders["test"]))
    return inputs, input_marks, input_dim


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "base_model_type,expected_pcc,expected_router_pcc",
    [
        ("dlinear", 0.995, 0.995),
        ("rlinear", 0.995, 0.995),
        ("rmlp", 0.995, 0.995),
    ],
)
def test_mole_e2e_real_dataset_weather(device, base_model_type, expected_pcc, expected_router_pcc):
    torch.manual_seed(0)

    batch_size = 8
    seq_len = 96
    pred_len = 24
    num_experts = 4
    inputs, input_marks, input_dim = _load_real_eval_batch(
        dataset_name="weather",
        batch_size=batch_size,
        seq_len=seq_len,
        pred_len=pred_len,
    )

    config = MoLEConfig(
        seq_len=seq_len,
        pred_len=pred_len,
        input_dim=input_dim,
        base_model_type=base_model_type,
        num_experts=num_experts,
    )
    reference_model = MixtureOfLinearExperts(config).eval()

    torch_output, torch_router_weights = reference_model(inputs, input_marks)

    tt_model = TtMoLE(config, reference_model=reference_model, device=device)
    tt_output, tt_router_output = tt_model.forward_from_torch_input(inputs, input_marks=input_marks, device=device)
    tt_output = ttnn.to_torch(tt_output).squeeze(0)
    tt_router_output = ttnn.to_torch(tt_router_output).squeeze(0).squeeze(0)
    torch_router_weights_averaged = torch_router_weights.mean(dim=2)

    assert_with_pcc(torch_router_weights_averaged, tt_router_output, pcc=expected_router_pcc)
    assert_with_pcc(torch_output, tt_output, pcc=expected_pcc)
