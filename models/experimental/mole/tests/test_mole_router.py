# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.mole.reference.config import MoLEConfig
from models.experimental.mole.reference.mole import MixtureOfLinearExperts
from models.experimental.mole.tt.common import time_marks_input_to_device, timeseries_input_to_device
from models.experimental.mole.tt.mole import TtMoLE
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("base_model_type", ["dlinear", "rlinear", "rmlp"])
def test_reference_router_depends_only_on_initial_timestamp(base_model_type):
    torch.manual_seed(0)

    config = MoLEConfig(
        seq_len=96,
        pred_len=24,
        input_dim=7,
        base_model_type=base_model_type,
        num_experts=4,
    )
    model = MixtureOfLinearExperts(config).eval()

    inputs_a = torch.randn(2, config.seq_len, config.input_dim)
    inputs_b = torch.randn(2, config.seq_len, config.input_dim)
    marks_a = torch.randn(2, config.seq_len, 4)
    marks_b = torch.randn(2, config.seq_len, 4)
    marks_b[:, 0, :] = marks_a[:, 0, :]

    outputs_a = model._forward_outputs(inputs_a, marks_a)
    outputs_b = model._forward_outputs(inputs_b, marks_b)

    torch.testing.assert_close(outputs_a.gating_weights, outputs_b.gating_weights, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "base_model_type,expected_router_pcc",
    [("dlinear", 0.998), ("rlinear", 0.996), ("rmlp", 0.995)],
)
def test_tt_router_channelwise_matches_reference(device, base_model_type, expected_router_pcc):
    torch.manual_seed(0)

    config = MoLEConfig(
        seq_len=96,
        pred_len=24,
        input_dim=7,
        base_model_type=base_model_type,
        num_experts=4,
    )
    reference_model = MixtureOfLinearExperts(config).eval()

    inputs = torch.randn(2, config.seq_len, config.input_dim)
    input_marks = torch.randn(2, config.seq_len, 4)
    reference_outputs = reference_model._forward_outputs(inputs, input_marks)

    tt_model = TtMoLE(config, reference_model=reference_model, device=device)
    tt_inputs = timeseries_input_to_device(
        inputs,
        device=device,
        dtype=tt_model.dtype,
        memory_config=tt_model.memory_config,
    )
    tt_marks = time_marks_input_to_device(
        input_marks,
        device=device,
        dtype=tt_model.dtype,
        memory_config=tt_model.memory_config,
    )
    _, tt_gating_weights, _ = tt_model.model._forward_outputs(tt_inputs, tt_marks)
    tt_gating_weights = ttnn.to_torch(tt_gating_weights).squeeze(0)

    assert_with_pcc(reference_outputs.gating_weights, tt_gating_weights, pcc=expected_router_pcc)
