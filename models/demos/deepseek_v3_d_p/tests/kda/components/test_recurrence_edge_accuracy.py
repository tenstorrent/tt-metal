# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure the device KDA recurrence on tier-C edge inputs against the FP64 golden.

Each edge case (``tests/kda/edge_inputs.py``) is cast to the device formats (BF16 q/k/v/gate, FP32 beta and
state) and the golden is the FP64 recurrence of exactly those cast values, so the reported error is the
device's alone. The result is a ``KDA_RECURRENCE_EDGE_ACCURACY=`` JSON line per case; the test fails only on
non-finite results, since budgets are calibrated separately. Golden cost is about a second of CPU per case.
"""

import json

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_recurrent_reference
from models.demos.deepseek_v3_d_p.tests.kda.accuracy_metric import slice_nrmse
from models.demos.deepseek_v3_d_p.tests.kda.components.test_recurrence import _run_recurrence, _to_device
from models.demos.deepseek_v3_d_p.tests.kda.edge_inputs import EDGE_CASES, recurrence_edge_case

pytestmark = run_for_blackhole()

HEADS, KEY_DIM, VALUE_DIM = 2, 128, 128


@pytest.mark.parametrize("sequence,summary_group_chunks", [(256, 2), (2048, 8)], ids=["T256", "T2048"])
@pytest.mark.parametrize("name", EDGE_CASES)
@pytest.mark.use_module_device
def test_recurrence_edge_case_against_fp64_golden(
    device: ttnn.Device, name: str, sequence: int, summary_group_chunks: int
) -> None:
    case = recurrence_edge_case(name, sequence, heads=HEADS, key_dim=KEY_DIM, value_dim=VALUE_DIM, seed=sequence)
    # The device receives BF16 q/k/v/gate and FP32 beta/state; the golden starts from exactly those values.
    device_inputs = {key: case[key].bfloat16().float() for key in ("q", "k", "v", "gate")}
    device_inputs.update(beta=case["beta"].float(), initial_state=case["initial_state"].float())
    golden_output, golden_state = kda_recurrent_reference(
        *(device_inputs[key] for key in ("q", "k", "v", "gate", "beta", "initial_state")), dtype=torch.float64
    )

    def flat(key: str) -> torch.Tensor:
        return device_inputs[key].reshape(1, sequence, -1)

    with ttnn.manage_config("throw_exception_on_fallback", True):
        state_tt, output_tt = _run_recurrence(
            device,
            _to_device(flat("q"), device, ttnn.bfloat16),
            _to_device(flat("k"), device, ttnn.bfloat16),
            _to_device(flat("v"), device, ttnn.bfloat16),
            _to_device(flat("gate"), device, ttnn.bfloat16),
            _to_device(device_inputs["beta"], device, ttnn.float32),
            _to_device(device_inputs["initial_state"], device, ttnn.float32),
            summary_group_chunks=summary_group_chunks,
            local_scan_strategy="grouped",
        )
    output = ttnn.to_torch(output_tt).reshape(1, HEADS, sequence, VALUE_DIM).permute(0, 2, 1, 3)
    state = ttnn.to_torch(state_tt)
    assert torch.isfinite(output).all() and torch.isfinite(state).all(), f"{name}: non-finite device result"

    result = {"case": name, "sequence": sequence, "heads": HEADS, "key_dim": KEY_DIM, "value_dim": VALUE_DIM}
    if name == "zero_values":
        # The exact output and state are zero; only absolute error is meaningful.
        result.update(output_abs_max=float(output.abs().max()), state_abs_max=float(state.abs().max()))
    else:
        output_error = slice_nrmse(output, golden_output, keep_dims=(2,))  # per head
        token_error = slice_nrmse(output, golden_output, keep_dims=(1, 2))  # per token and head
        state_error = slice_nrmse(state, golden_state, keep_dims=(1,))  # per head
        result.update(
            output_nrmse_per_head=[float(x) for x in output_error],
            output_worst_token_nrmse=float(token_error.max()),
            state_nrmse_per_head=[float(x) for x in state_error],
        )
    print("KDA_RECURRENCE_EDGE_ACCURACY=" + json.dumps(result, sort_keys=True))
