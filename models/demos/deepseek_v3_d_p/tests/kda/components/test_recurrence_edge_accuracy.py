# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The device KDA recurrence on tier-C edge inputs against the FP64 golden of the same inputs.

Each edge case (``tests/kda/edge_inputs.py``) is cast to the device formats (BF16 q/k/v/gate/beta logits, FP32 state)
and the golden is the FP64 recurrence of exactly those cast values, so the measured error is the device's alone.
Output and final state are gated per head (PCC + NRMSE), so one head's drift is not averaged away; zero values
have an exact zero answer and are checked absolutely. The golden costs about a second of CPU per case.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_recurrent_reference
from models.demos.deepseek_v3_d_p.tests.kda.components.test_recurrence import _run_recurrence, _to_device
from models.demos.deepseek_v3_d_p.tests.kda.edge_inputs import EDGE_CASES, recurrence_edge_case
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate

pytestmark = run_for_blackhole()

HEADS, KEY_DIM, VALUE_DIM = 2, 128, 128

# (case, tokens) -> (output, state) worst-head NRMSE thresholds: 1.5x the values measured on main.
_THRESHOLDS: dict[tuple[str, int], tuple[float, float]] = {
    ("bounded", 256): (0.019, 0.042),  # measured 1.24e-02, 2.80e-02
    ("bounded", 2048): (0.02, 0.051),  # measured 1.28e-02, 3.38e-02
    ("ceiling", 256): (0.019, 0.018),  # measured 1.24e-02, 1.13e-02
    ("ceiling", 2048): (0.023, 0.016),  # measured 1.53e-02, 1.03e-02
    ("ceiling_with_long_memory", 256): (0.017, 0.014),  # measured 1.09e-02, 8.77e-03
    ("ceiling_with_long_memory", 2048): (0.018, 0.015),  # measured 1.19e-02, 9.95e-03
    ("long_memory", 256): (0.019, 0.015),  # measured 1.21e-02, 9.92e-03
    ("long_memory", 2048): (0.023, 0.02),  # measured 1.49e-02, 1.33e-02
    ("repeated_keys", 256): (0.02, 0.043),  # measured 1.32e-02, 2.86e-02
    ("repeated_keys", 2048): (0.02, 0.043),  # measured 1.30e-02, 2.83e-02
}


@pytest.mark.parametrize("sequence,summary_group_chunks", [(256, 2), (2048, 8)], ids=["T256", "T2048"])
@pytest.mark.parametrize("name", EDGE_CASES)
@pytest.mark.use_module_device
def test_recurrence_edge_case_against_fp64_golden(
    device: ttnn.Device, name: str, sequence: int, summary_group_chunks: int
) -> None:
    case = recurrence_edge_case(name, sequence, heads=HEADS, key_dim=KEY_DIM, value_dim=VALUE_DIM, seed=sequence)
    device_inputs = {key: case[key].bfloat16() for key in ("q", "k", "v", "gate", "beta_logits")}
    golden_output, golden_state = kda_recurrent_reference(
        *(device_inputs[key].double() for key in ("q", "k", "v", "gate")),
        torch.sigmoid(device_inputs["beta_logits"].double()),
        case["initial_state"].double(),
        dtype=torch.float64,
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
            _to_device(device_inputs["beta_logits"], device, ttnn.bfloat16),
            _to_device(case["initial_state"].float(), device, ttnn.float32),
            summary_group_chunks=summary_group_chunks,
            local_scan_strategy="grouped",
        )
    output = ttnn.to_torch(output_tt).reshape(1, HEADS, sequence, VALUE_DIM).permute(0, 2, 1, 3).double()
    state = ttnn.to_torch(state_tt).double()

    if name == "zero_values":
        # The exact output and state are zero.
        assert not output.any() and not state.any(), (float(output.abs().max()), float(state.abs().max()))
        return
    output_threshold, state_threshold = _THRESHOLDS[(name, sequence)]
    assert_accurate(
        golden_output,
        output,
        name=f"{name} T{sequence} output",
        nrmse_threshold=output_threshold,
        nrmse_keep_dims=(2,),
    )
    assert_accurate(
        golden_state,
        state,
        name=f"{name} T{sequence} state",
        nrmse_threshold=state_threshold,
        nrmse_keep_dims=(1,),
    )
