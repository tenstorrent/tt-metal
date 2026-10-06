# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the D5 per-V-head state gate (design gdn-on-kda §6.3, floor from tt_metal_tracker-g1b.5.18).

D5 divides each head's state RMSE by max(head reference RMS, 1 % x median head reference RMS). Every state here has
K = 1 and V = 4 with constant heads, so each head's RMS and RMSE are the constant itself.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.gdn import GDNReferenceState
from models.demos.deepseek_v3_d_p.tests.gdn.device_utils import (
    HEAD_STATE_REL_RMSE_THRESHOLD,
    chunk_gate_rows,
    per_head_state_errors,
)


def _state(*head_values: float) -> torch.Tensor:
    """``[HV, 1, 4]`` state whose head ``h`` is the constant ``head_values[h]``."""
    return torch.tensor(head_values, dtype=torch.float32).reshape(-1, 1, 1).expand(-1, 1, 4).clone()


def test_floor_applies_to_near_zero_head_only() -> None:
    """Heads RMS 1, 2, 1e-6: median 1, floor 0.01; only the near-zero head is divided by the floor."""
    expected = _state(1.0, 2.0, 1e-6)
    actual = _state(1.01, 2.0, 1e-6 + 5e-4)
    raw, d5 = per_head_state_errors(expected, actual)
    assert raw.tolist() == pytest.approx([0.01, 0.0, 500.0], rel=1e-3)
    # 5e-4 / max(1e-6, 0.01) = 0.05; normal heads keep their own RMS.
    assert d5.tolist() == pytest.approx([0.01, 0.0, 0.05], rel=1e-3)


def test_floor_inactive_above_one_percent_of_median() -> None:
    """A small head above the floor (RMS 0.02 > 0.01) keeps its own RMS: raw == D5 = 1e-3 / 0.02 = 0.05."""
    expected = _state(1.0, 2.0, 0.02)
    actual = _state(1.0, 2.0, 0.021)
    raw, d5 = per_head_state_errors(expected, actual)
    assert torch.equal(raw, d5)
    assert d5.tolist() == pytest.approx([0.0, 0.0, 0.05], rel=1e-3)


def _gate(expected_recurrent: torch.Tensor, actual_by_rank: list[torch.Tensor]) -> tuple[dict, list[str]]:
    """D5 row and failures of one chunk whose output and convolution carry are exact."""
    output = torch.arange(1.0, 9.0).reshape(2, 4)
    state = GDNReferenceState(conv=torch.arange(1.0, 7.0).reshape(2, 3), recurrent=expected_recurrent)
    snapshot = {"output": output.bfloat16()}
    for rank, actual in enumerate(actual_by_rank):
        snapshot[f"recurrent_sp{rank}"] = actual
        snapshot[f"convolution_sp{rank}"] = state.conv.bfloat16()
    rows, failures = chunk_gate_rows(0, snapshot, output, state)
    (recurrent,) = (row for row in rows if row["tensor"] == "recurrent")
    return recurrent, failures


def test_gate_passes_near_zero_head_and_reports_raw() -> None:
    """The g1b.5.18 shape: raw 500 on a ~0 head, D5 0.05 passes; the row records both."""
    row, failures = _gate(_state(1.0, 2.0, 1e-6), [_state(1.0, 2.0, 1e-6 + 5e-4)])
    assert failures == []
    assert row["worst_head"] == 2 and row["worst_raw_head"] == 2
    assert row["worst_head_d5"] == pytest.approx(0.05, rel=1e-3)
    assert row["worst_head_rel_rmse_raw"] == pytest.approx(500.0, rel=1e-3)
    assert row["head_d5"] == pytest.approx([0.0, 0.0, 0.05], rel=1e-3)
    assert row["head_rel_rmse_raw"] == pytest.approx([0.0, 0.0, 500.0], rel=1e-3)


def test_gate_fails_large_absolute_error_on_near_zero_head() -> None:
    """The floor bounds the denominator, not the error: 2e-3 on a ~0 head is D5 0.2 > 0.10."""
    _, failures = _gate(_state(1.0, 2.0, 1e-6), [_state(1.0, 2.0, 1e-6 + 2e-3)])
    assert len(failures) == 1 and "V head 2 state D5 0.2000" in failures[0]


def test_gate_fails_localized_fault_on_normal_head_at_one_rank() -> None:
    """A localized fault on a normal head at one SP rank (head 0, rank 1: 1.0 -> 1.2) fails with D5 = raw = 0.2."""
    expected = _state(1.0, 2.0, 1e-6)
    row, failures = _gate(expected, [expected.clone(), _state(1.2, 2.0, 1e-6)])
    assert row["worst_head"] == 0 and row["worst_head_rank"] == "recurrent_sp1"
    assert row["worst_head_d5"] == pytest.approx(0.2, rel=1e-3)
    assert row["worst_head_d5"] > HEAD_STATE_REL_RMSE_THRESHOLD
    (d5_failure,) = (failure for failure in failures if "D5" in failure)  # recurrent PCC fails here too
    assert "V head 0 state D5 0.2000" in d5_failure and "raw rel RMSE 0.2000" in d5_failure
