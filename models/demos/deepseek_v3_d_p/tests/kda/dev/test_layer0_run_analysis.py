# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the run-sum metrics with hand-verifiable values."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from models.demos.deepseek_v3_d_p.tests.kda.dev.layer0_run_analysis import (
    HEADS,
    RunWriter,
    analyze,
    call_summary,
    error_sums,
    load_run,
)


def test_error_sums_hand_values() -> None:
    golden = torch.tensor([[3.0, 4.0]])  # |g|² = 25
    actual = torch.tensor([[3.0, 3.0]])  # error (0, -1): 1; cross 9 + 12 = 21; |a|² = 18
    assert error_sums(golden, actual).tolist() == [[1.0, 25.0, 21.0, 18.0]]


def test_half_scale_output_has_gain_half_and_nrmse_half(tmp_path) -> None:
    tokens, hidden = 2048, 8
    golden = torch.ones(tokens, hidden)
    writer = RunWriter(tmp_path, tokens, 1, save_state=False)
    state_golden = torch.ones(HEADS, 2, 2)
    state = state_golden.clone()
    state[6] *= 0.9  # head 6 contracts by 10%
    token_sums = error_sums(golden, 0.5 * golden)
    state_sums = error_sums(state_golden, state, dim=(-2, -1))
    writer.write_call(0, tokens, 0, token_sums, state_sums, None)
    writer.flush()
    (tmp_path / "DONE").write_text("")
    result = analyze(load_run(tmp_path), first_call=1024)
    assert result["output"]["nrmse"] == pytest.approx(0.5)
    assert result["output"]["gain"] == pytest.approx(0.5)
    assert result["output_windows"]["nrmse"] == pytest.approx([0.5, 0.5])
    assert result["state"]["heads"]["6"]["nrmse_mean"] == pytest.approx(0.1)
    assert result["state"]["heads"]["6"]["gain_final"] == pytest.approx(0.9)
    assert result["state"]["without_h6_mean"] == pytest.approx(0.0)
    assert result["state"]["median_head_mean"] == pytest.approx(0.0)
    # pooled: error 0.01·4 over golden 4·96
    assert result["state"]["pooled_mean"] == pytest.approx(np.sqrt(0.04 / (4 * HEADS)))
    summary = call_summary(token_sums, state_sums)
    assert summary["h6_gain"] == pytest.approx(0.9) and summary["worst_head"] == 6


def test_interrupted_run_reports_only_finished_calls(tmp_path) -> None:
    writer = RunWriter(tmp_path, 4096, 2, save_state=False)
    golden = torch.ones(2048, 4)
    writer.write_call(
        0, 2048, 0, error_sums(golden, golden), error_sums(torch.ones(HEADS, 1), torch.ones(HEADS, 1), dim=-1), None
    )
    writer.flush()
    run = load_run(tmp_path)
    assert len(run["token_sums"]) == 2048 and len(run["state_sums"]) == 1 and not run["complete"]
    assert json.loads((tmp_path / "state_tokens.json").read_text()) == [2048]
