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
    distributions,
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


def test_distributions_hand_values() -> None:
    # 4 tokens with golden norms 1, 1, 2, 2 and error norms 0.1, 0.2, 0.2, 0.8
    t = np.array([[0.01, 1, 1, 1], [0.04, 1, 1, 1], [0.04, 4, 4, 4], [0.64, 4, 4, 4]], dtype=np.float64)
    s = np.zeros((4, HEADS, 4))
    s[..., 1] = 1.0
    s[..., 0] = np.array([0.1, 0.2, 0.4, 0.8])[:, None] ** 2  # every head doubles its error each snapshot
    s[3, 6, 0] = 0.9**2  # head 6 is the worst at the end
    s[1, 5, 0] = 0.95**2  # head 5 peaks early, then recovers
    d = distributions(t, s, [100_000, 200_000, 300_000, 400_000], floor=1e-9, windows=(1, 2))
    own = d["overall"]["token"]["own_norm"]  # 0.1, 0.2, 0.1, 0.4
    assert own["max"] == pytest.approx(0.4) and own["argmax"] == 3
    assert own["p50"] == pytest.approx(0.15)
    assert d["overall"]["token"]["span_rms"]["max"] == pytest.approx(0.8 / np.sqrt(2.5))  # RMS norm √2.5
    pairs = d["token_windows"]["2"]
    assert pairs["end"] == [2, 4] and pairs["own_norm"]["max"] == pytest.approx([0.2, 0.4])
    assert pairs["own_norm"]["argmax"] == [1, 3]
    assert pairs["span_rms"]["max"] == pytest.approx([0.2, 0.8 / np.sqrt(4)])  # window RMS norms 1 and 2
    assert d["state_over_position"]["p50"] == pytest.approx([0.1, 0.2, 0.4, 0.8])
    assert d["overall"]["state_final"]["worst3"][0] == [6, pytest.approx(0.9)]
    assert d["overall"]["state_worst_snapshot"]["worst3"][0] == [5, pytest.approx(0.95)]
    assert d["drift_per_100k_tokens"]["state_p50"]["whole"] == pytest.approx(2.0)
    assert d["drift_per_100k_tokens"]["state_p50"]["second_half"] == pytest.approx(2.0)
