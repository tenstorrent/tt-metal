# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the run-sum metrics with hand-verifiable values."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from models.demos.deepseek_v3_d_p.tests.kda.dev.layer0_run_analysis import (
    RunWriter,
    analyze,
    error_sums,
    from_trace,
    golden_token_floor,
    load_run,
    report,
    sums_token_floor,
    tiers,
)
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import MANIFEST, token_ids_sha256

HEADS = 96


def _run(path, tokens: int, snapshots: int) -> RunWriter:
    path.mkdir(exist_ok=True)
    (path / "run.json").write_text(json.dumps({"token_floor": 1e-9, "fabric_config": "FABRIC_2D"}))
    return RunWriter(path, tokens, snapshots, HEADS)


def test_error_sums_hand_values() -> None:
    golden = torch.tensor([[3.0, 4.0]])  # |g|² = 25
    actual = torch.tensor([[3.0, 3.0]])  # error (0, -1): 1; cross 9 + 12 = 21
    assert error_sums(golden, actual).tolist() == [[1.0, 25.0, 21.0]]


def test_half_scale_output_through_the_run_files(tmp_path) -> None:
    tokens, hidden = 2048, 8
    golden = torch.ones(tokens, hidden)
    state_golden = torch.ones(HEADS, 2, 2)
    state = state_golden.clone()
    state[6] *= 0.9  # head 6 contracts by 10%
    _run(tmp_path, tokens, 1).write_call(
        0, tokens, error_sums(golden, 0.5 * golden), error_sums(state_golden, state, dim=(-2, -1))
    )
    (tmp_path / "DONE").write_text("")
    result = analyze(load_run(tmp_path))
    assert (result["tokens"], result["complete"], result["fabric_config"]) == (tokens, True, "FABRIC_2D")
    assert result["tiers"]["tier0"]["output_nrmse"] == pytest.approx(0.5)
    assert result["tiers"]["tier2"]["output_gain"] == pytest.approx(0.5)
    assert result["series"]["window_nrmse"] == pytest.approx([0.5, 0.5])
    assert result["series"]["state_head_nrmse"][0][6] == pytest.approx(0.1)
    assert result["tiers"]["tier2"]["state_head_gain_final"]["worst_gain"] == pytest.approx(0.9)


def test_interrupted_run_reports_only_finished_calls_and_keeps_non_finite_rows(tmp_path) -> None:
    writer = _run(tmp_path, 4096, 2)
    actual = torch.ones(2048, 4)
    actual[-1, 0] = float("nan")  # a broken last token of a finished call must stay in the run
    writer.write_call(0, 2048, error_sums(torch.ones(2048, 4), actual), error_sums(*[torch.ones(HEADS, 1)] * 2))
    run = load_run(tmp_path)
    assert len(run["token_sums"]) == 2048 and len(run["state_sums"]) == 1 and not run["complete"]
    assert np.isnan(run["token_sums"][-1, 0])


def test_tiers_hand_values() -> None:
    # Output: two 1,024-token windows, every token golden norm 1; window 0 errors 0.2, window 1 errors 0.1; cross 0.9.
    t = np.zeros((2048, 3))
    t[:, 1] = 1.0
    t[:, 2] = 0.9
    t[:1024, 0] = 0.2**2
    t[1024:, 0] = 0.1**2
    # State: two snapshots, golden norm 1 per head. Snapshot 0: every head 0.1. Snapshot 1: exact except head 6 (0.5,
    # gain 0.8) and head 7 (0.3).
    s = np.zeros((2, HEADS, 3))
    s[..., 1] = 1.0
    s[..., 2] = 1.0
    s[0, :, 0] = 0.1**2
    s[1, 6] = [0.5**2, 1.0, 0.8]
    s[1, 7, 0] = 0.3**2
    got = tiers(t, s, [1024, 2048], floor=1e-9)
    assert got["tier0"]["output_nrmse"] == pytest.approx(np.sqrt(0.025))
    assert got["tier0"]["state_final_nrmse"] == pytest.approx(np.sqrt(0.34 / HEADS))
    token = got["tier1"]["output_token"]  # 1,024 x 0.1 then 1,024 x 0.2: the median interpolates between them
    assert (token["p50"], token["p95"], token["p99"]) == pytest.approx((0.15, 0.2, 0.2))
    assert (token["worst"], token["worst_at"]) == (pytest.approx(0.2), 0)
    window = got["tier1"]["output_window"]
    assert (window["p50"], window["worst"], window["worst_at"]) == (pytest.approx(0.15), pytest.approx(0.2), 0)
    head = got["tier1"]["state_head_final"]  # 94 x 0, 0.3, 0.5: p99 sits 5% of the way from 0.3 to 0.5
    assert (head["p50"], head["p95"], head["p99"]) == pytest.approx((0.0, 0.0, 0.31))
    assert (head["worst"], head["worst_at"]) == (pytest.approx(0.5), 6)
    call = got["tier1"]["state_window"]
    assert (call["worst"], call["worst_at"]) == (pytest.approx(0.1), 1024)
    assert got["tier2"]["output_gain"] == pytest.approx(0.9)
    assert got["tier2"]["state_gain_final"] == pytest.approx((95 + 0.8) / HEADS)
    gain = got["tier2"]["state_head_gain_final"]  # |1 - gain|: 95 x 0, 0.2; p99 sits 5% of the way to 0.2
    assert (gain["p50"], gain["p95"], gain["p99"]) == pytest.approx((0.0, 0.0, 0.01))
    assert (gain["worst"], gain["worst_at"], gain["worst_gain"]) == (pytest.approx(0.2), 6, pytest.approx(0.8))
    per_head = got["tier2"]["state_head_window"]  # 94 x 0, 96 x 0.1, 0.3, 0.5
    assert per_head["p50"] == pytest.approx(0.1)
    assert (per_head["worst"], per_head["worst_at"]) == (pytest.approx(0.5), [2048, 6])


def test_token_floor_floors_small_golden_tokens() -> None:
    t = np.array([[0.01, 1e-6, 0.0], [0.01, 1.0, 0.0]])  # error 0.1 on a near-zero and on a unit golden token
    assert tiers(t, np.ones((1, HEADS, 3)), [2], floor=0.5)["tier1"]["output_token"]["worst"] == pytest.approx(0.2)


def test_report_has_every_tier_and_well_formed_rows(tmp_path) -> None:
    golden = torch.ones(1024, 4)
    _run(tmp_path, 1024, 1).write_call(
        0, 1024, error_sums(golden, 0.5 * golden), error_sums(*[torch.ones(HEADS, 1)] * 2)
    )
    text = report({"a": analyze(load_run(tmp_path))})
    for line in ("tier 0", "tier 1", "tier 2", "| output NRMSE | 50.000% |", "| tokens | 1,024 |", "| complete | no |"):
        assert line in text
    assert {row.count("|") for row in text.splitlines() if row.startswith("|")} == {3}


def _golden_and_trace(tmp_path, metadata: dict | None = None):
    """A 2-segment golden (4 tokens each, hidden 3, 8 heads of 2x2) and a trace of half its output, state stored V x K."""
    golden, trace = tmp_path / "golden", tmp_path / "trace"
    (trace / "kda").mkdir(parents=True)
    golden.mkdir()
    token_ids = list(range(8))
    output = -torch.arange(24, dtype=torch.float64).reshape(8, 3) / 4
    states = torch.arange(64, dtype=torch.float64).reshape(2, 8, 2, 2) - 5  # [snapshot, head, k, v]
    for i in range(2):
        state = {name: torch.zeros(1) for name in ("q_convolution", "k_convolution", "v_convolution")}
        state["recurrent"] = states[i][None]
        payload = {"start": 4 * i, "end": 4 * i + 4, "output": output[4 * i : 4 * i + 4], "state": state}
        torch.save(payload, golden / f"segment_{i:06d}.pt")
    manifest = {"tokens": 8, "segment": 4, "token_ids_sha256": token_ids_sha256(token_ids)}
    manifest["segments"] = [{"start": 0, "end": 4}, {"start": 4, "end": 8}]
    (golden / MANIFEST).write_text(json.dumps(manifest))
    save_file({"o": (output / 2).bfloat16()}, trace / "kda" / "kda_output_layer_0.safetensors")
    save_file(
        {"s": states.transpose(-1, -2).float().contiguous()}, trace / "kda" / "kda_recurrent_state_layer_0.safetensors"
    )
    meta = {"token_ids": token_ids, "kda_state_every": 4, "row_window": {"enabled": False}, **(metadata or {})}
    (trace / "metadata.json").write_text(json.dumps(meta))
    return golden, trace


def test_from_trace_scores_a_transposed_half_scale_trace(tmp_path) -> None:
    golden, trace = _golden_and_trace(tmp_path)
    from_trace(trace, golden, tmp_path / "run")
    run = load_run(tmp_path / "run")
    assert run["state_tokens"] == [4, 8] and run["complete"]
    assert json.loads((tmp_path / "run" / "run.json").read_text())["state_orientation"] == "transposed"
    result = analyze(run)
    assert result["tiers"]["tier0"]["output_nrmse"] == pytest.approx(0.5)
    assert result["tiers"]["tier0"]["state_final_nrmse"] == 0.0
    assert run["token_floor"] == golden_token_floor(golden) == sums_token_floor(run["token_sums"])


@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"row_window": {"enabled": True}}, "row window"),
        ({"kda_snapshot_grid": {"enabled": True}}, "head/tail"),
        ({"kda_state_every": 2}, "segments"),
        ({"token_ids": list(range(1, 9))}, "golden's tokens"),
    ],
    ids=["row-window", "snapshot-grid", "snapshot-positions", "tokens"],
)
def test_from_trace_refuses_what_it_cannot_score(tmp_path, expect_error, metadata, message) -> None:
    golden, trace = _golden_and_trace(tmp_path, metadata)
    with expect_error(ValueError, message):
        from_trace(trace, golden, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_from_trace_refuses_an_existing_run_dir(tmp_path, expect_error) -> None:
    golden, trace = _golden_and_trace(tmp_path)
    (tmp_path / "run").mkdir()
    with expect_error(FileExistsError, "run"):
        from_trace(trace, golden, tmp_path / "run")
