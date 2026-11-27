# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA metric tiers (``kda_tiers``) on hand-computed sums, and ``KdaScorer`` through ``run_case`` with a fake KDA
Block and a tiny golden: tier values, bit equality with the shared NRMSE, per-call stop, fail-closed state PCC. CPU
only."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch

from models.demos.common.prefill.long_context_accuracy.acceptance_bars import AcceptanceBar
from models.demos.common.prefill.long_context_accuracy.block import (
    BlockOutputs,
    ContextBars,
    GoldenIdentity,
    LongContextCase,
    PreparedCase,
)
from models.demos.common.prefill.long_context_accuracy.fp64_golden import (
    ROOT_ENV,
    Fp64Golden,
    product_schema,
    write_manifest,
    write_token_ids,
)
from models.demos.common.prefill.long_context_accuracy.kda.scorer import KdaScorer, kda_tiers
from models.demos.common.prefill.long_context_accuracy.run import run_case
from models.demos.common.prefill.long_context_accuracy.windowed_errors import nrmse, row_sums

HEADS = 96


def _tiers(t: np.ndarray, s: np.ndarray, state_ends: list[int], floor: float) -> dict:
    """``kda_tiers`` of ``[rows, 3]`` and ``[snapshots, heads, 3]`` sums, as one flat dict of metrics."""
    t = torch.from_numpy(t)
    tiers = kda_tiers(t[:, 0], t[:, 1], t[:, 2], s, state_ends, floor)
    assert sorted(tiers) == [0, 1, 2] and all(len(m) == 1 for m in tiers.values())
    assert all(m[0]["start"] == 0 and m[0]["end"] == m[0]["rows"] == len(t) for m in tiers.values())
    return {k: v for m in tiers.values() for k, v in m[0].items()}


def test_kda_tiers_hand_values() -> None:
    # Output: two 1,024-row windows, every row golden norm 1; window 0 errors 0.2, window 1 errors 0.1; cross 0.9.
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
    got = _tiers(t, s, [1024, 2048], floor=1e-9)
    assert got["output_nrmse"] == pytest.approx(np.sqrt(0.025))
    assert got["state_final_nrmse"] == pytest.approx(np.sqrt(0.34 / HEADS))
    # 1,024 x 0.1 then 1,024 x 0.2: the median interpolates between them
    assert (got["output_token_p50"], got["output_token_p95"], got["output_token_p99"]) == pytest.approx(
        (0.15, 0.2, 0.2)
    )
    assert (got["output_token_worst"], got["output_token_worst_at"]) == (pytest.approx(0.2), 0)
    assert (got["output_window_p50"], got["output_window_worst"]) == pytest.approx((0.15, 0.2))
    assert got["output_window_worst_at"] == 0
    # 94 x 0, 0.3, 0.5: p99 sits 5% of the way from 0.3 to 0.5
    head = [got[f"state_head_final_p{p}"] for p in (50, 95, 99)]
    assert head == pytest.approx([0.0, 0.0, 0.31])
    assert (got["state_head_final_worst"], got["state_head_final_worst_at"]) == (pytest.approx(0.5), 6)
    assert (got["state_window_worst"], got["state_window_worst_at"]) == (pytest.approx(0.1), 1024)
    assert got["output_gain"] == pytest.approx(0.9)
    assert got["state_gain_final"] == pytest.approx((95 + 0.8) / HEADS)
    # |1 - gain|: 95 x 0, 0.2; p99 sits 5% of the way to 0.2
    gain = [got[f"state_head_gain_final_p{p}"] for p in (50, 95, 99)]
    assert gain == pytest.approx([0.0, 0.0, 0.01])
    assert (got["state_head_gain_final_worst"], got["state_head_gain_final_worst_at"]) == (pytest.approx(0.2), 6)
    assert got["state_head_gain_final_worst_gain"] == pytest.approx(0.8)
    # 94 x 0, 96 x 0.1, 0.3, 0.5
    assert got["state_head_window_p50"] == pytest.approx(0.1)
    assert got["state_head_window_worst"] == pytest.approx(0.5)
    assert (got["state_head_window_worst_row"], got["state_head_window_worst_head"]) == (2048, 6)


def test_kda_tiers_floor_the_row_norm() -> None:
    t = np.array([[0.01, 1e-6, 0.0], [0.01, 1.0, 0.0]])  # error 0.1 on a near-zero and on a unit golden row
    assert _tiers(t, np.ones((1, HEADS, 3)), [2], floor=0.5)["output_token_worst"] == pytest.approx(0.2)


def test_kda_tiers_refuse_mismatched_sums(expect_error) -> None:
    t = torch.zeros(4)
    with expect_error(ValueError, "snapshots"):
        kda_tiers(t, t, t, np.ones((2, HEADS, 3)), [4], floor=0.0)
    with expect_error(ValueError, "per-row sums"):
        kda_tiers(t, t, torch.zeros(3), np.ones((1, HEADS, 3)), [4], floor=0.0)


# ---------------------------------------------------------------------------------------------- through run_case

ROWS, SEGMENT, CHUNK, HIDDEN, STATE_HEADS = 8, 4, 4, 2, 2
GOLDEN_ID = "tiny-kda-fp64-8"


def _golden_output(lo: int, hi: int) -> torch.Tensor:
    """Row r is ``[r + 1, 2 (r + 1)]``: Σy² = 5 (r + 1)²."""
    r = torch.arange(lo + 1, hi + 1, dtype=torch.float64).unsqueeze(1)
    return torch.cat([r, 2 * r], 1)


def _golden_state(end: int) -> torch.Tensor:
    """After row ``end``, head h's 1x2 recurrent state is ``[end, h + 1]``."""
    return torch.tensor([[[[float(end), h + 1.0]] for h in range(STATE_HEADS)]], dtype=torch.float64)


class TinyKdaGolden(Fp64Golden):
    producer = "tests.tiny_kda"
    producer_version = 1

    def _read_rows(self, names, start, end):
        return {"output": _golden_output(start, end)}

    def _segment_end(self, names, index):
        return {"state.recurrent": _golden_state(self.manifest["segments"][index]["end"])}


@pytest.fixture
def golden_root(tmp_path, monkeypatch):
    directory = tmp_path / GOLDEN_ID
    directory.mkdir()
    write_manifest(
        directory,
        {
            "golden_id": GOLDEN_ID,
            "producer": TinyKdaGolden.producer,
            "producer_version": TinyKdaGolden.producer_version,
            "identity": {"model": "tiny", "token_source": "test", "weights_sha256": "w"},
            "token_ids_sha256": write_token_ids(directory, torch.arange(ROWS)),
            "rows": ROWS,
            "segment": SEGMENT,
            "segments": [{"start": s, "end": s + SEGMENT} for s in range(0, ROWS, SEGMENT)],
            "products": {
                "output": product_schema("row", torch.zeros(1, HIDDEN, dtype=torch.float64)),
                "state.recurrent": product_schema("segment_end", _golden_state(0)),
            },
        },
    )
    monkeypatch.setenv(ROOT_ENV, str(tmp_path))
    return tmp_path


@dataclass
class FakeKdaBlock:
    """Returns its input (the golden output rows) times ``output_scale[lo]``, and the golden state after the call
    with head h times ``state_scale[(hi, h)]``."""

    output_scale: dict[int, float] = field(default_factory=dict)
    state_scale: dict[tuple[int, int], float] = field(default_factory=dict)

    def build(self, mesh, context_len, chunk):
        pass

    def forward(self, x, lo, hi):
        state = _golden_state(hi)[0].clone()
        for h in range(STATE_HEADS):
            state[h] *= self.state_scale.get((hi, h), 1.0)
        return BlockOutputs(x * self.output_scale.get(lo, 1.0), {"recurrent": state})

    def carried_state(self):
        return {}

    def release(self):
        pass


@dataclass
class Mesh:
    shape: tuple = (1, 1)


BARS = ContextBars(
    chunk=(AcceptanceBar("call_pcc", "min", 0.999), AcceptanceBar("call_nrmse", "max", 0.2)),
    run=(
        AcceptanceBar("output_nrmse", "max", 0.1),
        AcceptanceBar("window_nrmse", "max", 0.1),
        AcceptanceBar("state_head_final_worst", "max", 0.2),
        AcceptanceBar("state_head_final_pcc_min", "min", 0.99),
    ),
)


def _run(golden_root, block: FakeKdaBlock):
    def prepare(golden, checkpoint, context_len):
        return PreparedCase(
            identity=GoldenIdentity({"model": "tiny"}),
            scorer=KdaScorer(token_floor=1.0),
            input_tensors=(),
            block_input=lambda lo, hi, rows: rows["output"],
            device_block=block,
        )

    case = LongContextCase(
        id="tiny-kda",
        golden_id=GOLDEN_ID,
        golden_reader=TinyKdaGolden,
        placement=pytest.param((1, 1), {}),
        chunk=CHUNK,
        bars={ROWS: BARS},
        checkpoint=lambda: Path("/nonexistent"),
        prepare=prepare,
    )
    return run_case(case, ROWS, Mesh(), Path("/nonexistent"))


def _metrics(tier: list[dict], name: str) -> list:
    return [m[name] for m in tier if name in m]


def test_kda_scorer_tiers_through_run_case(golden_root) -> None:
    """Call 1 (rows [0, 4)) is 10% large, call 2 exact; after call 2 state head 1 is 10% small (gain 0.9)."""
    result = _run(golden_root, FakeKdaBlock(output_scale={0: 1.1}, state_scale={(8, 1): 0.9}))
    assert result.passed, result.violations
    tiers = result.metric_tiers
    # Σy² rows 0..3: 5 (1 + 4 + 9 + 16) = 150; rows 0..7: 5 * 204 = 1020; error 0.01 * 150 on call 1 only.
    (whole,) = tiers[0]
    assert whole["output_nrmse"] == pytest.approx(math.sqrt(1.5 / 1020))
    # Same bits as the shared NRMSE on the same rows (the CI result keys before the scorer existed).
    golden, device = _golden_output(0, ROWS), torch.cat([1.1 * _golden_output(0, 4), _golden_output(4, 8)])
    assert whole["output_nrmse"] == nrmse(row_sums(golden, device))
    assert whole["output_rms_err"] == pytest.approx(math.sqrt(1.5 / (ROWS * HIDDEN)))
    assert whole["output_rms_ref"] == pytest.approx(math.sqrt(1020 / (ROWS * HIDDEN)))
    # State after row 8: head 0 [8, 1] (Σy² 65) exact, head 1 [8, 2] (Σy² 68) off by 10%.
    assert whole["state_final_nrmse"] == pytest.approx(math.sqrt(0.68 / 133))
    assert _metrics(tiers[1], "call_nrmse") == pytest.approx([0.1, 0.0])
    assert _metrics(tiers[1], "window_nrmse") == pytest.approx([math.sqrt(1.5 / 1020)])  # one partial window
    summary = tiers[1][0]
    assert (summary["state_head_final_worst"], summary["state_head_final_worst_at"]) == (pytest.approx(0.1), 1)
    assert summary["state_head_final_pcc_min"] == pytest.approx(1.0)  # a scaled head still correlates
    assert summary["state_window_worst_at"] == 8
    assert tiers[2][0]["state_head_gain_final_worst_gain"] == pytest.approx(0.9)


def test_kda_scorer_stops_at_a_call_over_its_bar_and_on_non_finite_output_or_state(golden_root) -> None:
    result = _run(golden_root, FakeKdaBlock(output_scale={4: 1.3}))
    assert (result.stopped_at, result.metric_tiers) == (4, {})
    assert result.violations == ["rows [4, 8): call_nrmse 0.3 > 0.2 in [4, 8)"]
    result = _run(golden_root, FakeKdaBlock(output_scale={0: math.nan}))
    assert result.stopped_at == 0
    assert result.violations[0] == "rows [0, 4): output has 8 non-finite values"
    result = _run(golden_root, FakeKdaBlock(state_scale={(4, 1): math.nan}))
    assert (result.stopped_at, result.metric_tiers) == (0, {})
    assert result.violations[0] == "rows [0, 4): recurrent state has 2 non-finite values"


def test_kda_scorer_fails_a_constant_final_state_head(golden_root) -> None:
    """A head read back as zeros has PCC 0 (fail closed), not an undefined PCC left out of the bar."""
    result = _run(golden_root, FakeKdaBlock(state_scale={(8, 0): 0.0}))
    assert result.metric_tiers[1][0]["state_head_final_pcc_min_at"] == 0
    assert "state_head_final_pcc_min 0 < 0.99 in [0, 8)" in result.violations
