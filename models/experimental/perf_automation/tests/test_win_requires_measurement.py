# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A ✓win must mean "measured faster and committed", never merely "a commit happened".

_record_committed_win set beat_baseline=True on EVERY successful git_commit, with measured_ms taken
from the target -- often None. The agent uses git_commit for housekeeping too, so
"refresh the generated RUN_REPORT", "checkpoint the perf test" and a comment-only
"record the measured dead ends" all rendered as wins. On llama3_1_8b_p150 that was 47 of 73 wins in
one run, and it put a ✓ in the fidelity column while both real fidelity measurements showed no gain.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _pm(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(tmp_path / "m.json"))
    (tmp_path / "m.json").write_text('{"config": {}, "perf_test_resolved": {"path": "t.py"}}')
    monkeypatch.setenv("PERF_MCP_KERNEL_LOG", str(tmp_path / "kl.json"))
    spec = importlib.util.spec_from_file_location("pm_win_ut", _ROOT / "cc_optimize" / "perf_mcp.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["pm_win_ut"] = m
    spec.loader.exec_module(m)
    return m


def _summary():
    spec = importlib.util.spec_from_file_location("sm_win_ut", _ROOT / "cc_optimize" / "summary.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["sm_win_ut"] = m
    spec.loader.exec_module(m)
    return m


def test_an_unmeasured_commit_is_not_recorded_as_a_win(tmp_path, monkeypatch):
    """THE REGRESSION: a comment-only commit became a ✓win."""
    pm = _pm(tmp_path, monkeypatch)
    monkeypatch.setattr(pm, "_load_target", lambda: {"op": "MatmulDeviceOperation", "rung": "knob:fidelity"})
    pm._record_committed_win("record the measured dead ends  Comment-only.")
    rows = json.loads((tmp_path / "kl.json").read_text()) if (tmp_path / "kl.json").exists() else []
    assert not [r for r in rows if r.get("beat_baseline")], rows


def test_a_measured_commit_is_still_recorded_as_a_win(tmp_path, monkeypatch):
    """The original intent must survive: a genuinely banked lever still gets its ✓."""
    pm = _pm(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pm, "_load_target", lambda: {"op": "MatmulDeviceOperation", "rung": "knob:dtype", "measured_ms": 648.17}
    )
    pm._record_committed_win("put the LM head weight on bf4_b")
    rows = json.loads((tmp_path / "kl.json").read_text())
    wins = [r for r in rows if r.get("beat_baseline")]
    assert len(wins) == 1 and wins[0]["measured_ms"] == 648.17


def test_the_renderer_refuses_an_unmeasured_win_from_an_old_log(tmp_path):
    """Logs already on disk carry these rows, so the renderer must refuse them too -- otherwise every
    previously written kernel log keeps producing inflated ✓ marks."""
    sm = _summary()
    kl = tmp_path / "kl.json"
    kl.write_text(
        json.dumps(
            [
                {"op_signature": "Matmul", "kernel_kind": "fidelity", "measured_ms": None, "beat_baseline": True},
                {"op_signature": "Matmul", "kernel_kind": "fidelity", "measured_ms": 664.13, "beat_baseline": False},
            ]
        )
    )
    out = sm.render_summary(kl, model="m", task="main", finalized=True)
    row = next(l for l in out.splitlines() if l.lstrip().startswith("Matmul"))
    assert "✓win" not in row, row
    assert "·try" in row, row


def test_a_measured_win_still_renders_a_tick(tmp_path):
    sm = _summary()
    kl = tmp_path / "kl.json"
    kl.write_text(
        json.dumps([{"op_signature": "Matmul", "kernel_kind": "dtype", "measured_ms": 648.17, "beat_baseline": True}])
    )
    out = sm.render_summary(kl, model="m", task="main", finalized=True)
    row = next(l for l in out.splitlines() if l.lstrip().startswith("Matmul"))
    assert "✓win" in row, row


def test_a_committed_win_updates_the_ledger(tmp_path, monkeypatch):
    """The ledger saw only profile_model readings, so it lagged the real state: llama3_1_8b_p150's
    headline rendered "-> 664.17 ms" (the last full profile) while the run had already committed
    654.43 and then 615.69. A commit is when the current state changes, so it is when the ledger
    must learn the number."""
    import importlib.util as ilu

    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "led.jsonl"))
    monkeypatch.setenv("TT_PERF_LAYERS", "16")
    pm = _pm(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pm, "_load_target", lambda: {"op": "MatmulDeviceOperation", "rung": "knob:grid", "measured_ms": 615.69}
    )
    pm._record_committed_win("run ff1/ff3 on the full core grid")

    spec = ilu.spec_from_file_location("meas_lag_ut", _ROOT / "cc_optimize" / "measurements.py")
    led = ilu.module_from_spec(spec)
    spec.loader.exec_module(led)
    row = led.last(led.KIND_EAGER, led.PHASE_AFTER)
    assert row and row["value_ms"] == 615.69, row
    assert row["source"] == "git_commit" and row["depth"] == "16"


def test_an_unmeasured_commit_does_not_touch_the_ledger(tmp_path, monkeypatch):
    """The same guard as the win mark: no measurement, no ledger row."""

    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "led.jsonl"))
    pm = _pm(tmp_path, monkeypatch)
    monkeypatch.setattr(pm, "_load_target", lambda: {"op": "Matmul", "rung": "knob:fidelity"})
    pm._record_committed_win("record the measured dead ends  Comment-only.")
    assert not (tmp_path / "led.jsonl").exists()


def test_the_original_before_survives_committed_wins(tmp_path, monkeypatch):
    """Committed wins append as 'after'; the seeded original must stay the anchor."""
    import importlib.util as ilu

    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "led.jsonl"))
    monkeypatch.setenv("TT_PERF_LAYERS", "16")
    spec = ilu.spec_from_file_location("meas_lag2_ut", _ROOT / "cc_optimize" / "measurements.py")
    led = ilu.module_from_spec(spec)
    spec.loader.exec_module(led)
    led.record(led.KIND_EAGER, led.PHASE_BEFORE, 2464.18, depth="16", mode="eager", source="seed")

    pm = _pm(tmp_path, monkeypatch)
    for ms in (654.43, 615.69):
        monkeypatch.setattr(pm, "_load_target", lambda ms=ms: {"op": "M", "rung": "knob:grid", "measured_ms": ms})
        pm._record_committed_win("win")
    assert led.first(led.KIND_EAGER, led.PHASE_BEFORE)["value_ms"] == 2464.18
    assert led.last(led.KIND_EAGER, led.PHASE_AFTER)["value_ms"] == 615.69


def test_both_sections_agree_on_what_a_win_is(tmp_path):
    """THE DEFECT: the ladder matrix and the attempts table each decided 'is this a win' for
    themselves, and only the matrix got the measurement check -- so one unmeasured commit rendered
    '·try' in the matrix and '✓ win' in the attempts table of the SAME report.
    """
    sm = _summary()
    kl = tmp_path / "kl.json"
    kl.write_text(
        json.dumps(
            [
                {
                    "op_signature": "MatmulDeviceOperation",
                    "kernel_kind": "dtype",
                    "measured_ms": None,
                    "beat_baseline": True,
                    "note": "committed: write the prefill MLP intermediates as bf8_b",
                }
            ]
        )
    )
    out = sm.render_summary(kl, model="m", task="main", finalized=True)
    # The legend explains what ✓win MEANS, so it is not a claim about an attempt. Everything from
    # its heading on is legend, which is what this drops -- it used to drop the single line starting
    # `levels:`, and silently stopped covering the marks when the legend was wrapped onto a second
    # line, so the explanation of ✓win read as an attempt claiming one.
    _legend = next((i for i, l in enumerate(out.splitlines()) if l.strip() == "Legend"), None)
    body = out.splitlines()[:_legend]
    assert _legend is not None, out
    assert not [l for l in body if "✓win" in l or "✓ win" in l], "\n".join(body)
    # the attempts table no longer carries the note text, so find the row by its op and verdict
    attempt_rows = [l for l in body if l.lstrip().startswith("Matmul") and "· no gain" in l]
    assert attempt_rows, out


def test_is_win_is_the_only_definition(tmp_path):
    """A zero or negative measured ms is not a speedup either."""
    sm = _summary()
    assert sm._is_win({"beat_baseline": True, "measured_ms": 648.17})
    for bad in (None, 0, -1.0, "648", float("nan")):
        assert not sm._is_win({"beat_baseline": True, "measured_ms": bad}), bad
    assert not sm._is_win({"beat_baseline": False, "measured_ms": 648.17})
    assert not sm._is_win({})
    assert not sm._is_win(None)


# --- a win must have REDUCED the time, not merely been committed and timed --------------------------


def _led():
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location(
        "led_wins_ut", _P(__file__).resolve().parents[1] / "cc_optimize" / "measurements.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _a(ms, flag=True):
    return {"op_signature": "Matmul", "kernel_kind": "grid", "measured_ms": ms, "beat_baseline": flag}


def test_only_new_bests_count_as_wins():
    """THE DEFECT: every committed+timed attempt got a tick, so a run whose end-to-end time moved 4
    times showed 16 wins. A win must be strictly faster than the baseline AND than every win before
    it -- the staircase the run actually walked down."""
    led = _led()
    att = [_a(654.43), _a(700.0), _a(615.69), _a(620.0), _a(567.94), _a(534.44), _a(540.0)]
    assert led.winning_indices(att, 2464.18) == {0, 2, 4, 5}


def test_an_attempt_that_did_not_improve_is_not_a_win_however_it_was_flagged():
    led = _led()
    att = [_a(600.0), _a(600.0), _a(601.0)]
    assert led.winning_indices(att, 2464.18) == {0}


def test_nothing_beating_the_baseline_yields_no_wins():
    led = _led()
    assert led.winning_indices([_a(3000.0), _a(2600.0)], 2464.18) == set()


def test_unmeasured_commits_cannot_enter_the_staircase():
    led = _led()
    att = [_a(None), _a(0), _a(600.0), _a(None)]
    assert led.winning_indices(att, 2464.18) == {2}


def test_unflagged_measurements_do_not_lower_the_bar():
    """A faster reading that was NOT kept must not raise the bar for later real wins -- otherwise a
    reverted experiment silently disqualifies the win that follows it."""
    led = _led()
    att = [_a(400.0, flag=False), _a(600.0)]
    assert led.winning_indices(att, 2464.18) == {1}


def test_order_is_respected():
    led = _led()
    assert led.winning_indices([_a(500.0), _a(600.0)], 2464.18) == {0}
    assert led.winning_indices([_a(600.0), _a(500.0)], 2464.18) == {0, 1}


def test_without_a_baseline_the_first_timed_commit_starts_the_staircase():
    led = _led()
    assert led.winning_indices([_a(600.0), _a(700.0), _a(550.0)], None) == {0, 2}


def test_junk_rows_and_junk_baselines_never_raise():
    led = _led()
    for bad_base in (None, 0, -1, "x", float("nan"), float("inf")):
        assert isinstance(led.winning_indices([_a(600.0)], bad_base), set), bad_base
    assert led.winning_indices([None, "x", 5, _a(600.0)], 2464.18) == {3}
    assert led.winning_indices(None, 100.0) == set()


def test_every_report_section_reads_the_same_win_set(tmp_path):
    """The matrix, the per-attempt table and the limitations section must agree: one of them judging
    rows for itself is how a ✓ appeared in one section and 'no gain' in another.

    The code-changes list used to be a fourth reader of the same set; it was removed from the report
    (source diffs belong in the kernel log and in git), so there are three."""
    sm = _summary()
    kl = tmp_path / "kl.json"
    kl.write_text(
        json.dumps(
            [
                {
                    "op_signature": "Matmul A",
                    "kernel_kind": "grid",
                    "measured_ms": 600.0,
                    "beat_baseline": True,
                    "note": "real improvement",
                    "diff": "--- a\n+++ b\n+x",
                },
                {
                    "op_signature": "Matmul A",
                    "kernel_kind": "dtype",
                    "measured_ms": 650.0,
                    "beat_baseline": True,
                    "note": "committed but slower",
                    "diff": "--- a\n+++ b\n+y",
                },
            ]
        )
    )
    out = sm.render_summary(kl, model="m", task="main", finalized=True)
    body = out.splitlines()
    matrix = [l for l in body if l.lstrip().startswith("Matmul A") and ("✓win" in l or "·try" in l)]
    attempts_rows = [l for l in body if l.lstrip().startswith("Matmul A") and ("✓ win" in l or "· no gain" in l)]
    # the matrix marks the winning lever and only that lever
    assert len(matrix) == 1 and "✓win" in matrix[0] and "·try" in matrix[0], matrix
    # the per-attempt table: the faster attempt won, the slower committed one did not
    assert [("✓ win" in r) for r in attempts_rows] == [True, False], attempts_rows
    # and no section prints source diffs any more
    assert not [l for l in body if l.startswith("[#")], body
