"""The watchdog heartbeat says what was actually tried, not just how long it has been running.

A run going for hours printed "· optimizing… Ns (agent transcript → <same path every time>)" once a
minute, forever -- hundreds of near-identical lines in the log, none of which said anything about
what the round was actually doing. The path never changes within a round, so it is now said once;
the repeating line now describes the most recent attempt (op, stack, lever, outcome) instead of a
bare timer, and prints less often (every 5 min, not every 1) since a real attempt takes minutes
anyway.

Model-agnostic by construction: the stack comes from stage_of_op against the model's own baseline
profile (the same call the report itself uses), never a name typed here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_run():
    spec = importlib.util.spec_from_file_location("ccrun_hb", _ROOT / "cc_optimize" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def run_mod():
    return _load_run()


def _row(op="MatmulDeviceOperation 32 x 2688 x 131072", kind="knob:shard", **kw):
    r = {"op_signature": op, "kernel_kind": kind}
    r.update(kw)
    return r


# ---------------------------------------------------------------- _fmt_elapsed


def test_seconds_stay_seconds_under_a_minute(run_mod):
    assert run_mod._fmt_elapsed(45) == "45s"


def test_minutes_drop_the_seconds(run_mod):
    assert run_mod._fmt_elapsed(300) == "5m"


def test_hours_and_minutes_both_show(run_mod):
    assert run_mod._fmt_elapsed(3661) == "1h1m"


def test_an_exact_hour_drops_the_zero_minutes(run_mod):
    assert run_mod._fmt_elapsed(7200) == "2h"


# ---------------------------------------------------------------- _last_attempt_summary


def test_a_missing_log_says_nothing(run_mod, tmp_path):
    assert run_mod._last_attempt_summary(str(tmp_path / "does_not_exist.json")) == ""


def test_an_empty_log_says_nothing(run_mod, tmp_path):
    p = tmp_path / "kl.json"
    p.write_text("[]")
    assert run_mod._last_attempt_summary(str(p)) == ""


def test_a_saved_win_is_labelled_as_such(run_mod, tmp_path, monkeypatch):
    p = tmp_path / "kl.json"
    p.write_text(json.dumps([_row(commit_record=True)]))
    monkeypatch.setattr(run_mod, "_perf_mcp", lambda: None)  # stage lookup unavailable -> "?" fallback
    out = run_mod._last_attempt_summary(str(p))
    assert "✓ win, saved" in out
    assert "shard" in out  # the "knob:" prefix must be stripped, same as everywhere else in the ladder


def test_a_wedge_is_labelled_as_such(run_mod, tmp_path, monkeypatch):
    p = tmp_path / "kl.json"
    p.write_text(json.dumps([_row(wedged=True)]))
    monkeypatch.setattr(run_mod, "_perf_mcp", lambda: None)
    assert "✗ wedged" in run_mod._last_attempt_summary(str(p))


def test_a_diverged_reading_is_labelled_as_uncounted(run_mod, tmp_path, monkeypatch):
    p = tmp_path / "kl.json"
    p.write_text(json.dumps([_row(diverged=True)]))
    monkeypatch.setattr(run_mod, "_perf_mcp", lambda: None)
    out = run_mod._last_attempt_summary(str(p))
    assert "diverged" in out and "uncounted" in out


def test_a_plain_attempt_is_no_gain(run_mod, tmp_path, monkeypatch):
    p = tmp_path / "kl.json"
    p.write_text(json.dumps([_row()]))
    monkeypatch.setattr(run_mod, "_perf_mcp", lambda: None)
    assert "no gain" in run_mod._last_attempt_summary(str(p))


def test_it_describes_the_last_row_not_the_first(run_mod, tmp_path, monkeypatch):
    p = tmp_path / "kl.json"
    p.write_text(json.dumps([_row(op="LayerNormDeviceOperation", kind="grid"), _row(commit_record=True)]))
    monkeypatch.setattr(run_mod, "_perf_mcp", lambda: None)
    out = run_mod._last_attempt_summary(str(p))
    assert "MatmulDeviceOperation" in out and "LayerNormDeviceOperation" not in out


def test_the_stack_comes_from_stage_of_op_not_a_typed_name(run_mod, tmp_path, monkeypatch):
    """Model-agnostic: the stack label is whatever stage_of_op resolves, never a literal this file
    knows about -- swap in a fake resolver naming a stage nobody standardises on and it must show up
    verbatim."""
    p = tmp_path / "kl.json"
    p.write_text(json.dumps([_row()]))

    class _FakeMCP:
        def stage_of_op(self, op, prof):
            return "a_stage_this_model_made_up"

        def _read_baseline_profile(self):
            return {}

    monkeypatch.setattr(run_mod, "_perf_mcp", lambda: _FakeMCP())
    assert "a_stage_this_model_made_up" in run_mod._last_attempt_summary(str(p))


def test_a_broken_stage_lookup_falls_back_to_a_placeholder_not_a_crash(run_mod, tmp_path, monkeypatch):
    p = tmp_path / "kl.json"
    p.write_text(json.dumps([_row()]))

    def _boom():
        raise RuntimeError("no perf_mcp reachable here")

    monkeypatch.setattr(run_mod, "_perf_mcp", _boom)
    out = run_mod._last_attempt_summary(str(p))
    assert out and "?" in out


def test_a_long_op_signature_is_truncated_not_wrapped(run_mod, tmp_path, monkeypatch):
    long_op = "MatmulDeviceOperation " + "x" * 60
    p = tmp_path / "kl.json"
    p.write_text(json.dumps([_row(op=long_op)]))
    monkeypatch.setattr(run_mod, "_perf_mcp", lambda: None)
    out = run_mod._last_attempt_summary(str(p))
    assert "..." in out and len(out.split("[")[0]) <= 48
