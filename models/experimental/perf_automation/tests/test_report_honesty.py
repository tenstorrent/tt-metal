"""RED tests for report-honesty bugs (BUG 2 + BUG 3 of PERF_AUTOMATION_FIXES_PLAN.md).

Hermetic: no device, no agent. Evidence for both bugs came from the 2026-07-25
llama3_1_8b_p150 run and the ACE-Step module-level reports.
"""

from __future__ import annotations

import json

from cc_optimize import summary as S

NO_SPEEDUP = "No net speedup recorded"
AT_FLOOR = "at its ttnn floor"


def _log(tmp_path, attempts):
    p = tmp_path / "kernlog.json"
    p.write_text(json.dumps(attempts))
    return p


def _attempt(op="TopKDeviceOperation", kind="knob:grid", won=False, wedged=False, ms=None, note=""):
    a = {"op_signature": op, "kernel_kind": kind, "beat_baseline": won, "note": note}
    if wedged:
        a["wedged"] = True
    if ms is not None:
        a["measured_ms"] = ms
    return a


# --------------------------------------------------------------------------- BUG 3


def test_no_speedup_line_absent_when_nothing_was_measured(tmp_path):
    """llama 2026-07-25: 10 attempts, ALL unmeasured (profiler returned no ops).

    The report still claimed the model 'may already be at its ttnn floor' while it
    sat at 22% of floor with 521 ms of headroom. A claim about the floor requires a
    measurement; with zero valid measurements the result is inconclusive."""
    attempts = [_attempt(wedged=True) for _ in range(10)]
    out = S.render_summary(_log(tmp_path, attempts), 912.31, final_override_ms=912.31, model="llama3_1_8b_p150")
    assert NO_SPEEDUP not in out, "claimed 'no net speedup' when NOTHING was ever measured"
    assert AT_FLOOR not in out, "claimed 'at its ttnn floor' with zero valid measurements"


def test_no_speedup_line_absent_when_a_win_exists(tmp_path):
    """ACE ace_step_audio_tokenizer: 33.08 -> 11.82 ms (2.80x), 21 committed wins,
    and the report still printed 'No net speedup recorded' (all 5 ACE modules did)."""
    attempts = [_attempt(op="MatmulDeviceOperation", kind="knob:grid", won=True, ms=11.82)]
    out = S.render_summary(_log(tmp_path, attempts), 33.08, final_override_ms=11.82, model="ace_step1_5")
    assert NO_SPEEDUP not in out, "claimed 'no net speedup' on a module that improved 2.80x"


def test_no_speedup_line_present_when_measured_and_genuinely_no_gain(tmp_path):
    """The line is correct ONLY here: levers were really measured and none won."""
    attempts = [_attempt(kind="knob:grid", ms=912.0), _attempt(kind="knob:dtype", ms=915.0)]
    out = S.render_summary(_log(tmp_path, attempts), 912.31, final_override_ms=912.31, model="llama3_1_8b_p150")
    assert NO_SPEEDUP in out, "measured attempts that did not beat baseline SHOULD report no net speedup"


# --------------------------------------------------------------------------- BUG 2


def test_knob_prefixed_levers_land_in_their_own_column():
    """`knob:grid` fell through the exact-match list into the catch-all `return "host"`,
    so the lever matrix showed activity under `host` while the per-attempt row said
    `knob:grid` — the report contradicted itself (llama 2026-07-25)."""
    assert S._level_of("knob:grid") == "grid"
    assert S._level_of("knob:dtype") == "dtype"
    assert S._level_of("knob:shard") == "shard"
    assert S._level_of("knob:fidelity") == "fidelity"


def test_bare_lever_names_still_map():
    for k in ("grid", "dtype", "fidelity", "shard", "tt-lang", "cpp"):
        assert S._level_of(k) == k


def test_unknown_lever_is_not_silently_called_host():
    """The catch-all made every unrecognised name look like a host-lever attempt.
    An unknown name must be reported as unknown, never attributed to `host`."""
    for junk in ("totally-new-lever", "rung:whatever", "", "  ", "shard:width"):
        assert S._level_of(junk) != "host", f"{junk!r} was silently bucketed as host"
