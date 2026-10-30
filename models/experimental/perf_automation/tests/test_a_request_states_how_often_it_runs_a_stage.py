"""The full-pipeline line says how many times one request runs each stage, from the pipeline's own count.

The pass times each stage's step once. Qwen-Image-Edit's 35.7 s "trace+1CQ full-pipeline e2e" covered
1 of its 50 denoise steps and read as one image edit. A stage the pipeline replays per request states
how often through <stage>_trace_repeats; the report then says so and prices one request. Stage names
here are this file's own.
"""

from __future__ import annotations

import types
from pathlib import Path

from agent import perf_adapter as PA
from agent import stage_seams

_PA = Path(__file__).resolve().parents[1]


def test_the_seam_is_optional_and_the_stage_carries_it():
    assert stage_seams.REPEATS in stage_seams.OPTIONAL and stage_seams.REPEATS in stage_seams.ALL
    assert stage_seams.REPEATS not in stage_seams.REQUIRED
    assert PA._Stage("alpha", None, repeats=50).repeats == 50
    assert PA._Stage("alpha", None).repeats == 0, "unstated stays unstated"
    assert PA._Stage("alpha", None, seq_split=4).seq_split == 4, "the other seams are untouched"


def test_the_count_is_read_off_the_pipeline_like_every_counting_seam():
    p = types.SimpleNamespace(alpha_trace_repeats=lambda: 50, beta_trace_repeats=lambda: 1 / 0)
    assert PA._stated_count(p, "alpha", stage_seams.REPEATS) == 50
    assert PA._stated_count(p, "beta", stage_seams.REPEATS) == 0, "a failing count is not stated"
    assert PA._stated_count(p, "gamma", stage_seams.REPEATS) == 0
    src = (_PA / "agent" / "perf_adapter.py").read_text()
    assert "repeats=_stated_count(p, name, _seams.REPEATS)" in src


def test_the_marker_the_parser_and_the_pin_agree():
    replay = (_PA / "agent" / "trace_replay.py").read_text()
    assert 'print("TRACE_STAGE_REPEATS[%s]=%d" % (st.name, _rp), flush=True)' in replay and "if _rp > 1:" in replay
    mcp = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    assert '("TRACE_STAGE_REPEATS[", stage_repeats)' in mcp
    assert "_ledger().KIND_STAGE_REPEATS, stage_repeats" in mcp
    from cc_optimize import measurements

    assert measurements.KIND_STAGE_REPEATS == "stage_repeats"


def _note(monkeypatch, rows, stage_ms):
    """summary._per_request_note over these pinned rows, read through the REAL ledger reader."""
    import cc_optimize.measurements as M
    from cc_optimize import perf_mcp, summary

    monkeypatch.setattr(M, "rows", lambda kind="", phase="", model="", task="": [r for r in rows if r["kind"] == kind])
    monkeypatch.setattr(summary, "_ledger", lambda: M)
    monkeypatch.setattr(perf_mcp, "read_stage_ms", lambda **k: stage_ms)
    return summary._per_request_note("m", "t")


def test_a_repeated_stage_is_named_and_one_request_is_priced(monkeypatch):
    rows = [{"kind": "stage_repeats", "depth": "loop", "value_ms": 50.0}]
    note = _note(monkeypatch, rows, {"once": 3000.0, "loop": 20.0})
    assert "loop 50x" in note
    assert "~4000 ms per request" in note, "3000 x 1 + 20 x 50"


def test_the_first_pinned_count_wins_like_every_anchor(monkeypatch):
    rows = [
        {"kind": "stage_repeats", "depth": "loop", "value_ms": 50.0},
        {"kind": "stage_repeats", "depth": "loop", "value_ms": 2.0},
    ]
    assert "loop 50x" in _note(monkeypatch, rows, {"loop": 1.0})


def test_without_stage_times_it_still_says_what_the_pass_covers(monkeypatch):
    rows = [{"kind": "stage_repeats", "depth": "loop", "value_ms": 50.0}]
    note = _note(monkeypatch, rows, {})
    assert "loop 50x" in note and "per request" not in note


def test_nothing_is_printed_when_no_stage_repeats(monkeypatch):
    assert _note(monkeypatch, [], {"once": 3.0}) == ""
    assert _note(monkeypatch, [{"kind": "stage_repeats", "depth": "loop", "value_ms": 1.0}], {"loop": 1.0}) == ""


def test_a_ledger_that_cannot_be_read_prints_nothing(monkeypatch):
    from cc_optimize import summary

    def _boom():
        raise OSError("ledger gone")

    monkeypatch.setattr(summary, "_ledger", _boom)
    assert summary._per_request_note("m", "t") == ""


def test_the_note_sits_under_the_full_pipeline_line_only():
    src = (_PA / "cc_optimize" / "summary.py").read_text()
    i = src.index('_fp = _ledger_line(_ledger().KIND_FULLPIPE, "trace+1CQ %s" % _trace_scope, model, task)')
    assert "_per_request_note(model, task)" in src[i : i + 300]


def test_emit_e2e_tells_a_model_to_state_it():
    prompt = (_PA.parent.parent.parent / "scripts" / "tt_hw_planner" / "commands" / "emit_e2e.py").read_text()
    assert "<stage>_trace_repeats(): ZERO-ARG, OPTIONAL." in prompt


def test_a_pass_that_states_no_count_says_every_stage_is_priced_at_one_run(monkeypatch):
    """The silence itself is reported. Unstated and genuinely-once look identical to the ranking, so
    the line that carries the total says which of the two it assumed."""
    note = _note(monkeypatch, [], {"heavy": 42344.0, "light": 1297.0, "mid": 3012.0})
    assert "no stage states a <stage>_trace_repeats count" in note
    assert "priced at one run per request" in note


def test_that_clause_names_no_stage_and_guesses_at_none(monkeypatch):
    """Which stage loops is the pipeline's to say. A report that picked the biggest and called it a
    loop would be wrong for every model whose heaviest stage genuinely runs once."""
    note = _note(monkeypatch, [], {"heavy": 42344.0, "light": 1297.0})
    assert "heavy" not in note and "light" not in note


def test_a_stated_count_replaces_the_clause_rather_than_joining_it(monkeypatch):
    rows = [{"kind": "stage_repeats", "depth": "loop", "value_ms": 50.0}]
    note = _note(monkeypatch, rows, {"loop": 20.0, "once": 3000.0})
    assert "loop 50x" in note
    assert "no stage states" not in note, "the question is answered; the prompt goes away"


def test_one_measured_stage_is_left_alone(monkeypatch):
    """Nothing to rank against, so the missing count skews nothing worth a line."""
    from cc_optimize import summary

    assert summary._MIN_STAGES_FOR_REPEATS_NOTE == 2
    assert _note(monkeypatch, [], {"only": 3000.0}) == ""
    assert _note(monkeypatch, [], {"only": 3000.0, "unmeasured": 0.0}) == ""


def test_the_clause_spells_the_seam_the_pipeline_must_add(monkeypatch):
    """Named from stage_seams, not typed here: a renamed seam renames the advice with it."""
    note = _note(monkeypatch, [], {"a": 2.0, "b": 1.0})
    assert stage_seams.REPEATS in note
    src = (_PA / "cc_optimize" / "summary.py").read_text()
    assert "from agent.stage_seams import REPEATS as _repeats_seam" in src
