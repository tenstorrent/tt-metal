# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Work that is busy on purpose says so, and no window is tighter than a gap the run survived.

A trace measurement runs warmup, capture and a replay batch, and every one of those loops ran in
silence: no print, and -- because the replay batch is enqueued non-blocking and waited on in ONE
synchronize_device -- no syscalls and no stack movement either. Those are precisely the signals every
supervised loop in this tree judges liveness by (probes.progress_signature), so a stage whose step
costs tens of seconds looks identical to a device wedge. On 2026-09-29 one Qwen-Image-Edit stage was
killed three times that way, each kill reported as a wedge, and the retry loop re-ran the same
silence; the same stage completed the moment it was given a wider window by hand.

Two fixes, neither of which is a new number to tune:

  * the work announces itself, on a cadence read from whatever window is supervising it -- growing
    the log is already a progress signal, so nothing downstream has to learn about this silence;
  * a stall window is never tighter than a multiple of the longest quiet stretch the run has ALREADY
    come back from. run._run_device_proc had worked this out and kept it to itself, so the other two
    supervised loops still killed on the typed number. The rule now lives in ProgressWatch, whose own
    docstring is about exactly this exact three-way copy.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from agent import probes as PR  # noqa: E402


def _trace_replay(monkeypatch):
    """trace_replay imports ttnn at module scope; the loops under test never call into it."""
    monkeypatch.setitem(sys.modules, "ttnn", types.SimpleNamespace())
    sys.modules.pop("agent.trace_replay", None)
    import importlib

    return importlib.import_module("agent.trace_replay")


# --- the work announces itself -------------------------------------------------------------------


def test_a_slow_loop_breaks_its_own_silence(monkeypatch, capsys):
    """THE FAILURE: iterations long enough to outlast the window, with nothing printed."""
    TR = _trace_replay(monkeypatch)
    monkeypatch.setenv("PERF_MCP_VALIDATE_STALL_SEC", "40")  # beat every 10s
    clock = [0.0]
    monkeypatch.setattr(TR.time, "monotonic", lambda: clock[0])

    def _step():
        clock[0] += 25.0  # one iteration outlasts the beat
        return "sample"

    TR._iterate(_step, 4, "s")
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "TRACE_STAGE_ITER" in ln]
    assert len(lines) == 4, "every iteration outlasted the cadence, so every one must report"
    assert "TRACE_STAGE_ITER[s]=4/4" in lines[-1]


def test_a_fast_loop_does_not_spam_the_log(monkeypatch, capsys):
    """It sits inside a timed region; a fast stage must not pay for thousands of prints."""
    TR = _trace_replay(monkeypatch)
    monkeypatch.setenv("PERF_MCP_VALIDATE_STALL_SEC", "400")
    monkeypatch.setattr(TR.time, "monotonic", lambda: 0.0)  # no time passes
    TR._iterate(lambda: None, 500, "s")
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "TRACE_STAGE_ITER" in ln]
    assert lines == ["TRACE_STAGE_ITER[s]=500/500"], "only the final line is owed"


def test_the_replay_loop_keeps_nothing_it_does_not_need(monkeypatch):
    """The timed loop drops each result so the device buffers are freed; warmup needs them."""
    TR = _trace_replay(monkeypatch)
    monkeypatch.setattr(TR.time, "monotonic", lambda: 0.0)
    assert TR._iterate(lambda: "r", 3, "s") == []
    assert TR._iterate(lambda: "r", 3, "s", keep=True) == ["r", "r", "r"]


def test_warmup_still_hands_its_samples_to_the_advance_check(monkeypatch):
    TR = _trace_replay(monkeypatch)
    monkeypatch.setattr(TR.time, "monotonic", lambda: 0.0)
    assert TR._warm(lambda: "tok", 3) == ["tok", "tok", "tok"]
    assert TR._warm(lambda: "tok", 0) == []


def test_a_blocking_device_wait_still_reports_liveness(monkeypatch, capsys):
    """The replay batch is enqueued non-blocking ON PURPOSE, so the wait is one silent block."""
    TR = _trace_replay(monkeypatch)
    monkeypatch.setenv("PERF_MCP_VALIDATE_STALL_SEC", "4")  # beat every 1s
    with TR._Alive("replay x16"):
        import time as _t

        _t.sleep(2.2)
    out = capsys.readouterr().out
    assert "TRACE_STAGE_WAITING[replay x16]" in out


def test_the_beat_is_read_from_the_supervising_window_not_typed(monkeypatch):
    TR = _trace_replay(monkeypatch)
    for env in TR._STALL_WINDOW_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("PERF_MCP_VALIDATE_STALL_SEC", "600")
    assert TR._heartbeat_s() == 600 / TR._HEARTBEAT_DIVISOR
    monkeypatch.setenv("PERF_MCP_MEASURE_STALL_SEC", "120")
    assert TR._heartbeat_s() == 120 / TR._HEARTBEAT_DIVISOR, "the TIGHTEST window is the one to stay inside"
    monkeypatch.setenv("PERF_MCP_VALIDATE_STALL_SEC", "not a number")
    assert TR._heartbeat_s() == 120 / TR._HEARTBEAT_DIVISOR


def test_with_nothing_supervising_it_falls_back_to_what_execute_uses(monkeypatch):
    """No caller narrowed the window, so the cadence comes off _execute's own default."""
    import inspect

    TR = _trace_replay(monkeypatch)
    for env in TR._STALL_WINDOW_ENVS:
        monkeypatch.delenv(env, raising=False)
    default = float(inspect.signature(PR._execute).parameters["stall_timeout_s"].default)
    assert TR._heartbeat_s() == default / TR._HEARTBEAT_DIVISOR


def test_an_unsupervised_caller_is_left_quiet(monkeypatch, capsys):
    TR = _trace_replay(monkeypatch)
    for env in TR._STALL_WINDOW_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(TR, "_heartbeat_s", lambda: 0.0)
    with TR._Alive("x"):
        pass
    assert "TRACE_STAGE_WAITING" not in capsys.readouterr().out


# --- no window is tighter than a gap the run survived --------------------------------------------


def test_the_window_starts_at_what_the_caller_asked_for():
    w = PR.ProgressWatch(os.getpgrp(), None, 600.0)
    assert w.limit() == 600.0


def test_a_survived_gap_widens_the_window():
    """THE FAILURE: a run that has already gone quiet for 300s and come back is not wedging at 600s."""
    w = PR.ProgressWatch(os.getpgrp(), None, 600.0)
    w.note_progress(300.0, 0.0)  # quiet for 300s, then real progress
    assert w.limit() == PR._GAP_MULT * 300.0 > 600.0


def test_the_window_only_ever_grows():
    w = PR.ProgressWatch(os.getpgrp(), None, 100.0)
    w.note_progress(90.0, 0.0)
    wide = w.limit()
    w.note_progress(91.0, 90.0)  # a short gap must not shrink it back
    assert w.limit() == wide


def test_a_run_that_never_goes_quiet_keeps_the_callers_window():
    w = PR.ProgressWatch(os.getpgrp(), None, 600.0)
    for t in range(0, 100, 5):
        w.note_progress(float(t + 5), float(t))
    assert w.limit() == 600.0


def test_every_supervised_loop_asks_the_same_owner():
    """Same rule, three copies, already diverging -- the class docstring's own warning."""
    import inspect

    from scripts.tt_hw_planner import cc_harness as CH

    sources = [
        inspect.getsource(PR._execute),
        inspect.getsource(CH._supervised_gate_run),
    ]
    for src in sources:
        assert ".limit()" in src, "this loop still decides the window by itself"
        assert ".note_progress(" in src, "this loop never tells the owner it saw progress"


def test_the_blind_fallback_answers_the_same_questions(monkeypatch):
    """When probes cannot be imported the watch is a stand-in; it must not raise on the new calls."""
    from cc_optimize import run as RUN

    monkeypatch.setitem(sys.modules, "agent.probes", None)
    w = RUN._progress_watch(os.getpgrp(), None, 42.0)
    w.note_progress(10.0, 0.0)
    assert isinstance(w.limit(), float)


def test_it_names_no_model_or_stage(monkeypatch):
    """Constraint: no model, stage or box may be assumed. Executable bodies only, as the repo does."""
    import ast
    import inspect
    import textwrap

    TR = _trace_replay(monkeypatch)
    for fn in (TR._heartbeat_s, TR._iterate, PR.ProgressWatch.note_progress, PR.ProgressWatch.limit):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        node = tree.body[0]
        if ast.get_docstring(node) is not None:
            node.body = node.body[1:]
        lowered = ast.unparse(node).lower()
        for name in ("qwen", "denoise", "prefill", "vision", "encoder", "decoder", "vae"):
            assert name not in lowered, f"{name!r} in {fn.__name__} assumes the model's vocabulary"


def test_one_iteration_longer_than_the_window_is_still_not_silent(monkeypatch, capsys):
    """THE HOLE IN REPORTING PER ITERATION: the loop can only speak between steps, and a stage whose
    single step outlasts the supervising window is silent for the whole of it."""
    import time as _t

    TR = _trace_replay(monkeypatch)
    monkeypatch.setenv("PERF_MCP_VALIDATE_STALL_SEC", "4")  # beat every 1s

    def _one_very_long_step():
        _t.sleep(2.5)  # longer than the beat, and it is the ONLY iteration

    TR._iterate(_one_very_long_step, 1, "s")
    out = capsys.readouterr().out
    assert "TRACE_STAGE_WAITING[s]" in out, "nothing spoke while the single step ran"
