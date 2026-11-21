# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A process holding the device at the safety ceiling is ended, not left waiting -- and the step it
belonged to is re-run from a cool board, keeping the trace stages it had already measured.

Measured on a 4-chip p300c, 2026-10-13, nothing else on the box: the mesh opened and left idle took
every chip from ~20 W to 85-91 W and the board from 58C to 92.5C in fifteen minutes; closing the mesh
inside the still-living process changed nothing (the device files stayed open, 93.9C and rising);
only the process EXITING let it cool, and a reset brought the chips back to ~25 W. The in-process wait
at the ceiling (probes.thermal_yield) therefore never cooled anything: Devstral's perf test sat in it
for forty-five minutes climbing 91.5C -> 94.7C.

Real pytest and real subprocesses throughout; only the thermometer and the reset are faked, so no test
reads or resets the real board.
"""
from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from agent import perf_test_gen as PTG
from agent import probes as PR
from agent import thermal_pytest_plugin as PL
from agent.perf_adapter import PipelineStageAdapter

_PA = Path(__file__).resolve().parents[1]
_REPO = _PA.parents[2]


def _child_env(**extra):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(_PA), str(_REPO)])}
    for k in (PL.RUN_ENV, PR.STAGES_KEPT_ENV, "PYTEST_PLUGINS", "PYTEST_ADDOPTS"):
        env.pop(k, None)
    env.update(extra)
    return env


# ---- the yield at a stage boundary -------------------------------------------------------------------

_YIELD_CHILD = """
import sys
from agent import probes, thermal_pytest_plugin as PL
pm = probes._cc_optimize("perf_mcp")
PL.holds_device = lambda: {holds}
pm.board_over_safety_ceiling = lambda: ({hot}, 91.0)
pm.cool_if_over_safety_ceiling = lambda label="": print("WAITED", flush=True)
probes.thermal_yield("prefill")
print("CONTINUED", flush=True)
"""


def _yield_in_child(tmp_path, holds, hot, record=True):
    rec = tmp_path / "record.log"
    env = _child_env(**({PL.RUN_ENV: str(rec)} if record else {}))
    r = subprocess.run(
        [sys.executable, "-c", _YIELD_CHILD.format(holds=holds, hot=hot)],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return r, (rec.read_text() if rec.exists() else "")


def test_holding_the_device_at_the_ceiling_ends_the_process_instead_of_waiting(tmp_path):
    r, record = _yield_in_child(tmp_path, holds=True, hot=True)
    assert r.returncode == PL.ABORT_EXIT_CODE, r.stdout + r.stderr
    assert "WAITED" not in r.stdout and "CONTINUED" not in r.stdout
    assert "[thermal-handoff] prefill: board at 91.0C" in r.stderr
    # the run's watcher sees it exactly as it sees an abort: one ABORT line, from this process
    assert record.count(PL.ABORT_MARK + " pid=") == 1 and "by=handoff" in record


def test_a_process_that_does_not_hold_the_device_still_waits(tmp_path):
    r, record = _yield_in_child(tmp_path, holds=False, hot=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["WAITED", "CONTINUED"] and record == ""


def test_below_the_ceiling_a_holder_carries_on(tmp_path):
    r, record = _yield_in_child(tmp_path, holds=True, hot=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["WAITED", "CONTINUED"] and record == ""


def test_outside_a_run_with_a_record_the_exit_alone_says_it(tmp_path):
    """optimize keeps no abort record: the exit status is the whole message, and nothing is written."""
    r, record = _yield_in_child(tmp_path, holds=True, hot=True, record=False)
    assert r.returncode == PL.ABORT_EXIT_CODE and PR.ended_for_heat(r.returncode)
    assert record == ""


def test_an_ordinary_exit_status_is_not_a_heat_stop():
    assert not any(PR.ended_for_heat(rc) for rc in (0, 1, 2, -9, None))


# ---- the stages a step keeps across its attempts ------------------------------------------------------


def _fake_ttnn():
    m = types.ModuleType("ttnn")
    m._tid = 0

    def begin_trace_capture(device, cq_id=0):
        m._tid += 1
        return m._tid

    m.begin_trace_capture = begin_trace_capture
    m.end_trace_capture = lambda device, tid, cq_id=0: None
    m.execute_trace = lambda device, tid, cq_id=0, blocking=False: None
    m.synchronize_device = lambda device: None
    m.release_trace = lambda device, tid: None
    m.record_event = lambda device, cq: ("ev", cq)
    m.wait_for_event = lambda cq, ev: None
    return m


@pytest.fixture
def trace_replay(monkeypatch):
    monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn())
    sys.modules.pop("agent.trace_replay", None)  # re-import so `import ttnn` binds the fake
    import agent.trace_replay as tr

    monkeypatch.setattr(PR, "thermal_yield", lambda label="": None)
    return tr


class _Dev:
    def num_command_queues(self):
        return 1


class _ThreeStagePipe:
    """Three independent stages, each preparing its own inputs, as emit-e2e emits them."""

    PIPELINE_STAGES = ["a", "b", "c"]

    def __init__(self, stepped):
        self._stepped = stepped

    def a_trace_setup(self, inputs=None):
        pass

    def a_trace_step(self):
        self._stepped.append("a")

    def a_trace_items(self):
        return 7

    def b_trace_setup(self, inputs=None):
        pass

    def b_trace_step(self):
        self._stepped.append("b")

    def b_trace_items(self):
        return 1

    def c_trace_setup(self, inputs=None):
        pass

    def c_trace_step(self):
        self._stepped.append("c")

    def c_trace_items(self):
        return 1


def _three_stage_pipe(stepped):
    return _ThreeStagePipe(stepped)


def _measure(tr, stepped):
    return tr.measure_adapter(PipelineStageAdapter(lambda dev: _three_stage_pipe(stepped), batch=1), _Dev())


def _stage_ms_lines(out):
    return [ln for ln in out.splitlines() if ln.startswith("TRACE_STAGE_MS[")]


def test_a_kept_stage_is_replayed_not_measured_again(trace_replay, monkeypatch, capsys):
    with PR.stages_kept_across_retries() as kept:
        first = []
        _measure(trace_replay, first)
        out1 = capsys.readouterr().out
        assert kept() == 3 and set(first) == {"a", "b", "c"}
        # keep only `a` from that attempt, as if the board ended it at the b boundary
        rec = Path(os.environ[PR.STAGES_KEPT_ENV])
        rec.write_text(rec.read_text().splitlines()[0] + "\n")
        second = []
        _measure(trace_replay, second)
        out2 = capsys.readouterr().out
    assert set(second) == {"b", "c"}, "a stage kept by an earlier attempt was measured again"
    # the reader sees every stage exactly once, the kept one carrying its first attempt's own lines
    assert [ln.split("]")[0] for ln in _stage_ms_lines(out2)] == [
        "TRACE_STAGE_MS[a",
        "TRACE_STAGE_MS[b",
        "TRACE_STAGE_MS[c",
    ]
    assert _stage_ms_lines(out1)[0] in out2
    assert out2.count("TRACE_STAGE_ITEMS[a]=7") == 1, "the kept stage's own rows did not come back with it"
    assert "TRACE_STAGES=3" in out2
    assert PR.STAGES_KEPT_ENV not in os.environ, "the record outlived its step"


def test_without_a_record_every_stage_is_measured_as_before(trace_replay, capsys):
    assert PR.STAGES_KEPT_ENV not in os.environ
    for _ in range(2):
        stepped = []
        _measure(trace_replay, stepped)
        assert set(stepped) == {"a", "b", "c"}
        assert len(_stage_ms_lines(capsys.readouterr().out)) == 3


def test_kept_lines_are_results_not_heartbeats(trace_replay, capsys):
    with PR.stages_kept_across_retries():
        _measure(trace_replay, [])
        lines = [ln for r in PR._kept_records() for ln in r["lines"]]
    assert lines and all(ln.startswith(trace_replay._MARK_PREFIX) for ln in lines)
    assert not any(ln.startswith(trace_replay._HEARTBEAT_MARK) for ln in lines)


def test_a_kept_stage_belongs_to_its_own_test_node(trace_replay, monkeypatch):
    with PR.stages_kept_across_retries():
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "t.py::test_one (call)")
        _measure(trace_replay, [])
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "t.py::test_two (call)")
        stepped = []
        _measure(trace_replay, stepped)
    assert set(stepped) == {"a", "b", "c"}, "another test's measurement was reused"


def test_a_nested_owner_shares_the_outer_record():
    with PR.stages_kept_across_retries():
        outer = os.environ[PR.STAGES_KEPT_ENV]
        with PR.stages_kept_across_retries():
            assert os.environ[PR.STAGES_KEPT_ENV] == outer
        assert os.environ[PR.STAGES_KEPT_ENV] == outer and Path(outer).exists()
    assert PR.STAGES_KEPT_ENV not in os.environ and not Path(outer).exists()


def test_a_launcher_given_an_older_env_still_passes_the_record_on():
    old_env = {"A": "1"}
    with PR.stages_kept_across_retries():
        env = PR.with_kept_stages(old_env)
        assert env[PR.STAGES_KEPT_ENV] == os.environ[PR.STAGES_KEPT_ENV]
    assert old_env == {"A": "1"}, "the caller's env was changed"
    assert PR.with_kept_stages(old_env) is old_env


def test_a_line_cut_short_by_an_ended_process_is_not_a_measurement(tmp_path, monkeypatch):
    rec = tmp_path / "kept.jsonl"
    rec.write_text('{"node": "", "stage": "a", "ms": 1.0, "path": "p", "lines": []}\n{"node": "", "sta')
    monkeypatch.setenv(PR.STAGES_KEPT_ENV, str(rec))
    assert list(PR.kept_stages("")) == ["a"] and PR.kept_stage_count() == 1


# ---- the one re-run rule -----------------------------------------------------------------------------


def _run_mod():
    import importlib.util

    spec = importlib.util.spec_from_file_location("cc_optimize_run_t", _PA / "cc_optimize" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _attempts(stages_per_attempt):
    """An attempt that keeps the given number of new stages and is then ended for heat; the last one
    completes. Returns (attempt, was_aborted, calls)."""
    calls = []

    def attempt():
        n = len(calls)
        calls.append(os.environ.get(PR.STAGES_KEPT_ENV))
        for i in range(stages_per_attempt[n] if n < len(stages_per_attempt) else 0):
            PR.keep_stage("", "s%d_%d" % (n, i), 1.0, "p", [])
        return n

    return attempt, (lambda: len(calls) <= len(stages_per_attempt)), calls


def test_an_attempt_that_kept_a_stage_does_not_use_up_a_retry():
    run = _run_mod()
    cooled = []
    progress = [1] * (run._THERMAL_ABORT_RETRIES + 3)  # more ended attempts than retries, each progressing
    attempt, aborted, calls = _attempts(progress)
    assert run.rerun_after_thermal_abort(attempt, aborted, "x", cool=lambda: cooled.append(1)) == len(progress)
    assert len(calls) == len(progress) + 1 and len(cooled) == len(progress)
    assert len(set(calls)) == 1 and calls[0], "the attempts did not share one record"
    assert PR.STAGES_KEPT_ENV not in os.environ


def test_attempts_that_keep_nothing_are_still_bounded():
    run = _run_mod()
    attempt, _, calls = _attempts([])
    run.rerun_after_thermal_abort(attempt, lambda: True, "x", cool=lambda: None)
    assert len(calls) == 1 + run._THERMAL_ABORT_RETRIES


def test_every_rerun_starts_from_the_cool_back_target(monkeypatch):
    """It waited only at the safety ceiling, so a board released at 95C that had dropped to the high 80s
    was re-run from there and ended again before measuring anything."""
    run = _run_mod()
    waits = []
    pm = types.SimpleNamespace(
        _COOL_BACK_TO_C=65.0,
        _cooldown_after_clamp=lambda t, give_up_on_plateau=True: waits.append((t, give_up_on_plateau)),
    )
    monkeypatch.setattr(run, "_perf_mcp", lambda: pm)
    attempt, aborted, _ = _attempts([0])
    run.rerun_after_thermal_abort(attempt, aborted, "x")
    assert waits == [(65.0, False)]


def test_a_step_that_is_never_ended_runs_once_and_waits_for_nothing():
    run = _run_mod()
    cooled = []
    attempt, _, calls = _attempts([])
    assert run.rerun_after_thermal_abort(attempt, lambda: False, "x", cool=lambda: cooled.append(1)) == 0
    assert len(calls) == 1 and cooled == []


# ---- optimize's launcher: a child that ended itself for heat ----------------------------------------


def _device_proc(run, monkeypatch, code, record=None):
    monkeypatch.setattr(run, "_wait_for_memory_headroom_before_device_work", lambda label="": None)
    monkeypatch.setattr(run, "_wait_for_thermal_headroom_before_device_work", lambda label="": None)
    monkeypatch.setattr(run, "_thermal_watch_sample", lambda *a, **k: None)
    monkeypatch.setattr(run, "_device_answers", lambda: True)
    monkeypatch.setattr(run, "_reset_is_mandatory_after_kill", lambda devices, env=None: True)
    resets = []
    monkeypatch.setattr(
        run, "_reclaim_device", lambda devices, error_text="", after_kill=False: resets.append(1) or "reset"
    )
    if record:
        monkeypatch.setenv(PL.RUN_ENV, record)
    else:
        monkeypatch.delenv(PL.RUN_ENV, raising=False)
    run._THERMAL_ABORTED[0] = False
    rc, _ = run._run_device_proc(
        [sys.executable, "-c", code], Path.cwd(), _child_env(), "0", 60, label="t", capture=True
    )
    return rc, resets


def test_optimize_reruns_a_child_that_ended_for_heat_and_resets_the_board(monkeypatch):
    run = _run_mod()
    rc, resets = _device_proc(run, monkeypatch, "import os; os._exit(%d)" % PL.ABORT_EXIT_CODE)
    assert rc == PL.ABORT_EXIT_CODE and run._THERMAL_ABORTED[0] is True and resets == [1]


def test_an_ordinary_failure_is_not_taken_for_heat(monkeypatch):
    run = _run_mod()
    rc, resets = _device_proc(run, monkeypatch, "import sys; sys.exit(1)")
    assert rc == 1 and run._THERMAL_ABORTED[0] is False and resets == []


def test_inside_a_run_whose_watcher_answers_aborts_the_launcher_leaves_it_alone(monkeypatch, tmp_path):
    """emit-e2e's watcher resets and re-runs on the ABORT line; doing it here too would reset twice."""
    run = _run_mod()
    rc, resets = _device_proc(
        run, monkeypatch, "import os; os._exit(%d)" % PL.ABORT_EXIT_CODE, record=str(tmp_path / "rec")
    )
    assert rc == PL.ABORT_EXIT_CODE and run._THERMAL_ABORTED[0] is False and resets == []


# ---- end to end: a real perf test, through the real launcher ------------------------------------------

_PERF_TEST = """
import os, sys, types
m = types.ModuleType("ttnn"); m._tid = 0
def _b(device, cq_id=0):
    m._tid += 1
    return m._tid
m.begin_trace_capture = _b
m.end_trace_capture = lambda device, tid, cq_id=0: None
m.execute_trace = lambda device, tid, cq_id=0, blocking=False: None
m.synchronize_device = lambda device: None
m.release_trace = lambda device, tid: None
m.record_event = lambda device, cq: ("ev", cq)
m.wait_for_event = lambda cq, ev: None
sys.modules["ttnn"] = m
from agent import probes, thermal_pytest_plugin as PL
import agent.trace_replay as tr
from agent.perf_adapter import PipelineStageAdapter

probes._THERMAL_YIELD_MIN_GAP_S = 0  # stages here take microseconds, not minutes
pm = probes._cc_optimize("perf_mcp")
PL.holds_device = lambda: True
FIRST, RUNS = os.environ["T_FIRST"], os.environ["T_RUNS"]
_boundaries = []

def _hot():
    # the board reaches the ceiling at the second stage boundary of the FIRST attempt only
    _boundaries.append(1)
    if len(_boundaries) == 2 and not os.path.exists(FIRST):
        open(FIRST, "w").write("1")
        return True, 91.0
    return False, 60.0

pm.board_over_safety_ceiling = _hot

def _ran(name):
    with open(RUNS, "a") as fh:
        fh.write(name + "\\n")  # returns nothing, like a real trace step
pm.cool_if_over_safety_ceiling = lambda label="": None

class _Pipe:
    PIPELINE_STAGES = ["a", "b", "c"]

    def a_trace_setup(self, inputs=None):
        pass

    def a_trace_step(self):
        _ran("a")

    def a_trace_items(self):
        return 7

    def b_trace_setup(self, inputs=None):
        pass

    def b_trace_step(self):
        _ran("b")

    def b_trace_items(self):
        return 1

    def c_trace_setup(self, inputs=None):
        pass

    def c_trace_step(self):
        _ran("c")

    def c_trace_items(self):
        return 1

class _Dev:
    def num_command_queues(self):
        return 1

def test_perf():
    tr.measure_adapter(PipelineStageAdapter(lambda dev: _Pipe(), batch=1), _Dev())
"""


def _launch(tmp_path, monkeypatch, record=None):
    test = tmp_path / "t" / "test_perf_x.py"
    test.parent.mkdir()
    test.write_text(_PERF_TEST)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(_PA), str(_REPO)]))
    monkeypatch.setenv("T_FIRST", str(tmp_path / "first"))
    monkeypatch.setenv("T_RUNS", str(tmp_path / "runs.log"))
    for k in ("PYTEST_PLUGINS", "PYTEST_ADDOPTS", PR.STAGES_KEPT_ENV):
        monkeypatch.delenv(k, raising=False)
    if record:
        monkeypatch.setenv(PL.RUN_ENV, record)
    else:
        monkeypatch.delenv(PL.RUN_ENV, raising=False)
    monkeypatch.setattr(PR, "wait_for_memory_headroom_before_device_work", lambda label="": None)
    run = PR._cc_optimize("run")
    monkeypatch.setattr(run, "_wait_for_thermal_headroom_before_device_work", lambda label="": None)
    monkeypatch.setattr(run, "_cool_before_rerun", lambda: None)
    resets = []
    monkeypatch.setattr(PR, "reset_after_ended_child", lambda error_text="", env=None: resets.append(1) or True)
    monkeypatch.chdir(tmp_path)
    rc, out = PTG._run_perf_node(str(test) + "::test_perf", {}, timeout_s=120)
    runs = (tmp_path / "runs.log").read_text().split() if (tmp_path / "runs.log").exists() else []
    return rc, out, runs, resets


def _order(runs):
    """Each stage's step runs several times (count, warm, capture); keep the order stages were reached."""
    seen = []
    for n in runs:
        if not seen or seen[-1] != n:
            seen.append(n)
    return seen


def test_a_perf_test_ended_for_heat_is_rerun_and_measures_only_what_is_left(tmp_path, monkeypatch):
    rc, out, runs, resets = _launch(tmp_path, monkeypatch)
    # attempt 1: a, then ended at the b boundary; attempt 2: a kept, so only b and c
    assert _order(runs) == ["a", "b", "c"], runs
    assert rc == 0, out[-3000:]
    assert [ln.split("]")[0] for ln in _stage_ms_lines(out)] == [
        "TRACE_STAGE_MS[a",
        "TRACE_STAGE_MS[b",
        "TRACE_STAGE_MS[c",
    ]
    assert "TRACE_STAGES=3" in out
    assert resets == [1], "the board was not reset once after the heat exit"


def test_inside_emit_e2e_the_perf_test_hands_the_heat_exit_to_the_run(tmp_path, monkeypatch):
    rec = tmp_path / "record.log"
    rc, out, runs, resets = _launch(tmp_path, monkeypatch, record=str(rec))
    assert rc == PL.ABORT_EXIT_CODE and _order(runs) == ["a"]
    assert resets == [], "the launcher reset the board under the run's own watcher"
    assert "by=handoff" in rec.read_text()


# ---- optimize's full-pipeline measurement ------------------------------------------------------------


def _perf_mcp(monkeypatch, ended_first):
    pm = PR._cc_optimize("perf_mcp")
    calls, resets = [], []

    def _full():
        calls.append(os.environ.get(PR.STAGES_KEPT_ENV))
        if ended_first and len(calls) == 1:
            pm._LAST_RUN_ENDED_FOR_HEAT = True
            return None, None, "ended for heat", None
        return 12.5, "trace", None, "trace+1cq"

    monkeypatch.setattr(pm, "_run_full_pipeline_ms", _full)
    monkeypatch.setattr(pm, "cool_if_over_safety_ceiling", lambda label="": False)
    monkeypatch.setattr(pm, "_cooldown_after_clamp", lambda *a, **k: (True, 60.0))
    monkeypatch.setattr(pm, "_record_thermal_observation", lambda *a, **k: None)
    monkeypatch.setattr(pm, "_read_die_temp_c", lambda: 60.0)
    monkeypatch.setattr(PR, "reset_after_ended_child", lambda error_text="", env=None: resets.append(1) or True)
    monkeypatch.delenv(PL.RUN_ENV, raising=False)
    return pm, calls, resets


def test_a_full_pipeline_reading_ended_for_heat_is_taken_again(monkeypatch):
    pm, calls, resets = _perf_mcp(monkeypatch, ended_first=True)
    assert pm._measure_full_pipeline_guarded() == (12.5, "trace", None, "trace+1cq")
    assert len(calls) == 2 and calls[0] and calls[0] == calls[1] and resets == [1]


def test_a_full_pipeline_reading_is_otherwise_unchanged(monkeypatch):
    pm, calls, resets = _perf_mcp(monkeypatch, ended_first=False)
    assert pm._measure_full_pipeline_guarded() == (12.5, "trace", None, "trace+1cq")
    assert len(calls) == 1 and resets == []


def _full_pipeline_launches(monkeypatch, rc):
    pm = PR._cc_optimize("perf_mcp")
    launches = []
    monkeypatch.setattr(pm, "_MANIFEST", {"perf_test_resolved": {"path": "t.py::test_perf"}})
    monkeypatch.setattr(pm, "_FULLPIPE_SAMPLES", 3)
    monkeypatch.setattr(pm, "_adaptive_run", lambda *a, **k: launches.append(1) or pm._AdaptiveResult(rc, "out"))
    monkeypatch.delenv(PL.RUN_ENV, raising=False)
    monkeypatch.setattr(pm, "_LAST_RUN_ENDED_FOR_HEAT", False)
    ms = pm._run_full_pipeline_ms()[0]
    return pm, ms, launches


def test_a_heat_exit_ends_the_samples_and_is_flagged_for_a_rerun(monkeypatch):
    """No next sample on a board this hot: the re-run, from a cool board, takes them."""
    pm, ms, launches = _full_pipeline_launches(monkeypatch, PL.ABORT_EXIT_CODE)
    assert ms is None and launches == [1] and pm._LAST_RUN_ENDED_FOR_HEAT is True


def test_a_heartbeat_said_during_a_stage_is_not_kept(trace_replay, monkeypatch):
    def _stage(device, st, budget_s=0.0, tables=None):
        print("%s[replay x2] 30s on device" % trace_replay._HEARTBEAT_MARK, flush=True)
        print("TRACE_STAGE_REPLAYS=2 of 2 requested (1.0s each)", flush=True)
        return 1.0, "trace+1cq"

    monkeypatch.setattr(trace_replay, "_measure_stage", _stage)
    with PR.stages_kept_across_retries():
        _measure(trace_replay, [])
        lines = [ln for r in PR._kept_records() for ln in r["lines"]]
    assert any(ln.startswith("TRACE_STAGE_REPLAYS=") for ln in lines)
    assert not any(ln.startswith(trace_replay._HEARTBEAT_MARK) for ln in lines)
