# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""emit-e2e gets optimize's four thermal layers: hold, cooldown, watch, abort.

2026-10-10, 4-chip p300c: a Devstral emit-e2e run held 88-94C for hours with not one [thermal-*] line
in its log. The builder ran its own pytest through Bash, which none of optimize's layers reach; at
~18:41 a chip dropped off the bus and the reset after it took all four.

The thermometer is faked (the real one is the board's); the processes, /proc reads, environment
propagation and pytest sessions are real.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from scripts.tt_hw_planner import e2e_thermal as T
from scripts.tt_hw_planner.commands import emit_e2e as E
from models.experimental.perf_automation.agent import device_recovery as DR
from models.experimental.perf_automation.agent import probes as PR
from models.experimental.perf_automation.agent import thermal_pytest_plugin as PL

# Captured at import, before the suite conftest swaps them out for each test.
_REAL_HOLD = T.hold_if_hot
REPO = Path(T.__file__).resolve().parents[2]


def _pm():
    return PR._cc_optimize("perf_mcp")


def _run():
    return PR._cc_optimize("run")


def _thermometer(monkeypatch, readings):
    """perf_mcp reads `readings` in order (the last one repeats); sleeping is recorded, not done."""
    pm = _pm()
    it = list(readings)
    seen, slept = [], []

    def _read():
        v = it.pop(0) if len(it) > 1 else it[0]
        seen.append(v)
        return v

    monkeypatch.setattr(pm, "_read_die_temp_c", _read)
    monkeypatch.setattr(pm.time, "sleep", lambda s: slept.append(s))
    return seen, slept


# ---- hold / cooldown: optimize's launch gate, its own numbers ---------------------------------------


def test_at_the_ceiling_the_hold_waits_until_the_board_is_back_to_the_cool_target(monkeypatch, capsys):
    pm = _pm()
    hot, cool = pm._SAFETY_CEILING_C + 2, pm._COOL_BACK_TO_C
    seen, slept = _thermometer(monkeypatch, [hot, hot, cool + 10, cool + 3, cool - 1])
    _REAL_HOLD("G2/G3 tests/e2e")
    err = capsys.readouterr().err
    assert "[thermal-ceiling] G2/G3 tests/e2e" in err
    assert seen[-1] <= cool, "the hold returned before the board reached the cool-back target"
    assert len(slept) >= 3, "the hold did not actually wait between readings"


def test_below_the_ceiling_the_hold_does_not_wait(monkeypatch, capsys):
    seen, slept = _thermometer(monkeypatch, [_pm()._SAFETY_CEILING_C - 0.5])
    _REAL_HOLD("G6 trace")
    assert slept == [] and "[thermal-ceiling]" not in capsys.readouterr().err


def test_the_hold_has_no_deadline(monkeypatch, capsys):
    """A board that takes a long time to cool is still waited for: proceeding at the ceiling risks it."""
    pm = _pm()
    readings = [pm._SAFETY_CEILING_C + 1] * 200 + [pm._COOL_BACK_TO_C - 1]
    seen, slept = _thermometer(monkeypatch, readings)
    _REAL_HOLD("long cool")
    assert seen[-1] <= pm._COOL_BACK_TO_C and len(slept) >= 199


def test_an_unreadable_thermometer_never_stops_the_work(monkeypatch):
    _thermometer(monkeypatch, [None])
    _REAL_HOLD("no sensor")  # returns: a missing sensor is not a hot board


# ---- device_step: hold before, cool after, re-run an aborted step -----------------------------------


@pytest.fixture
def record(monkeypatch, tmp_path):
    rec = tmp_path / "thermal.log"
    rec.write_text("")
    monkeypatch.setenv(T.RUN_ENV, str(rec))
    return rec


def _holds(monkeypatch):
    holds = []
    monkeypatch.setattr(T, "hold_if_hot", lambda label: holds.append(label))
    return holds


def test_a_step_holds_before_and_cools_after(monkeypatch, record):
    holds = _holds(monkeypatch)
    assert T.device_step("G6 trace", lambda: "verdict") == "verdict"
    assert holds == ["G6 trace", "G6 trace (post-run cooldown)"]


def test_a_step_the_board_aborted_is_rerun_after_recovery_and_cooling(monkeypatch, record):
    holds = _holds(monkeypatch)
    calls = []

    def once():
        calls.append(time.monotonic())
        if len(calls) == 1:
            PL.record_abort(str(record), "ABORT pid=1 temp=96 by=pytest-plugin")
            PL.record_abort(str(record), "RECOVERED upto=1 t=0 verified")
        return len(calls)

    assert T.device_step("G2/G3 tests/e2e", once) == 2
    assert holds == ["G2/G3 tests/e2e", "G2/G3 tests/e2e (post-run cooldown)"] * 2


def test_the_rerun_does_not_start_on_top_of_the_recovery(monkeypatch, record):
    _holds(monkeypatch)
    monkeypatch.setattr(T, "WATCH_POLL_S", 0.05)
    recovered_at = []
    calls = []

    def once():
        calls.append(time.monotonic())
        if len(calls) == 1:
            PL.record_abort(str(record), "ABORT pid=1 temp=96 by=e2e-watcher")

            def _recover_later():
                time.sleep(0.5)
                recovered_at.append(time.monotonic())
                PL.record_abort(str(record), "RECOVERED upto=1 t=0 verified")

            threading.Thread(target=_recover_later).start()
        return len(calls)

    T.device_step("G6 block stacks", once)
    assert len(calls) == 2 and calls[1] >= recovered_at[0]


def test_a_board_that_aborts_every_attempt_is_not_retried_forever(monkeypatch, record):
    _holds(monkeypatch)
    calls = []

    def once():
        calls.append(1)
        n = len(calls)
        PL.record_abort(str(record), "ABORT pid=%d temp=96 by=pytest-plugin" % n)
        PL.record_abort(str(record), "RECOVERED upto=%d t=0 verified" % n)
        return n

    assert T.device_step("G2/G3 tests/e2e", once) == 1 + _run()._THERMAL_ABORT_RETRIES
    assert len(calls) == 1 + _run()._THERMAL_ABORT_RETRIES


# ---- watch / abort: only this run's holders, then the shared recovery -------------------------------


class _HotBoard:
    _ABORT_CEILING_C = 95.0

    @staticmethod
    def board_over_abort_limit():
        return True, 96.2


class _Run:
    samples: list = []

    @staticmethod
    def _thermal_watch_sample(state, label):
        _Run.samples.append(label)


def _sleeper(env):
    return subprocess.Popen(["sleep", "60"], env=env)


def test_the_watch_ends_only_this_runs_holder_and_recovers_by_the_shared_rule(monkeypatch, record):
    mine = _sleeper({**os.environ, T.RUN_ENV: str(record)})
    other = _sleeper({k: v for k, v in os.environ.items() if k != T.RUN_ENV})  # someone else's work
    try:
        monkeypatch.setattr(DR, "device_holders", lambda: {mine.pid, other.pid})
        monkeypatch.setattr(T, "_ABORT_GRACE_S", 0.0)
        certain = []
        monkeypatch.setitem(T._STATE, "recover", lambda text, c: certain.append(c) or (True, "verified"))
        ctx = {"state": {}, "hot_since": None, "handled": 0}
        T._watch_tick(_Run, _HotBoard, ctx)
        assert mine.wait(timeout=5) == -9, "this run's holder was not ended at the abort limit"
        assert other.poll() is None, "a holder that is not this run's was touched"
        lines = record.read_text().splitlines()
        assert lines[0].startswith("ABORT pid=%d" % mine.pid) and "by=e2e-watcher" in lines[0]
        assert lines[1].startswith("RECOVERED upto=1")
        assert certain == [True], "an abort must get the reset, as a certain fault"
        assert _Run.samples, "the over-clamp watch was not sampled"
        T._watch_tick(_Run, _HotBoard, ctx)
        assert certain == [True], "the same abort was recovered twice"
    finally:
        for p in (mine, other):
            p.kill()


def test_a_single_chip_abort_is_still_reset(monkeypatch, record):
    """Measured 2026-10-10: a single-chip run ended mid-matmul left chip 2 failing FW init while its ARC
    still answered, so the shared single-chip veto would have left it broken. An abort always resets."""
    certain = []
    monkeypatch.setitem(T._STATE, "recover", lambda text, c: certain.append(c) or (True, "verified"))
    PL.record_abort(str(record), "ABORT pid=1 temp=96 by=pytest-plugin")
    cool = type("Cool", (), {"board_over_abort_limit": staticmethod(lambda: (False, 60.0))})
    T._watch_tick(_Run, cool, {"state": {}})
    assert certain == [True]
    assert record.read_text().splitlines()[-1].startswith("RECOVERED upto=1 ")
    assert T._counts() == (1, 1)


def test_two_aborts_found_together_get_one_reset(monkeypatch, record):
    certain = []
    monkeypatch.setitem(T._STATE, "recover", lambda text, c: certain.append(c) or (True, "verified"))
    PL.record_abort(str(record), "ABORT pid=1 temp=96 by=pytest-plugin")
    PL.record_abort(str(record), "ABORT pid=2 temp=96 by=e2e-watcher")
    cool = type("Cool", (), {"board_over_abort_limit": staticmethod(lambda: (False, 60.0))})
    T._watch_tick(_Run, cool, {"state": {}})
    assert certain == [True] and T._counts() == (2, 2)


def test_the_watch_gives_the_plugin_its_grace_before_ending_a_holder(monkeypatch, record):
    mine = _sleeper({**os.environ, T.RUN_ENV: str(record)})
    try:
        monkeypatch.setattr(DR, "device_holders", lambda: {mine.pid})
        monkeypatch.setitem(T._STATE, "recover", lambda text, c: (True, "verified"))
        ctx = {"state": {}, "hot_since": None, "handled": 0}
        T._watch_tick(_Run, _HotBoard, ctx)
        time.sleep(0.2)
        assert mine.poll() is None and record.read_text() == "", "ended before the plugin's grace"
    finally:
        mine.kill()


# ---- the environment every child of the run gets -----------------------------------------------------


def test_a_child_env_carries_the_tag_the_plugin_and_the_repo_root(record):
    env = T.agent_env({"PYTHONPATH": "/x", "PYTEST_PLUGINS": "other_plugin"})
    assert env[T.RUN_ENV] == str(record)
    assert env["PYTEST_PLUGINS"] == "other_plugin," + PL.PLUGIN_MODULE
    assert env["PYTHONPATH"].split(os.pathsep) == [str(REPO), "/x"]
    assert T.agent_env(env) == env, "applying it twice must change nothing"


def test_phase_3_envs_carry_the_protection():
    """The harness rebuilds the agent env with PYTHONPATH=repo root and the MCP server env from scratch."""
    import inspect

    src = inspect.getsource(E._run_emit_e2e_cc)
    assert "mcp_env = _thermal_env(mcp_env)" in src
    assert src.index('env["PYTHONPATH"] = str(repo_root)') < src.index("env = _thermal_env(env)")


def _pytest_session(tmp_path, test_body, conftest="", env_extra=None):
    """A real pytest session in a directory OUTSIDE the repo, with only the env emit-e2e hands a child."""
    d = tmp_path / "session"
    d.mkdir()
    (d / "test_s.py").write_text(textwrap.dedent(test_body))
    if conftest:
        (d / "conftest.py").write_text(textwrap.dedent(conftest))
    base = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", ""), "PYTHONPATH": str(REPO)}
    env = T.agent_env({**base, **(env_extra or {})})
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider", "test_s.py"],
        cwd=str(d),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_plugin_loads_in_a_pytest_the_agent_starts_anywhere(tmp_path, record):
    r = _pytest_session(
        tmp_path,
        """
        def test_loaded(request):
            assert request.config.pluginmanager.has_plugin("%s")
        """
        % PL.PLUGIN_MODULE,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_plugin_holds_the_session_before_the_first_test_on_a_hot_board(tmp_path, record):
    order = tmp_path / "order.log"
    r = _pytest_session(
        tmp_path,
        """
        def test_body():
            open(%r, "a").write("test ran\\n")
        """
        % str(order),
        conftest="""
        from models.experimental.perf_automation.agent import probes as PR
        pm = PR._cc_optimize("perf_mcp")
        _r = [pm._SAFETY_CEILING_C + 3, pm._SAFETY_CEILING_C + 3, pm._COOL_BACK_TO_C + 4, pm._COOL_BACK_TO_C - 2]
        def _read():
            v = _r.pop(0) if len(_r) > 1 else _r[0]
            open(%r, "a").write("read %%s\\n" %% v)
            return v
        pm._read_die_temp_c = _read
        pm._COOLDOWN_POLL_S = 0.0
        """
        % str(order),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[thermal-ceiling] pytest session start" in r.stderr
    lines = order.read_text().splitlines()
    first_test = lines.index("test ran")
    assert float(lines[first_test - 1].split()[1]) <= _pm()._COOL_BACK_TO_C, lines


def test_the_plugin_ends_its_own_run_at_the_abort_limit_and_says_why(tmp_path, record):
    r = _pytest_session(
        tmp_path,
        """
        import time
        def test_long_device_run():
            time.sleep(30)
        """,
        conftest="""
        from models.experimental.perf_automation.agent import thermal_pytest_plugin as PL
        from models.experimental.perf_automation.agent import probes as PR
        pm = PR._cc_optimize("perf_mcp")
        PL.WATCH_POLL_S = 0.2
        PL.holds_device = lambda: True
        pm.board_over_abort_limit = lambda: (True, 96.4)
        pm.cool_if_over_safety_ceiling = lambda label="": False
        """,
    )
    assert r.returncode == PL.ABORT_EXIT_CODE, r.stdout + r.stderr
    assert "[thermal-abort]" in r.stderr and "NOT a code failure" in r.stderr
    assert "by=pytest-plugin" in record.read_text().splitlines()[0]


def test_the_plugin_ignores_a_hot_board_while_it_does_not_hold_the_device(tmp_path, record):
    r = _pytest_session(
        tmp_path,
        """
        import time
        def test_cpu_only():
            time.sleep(1.5)
        """,
        conftest="""
        from models.experimental.perf_automation.agent import thermal_pytest_plugin as PL
        from models.experimental.perf_automation.agent import probes as PR
        pm = PR._cc_optimize("perf_mcp")
        PL.WATCH_POLL_S = 0.2
        pm.board_over_abort_limit = lambda: (True, 96.4)
        pm.cool_if_over_safety_ceiling = lambda label="": False
        """,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert record.read_text() == ""


def test_a_plugin_that_cannot_reach_the_gates_warns_and_lets_the_tests_run(tmp_path, record):
    r = _pytest_session(
        tmp_path,
        """
        def test_still_runs():
            pass
        """,
        conftest="""
        from models.experimental.perf_automation.agent import probes as PR
        def _broken(name):
            raise ImportError("cc_optimize.%s is not reachable" % name)
        PR._cc_optimize = _broken
        """,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "temperature protection is INERT" in r.stderr


# ---- install: one watcher per run, inherited by every child ------------------------------------------


def test_install_starts_one_watcher_and_a_child_inherits_rather_than_starting_another(tmp_path):
    probe = textwrap.dedent(
        """
        import json, os, subprocess, sys, threading
        from scripts.tt_hw_planner import e2e_thermal as T
        path = T.install()
        again = T.install()
        child = subprocess.run([sys.executable, "-c", (
            "import json, os, threading;"
            "from scripts.tt_hw_planner import e2e_thermal as T;"
            "p = T.install();"
            "print(json.dumps({'path': p, 'threads': [t.name for t in threading.enumerate()],"
            " 'plugins': os.environ.get('PYTEST_PLUGINS', '')}))")],
            capture_output=True, text=True, env=os.environ.copy(), cwd=%r)
        print(json.dumps({
            "path": path, "again": again,
            "watchers": [t.name for t in threading.enumerate()].count("e2e-thermal-watch"),
            "plugins": os.environ.get("PYTEST_PLUGINS", ""),
            "child": json.loads(child.stdout.strip().splitlines()[-1]),
        }))
        """
        % str(REPO)
    )
    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", ""), "PYTHONPATH": str(REPO)}
    env["TMPDIR"] = str(tmp_path)
    r = subprocess.run(
        [sys.executable, "-c", probe], cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0, r.stdout + r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["path"] and out["again"] == out["path"]
    assert out["watchers"] == 1
    assert PL.PLUGIN_MODULE in out["plugins"].split(",")
    assert out["child"]["path"] == out["path"], "a child must join its parent's run, not start its own"
    assert "e2e-thermal-watch" not in out["child"]["threads"]
    assert PL.PLUGIN_MODULE in out["child"]["plugins"].split(",")


# ---- the gate's device steps go through it --------------------------------------------------------------


def test_the_e2e_gate_step_is_held_before_and_cooled_after(monkeypatch, tmp_path, record):
    holds = _holds(monkeypatch)
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tests" / "e2e").mkdir(parents=True)
    (demo / "tests" / "e2e" / "test_e2e_m.py").write_text("def test_e2e():\n    pass\n")
    monkeypatch.setenv("E2E_REQUIRE_ON_DEVICE", "0")
    order = []

    def _exec(cmd, cwd, env, timeout_s, log_path, **k):
        order.append("execute")
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("AssertionError: Gate 3: PCC 0.5 < 0.99\n1 failed in 1.0s")
        return 1

    monkeypatch.setattr(PR, "_execute", _exec)
    monkeypatch.setattr(T, "hold_if_hot", lambda label: (holds.append(label), order.append(label)))
    E._run_deterministic_gates(demo, 0.99, 60)
    i = order.index("execute")
    assert order[i - 1] == "G2/G3 tests/e2e" and order[i + 1] == "G2/G3 tests/e2e (post-run cooldown)", order


def test_every_device_launch_in_the_gate_is_a_thermal_step():
    """Each place emit-e2e and its MCP server start device work is wrapped; a new one must be too."""
    import inspect
    import re

    from scripts.tt_hw_planner import e2e_mcp as M

    gate_src = inspect.getsource(E)
    for label in (
        '"G6 block stacks"',
        '"host-op observer probe"',
        '"G2/G3 tests/e2e"',
        '"G6 trace"',
        '"trace gate"',
        '"trace-gate overflow fix-loop"',
    ):
        assert re.search(r"_(thermal|device_gate)_step\(\s*" + re.escape(label), gate_src), label
    assert re.search(r"_thermal_step\(\s*\"trace-capture probe\"", inspect.getsource(M._run_probe))
