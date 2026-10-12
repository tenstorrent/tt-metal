# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A retry of the e2e gate's test step runs only the tests that have not passed yet.

2026-10-12, 4-chip p300c: Devstral's four e2e tests take ~12 min back to back; the board reached the
95C abort limit at ~11-12 min even from 65C, and every retry started again from the first test, so
twelve retries in a row were ended near the end. The same four tests passed one at a time.

Everything here is real pytest: the mark plugin in real sessions, and the gate driven through the real
probes._execute launcher. Only the thermometer is faked (a cool board), so no test waits on the real one.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.tt_hw_planner.commands import emit_e2e as E
from scripts.tt_hw_planner.gate_pytest_plugins import tt_hw_planner_gate_marks as MK
from scripts.tt_hw_planner import e2e_thermal as T
from models.experimental.perf_automation.agent import probes as PR
from models.experimental.perf_automation.agent import thermal_pytest_plugin as PL

PLUGIN_DIR = Path(MK.__file__).parent
PLUGIN_NAME = MK.__name__.rsplit(".", 1)[-1]


def _session(tmp_path, body, earned=(), conftest=""):
    """A real pytest session with the mark plugin loaded the way the gate loads it."""
    d = tmp_path / "s"
    d.mkdir(exist_ok=True)
    (d / "test_s.py").write_text(textwrap.dedent(body))
    if conftest:
        (d / "conftest.py").write_text(textwrap.dedent(conftest))
    ef = tmp_path / "earned.txt"
    ef.write_text("".join(n + "\n" for n in earned))
    env = {**os.environ, "PYTHONPATH": str(PLUGIN_DIR), MK.EARNED_ENV: str(ef)}
    env.pop("PYTEST_PLUGINS", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", PLUGIN_NAME, "-rA", "-s", "test_s.py"],
        cwd=str(d),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


# ---- the mark plugin ------------------------------------------------------------------------------------


def test_only_a_test_that_passed_every_phase_is_marked_passed(tmp_path):
    r = _session(
        tmp_path,
        """
        import pytest

        @pytest.fixture
        def bad_setup():
            raise RuntimeError("setup")

        @pytest.fixture
        def bad_teardown():
            yield
            raise RuntimeError("teardown")

        def test_ok(): pass
        def test_fail(): assert False
        def test_skip(): pytest.skip("no")
        @pytest.mark.xfail
        def test_xfail(): assert False
        @pytest.mark.xfail
        def test_xpass(): pass
        def test_setup_error(bad_setup): pass
        def test_teardown_error(bad_teardown): pass
        """,
    )
    passed = set(E._passed_segments(r.stdout))
    assert passed == {"test_s.py::test_ok"}, r.stdout[-3000:]
    started = {ln.split(MK.START_MARK, 1)[1].strip() for ln in r.stdout.splitlines() if MK.START_MARK in ln}
    assert len(started) == 7


def test_earned_tests_are_left_out_by_exact_id_not_by_prefix(tmp_path):
    r = _session(
        tmp_path,
        """
        import pytest
        def test_a(): print("RAN test_a")
        def test_ab(): print("RAN test_ab")
        @pytest.mark.parametrize("n", [1, 10])
        def test_p(n): print("RAN test_p", n)
        """,
        earned=["test_s.py::test_a", "test_s.py::test_p[1]"],
    )
    assert r.returncode == 0, r.stdout[-3000:]
    ran = [ln for ln in r.stdout.splitlines() if ln.startswith("RAN")]
    assert ran == ["RAN test_ab", "RAN test_p 10"], ran
    assert "2 deselected" in r.stdout


def test_everything_already_earned_is_pytests_no_tests_ran(tmp_path):
    r = _session(tmp_path, "def test_a(): pass\n", earned=["test_s.py::test_a"])
    assert r.returncode == pytest.ExitCode.NO_TESTS_COLLECTED, r.stdout[-2000:]


def test_a_session_ended_from_outside_keeps_the_passes_it_already_said(tmp_path):
    r = _session(
        tmp_path,
        """
        import os
        def test_1(): print("PCC=0.999")
        def test_2(): os._exit(75)
        def test_3(): pass
        """,
    )
    assert r.returncode == 75
    segs = E._passed_segments(r.stdout)
    assert set(segs) == {"test_s.py::test_1"}
    assert "PCC=0.999" in segs["test_s.py::test_1"]


def test_a_mark_sharing_a_line_with_pytest_progress_is_still_read():
    out = "test_s.py GATE_TEST_START=test_s.py::test_1\nhello\n.GATE_TEST_PASSED=test_s.py::test_1\n"
    assert E._passed_segments(out) == {"test_s.py::test_1": out.rstrip("\n")}


def test_a_passed_mark_for_a_different_test_is_not_credited():
    out = "GATE_TEST_START=a::t1\nGATE_TEST_PASSED=a::t2\n"
    assert E._passed_segments(out) == {}


# ---- the gate: real pytest through the real launcher ------------------------------------------------------


@pytest.fixture
def cool_board(monkeypatch):
    monkeypatch.setattr(PR._cc_optimize("perf_mcp"), "_read_die_temp_c", lambda: 40.0)


@pytest.fixture
def record(monkeypatch, tmp_path):
    rec = tmp_path / "thermal.log"
    rec.write_text("")
    monkeypatch.setenv(T.RUN_ENV, str(rec))
    return rec


def _demo(tmp_path, body):
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tests" / "e2e").mkdir(parents=True)
    (demo / "tests" / "e2e" / "test_e2e_m.py").write_text(textwrap.dedent(body))
    return demo


def _runs(tmp_path):
    f = tmp_path / "runs.log"
    return f.read_text().split() if f.exists() else []


_THREE_TESTS = """
import os
RUNS = %(runs)r
FIRST = %(first)r
REC = %(rec)r

def _ran(name):
    open(RUNS, "a").write(name + "\\n")

def test_1():
    _ran("test_1")
    print("e2e PCC=0.42")  # evidence a later check must still read

def test_2():
    _ran("test_2")
    if not os.path.exists(FIRST):
        open(FIRST, "w").write("1")
        # what the thermal plugin does at the abort limit: say so in the record, then end the session
        open(REC, "a").write("ABORT pid=%%d temp=95.2 by=pytest-plugin t=0\\n" %% os.getpid())
        open(REC, "a").write("RECOVERED upto=1 t=0 verified\\n")
        os._exit(75)

def test_3():
    _ran("test_3")
"""


def test_a_retry_after_an_abort_runs_only_the_tests_not_yet_passed(monkeypatch, tmp_path, cool_board, record):
    demo = _demo(
        tmp_path,
        _THREE_TESTS % {"runs": str(tmp_path / "runs.log"), "first": str(tmp_path / "first"), "rec": str(record)},
    )
    monkeypatch.setenv("E2E_REQUIRE_ON_DEVICE", "0")
    ok, reasons = E._run_deterministic_gates(demo, 0.99, 120)
    assert _runs(tmp_path) == ["test_1", "test_2", "test_2", "test_3"], _runs(tmp_path)
    g23 = [r for r in reasons if r.startswith("G2/G3")]
    assert g23 == [], g23
    # test_1 passed only in the first attempt; what it printed is still checked
    assert any("measured PCC 0.42" in r for r in reasons), reasons


def test_without_a_retry_nothing_changes(monkeypatch, tmp_path, cool_board, record):
    demo = _demo(
        tmp_path,
        """
        RUNS = %r
        def test_1(): open(RUNS, "a").write("test_1\\n")
        def test_2(): open(RUNS, "a").write("test_2\\n")
        """
        % str(tmp_path / "runs.log"),
    )
    monkeypatch.setenv("E2E_REQUIRE_ON_DEVICE", "0")
    _, reasons = E._run_deterministic_gates(demo, 0.99, 120)
    assert _runs(tmp_path) == ["test_1", "test_2"]
    assert not [r for r in reasons if r.startswith("G2/G3")], reasons


def test_a_real_failure_still_fails_and_passed_tests_are_not_rerun(monkeypatch, tmp_path, cool_board, record):
    """A failing test is retried (here: the board is reported wedged), and only it is."""
    demo = _demo(
        tmp_path,
        """
        RUNS = %r
        def test_1(): open(RUNS, "a").write("test_1\\n")
        def test_2():
            open(RUNS, "a").write("test_2\\n")
            raise RuntimeError("NOC0 is hung on PCIe device ID 9")
        """
        % str(tmp_path / "runs.log"),
    )
    monkeypatch.setenv("E2E_REQUIRE_ON_DEVICE", "0")
    monkeypatch.setattr(E, "_recover_board", lambda *a, **k: (True, "recovered (test)"))
    ok, reasons = E._run_deterministic_gates(demo, 0.99, 120)
    assert _runs(tmp_path) == ["test_1", "test_2", "test_2"], _runs(tmp_path)
    assert not ok and any("G2/G3: tests/e2e did not pass" in r for r in reasons), reasons


def test_a_session_ended_after_its_last_pass_needs_nothing_more(monkeypatch, tmp_path, cool_board, record):
    """Ended after every test passed (here: at session finish): the retry has nothing left to run,
    which pytest reports as "no tests ran" -- and that is a pass."""
    demo = _demo(
        tmp_path,
        """
        RUNS = %r
        def test_1(): open(RUNS, "a").write("test_1\\n")
        """
        % str(tmp_path / "runs.log"),
    )
    (demo / "tests" / "e2e" / "conftest.py").write_text(
        textwrap.dedent(
            """
            import os
            FIRST, REC = %r, %r
            def pytest_sessionfinish(session):
                if not os.path.exists(FIRST):
                    open(FIRST, "w").write("1")
                    open(REC, "a").write("ABORT pid=1 temp=95.1 by=pytest-plugin t=0\\n")
                    open(REC, "a").write("RECOVERED upto=1 t=0 verified\\n")
                    os._exit(75)
            """
            % (str(tmp_path / "first"), str(record))
        )
    )
    monkeypatch.setenv("E2E_REQUIRE_ON_DEVICE", "0")
    _, reasons = E._run_deterministic_gates(demo, 0.99, 120)
    assert _runs(tmp_path) == ["test_1"]
    assert not [r for r in reasons if r.startswith("G2/G3")], reasons


def test_the_plugin_directory_holds_only_plugins():
    """It goes on the gate's PYTHONPATH: any other module here would shadow the model's imports."""
    assert sorted(p.name for p in PLUGIN_DIR.glob("*.py")) == ["__init__.py", PLUGIN_NAME + ".py"]
