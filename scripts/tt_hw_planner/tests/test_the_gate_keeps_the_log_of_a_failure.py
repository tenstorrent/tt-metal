# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pin: a gate step that FAILED keeps its output; a step that passed cleans up.

The gate streams pytest to a temp log and then removed that directory unconditionally in a
`finally:`. So the only surviving trace of a failure was the 15-line tail quoted into `reasons` --
and for a step killed mid-flight there is no useful tail at all.

Measured 2026-09-25: a gate killed at its budget reported

    G2/G3: tests/e2e exceeded 14400s with no verdict (likely device/fabric hang)

and the pytest output that would have shown where it actually was had already been deleted by that
line. Neither operator, on either box, could produce it afterwards; the failure could only be
argued about. A tool that destroys the evidence of its own verdict cannot be debugged.

Nothing about the verdict changes here -- same pass/fail, same reasons -- only whether the file
survives long enough to read, plus one line saying where it is.
"""

from __future__ import annotations

import glob
import inspect
import os
import tempfile
from pathlib import Path

import pytest

from models.experimental.perf_automation.agent import probes as _PR
from scripts.tt_hw_planner.commands import emit_e2e as E


@pytest.fixture()
def demo(tmp_path, monkeypatch):
    d = tmp_path / "models" / "demos" / "m"
    (d / "tests" / "e2e").mkdir(parents=True)
    (d / "tests" / "e2e" / "test_e2e_m.py").write_text("def test_e2e():\n    pass\n")
    monkeypatch.setenv("E2E_REQUIRE_ON_DEVICE", "0")
    return d


def _stub(monkeypatch, rc=0, stall=False, text="1 passed"):
    """Make the gate's pytest write `text` and return rc (or stall). Records the log path."""
    seen = {}

    def _exec(cmd, cwd, env, timeout_s, log_path, **k):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(text)
        seen["path"] = Path(log_path)
        if stall:
            raise _PR.TracyHangError("tracy run made no forward progress for 600s; log: %s" % log_path)
        return rc

    monkeypatch.setattr(_PR, "_execute", _exec)
    return seen


def test_a_failed_step_keeps_its_log(demo, monkeypatch):
    """THE BUG: the real error was written, then deleted."""
    seen = _stub(monkeypatch, rc=1, text="E RuntimeError: the real reason\n1 error in 12s")
    ok, reasons = E._run_deterministic_gates(demo, 0.99, 60)
    p = seen["path"]
    assert p.exists(), "the failing run's output was deleted -- the failure cannot be diagnosed"
    assert "the real reason" in p.read_text()


def test_a_failed_step_says_where_its_log_is(demo, monkeypatch):
    """Keeping it is useless if nothing says where it is."""
    seen = _stub(monkeypatch, rc=1, text="E RuntimeError: boom\n1 error")
    ok, reasons = E._run_deterministic_gates(demo, 0.99, 60)
    assert any(str(seen["path"]) in r for r in reasons), reasons


def test_a_stalled_step_keeps_its_log_too(demo, monkeypatch):
    """The case that motivated this: killed mid-flight, so the tail is worthless and the partial
    output is the ONLY evidence of where it had got to."""
    seen = _stub(monkeypatch, stall=True, text="[e2e] denoise step 37/50 done\n")
    ok, reasons = E._run_deterministic_gates(demo, 0.99, 60)
    assert seen["path"].exists(), "a killed run's partial output was deleted"
    assert "denoise step 37/50" in seen["path"].read_text()


def test_a_passing_step_cleans_up(demo, monkeypatch):
    """No leak: success leaves nothing behind."""
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "e2e_gate_*")))
    seen = _stub(monkeypatch, rc=0, text="1 passed")
    E._run_deterministic_gates(demo, 0.99, 60)
    assert not seen["path"].parent.exists(), "a passing step left its temp dir behind"
    after = set(glob.glob(os.path.join(tempfile.gettempdir(), "e2e_gate_*")))
    assert after <= before, "a passing step leaked a temp dir"


def test_the_cleanup_is_conditional_not_unconditional():
    """Pin the shape: the rmtree must not sit bare in the finally again."""
    code = inspect.getsource(E._run_deterministic_gates)
    i = code.index("shutil.rmtree(_gate_log.parent")
    window = code[max(0, i - 220) : i]
    assert "_e2e.ok" in window, "cleanup is not gated on the step having passed"


def test_an_exception_still_cleans_up():
    """If the step raised, there is no verdict to diagnose -- do not leak a temp dir for it."""
    code = inspect.getsource(E._run_deterministic_gates)
    i = code.index("shutil.rmtree(_gate_log.parent")
    assert "_e2e is None" in code[max(0, i - 220) : i]
