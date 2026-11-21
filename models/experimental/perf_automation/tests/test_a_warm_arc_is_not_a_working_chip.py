# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A chip whose ARC answers can still be unable to run anything, and the recovery could not tell.

2026-10-13/14, QB2, MiniCPM5-2B optimize on chip 3. After one hang, every device run on that chip hung
-- the profiler, and the trace run without it -- for fifteen hours across four rounds. Each recovery
printed "no reset issued": the run held ONE chip, so no kill made the reset mandatory, and both
liveness signals said the board was fine -- tt-smi answered, and every chip reported a die
temperature. Both come from the ARC, which keeps running while the Tensix cores are hung. A manual
`tt-smi -r` of the board cleared it.

THE FIX ADDS A QUESTION, NOT A RESET. Where the ARC signals would skip the reset, the device is asked
to run one op (probes.device_runs_work). Only a definite failure (False) turns the skip into a reset;
a pass, or no answer at all (None), leaves the decision exactly as it was -- so a healthy board is
still never reset, which is what the 2026-08-17 veto exists to guarantee.
"""

import sys
import time
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

from agent import probes  # noqa: E402

from models.experimental.perf_automation.cc_optimize import run as run_mod  # noqa: E402

# Captured at import, before the suite's autouse stub replaces it for every test.
_REAL_DEVICE_RUNS_WORK = probes.device_runs_work


# ------------------------------------------------------------------ the probe's three answers


def _probe(monkeypatch, code, timeout_s=30, attempts=1):
    """Run the real probe with a stand-in for the code it hands the interpreter."""
    monkeypatch.setattr(probes, "_WORK_PROBE_CODE", code)
    return _REAL_DEVICE_RUNS_WORK(timeout_s, attempts)


def _say(*markers):
    return "".join("print(%r, flush=True)\n" % m for m in markers)


def test_a_chip_that_runs_the_op_is_working(monkeypatch):
    assert _probe(monkeypatch, _say(probes._WORK_PROBE_READY, probes._WORK_PROBE_OK)) is True


def test_a_chip_that_fails_the_op_is_not_working(monkeypatch):
    code = _say(probes._WORK_PROBE_READY) + "raise SystemExit(1)\n"
    assert _probe(monkeypatch, code) is False


def test_a_chip_that_hangs_is_not_working_and_costs_only_the_timeout(monkeypatch):
    """THE 2026-10-13 CASE: the op never comes back."""
    code = _say(probes._WORK_PROBE_READY) + "import time\ntime.sleep(600)\n"
    t0 = time.monotonic()
    assert _probe(monkeypatch, code, timeout_s=3) is False
    assert time.monotonic() - t0 < 60, "the hung probe was not killed at its timeout"


def test_a_cold_build_longer_than_the_timeout_is_not_a_hang(monkeypatch):
    """2026-10-14: a healthy chip's cold probe was still compiling at the timeout. A build runs in
    child processes; while one is alive the hang clock does not run."""
    monkeypatch.setattr(probes, "_WORK_PROBE_POLL_S", 0.5)
    code = (
        _say(probes._WORK_PROBE_READY)
        + "import subprocess, sys\n"
        + "subprocess.run([sys.executable, '-c', 'import time; time.sleep(6)'])\n"
        + _say(probes._WORK_PROBE_OK)
    )
    assert _probe(monkeypatch, code, timeout_s=3) is True


def test_a_build_still_running_at_the_ceiling_is_no_evidence(monkeypatch):
    """Bounded all the same -- and a build cut off is a slow build, not a hung chip."""
    monkeypatch.setattr(probes, "_WORK_PROBE_POLL_S", 0.5)
    monkeypatch.setattr(probes, "_WORK_PROBE_CEILING_MULT", 2)
    code = (
        _say(probes._WORK_PROBE_READY)
        + "import subprocess, sys\n"
        + "subprocess.run([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
    )
    t0 = time.monotonic()
    assert _probe(monkeypatch, code, timeout_s=2) is None
    assert time.monotonic() - t0 < 60, "the build was not cut off at the ceiling"


def test_an_ok_without_a_clean_exit_is_not_working(monkeypatch):
    code = _say(probes._WORK_PROBE_READY, probes._WORK_PROBE_OK) + "raise SystemExit(3)\n"
    assert _probe(monkeypatch, code) is False


def test_no_ttnn_is_no_evidence_not_a_dead_chip(monkeypatch):
    """A probe that could not be run says nothing about the device, so it must not cause a reset."""
    assert _probe(monkeypatch, "raise ImportError('no ttnn here')\n") is None


def test_a_hang_before_ttnn_imported_is_no_evidence(monkeypatch):
    assert _probe(monkeypatch, "import time\ntime.sleep(600)\n", timeout_s=3) is None


def test_a_zero_timeout_switches_the_probe_off(monkeypatch):
    assert _probe(monkeypatch, _say(probes._WORK_PROBE_READY), timeout_s=0) is None


# ------------------------------------------------------------------ one failure is not a verdict


def _sequence(monkeypatch, answers):
    """device_runs_work over a scripted run of single attempts; returns (verdict, attempts made)."""
    seen = []
    script = iter(answers)
    monkeypatch.setattr(probes, "_WORK_PROBE_POLL_S", 0)
    monkeypatch.setattr(probes, "_work_probe_once", lambda t: seen.append(t) or next(script))
    return _REAL_DEVICE_RUNS_WORK(5, len(answers)), len(seen)


def test_a_failure_then_a_pass_is_a_working_chip(monkeypatch):
    """2026-10-14: one probe in three hung on a healthy chip while tt-smi polled it."""
    assert _sequence(monkeypatch, [False, True, False]) == (True, 2)


def test_only_a_failure_on_every_attempt_is_a_dead_chip(monkeypatch):
    assert _sequence(monkeypatch, [False, False, False]) == (False, 3)


def test_a_failure_mixed_with_no_evidence_is_no_verdict(monkeypatch):
    assert _sequence(monkeypatch, [False, None, False])[0] is None


def test_the_default_needs_more_than_one_failure():
    assert probes._WORK_PROBE_ATTEMPTS > 1


def test_the_probe_runs_a_real_op_on_every_visible_chip():
    """What the real code asks: every chip the run's scope makes visible, an op, a checked answer."""
    code = probes._WORK_PROBE_CODE
    assert "get_device_ids()" in code and "open_device" in code and "close_device" in code
    assert "ttnn.add" in code and "torch.equal" in code
    assert code.index(probes._WORK_PROBE_READY) < code.index("open_device"), "READY must precede the device"


# ------------------------------------------------------------------ recover() consults it


def _recover(monkeypatch, works, temps=(55.0, 56.0, 57.0, 58.0)):
    """recover() with every ARC signal saying 'alive'; returns the reset targets it attempted."""
    from agent import device_recovery as dr

    monkeypatch.setattr(probes, "board_telemetry", lambda: (list(temps), []))
    monkeypatch.setattr(probes, "device_runs_work", lambda *a, **k: works)
    monkeypatch.setattr(dr, "reap_device_holders", lambda: [])
    monkeypatch.setattr(dr, "recovery_exhausted", lambda: False)
    monkeypatch.setattr(dr, "device_is_healthy", lambda: True)
    monkeypatch.setattr(dr, "targets_for", lambda *a, **k: ["2,3"])
    tried, said = [], []
    ok = dr.recover("test", lambda tgt: tried.append(tgt), config_target="3", log=said.append)
    return tried, ok, said


def test_a_warm_chip_that_cannot_run_work_IS_reset(monkeypatch):
    tried, ok, said = _recover(monkeypatch, works=False)
    assert tried == ["2,3"] and ok is True
    assert any("could not run a single op" in s for s in said)


def test_a_warm_chip_that_runs_work_is_still_not_reset(monkeypatch):
    """The 2026-08-17 guarantee is untouched: a healthy board is never reset."""
    tried, ok, said = _recover(monkeypatch, works=True)
    assert tried == [] and ok is True
    assert any("reset SKIPPED" in s for s in said)


def test_no_probe_answer_keeps_the_old_skip(monkeypatch):
    tried, ok, _ = _recover(monkeypatch, works=None)
    assert tried == [] and ok is True


def test_the_probe_is_not_run_when_the_reset_is_already_decided(monkeypatch):
    """A silent chip or a certain fault resets without it: the probe never adds a device run there."""
    from agent import device_recovery as dr

    asked = []
    monkeypatch.setattr(probes, "board_telemetry", lambda: ([], []))
    monkeypatch.setattr(probes, "device_runs_work", lambda *a, **k: asked.append(1) or True)
    monkeypatch.setattr(dr, "reap_device_holders", lambda: [])
    monkeypatch.setattr(dr, "recovery_exhausted", lambda: False)
    monkeypatch.setattr(dr, "device_is_healthy", lambda: True)
    monkeypatch.setattr(dr, "targets_for", lambda *a, **k: ["2,3"])
    tried = []
    dr.recover("test", lambda tgt: tried.append(tgt))
    dr.recover("test", lambda tgt: tried.append(tgt), fault_is_certain=True)
    assert tried == ["2,3", "2,3"] and asked == []


# ------------------------------------------------------------------ the timeout path consults it


def _answers(monkeypatch, responsive, works):
    monkeypatch.setattr(probes, "device_is_responsive", lambda *a, **k: responsive)
    asked = []
    monkeypatch.setattr(probes, "device_runs_work", lambda *a, **k: asked.append(1) or works)
    return run_mod._device_answers(), asked


def test_an_answering_board_that_cannot_run_work_does_not_count_as_answering(monkeypatch):
    assert _answers(monkeypatch, True, False)[0] is False


def test_an_answering_board_that_runs_work_still_answers(monkeypatch):
    assert _answers(monkeypatch, True, True)[0] is True
    assert _answers(monkeypatch, True, None)[0] is True


def test_a_silent_board_is_not_also_asked_to_run_work(monkeypatch):
    answered, asked = _answers(monkeypatch, False, True)
    assert answered is False and asked == []
