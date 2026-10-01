# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pin: after SIGKILLing a multi-chip run, reset -- do not ask whether the board is "alive".

_device_answers() asks tt-smi, which reports the ARC. A multi-chip ETH fabric can be wedged solid
while every ARC still answers, so "the board replied" is evidence about a different component.

REPRODUCED 2026-09-25 on a T3K, deliberately, with nothing else on the box:

    a workload ran 5180 all_gathers across the 2x4 mesh, healthy throughout;
    SIGKILL mid-collective -- exactly what _run_device_proc does on a timeout;
    the very next mesh open:
        RuntimeError: Firmware startup error on device 0 at core 0-10 over NOC0:
                      scratch_status=0xffffffff, postcode=0xffffffff

Open/close cycling was ruled out in the same session: 10 in-process cycles and 8 consecutive fresh
processes, fabric on, all clean. It is the KILL that wedges the fabric. The perf-test builder
recorded the consequence verbatim: "First run: VERDICT=WEDGE (rc=124, a timeout) ... Next three
runs: all FAIL at device open" -- 51 minutes spent for nothing.

The liveness gate stays for SINGLE-CHIP runs, which is the case it was added for ("On 2026-08-15
that reset four HEALTHY chips because an op ran long"): with one chip there is no fabric to wedge.
"""

from __future__ import annotations

import inspect

import pytest

from models.experimental.perf_automation.cc_optimize import run as run_mod

MANDATORY = run_mod._reset_is_mandatory_after_kill


@pytest.mark.parametrize("devices", ["all", "", "0,1", "0,1,2,3"])
def test_a_multi_chip_run_always_resets_after_a_kill(devices, monkeypatch):
    """The fabric can only be wedged when more than one chip was in play."""
    monkeypatch.setattr(run_mod, "_chip_count", lambda d: 8)
    assert MANDATORY(devices, {}) is True


@pytest.mark.parametrize("devices", ["single", "0", "3"])
def test_a_single_chip_run_keeps_the_liveness_gate(devices, monkeypatch):
    """The 2026-08-15 protection, unchanged: one chip, no fabric, the probe is the whole truth."""
    monkeypatch.setattr(run_mod, "_chip_count", lambda d: 1)
    assert MANDATORY(devices, {}) is False


def test_the_childs_own_environment_is_believed_first(monkeypatch):
    """Answered without opening a device -- the question is asked the moment one just died."""
    opened = {"n": 0}

    def _boom(d):
        opened["n"] += 1
        return 1

    monkeypatch.setattr(run_mod, "_chip_count", _boom)
    assert MANDATORY("all", {"device_count": "8"}) is True
    assert MANDATORY("all", {"mesh_chips": "4"}) is True
    assert opened["n"] == 0, "the env answered, so the device spec must not be consulted"


def test_a_wedged_board_does_not_talk_itself_out_of_the_reset(monkeypatch):
    """THE BUG THE FIRST VERSION OF THIS FIX HAD, pinned.

    _chip_count("all") asks ttnn.GetNumAvailableDevices() and returns 1 when that raises -- which is
    exactly what it does on a wedged board. Measured while writing the fix: fabric wedged,
    devices="all", the predicate answered False, so the one case it exists for would have skipped
    its reset. A count that cannot be taken is UNKNOWN, not one."""

    def _as_on_a_wedged_board(_devices):
        raise AssertionError("the device must not be consulted the moment it just died")

    monkeypatch.setattr(run_mod, "_chip_count", _as_on_a_wedged_board)
    assert MANDATORY("all", {}) is True
    assert MANDATORY("", {}) is True


def test_an_explicit_single_chip_spec_needs_no_device_either(monkeypatch):
    """The gate is kept by parsing the spec, not by asking the hardware."""
    monkeypatch.setattr(run_mod, "_chip_count", lambda d: 1 if d in ("single", "0", "3") else 8)
    assert MANDATORY("0", {}) is False
    assert MANDATORY("0,1", {}) is True


def test_an_explicit_env_count_overrides_the_spec(monkeypatch):
    """A one-chip box whose spec says "all" still keeps the 2026-08-15 gate."""
    monkeypatch.setattr(run_mod, "_chip_count", lambda d: 8)
    assert MANDATORY("all", {"device_count": "1"}) is False


def test_an_unusable_env_value_falls_back_instead_of_raising(monkeypatch):
    monkeypatch.setattr(run_mod, "_chip_count", lambda d: 8)
    for bad in ("not-a-number", None, "", {}):
        assert MANDATORY("all", {"device_count": bad}) is True, repr(bad)
        assert MANDATORY("0,1", {"device_count": bad}) is True, repr(bad)
    assert MANDATORY("all", None) is True


def test_the_timeout_path_actually_consults_it():
    """A predicate nothing calls fixes nothing."""
    src = inspect.getsource(run_mod._run_device_proc)
    assert "_reset_is_mandatory_after_kill(devices, env)" in src
    i = src.index("_reset_is_mandatory_after_kill(devices, env)")
    window = src[i : i + 200]
    assert "_device_answers()" in window, "the liveness probe must still run for the single-chip case"
    assert "not _reset_is_mandatory_after_kill" in src, "mandatory must SKIP the probe, not require it"


def test_the_reclaim_still_happens_through_the_shared_primitive():
    """No parallel reset: the kill path keeps routing through _reclaim_device."""
    src = inspect.getsource(run_mod._run_device_proc)
    assert "_reclaim_device(devices, error_text=out, after_kill=True)" in src
    # The predicate DECIDES; it must not act. Checked against the code, not the docstring -- which
    # legitimately explains what the liveness probe measures.
    code = inspect.getsource(MANDATORY)
    doc = inspect.getdoc(MANDATORY) or ""
    for line in doc.splitlines():
        code = code.replace(line, "")
    for forbidden in ("subprocess", "_reclaim_device", "_device_reset", "tt-smi", "tt_smi"):
        assert forbidden not in code, f"the predicate must not {forbidden}; it only decides"


# ------------------------------------------------------- the decision must reach the ACTION
#
# Measured 2026-09-25 with the predicate above already in place: the caller correctly decided
# "reset mandatory", and recover() still answered "no reset issued" -- every ARC was warm and the
# killed process had left no text, so the telemetry veto cancelled it. The two halves disagreed,
# and the board stayed wedged. fault_is_certain is how the decision reaches the action.


def _recover_with(monkeypatch, **kw):
    from agent import device_recovery as dr
    from agent import probes

    issued = []
    monkeypatch.setattr(probes, "board_telemetry", lambda: ([60.0, 61.0, 60.5, 62.0], []))  # all warm
    monkeypatch.setattr(dr, "reap_device_holders", lambda: [])
    monkeypatch.setattr(dr, "recovery_exhausted", lambda: False)
    monkeypatch.setattr(dr, "targets_for", lambda *a, **k: ["all"])
    monkeypatch.setattr(dr, "device_is_healthy", lambda: True)
    dr.recover("test", lambda t: issued.append(t), **kw)
    return issued


def test_a_certain_fault_overrides_the_telemetry_veto(monkeypatch):
    """A warm ARC says nothing about a wedged fabric, and a killed process leaves no text."""
    assert _recover_with(monkeypatch, error_text="", fault_is_certain=True) == ["all"]


def test_without_certainty_the_veto_still_wins(monkeypatch):
    """Default is unchanged -- the 2026-08-17 protection is not weakened for anyone else."""
    assert _recover_with(monkeypatch, error_text="") == []
    assert _recover_with(monkeypatch, error_text="an op ran long") == []


def test_the_reclaim_passes_its_kill_through(monkeypatch):
    """_reclaim_device(after_kill=True) is the only thing that knows a kill happened."""
    src = inspect.getsource(run_mod._reclaim_device)
    assert "fault_is_certain=after_kill" in src


def test_the_new_parameter_is_additive():
    """Every existing caller of recover() keeps its behaviour without being edited."""
    from agent import device_recovery as dr

    sig = inspect.signature(dr.recover)
    assert sig.parameters["fault_is_certain"].default is False
    assert list(sig.parameters)[:2] == ["where", "reset"], "the positional contract is unchanged"
