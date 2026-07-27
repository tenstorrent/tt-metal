# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The planner's device reset obeys the shared recovery contract.

Three things had drifted here, all invisible because nothing tested them:
  * `agentic/actions.py` calls `_run_tt_smi_reset(reason=...)`, which a keyword-only signature
    without `reason` rejects with TypeError -- so the agentic reset action never reset anything.
  * the return value was tt-smi's exit code, which says the command ran, not that the card is back.
  * the target came from a default device list, ignoring the chip the failure names.
"""
from __future__ import annotations

import inspect

from scripts.tt_hw_planner import cli


def test_agentic_call_signature_is_accepted():
    """The exact kwargs agentic/actions.py passes must bind."""
    sig = inspect.signature(cli._run_tt_smi_reset)
    sig.bind(reason="agentic.RunDeviceReset")


class _FakeDR:
    def __init__(self, healthy):
        self._healthy = healthy

    def device_is_healthy(self, *a, **k):
        return self._healthy

    def dead_chip_from_error(self, text):
        return 3 if "PCIe ID 3" in str(text) else None

    def expand_to_boards(self, ids):
        return "2,3" if list(ids) == [3] else None


def _reset_with(monkeypatch, *, rc, healthy, error_text=""):
    calls = {}

    class _P:
        returncode = rc
        stdout = ""
        stderr = ""

    def _run(cmd, **kw):
        calls["cmd"] = list(cmd)
        return _P()

    monkeypatch.delenv("TT_PLANNER_NO_DEVICE_RESET", raising=False)
    monkeypatch.setattr(cli, "_DEVICE_RESET_COUNT", 0, raising=False)
    monkeypatch.setattr(cli, "_device_recovery", lambda: _FakeDR(healthy))
    import shutil as _sh
    import subprocess as _sp

    monkeypatch.setattr(_sh, "which", lambda n: "/usr/bin/tt-smi")
    monkeypatch.setattr(_sp, "run", _run)
    ok = cli._run_tt_smi_reset(context="ut", error_text=error_text)
    return ok, calls


def test_exit_zero_but_dead_card_is_a_failed_reset(monkeypatch):
    """tt-smi exiting 0 says the command ran, not that the device came back."""
    ok, _ = _reset_with(monkeypatch, rc=0, healthy=False)
    assert ok is False


def test_exit_zero_and_healthy_card_succeeds(monkeypatch):
    ok, _ = _reset_with(monkeypatch, rc=0, healthy=True)
    assert ok is True


def test_reason_is_accepted_and_behaves_like_context(monkeypatch, capsys):
    """The alias must actually work, not merely bind: agentic/actions.py has only ever passed it."""
    calls = {}

    class _P:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.delenv("TT_PLANNER_NO_DEVICE_RESET", raising=False)
    monkeypatch.setattr(cli, "_DEVICE_RESET_COUNT", 0, raising=False)
    monkeypatch.setattr(cli, "_device_recovery", lambda: _FakeDR(True))
    import shutil as _sh
    import subprocess as _sp

    monkeypatch.setattr(_sh, "which", lambda n: "/usr/bin/tt-smi")
    monkeypatch.setattr(_sp, "run", lambda cmd, **kw: calls.setdefault("cmd", list(cmd)) and None or _P())
    assert cli._run_tt_smi_reset(reason="agentic.RunDeviceReset") is True
    assert "agentic.RunDeviceReset" in capsys.readouterr().out


def test_the_chip_named_in_the_failure_selects_its_board(monkeypatch):
    """THE REGRESSION SHAPE: reset what the error names, widened to the whole board."""
    ok, calls = _reset_with(
        monkeypatch, rc=0, healthy=True, error_text="Read 0xffffffff over PCIe ID 3: board should be reset"
    )
    assert ok is True
    assert calls["cmd"][-1] == "2,3", calls
