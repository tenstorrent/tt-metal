# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Self-heal for claude-agent-sdk <-> CLI version drift: detect via smoke test, auto-upgrade,
re-test. Logic is tested with smoke_test / _pip_upgrade monkeypatched (no real SDK or pip)."""

from agent import sdk_health


def _seq_smoke(results):
    """smoke_test stub returning queued (ok, detail) tuples in order."""
    calls = {"n": 0}

    def stub(timeout=150):
        r = results[min(calls["n"], len(results) - 1)]
        calls["n"] += 1
        return r

    stub.calls = calls
    return stub


def test_healthy_no_install(monkeypatch):
    monkeypatch.setattr(sdk_health, "smoke_test", _seq_smoke([(True, "")]))
    pip = {"n": 0}
    monkeypatch.setattr(sdk_health, "_pip_upgrade", lambda: pip.__setitem__("n", pip["n"] + 1))
    r = sdk_health.ensure_compatible(log=lambda *a: None)
    assert r["ok"] and not r["healed"]
    assert pip["n"] == 0  # never touched pip on a healthy env


def test_mismatch_autosync_heals(monkeypatch):
    smoke = _seq_smoke([(False, "Claude Code returned an error result: success"), (True, "")])
    monkeypatch.setattr(sdk_health, "smoke_test", smoke)
    pip = {"n": 0}
    monkeypatch.setattr(sdk_health, "_pip_upgrade", lambda: pip.__setitem__("n", pip["n"] + 1))
    r = sdk_health.ensure_compatible(autosync=True, log=lambda *a: None)
    assert r["ok"] and r["healed"]
    assert pip["n"] == 1  # upgraded exactly once
    assert smoke.calls["n"] == 2  # tested, then re-tested


def test_mismatch_autosync_disabled_reports_fix(monkeypatch):
    monkeypatch.setattr(sdk_health, "smoke_test", _seq_smoke([(False, "Claude Code returned an error result")]))
    pip = {"n": 0}
    monkeypatch.setattr(sdk_health, "_pip_upgrade", lambda: pip.__setitem__("n", pip["n"] + 1))
    r = sdk_health.ensure_compatible(autosync=False, log=lambda *a: None)
    assert not r["ok"] and not r["healed"]
    assert pip["n"] == 0
    assert "pip install -U claude-agent-sdk" in r["reason"]


def test_non_mismatch_failure_does_not_pip(monkeypatch):
    # a network/creds failure must NOT trigger a pip upgrade
    monkeypatch.setattr(sdk_health, "smoke_test", _seq_smoke([(False, "Connection refused / no network")]))
    pip = {"n": 0}
    monkeypatch.setattr(sdk_health, "_pip_upgrade", lambda: pip.__setitem__("n", pip["n"] + 1))
    r = sdk_health.ensure_compatible(autosync=True, log=lambda *a: None)
    assert not r["ok"] and not r["healed"]
    assert pip["n"] == 0


def test_still_failing_after_upgrade(monkeypatch):
    bad = "Claude Code returned an error result"
    monkeypatch.setattr(sdk_health, "smoke_test", _seq_smoke([(False, bad), (False, bad)]))
    monkeypatch.setattr(sdk_health, "_pip_upgrade", lambda: None)
    r = sdk_health.ensure_compatible(autosync=True, log=lambda *a: None)
    assert not r["ok"] and r["healed"]  # tried to heal, still broken -> caller fails fast


def test_model_error_is_reported_not_pip_upgraded(monkeypatch):
    # a 404 / inaccessible-model error must be reported as a CONFIG problem, NOT pip-upgraded
    detail = "SMOKE_ERR status=404 result=There's an issue with the selected model (it may not exist)"
    monkeypatch.setattr(sdk_health, "smoke_test", _seq_smoke([(False, detail)]))
    pip = {"n": 0}
    monkeypatch.setattr(sdk_health, "_pip_upgrade", lambda: pip.__setitem__("n", pip["n"] + 1))
    r = sdk_health.ensure_compatible(autosync=True, log=lambda *a: None)
    assert not r["ok"] and not r["healed"]
    assert pip["n"] == 0  # never pip-upgrades over a model/config error
    assert "model name" in r["reason"].lower()


def test_is_mismatch_vs_model_error():
    # genuine version/control-protocol problems -> mismatch (pip may help)
    assert sdk_health.is_mismatch("Claude Code returned an error result")
    assert sdk_health.is_mismatch('{"type": "error"}')
    # model 404 -> NOT a mismatch (don't pip), classified as a model error instead
    assert not sdk_health.is_mismatch("SMOKE_ERR status=404 result=model may not exist")
    assert sdk_health.is_model_error("SMOKE_ERR status=404 result=you may not have access")
    assert not sdk_health.is_mismatch("Connection refused")
    assert not sdk_health.is_mismatch("")
