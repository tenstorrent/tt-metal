# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""cc_harness.run_cc_loop agent watchdog: a wedged `claude -p` (no round completion) must be killed
and, after max_consecutive_timeouts, abandoned (halted) — so bring-up self-recovers from an agent
hang the way the perf_automation loop already does. A normal fast agent must be unaffected."""
import os
import stat
import time

from scripts.tt_hw_planner import cc_harness
from scripts.tt_hw_planner.cc_harness import _resolve_agent_timeout_s, run_cc_loop


def _script(tmp_path, name, body):
    p = tmp_path / name
    p.write_text("#!/bin/sh\n" + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(p)


def test_resolve_timeout_env_and_defaults(monkeypatch):
    monkeypatch.delenv("TT_HW_PLANNER_CC_AGENT_TIMEOUT_S", raising=False)
    assert _resolve_agent_timeout_s(None) == 3600
    assert _resolve_agent_timeout_s(5) == 5
    assert _resolve_agent_timeout_s(0) is None
    assert _resolve_agent_timeout_s(-1) is None
    monkeypatch.setenv("TT_HW_PLANNER_CC_AGENT_TIMEOUT_S", "42")
    assert _resolve_agent_timeout_s(None) == 42


def test_wedged_agent_is_killed_and_abandoned(tmp_path):
    hang = _script(tmp_path, "claude_hang.sh", "sleep 30")
    started = time.monotonic()
    res = run_cc_loop(
        prompt="x",
        mcp_config_path=str(tmp_path / "cfg.json"),
        allowed_tools=["Read"],
        cwd=str(tmp_path),
        env=dict(os.environ),
        gate_fn=lambda: {},
        claude_bin=hang,
        agent_timeout_s=1,
        max_consecutive_timeouts=2,
    )
    elapsed = time.monotonic() - started
    assert res["halted"] is True
    assert res["can_stop"] is False
    assert res["rounds"] == 0
    assert elapsed < 20, f"watchdog didn't kill promptly (took {elapsed:.1f}s — waited on the 30s hangs)"


def test_fast_agent_completes_normally(tmp_path):
    fast = _script(tmp_path, "claude_fast.sh", "true")
    calls = {"n": 0}

    def gate():
        calls["n"] += 1
        return {"can_stop": True} if calls["n"] > 1 else {}

    res = run_cc_loop(
        prompt="x",
        mcp_config_path=str(tmp_path / "cfg.json"),
        allowed_tools=["Read"],
        cwd=str(tmp_path),
        env=dict(os.environ),
        gate_fn=gate,
        claude_bin=fast,
        agent_timeout_s=30,
    )
    assert res["can_stop"] is True
    assert res["halted"] is False
    assert res["rounds"] == 1


def test_transient_hang_then_recover_resets_counter(tmp_path):
    """One slow round (killed) then the gate says can_stop — must NOT abandon (counter only trips on
    CONSECUTIVE wedges), and the single kill shouldn't prevent a subsequent clean stop."""
    hang = _script(tmp_path, "claude_hang2.sh", "sleep 30")
    calls = {"n": 0}

    def gate():
        calls["n"] += 1
        return {"can_stop": True} if calls["n"] > 1 else {}

    res = run_cc_loop(
        prompt="x",
        mcp_config_path=str(tmp_path / "cfg.json"),
        allowed_tools=["Read"],
        cwd=str(tmp_path),
        env=dict(os.environ),
        gate_fn=gate,
        claude_bin=hang,
        agent_timeout_s=1,
        max_consecutive_timeouts=3,
    )
    assert res["can_stop"] is True and res["halted"] is False
