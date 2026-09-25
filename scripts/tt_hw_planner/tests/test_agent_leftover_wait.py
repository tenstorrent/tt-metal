# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""An agent's device run outlives its turn; the runner waits for it instead of moving on.

2026-09-25, WH Galaxy: the emit-e2e builder launched the full e2e gate with `nohup pytest ... &`, said
"I'll check back", and ended its turn. `claude -p` exits at the end of a turn, the runner went straight
to the next phase, and the device reclaim killed the run 18 minutes into its forward pass. The night
before, a run killed the same way mid-collective wedged the board.

The fake agents here do exactly that: start a detached child and exit at once. "Holding the device" is
stood in for by patching device_holders() to report those children, so no test needs hardware.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

import pytest

from scripts.tt_hw_planner import cc_harness as H
from models.experimental.perf_automation.agent import device_recovery as DR


def _pids_running(marker: str) -> set:
    out = set()
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                if marker in open("/proc/%s/cmdline" % pid).read().replace("\0", " "):
                    out.add(int(pid))
            except OSError:
                pass
    return out


@pytest.fixture
def child_seconds(tmp_path):
    """A unique `sleep` duration per test, so the 'device holders' are only this test's children."""
    return 2 + (time.time_ns() % 997) / 1000.0


@pytest.fixture
def holders_are(monkeypatch):
    """device_holders() reports every live `sleep <seconds>` process: the stand-in for /dev/tenstorrent."""

    def _set(seconds):
        monkeypatch.setattr(DR, "device_holders", lambda: _pids_running("sleep %s" % seconds))

    return _set


def _detached(seconds, env):
    """A detached device run, as the agent's `nohup ... &` leaves it (own session, reparented)."""
    return subprocess.Popen(["sleep", str(seconds)], env=env, start_new_session=True)


def test_the_tag_is_unique_and_keeps_the_environment():
    env1, tag1 = H.tag_agent_env({"A": "1"})
    env2, tag2 = H.tag_agent_env({"A": "1"})
    assert tag1 != tag2
    assert env1["A"] == "1" and env1[H.AGENT_RUN_ENV] == tag1
    env3, tag3 = H.tag_agent_env()
    assert env3[H.AGENT_RUN_ENV] == tag3 and env3.get("PATH") == os.environ.get("PATH")


def test_a_detached_child_still_carries_the_tag():
    env, tag = H.tag_agent_env()
    p = _detached(5, env)
    other = _detached(5, dict(os.environ))
    try:
        assert H._carries_tag(p.pid, tag)
        assert not H._carries_tag(other.pid, tag)
        assert not H._carries_tag(10**9, tag)  # a pid that does not exist
    finally:
        p.kill(), other.kill()


def test_the_wait_lasts_until_the_agents_device_run_finishes(holders_are, child_seconds):
    holders_are(child_seconds)
    env, tag = H.tag_agent_env()
    p = _detached(child_seconds, env)
    t0 = time.monotonic()
    assert H.wait_for_agent_device_work(tag, timeout_s=60, poll_s=0.2) == []
    assert time.monotonic() - t0 >= child_seconds - 0.5
    assert p.poll() is not None, "returned while the run was still going"


def test_someone_elses_device_run_is_not_waited_for(holders_are, child_seconds):
    holders_are(child_seconds)
    _, tag = H.tag_agent_env()
    p = _detached(child_seconds, dict(os.environ))  # a device user this agent did not start
    try:
        t0 = time.monotonic()
        assert H.wait_for_agent_device_work(tag, timeout_s=60, poll_s=0.2) == []
        assert time.monotonic() - t0 < 1
    finally:
        p.kill()


def test_the_wait_is_bounded(holders_are, child_seconds):
    holders_are(child_seconds + 30)
    env, tag = H.tag_agent_env()
    p = _detached(child_seconds + 30, env)
    try:
        t0 = time.monotonic()
        assert H.wait_for_agent_device_work(tag, timeout_s=1, poll_s=0.2) == [p.pid]
        assert time.monotonic() - t0 < 5
    finally:
        p.kill()


def test_the_bound_is_configurable(monkeypatch):
    monkeypatch.setenv("TT_HW_PLANNER_AGENT_LEFTOVER_WAIT_S", "12")
    assert H._leftover_wait_s() == 12
    monkeypatch.setenv("TT_HW_PLANNER_AGENT_LEFTOVER_WAIT_S", "not a number")
    assert H._leftover_wait_s() == H._DEFAULT_LEFTOVER_WAIT_S


def test_a_scan_that_cannot_run_skips_the_wait_instead_of_raising(monkeypatch):
    def _boom():
        raise RuntimeError("fuser missing")

    monkeypatch.setattr(DR, "device_holders", _boom)
    assert H.wait_for_agent_device_work("any", timeout_s=60, poll_s=0.1) == []


def test_the_reaper_uses_the_same_scan(monkeypatch, child_seconds):
    p = _detached(child_seconds + 30, dict(os.environ))
    monkeypatch.setattr(DR, "device_holders", lambda: {p.pid})
    assert DR.reap_device_holders() == [p.pid]
    p.wait(timeout=10)


def _fake_agent(tmp_path, seconds):
    """An 'agent' that leaves a detached device run behind and exits immediately."""
    script = tmp_path / "fake_agent.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import subprocess, sys
            subprocess.Popen(["sleep", "{seconds}"], start_new_session=True)
            print('{{"type": "result", "result": "started it; I will check back later"}}', flush=True)
            sys.exit(0)
            """
        )
    )
    wrapper = tmp_path / "fake_agent"
    wrapper.write_text("#!/bin/sh\nexec %s %s\n" % (sys.executable, script))
    wrapper.chmod(0o755)
    return str(wrapper)


def test_the_cc_loop_does_not_run_the_next_gate_while_the_agent_run_is_going(
    tmp_path, holders_are, child_seconds, monkeypatch
):
    holders_are(child_seconds)
    monkeypatch.setenv("TT_HW_PLANNER_AGENT_LEFTOVER_WAIT_S", "60")
    monkeypatch.setattr(H, "_LEFTOVER_POLL_S", 0.2)
    gate_calls = []

    def gate_fn():
        gate_calls.append(_pids_running("sleep %s" % child_seconds))
        return {"can_stop": len(gate_calls) > 1}

    res = H.run_cc_loop(
        prompt="p",
        mcp_config_path=tmp_path / "mcp.json",
        allowed_tools=[],
        cwd=tmp_path,
        env=dict(os.environ),
        gate_fn=gate_fn,
        max_rounds=3,
        claude_bin=_fake_agent(tmp_path, child_seconds),
    )
    assert res["can_stop"] and res["rounds"] == 1
    assert gate_calls[1] == set(), "the next gate ran while the agent's device run was still going"


def test_the_emit_e2e_builder_does_not_return_while_its_run_is_going(tmp_path, holders_are, child_seconds, monkeypatch):
    from scripts.tt_hw_planner.commands import emit_e2e as E

    holders_are(child_seconds)
    monkeypatch.setattr(H, "_LEFTOVER_POLL_S", 0.2)
    monkeypatch.chdir(tmp_path)
    rc, final = E._run_agent(prompt="p", agent_bin=_fake_agent(tmp_path, child_seconds), agent_model="m", timeout_s=60)
    assert rc == 0 and "check back" in final
    assert _pids_running("sleep %s" % child_seconds) == set(), "returned while the builder's run was still going"
