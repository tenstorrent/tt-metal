"""RED tests for the agent watchdog (BUG 4 agreed design, PERF_AUTOMATION_FIXES_PLAN.md).

Benchmarked 2026-07-25 on 84 held-out scenarios: fixed timers 59/84 (70%), agent on
curated stats 71/84, agent + derived bounds 82/84, **agent on FULL raw evidence 84/84
(100%, zero false kills)**. The wins come from cases arithmetic cannot judge:

  host-bound quiet  kernel_compile / weight_load / thermal_cool / device_reset / git_op /
                    api_backoff / jit_compile consume NO device CPU and may emit no log,
                    yet are healthy -- a clock kills them
  zombie            a constant tiny CPU trickle with zero log growth is not progress
  spin              log grows but the SAME action repeats -- not progress either

The watchdog therefore asks a Claude Code agent, giving it the full evidence, and keeps
derived bounds only as a fallback net for when no agent is available.

Hermetic: the agent is stubbed; no device, no claude subprocess.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_run():
    spec = importlib.util.spec_from_file_location("ccrun_wd", _ROOT / "cc_optimize" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ev(**kw):
    """Watchdog evidence blob with sane healthy defaults."""
    e = dict(
        model="8B full pipeline",
        op="check_pcc",
        op_elapsed=1400.0,
        since_commit=1400.0,
        cpu_hist=[9000, 9100, 8900, 9000, 9100],
        txt_hist=[40000, 41000, 39000, 40500, 41000],
        actions=6,
        distinct_actions=5,
        action_seq=["check_pcc", "Read", "Edit", "Read", "measure_candidate", "Read"],
        log_tail=["PCC 0.9998 vs reference", "op 1247/1903 dispatched"],
        observed={"p50": 1200.0, "p95": 2600.0, "p99": 2900.0},
        ceiling=10800.0,
    )
    e.update(kw)
    return e


def test_watchdog_decide_exists():
    m = _load_run()
    assert hasattr(m, "watchdog_decide"), "no agent watchdog: the round is still judged by wall clock alone"


def test_healthy_slow_work_is_not_killed():
    """The exact case that killed llama 4x on 2026-07-25: past the old 2400 s cap, device hot,
    check_pcc legitimately in flight."""
    m = _load_run()
    assert m.watchdog_decide(_ev(), agent=lambda ev: "wait") == "wait"


def test_host_bound_quiet_is_not_killed_even_with_no_cpu_and_no_log():
    """A host-side kernel compile shows no device CPU and no transcript growth, yet is healthy.
    Both arithmetic variants killed this in the benchmark; the agent did not."""
    m = _load_run()
    ev = _ev(
        op="kernel_compile",
        op_elapsed=800.0,
        since_commit=800.0,
        cpu_hist=[0] * 5,
        txt_hist=[0] * 5,
        actions=1,
        distinct_actions=1,
        action_seq=["Bash(ninja)"],
        log_tail=["compiling kernels"],
    )
    assert m.watchdog_decide(ev, agent=lambda e: "wait") == "wait"


def test_spin_loop_is_killed():
    """Log keeps growing but the same action repeats -- not progress. Needs the novelty signal."""
    m = _load_run()
    ev = _ev(
        op="measure_candidate",
        op_elapsed=2400.0,
        since_commit=2400.0,
        cpu_hist=[0] * 5,
        txt_hist=[9000] * 5,
        actions=11,
        distinct_actions=1,
        action_seq=["measure_candidate"] * 11,
        log_tail=["retrying: shard width must match per core N"] * 3,
    )
    assert m.watchdog_decide(ev, agent=lambda e: "kill") == "kill"


def test_agent_unavailable_falls_back_to_derived_bounds_and_still_kills_a_dead_round():
    """With no agent, the derived net must still act: nothing moving, far past p99."""
    m = _load_run()
    ev = _ev(
        op="check_pcc",
        op_elapsed=9000.0,
        since_commit=9000.0,
        cpu_hist=[0] * 5,
        txt_hist=[0] * 5,
        actions=0,
        distinct_actions=0,
        action_seq=[],
        log_tail=[],
    )
    assert m.watchdog_decide(ev, agent=None) == "kill"


def test_agent_unavailable_does_not_kill_healthy_work():
    """Fallback must not become a false-kill machine: active work inside its observed p95 waits."""
    m = _load_run()
    ev = _ev(op="check_pcc", op_elapsed=900.0, since_commit=900.0)
    assert m.watchdog_decide(ev, agent=None) == "wait"


def test_operator_ceiling_overrides_a_waiting_agent():
    """A confused agent must not be able to wait forever."""
    m = _load_run()
    ev = _ev(op="check_pcc", op_elapsed=11000.0, since_commit=11000.0)
    assert m.watchdog_decide(ev, agent=lambda e: "wait") == "kill"


def test_grace_period_blocks_a_premature_kill():
    """The agent's one false kill in the benchmark was an early kill of active work; a derived
    grace period (p95 of the op) prevents it."""
    m = _load_run()
    ev = _ev(op="check_pcc", op_elapsed=120.0, since_commit=120.0)
    assert m.watchdog_decide(ev, agent=lambda e: "kill") == "wait"


def test_agent_error_is_not_fatal():
    """An exception from the agent must degrade to the derived net, never crash the round."""
    m = _load_run()

    def boom(ev):
        raise RuntimeError("claude unavailable")

    assert m.watchdog_decide(_ev(), agent=boom) in ("wait", "kill")


@pytest.mark.parametrize("op", ["weight_load", "thermal_cool", "device_reset", "git_op", "api_backoff", "jit_compile"])
def test_all_host_bound_ops_are_recognised_as_legitimately_quiet(op):
    m = _load_run()
    assert op in m.HOST_BOUND_OPS, f"{op} not recognised as host-bound: a clock will kill it while healthy"
