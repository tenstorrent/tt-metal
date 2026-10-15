# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Where a run's time went (F70): the model, device runs, other tools, gates, and idle time.

    python -m models.demos.common.bringup.timing --spec S [--run run1] [--top 15]

Sources, all written during the run:
- the agents' transcripts (runs/<run>/agents/*.jsonl): the result event's wall and API time (duration_ms,
  duration_api_ms), and every tool call's time (the tool_use message's timestamp to its tool_result's). A Bash call
  through scripts/run_safe_pytest.sh or scripts/tt-probe.sh is a device run; any other tool call is "other tools";
  the rest of an agent's wall time (between messages) is its own overhead;
- state.json: every gate run's seconds (history entries carry `s` since F70; older runs have only the last run's
  duration_s), split device / CPU by the task's `device` flag in tasks.yaml;
- runs/<run>/timeline.jsonl (F70): the orchestrator's start and exit (with its exit code) and every agent and gate
  start and end. Idle time is the time between an exit and the next start, named by why it exited: waiting for a
  person (an approval or a decision), stopped (a task out of attempts), paused (by the overseer), or not running (a
  crash or a kill: no exit record);
- runs/<run>/device_timing.jsonl (scripts/lib/tt_safe_run.sh, TT_DEVICE_TIMING_LOG, set by the orchestrator since
  F70): per device run, the wait for the device lock and the run itself.
Writes runs/<run>/timing.json (outside the tree: it never counts against an agent) and prints the summary and the slowest tasks.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DEVICE_CMD = re.compile(r"run_safe_pytest\.sh|tt-probe\.sh")
TIMELINE = "timeline.jsonl"
DEVICE_LOG = "device_timing.jsonl"
EXIT_WHY = {0: "done", 1: "stopped", 3: "waiting for a person", 4: "paused"}


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def local_ts(s: str) -> float:
    """state.json / timeline times are local time without a zone."""
    return datetime.fromisoformat(s).timestamp()


def record(run_dir: Path, event: str, **kw) -> None:
    """Append one event to the run's timeline (the orchestrator calls this)."""
    line = {"t": datetime.now().isoformat(timespec="seconds"), "event": event, **kw}
    with open(Path(run_dir) / TIMELINE, "a") as f:
        f.write(json.dumps(line) + "\n")


def agent_times(path: Path) -> dict:
    """wall, api, device, tools (seconds) of one agent transcript; task / role / attempt from its file name."""
    pending, device, tools, first, last, wall, api = {}, 0.0, 0.0, None, None, None, None
    for line in path.read_text(errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = ts(ev["timestamp"]) if ev.get("timestamp") else None
        if t is not None:
            first, last = first or t, t
        if ev.get("type") == "assistant":
            for c in (ev.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_use" and t is not None:
                    pending[c.get("id")] = (t, json.dumps(c.get("input") or {}))
        elif ev.get("type") == "user":
            for c in (ev.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_result" and c.get("tool_use_id") in pending:
                    t0, inp = pending.pop(c["tool_use_id"])
                    dt = max(0.0, (t or t0) - t0)
                    if DEVICE_CMD.search(inp):
                        device += dt
                    else:
                        tools += dt
        elif ev.get("type") == "result":
            wall = (ev.get("duration_ms") or 0) / 1000
            api = (ev.get("duration_api_ms") or 0) / 1000
    if wall is None:  # killed before its result event
        wall = (last - first) if first else 0.0
    parts = path.name[: -len(".jsonl")].rsplit(".", 2)
    task, role, attempt = (parts + ["", ""])[:3] if len(parts) == 3 else (path.stem, "", "")
    return {
        "task": task,
        "role": role,
        "attempt": attempt,
        "wall": wall,
        "api": api or 0.0,
        "device": device,
        "tools": tools,
    }


def gate_times(state: dict, tasks: dict) -> list[dict]:
    out = []
    for tid, e in state.items():
        if tid.startswith("_") or not isinstance(e, dict):
            continue
        runs = [h["s"] for h in e.get("history") or [] if isinstance(h, dict) and "s" in h]
        secs = sum(runs) if runs else float(e.get("duration_s") or 0)
        out.append(
            {
                "task": tid,
                "seconds": secs,
                "runs": len(runs) or (1 if secs else 0),
                "device": bool(tasks.get(tid, {}).get("device")),
            }
        )
    return out


def idle_times(timeline: list[dict], now: float) -> dict:
    """Seconds between an orchestrator exit and the next start, by why it exited."""
    out, last_exit, last_t = defaultdict(float), None, None
    for ev in timeline:
        t = local_ts(ev["t"])
        if ev["event"] == "start":
            if last_exit is not None:
                out[last_exit[1]] += t - last_exit[0]
            elif last_t is not None:  # no exit record: the previous orchestrator was killed or crashed
                out["not running"] += t - last_t
            last_exit = None
        elif ev["event"] == "exit":
            last_exit = (t, EXIT_WHY.get(ev.get("code"), "not running"))
        last_t = t
    if last_exit is not None:
        out[last_exit[1] + " (now)"] += now - last_exit[0]
    return dict(out)


def device_log(path: Path) -> dict:
    if not path.exists():
        return {}
    wait = run = n = 0
    for line in path.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        wait += d.get("wait_ms", 0)
        run += d.get("run_ms", 0)
        n += 1
    return {"runs": n, "lock_wait": wait / 1000, "run": run / 1000}


def breakdown(bringup_dir: Path, run_dir: Path, tasks: dict, now: float | None = None) -> dict:
    now = now or datetime.now().timestamp()
    state = json.loads((Path(bringup_dir) / "state.json").read_text())
    agents = [agent_times(p) for p in sorted((Path(run_dir) / "agents").glob("*.jsonl"))]
    gates = gate_times(state, tasks)
    tl_path = Path(run_dir) / TIMELINE
    timeline = [json.loads(x) for x in tl_path.read_text().splitlines() if x.strip()] if tl_path.exists() else []
    created = (state.get("_run") or {}).get("created")
    elapsed = now - local_ts(created) if created else None
    tot = {
        "model (LLM API)": sum(a["api"] for a in agents),
        "agent device runs": sum(a["device"] for a in agents),
        "agent other tools": sum(a["tools"] for a in agents),
        "agent overhead": sum(max(0.0, a["wall"] - a["api"] - a["device"] - a["tools"]) for a in agents),
        "gates on device": sum(g["seconds"] for g in gates if g["device"]),
        "gates on CPU": sum(g["seconds"] for g in gates if not g["device"]),
    }
    idle = idle_times(timeline, now)
    measured = sum(tot.values()) + sum(idle.values())
    if elapsed is not None:
        tot.update({f"idle: {k}": v for k, v in idle.items()})
        tot["unaccounted (before the timeline, or orchestrator overhead)"] = max(0.0, elapsed - measured)
    per = defaultdict(lambda: defaultdict(float))
    for a in agents:
        p = per[a["task"]]
        p["model"] += a["api"]
        p["agent device"] += a["device"]
        p["agent tools"] += a["tools"]
        p["agent runs"] += 1
    for g in gates:
        per[g["task"]]["gate device" if g["device"] else "gate CPU"] += g["seconds"]
    return {
        "elapsed": elapsed,
        "totals": tot,
        "device_log": device_log(Path(run_dir) / DEVICE_LOG),
        "per_task": {k: dict(v) for k, v in per.items()},
        "agents": agents,
    }


def table(res: dict, top: int = 15) -> str:
    h = lambda s: f"{s / 3600:6.2f} h"  # noqa: E731
    lines = [f"elapsed {h(res['elapsed'])}" if res["elapsed"] is not None else "elapsed: unknown"]
    lines += [f"  {k:62s} {h(v)}" for k, v in res["totals"].items()]
    d = res["device_log"]
    if d:
        lines.append(
            f"  device runs logged: {d['runs']}, run {h(d['run'])}, waiting for the device lock {h(d['lock_wait'])}"
        )
    rows = sorted(res["per_task"].items(), key=lambda kv: -sum(v for k, v in kv[1].items() if k != "agent runs"))
    lines.append(
        f"slowest tasks (minutes): {'task':22s} {'model':>7} {'a.dev':>7} {'a.tools':>7} {'g.dev':>7} {'g.cpu':>7}"
    )
    for tid, v in rows[:top]:
        lines.append(
            f"  {tid:46s} {v.get('model', 0) / 60:7.1f} {v.get('agent device', 0) / 60:7.1f} "
            f"{v.get('agent tools', 0) / 60:7.1f} {v.get('gate device', 0) / 60:7.1f} {v.get('gate CPU', 0) / 60:7.1f}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    import yaml

    from models.demos.common.bringup.core.gate import run_name
    from models.demos.common.bringup.core.ledger import Ledger
    from models.demos.common.bringup.core.spec import Spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--run", help="run name (default: the current run)")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args(argv)
    spec = Spec.load(a.spec)
    run_dir = spec.run_dir(a.run or run_name(Ledger(spec.bringup_dir)))
    tasks = {t["id"]: t for t in (yaml.safe_load((spec.bringup_dir / "tasks.yaml").read_text()) or {}).get("tasks", [])}
    res = breakdown(spec.bringup_dir, run_dir, tasks)
    out = run_dir / "timing.json"
    out.write_text(json.dumps({k: v for k, v in res.items() if k != "agents"}, indent=1) + "\n")
    print(table(res, a.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
