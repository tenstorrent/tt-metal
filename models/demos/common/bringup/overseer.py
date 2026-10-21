# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Overseer notes and aborts (F69): how the supervising session steers agents without touching their work.

The overseer does not edit the model's code, knowledge files or run state. It writes notes; the agents act on them.

    <bringup_dir>/overseer.yaml (the overseer's; the agent path check ignores it)
    notes:
      - {id: OV-1, tasks: ["C.*"], roles: [implement], status: open, t: ..., text: "what is wrong, what to do"}

Every open note whose `tasks` globs match the step (and `roles`, when given) heads that step's brief. The agent answers
each in its knowledge-use report under `overseer`: [{"id": "OV-1", "answer": "done" | "n/a" | "disagree", "evidence":
"..."}]. The orchestrator echoes a disagreement or a missing answer (OVERSEER NOTE ...), which wakes the overseer's
watch. A note stays open until the overseer closes it.

`abort` stops the agent at work: it records the reason, adds it as a note for that task, and ends the agent's process
group. The orchestrator then counts the attempt as aborted (not against the task's budget, no gate) and starts the
next attempt with the note in its brief. A running agent cannot take messages; abort plus a note is how one reaches it.

    python -m models.demos.common.bringup.overseer note --spec S [--tasks GLOB ...] [--roles ROLE ...] "text"
    python -m models.demos.common.bringup.overseer close --spec S OV-1 [--why "..."]
    python -m models.demos.common.bringup.overseer list --spec S
    python -m models.demos.common.bringup.overseer abort --spec S --reason "..." [--tasks GLOB ...]
    python -m models.demos.common.bringup.overseer fix --spec S --gate TASK [--gate TASK ...] [--title T] [--hold WHY] "text"
    python -m models.demos.common.bringup.overseer release --spec S FIX.n
    python -m models.demos.common.bringup.overseer status --spec S ["why the run is stopped"]   (empty: clear)

`fix` (F77) queues a change for an agent to make now, when no failing test would start one (the code passes today:
a per-chunk weight rebuild, a precision raise a fixed limit forced). At its next task boundary the orchestrator turns
it into task FIX.<n> (step overseer_fix, role overseer: a bring-up engineer, the normal attempts) placed before the
first unpassed task. Its gate reruns the gates of the --gate tasks (the passed steps whose code the change touches;
their frozen tests must all still pass) and its paths are theirs. Fix tasks keep the plan approval; a fix that stops
stops the run. A fix that must wait for something (a framework change) is queued at once with --hold: it is on record
(`list`, the dashboard) and is placed only after `release`. A decision is queued when it is made, never only planned.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import signal
import sys
import time
from pathlib import Path

import yaml

FILE = "overseer.yaml"
AGENT_FILE = "AGENT.json"  # in the run dir while an agent works: {pid, pgid, task, role, attempt}
ABORT_FILE = "ABORT.json"  # in the run dir: the overseer's abort of the agent at work
ANSWERS = ("done", "n/a", "disagree")
MAX_ABORTS = 3  # per task: after this many, an abort stops the task for a person instead of retrying


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def path(bringup_dir: Path) -> Path:
    return Path(bringup_dir) / FILE


def _read(bringup_dir: Path) -> dict:
    p = path(bringup_dir)
    return (yaml.safe_load(p.read_text()) if p.exists() else None) or {}


def _write(bringup_dir: Path, data: dict) -> None:
    head = "# Overseer notes (models/demos/common/bringup/overseer.py): written by the overseer, read by every agent.\n"
    keep = {k: v for k, v in data.items() if v}
    path(bringup_dir).write_text(head + yaml.safe_dump({"notes": [], **keep}, sort_keys=False, width=118))


def load(bringup_dir: Path) -> list[dict]:
    return list(_read(bringup_dir).get("notes") or [])


def save(bringup_dir: Path, notes: list[dict]) -> None:
    _write(bringup_dir, {**_read(bringup_dir), "notes": notes})


# ---- fix jobs (F77): a change an agent makes now, gated by the passed steps it touches
FIX_STEP, FIX_ROLE = "overseer_fix", "overseer"


def fixes(bringup_dir: Path) -> list[dict]:
    return list(_read(bringup_dir).get("fixes") or [])


def queue_fix(
    bringup_dir: Path,
    text: str,
    gate_tasks: list[str],
    title: str = "",
    tasks: dict | None = None,
    hold: str = "",
    paths: list[str] | None = None,
) -> dict:
    """Queue a fix; ``tasks`` (the ledger's tasks by id), when given, validates the gate tasks."""
    if not gate_tasks:
        raise SystemExit("a fix needs at least one --gate task: the passed steps whose frozen tests must still pass")
    missing = [t for t in gate_tasks if tasks is not None and t not in tasks]
    if missing:
        raise SystemExit(f"no such tasks: {missing}")
    data = _read(bringup_dir)
    have = [int(str(x["id"]).split(".")[-1]) for x in data.get("fixes") or []]
    have += [int(t.split(".")[-1]) for t in (tasks or {}) if t.startswith("FIX.") and t.split(".")[-1].isdigit()]
    job = {
        "id": f"FIX.{max(have + [0]) + 1}",
        "title": title or text.strip().split(". ")[0][:100],
        "gate": list(gate_tasks),
        "status": "held" if hold else "queued",
        "t": now(),
        "text": text.strip(),
        **({"hold": hold} if hold else {}),
        **({"paths": list(paths)} if paths else {}),  # F127: where the fix may write, beyond its gate tasks' paths
    }
    data["fixes"] = list(data.get("fixes") or []) + [job]
    _write(bringup_dir, data)
    return job


def release(bringup_dir: Path, fix_id: str) -> dict:
    data = _read(bringup_dir)
    for x in data.get("fixes") or []:
        if x["id"] == fix_id:
            if x.get("status") != "held":
                raise SystemExit(f"{fix_id} is {x.get('status')}, not held")
            x.update(status="queued", released=now())
            _write(bringup_dir, data)
            return x
    raise SystemExit(f"no fix {fix_id} in {path(bringup_dir)}")


def fix_task(job: dict, tasks: dict) -> dict:
    """The ledger task of a queued fix: the --gate tasks' gate commands chained, their metrics and paths merged."""
    gates = [tasks[t] for t in job["gate"]]
    metrics, paths = {}, []
    for g in gates:
        metrics.update(g["gate"].get("metrics") or {})
        paths += [p for p in g.get("paths") or [] if p not in paths]
    paths += [p for p in job.get("paths") or [] if p not in paths]
    return {
        "id": job["id"],
        "title": job["title"],
        "step": FIX_STEP,
        "role": FIX_ROLE,
        "precheck": False,
        "deps": list(job["gate"]),
        # F79: every segment runs as its own task (its tests see the BRINGUP_TASK of their own gate: layer choice,
        # golden, limits), and the gate runner's BRINGUP_RECORD_TASK still files the metrics under FIX.<n>
        "gate": {"cmd": " && ".join(_as_task(g) for g in gates), "metrics": metrics},
        "device": any(g.get("device") for g in gates),
        "paths": paths,
        "brief": {"details": job["text"], "gate_tasks": list(job["gate"])},
    }


def _as_task(task: dict) -> str:
    segs = [x.strip() for x in task["gate"]["cmd"].split("&&") if x.strip()]
    return " && ".join(f"BRINGUP_TASK={task['id']} {x}" for x in segs)


def take_fixes(bringup_dir: Path) -> list[dict]:
    """The queued fixes, marked as placed (the orchestrator puts them in the ledger)."""
    data = _read(bringup_dir)
    out = [x for x in data.get("fixes") or [] if x.get("status") == "queued"]
    for x in out:
        x.update(status="placed", placed=now())
    if out:
        _write(bringup_dir, data)
    return out


def _globs(v) -> list[str]:
    return [v] if isinstance(v, str) else list(v or ["*"])


def add(bringup_dir: Path, text: str, tasks=None, roles=None) -> dict:
    notes = load(bringup_dir)
    n = max([int(str(x.get("id", "OV-0")).split("-")[-1]) for x in notes] + [0]) + 1
    note = {"id": f"OV-{n}", "tasks": _globs(tasks), "status": "open", "t": now(), "text": text.strip()}
    if roles:
        note["roles"] = list(roles)
    save(bringup_dir, notes + [note])
    return note


def close(bringup_dir: Path, note_id: str, why: str = "") -> dict:
    notes = load(bringup_dir)
    for x in notes:
        if x.get("id") == note_id:
            x.update(status="closed", closed=now(), **({"why_closed": why} if why else {}))
            save(bringup_dir, notes)
            return x
    raise SystemExit(f"no note {note_id} in {path(bringup_dir)}")


def for_task(bringup_dir: Path, tid: str, role: str) -> list[dict]:
    """The open notes a step of task `tid` in role `role` must answer."""
    return [
        x
        for x in load(bringup_dir)
        if x.get("status", "open") == "open"
        and any(fnmatch.fnmatchcase(tid, g) for g in _globs(x.get("tasks")))
        and (not x.get("roles") or role in x["roles"])
    ]


def brief_text(notes: list[dict]) -> str:
    if not notes:
        return ""
    body = "\n".join(f"- **{x['id']}** ({x['t']}): {x['text']}" for x in notes)
    return (
        "## Overseer notes (read first; answer every one)\n"
        "The overseer reviews the run and writes these for the steps they name. They come before the plan, the "
        "knowledge files and your own reading, but never before the owner's rules. Act on each, and answer each in "
        'your knowledge-use report under `"overseer"`: `{"id", "answer": "done" | "n/a" | "disagree", "evidence"}`. '
        "Disagree when the evidence (a test, a probe, the reference) says the note is wrong: say why; the overseer "
        "reads it.\n" + body + "\n"
    )


def answers(report: dict, shown: list[dict]) -> list[str]:
    """The echo lines for the notes a step was shown: unanswered, malformed, or disagreed with."""
    got = {a.get("id"): a for a in report.get("overseer") or [] if isinstance(a, dict)}
    out = []
    for x in shown:
        a = got.get(x["id"])
        if a is None:
            out.append(f"OVERSEER NOTE {x['id']} unanswered")
        elif a.get("answer") not in ANSWERS:
            out.append(f"OVERSEER NOTE {x['id']} answer {a.get('answer')!r} is not one of {ANSWERS}")
        elif a["answer"] == "disagree":
            out.append(f"OVERSEER NOTE {x['id']} DISAGREED: {a.get('evidence', '')}")
    return out


# ---- abort: the orchestrator records the agent at work; the overseer ends it
def agent_started(run_dir: Path, pid: int, task: str, role: str, attempt: int) -> None:
    (Path(run_dir) / AGENT_FILE).write_text(
        json.dumps({"pid": pid, "pgid": pid, "task": task, "role": role, "attempt": attempt, "t": now()})
    )


def agent_finished(run_dir: Path) -> None:
    (Path(run_dir) / AGENT_FILE).unlink(missing_ok=True)


def take_abort(run_dir: Path) -> dict | None:
    """The abort of the agent that just finished, if the overseer aborted it (consumed)."""
    p = Path(run_dir) / ABORT_FILE
    if not p.exists():
        return None
    rec = json.loads(p.read_text())
    p.unlink()
    return rec


def abort(bringup_dir: Path, run_dir: Path, reason: str, tasks=None, kill=os.killpg) -> dict:
    """Stop the agent at work: record the reason, add it as a note for its task, end its process group."""
    p = Path(run_dir) / AGENT_FILE
    if not p.exists():
        raise SystemExit("no agent is at work (no AGENT.json in the run dir); write a note instead")
    agent = json.loads(p.read_text())
    note = add(
        bringup_dir,
        f"The overseer stopped attempt {agent['attempt']} ({agent['role']}): {reason}",
        tasks or [agent["task"]],
    )
    (Path(run_dir) / ABORT_FILE).write_text(json.dumps({**agent, "reason": reason, "note": note["id"], "at": now()}))
    try:
        kill(agent["pgid"], signal.SIGTERM)
    except ProcessLookupError:
        pass
    return {**agent, "note": note["id"]}


STATUS_FILE = "OVERSEER_STATUS.json"  # in the run dir: why the overseer has the run stopped (the dashboard shows it)


def set_status(run_dir: Path, text: str) -> dict:
    rec = {"text": text.strip(), "t": now()}
    (Path(run_dir) / STATUS_FILE).write_text(json.dumps(rec))
    return rec


def status(run_dir: Path) -> dict | None:
    p = Path(run_dir) / STATUS_FILE
    return json.loads(p.read_text()) if p.exists() else None


def clear_status(run_dir: Path) -> None:
    (Path(run_dir) / STATUS_FILE).unlink(missing_ok=True)


def _dirs(spec_path: str):
    from models.demos.common.bringup.core.gate import run_name
    from models.demos.common.bringup.core.ledger import Ledger
    from models.demos.common.bringup.core.spec import Spec

    spec = Spec.load(spec_path)
    return spec.bringup_dir, spec.run_dir(run_name(Ledger(spec.bringup_dir)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("note", "close", "list", "abort", "fix", "release", "status"):
        p = sub.add_parser(name)
        p.add_argument("--spec", default=os.environ.get("BRINGUP_SPEC"))
        if name in ("note", "abort"):
            p.add_argument(
                "--tasks",
                action="append",
                help="a task id glob; repeat for more (default: every task / the aborted task)",
            )
        if name == "note":
            p.add_argument("--roles", action="append", help="a role; repeat for more")
            p.add_argument("text")
        if name == "close":
            p.add_argument("id")
            p.add_argument("--why", default="")
        if name == "abort":
            p.add_argument("--reason", required=True)
        if name == "fix":
            p.add_argument("--gate", action="append", required=True, help="a passed task whose gate must still pass")
            p.add_argument("--title", default="")
            p.add_argument("--hold", default="", help="queue it now, place it only after `release` (what it waits for)")
            p.add_argument(
                "--paths", action="append", default=None, help="where the fix may write (default: the model's tt/ dir)"
            )
            p.add_argument("text")
        if name == "release":
            p.add_argument("id")
        if name == "status":
            p.add_argument("text", nargs="?", default="", help="why the run is stopped; empty clears it")
    a = ap.parse_args(argv)
    bdir, rdir = _dirs(a.spec)
    if a.cmd == "note":
        x = add(bdir, a.text, a.tasks, a.roles)
        print(f"{x['id']} for {x['tasks']}: {x['text'][:100]}")
    elif a.cmd == "close":
        print(f"closed {close(bdir, a.id, a.why)['id']}")
    elif a.cmd == "list":
        for x in load(bdir):
            print(f"{x['id']} {x.get('status', 'open'):6} {x.get('tasks')} {x['text'][:100]}")
        for x in fixes(bdir):
            held = f" (held: {x['hold']})" if x.get("status") == "held" else ""
            print(f"{x['id']} {x.get('status'):6} gate {x.get('gate')}{held} {x['text'][:100]}")
    elif a.cmd == "fix":
        led_tasks = {t["id"]: t for t in (yaml.safe_load((bdir / "tasks.yaml").read_text()) or {}).get("tasks", [])}
        # F127: a fix changes the model's code even when its gate tasks own no paths (a scripted gate such as X.3)
        paths = a.paths or [str((bdir.parent / "tt").resolve().relative_to(Path.cwd().resolve()))]
        x = queue_fix(bdir, a.text, a.gate, a.title, led_tasks, a.hold, paths)
        when = f"held until `release` ({a.hold})" if a.hold else "the orchestrator starts it at its next task boundary"
        print(f"queued {x['id']} (gate {x['gate']}): {when}")
    elif a.cmd == "status":
        if a.text:
            print(f"status: {set_status(rdir, a.text)['text']}")
        else:
            clear_status(rdir)
            print("status cleared")
    elif a.cmd == "release":
        print(f"released {release(bdir, a.id)['id']}: placed at the orchestrator's next task boundary")
    else:
        r = abort(bdir, rdir, a.reason, a.tasks)
        print(f"aborted {r['task']} {r['role']} attempt {r['attempt']} (pid {r['pid']}); note {r['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
