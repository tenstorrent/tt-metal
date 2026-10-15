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


def load(bringup_dir: Path) -> list[dict]:
    p = path(bringup_dir)
    data = yaml.safe_load(p.read_text()) if p.exists() else None
    return list((data or {}).get("notes") or [])


def save(bringup_dir: Path, notes: list[dict]) -> None:
    head = "# Overseer notes (models/demos/common/bringup/overseer.py): written by the overseer, read by every agent.\n"
    path(bringup_dir).write_text(head + yaml.safe_dump({"notes": notes}, sort_keys=False, width=118))


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


def _dirs(spec_path: str):
    from models.demos.common.bringup.core.gate import run_name
    from models.demos.common.bringup.core.ledger import Ledger
    from models.demos.common.bringup.core.spec import Spec

    spec = Spec.load(spec_path)
    return spec.bringup_dir, spec.run_dir(run_name(Ledger(spec.bringup_dir)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("note", "close", "list", "abort"):
        p = sub.add_parser(name)
        p.add_argument("--spec", default=os.environ.get("BRINGUP_SPEC"))
        if name in ("note", "abort"):
            p.add_argument("--tasks", nargs="+", help="task id globs (default: every task / the aborted task)")
        if name == "note":
            p.add_argument("--roles", nargs="+")
            p.add_argument("text")
        if name == "close":
            p.add_argument("id")
            p.add_argument("--why", default="")
        if name == "abort":
            p.add_argument("--reason", required=True)
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
    else:
        r = abort(bdir, rdir, a.reason, a.tasks)
        print(f"aborted {r['task']} {r['role']} attempt {r['attempt']} (pid {r['pid']}); note {r['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
