# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Run a bring-up task's success gate and record the verdict.

    python models/demos/ernie45_d_p/bringup/gate.py P1.2            # run gate, update state.json
    python models/demos/ernie45_d_p/bringup/gate.py P1.2 --commit   # ... and git-commit on PASS
    python models/demos/ernie45_d_p/bringup/gate.py --status        # print the ledger
    python models/demos/ernie45_d_p/bringup/gate.py --next          # tasks whose deps are all PASS

Exit code: 0 PASS, 1 FAIL (gate cmd failed or a metric missed its threshold), 2 BLOCKED (deps not PASS).
A gate passes only if: cmd exits 0 AND every metric glob matches >= 1 recorded metric AND
every matched metric satisfies its threshold AND every declared artifact exists.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import fnmatch
import json
import operator
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

BRINGUP = Path(__file__).resolve().parent
REPO = BRINGUP.parents[3]
sys.path.insert(0, str(REPO))
from models.demos.ernie45_d_p.bringup import metrics as M  # noqa: E402

STATE = BRINGUP / "state.json"
LOGS = BRINGUP / "logs"
OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq}


def load_tasks() -> dict[str, dict]:
    spec = yaml.safe_load((BRINGUP / "tasks.yaml").read_text())
    return {t["id"]: t for t in spec["tasks"]}


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")


@contextlib.contextmanager
def locked():
    """Serialize state.json read-modify-write and git commits across concurrent gate runs."""
    with open(BRINGUP / ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def update_task_state(tid: str, **fields) -> None:
    with locked():
        state = load_state()
        entry = state.setdefault(tid, {})
        hist = fields.pop("history_add", None)
        entry.update(fields)
        if hist:
            entry.setdefault("history", []).append(hist)
        save_state(state)


def check_metrics(spec: dict[str, str], got: dict) -> tuple[bool, list[str]]:
    ok, lines = True, []
    for glob, cond in spec.items():
        op_s, val_s = cond.split()
        want = float(val_s)
        matched = {k: v["value"] for k, v in got.items() if fnmatch.fnmatchcase(k, glob)}
        if not matched:
            ok = False
            lines.append(f"  MISSING  {glob} (need {cond})")
            continue
        for k, v in sorted(matched.items()):
            good = v is not None and OPS[op_s](float(v), want)
            ok &= good
            lines.append(f"  {'ok  ' if good else 'FAIL'}     {k} = {v} (need {cond})")
    return ok, lines


def git_commit(task: dict, verdict: str, summary: str) -> str | None:
    # Stage only this task's footprint so concurrent work never leaks into a gate commit.
    rel = lambda p: str(Path(p).relative_to(REPO)) if Path(p).is_absolute() else p  # noqa: E731
    paths = [rel(STATE), rel(M.RESULTS_DIR / f"{task['id']}.json"), rel(BRINGUP / "BREADCRUMBS.md")]
    paths += [p for p in task.get("paths", []) if (REPO / p).exists()]
    subprocess.run(["git", "add", "-A", *paths], cwd=REPO, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode == 0:
        return None
    msg = (
        f"[ernie45_d_p][{task['id']}] {task['title']}\n\nGate: {verdict}\n{summary}\n\n"
        "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>\n"
    )
    # pre-commit hooks may rewrite files (black, EOF fixer): re-stage and retry once.
    if subprocess.run(["git", "commit", "-q", "-m", msg], cwd=REPO).returncode != 0:
        subprocess.run(["git", "add", "-A", *paths], cwd=REPO, check=True)
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=REPO, check=True)
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()


def run_gate(tid: str, commit: bool, force: bool) -> int:
    tasks, state = load_tasks(), load_state()
    task = tasks[tid]
    blocked = [d for d in task.get("deps", []) if state.get(d, {}).get("status") != "PASS"]
    if blocked and not force:
        print(f"BLOCKED {tid}: deps not PASS: {blocked}")
        return 2

    M.reset(tid)
    update_task_state(tid, status="RUNNING", started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    LOGS.mkdir(exist_ok=True)
    log = LOGS / f"{tid}.log"
    t0 = time.time()
    env = dict(os.environ, ERNIE_BRINGUP_TASK=tid)
    with open(log, "w") as f:
        f.write(f"$ {task['gate']['cmd']}\n")
        f.flush()
        rc = subprocess.run(
            task["gate"]["cmd"], shell=True, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT
        ).returncode
    dur = time.time() - t0

    got = M.load(tid)
    m_ok, lines = check_metrics(task["gate"].get("metrics", {}), got)
    missing_art = [a for a in task.get("artifacts", []) if not (REPO / a).exists()]
    verdict = "PASS" if (rc == 0 and m_ok and not missing_art) else "FAIL"
    summary = "\n".join(
        [f"cmd rc={rc} ({dur:.0f}s), log: {log.relative_to(REPO)}"]
        + lines
        + [f"  MISSING artifact {a}" for a in missing_art]
    )
    print(f"{verdict} {tid}: {task['title']}\n{summary}")

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    update_task_state(
        tid,
        status=verdict,
        last_run=now,
        duration_s=round(dur, 1),
        rc=rc,
        metrics={k: v["value"] for k, v in got.items()},
        log=str(log.relative_to(REPO)),
        history_add={"t": now, "status": verdict},
    )
    if commit and verdict == "PASS":
        # The commit for a task is found later via its tag: git log --grep "\[ernie45_d_p\]\[<id>\]"
        with locked():
            sha = git_commit(task, verdict, summary)
        if sha:
            print(f"committed {sha}")
    return 0 if verdict == "PASS" else 1


def task_commit(tid: str) -> str:
    out = subprocess.run(
        ["git", "log", "-1", "--format=%h", "--fixed-strings", f"--grep=[ernie45_d_p][{tid}]"],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return out


def print_status() -> None:
    tasks, state = load_tasks(), load_state()
    for tid, t in tasks.items():
        s = state.get(tid, {})
        print(f"{tid:6} {s.get('status', 'TODO'):6} {task_commit(tid):10} {t['title']}")


def next_tasks() -> list[str]:
    tasks, state = load_tasks(), load_state()
    return [
        tid
        for tid, t in tasks.items()
        if state.get(tid, {}).get("status") != "PASS"
        and all(state.get(d, {}).get("status") == "PASS" for d in t.get("deps", []))
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--force", action="store_true", help="run even if deps are not PASS")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--next", action="store_true")
    a = ap.parse_args()
    if a.status:
        print_status()
        return 0
    if a.next:
        print("\n".join(next_tasks()))
        return 0
    return run_gate(a.task, a.commit, a.force)


if __name__ == "__main__":
    sys.exit(main())
