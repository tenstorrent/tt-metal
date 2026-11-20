# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F278: an approved perf task must say what the ledger already settled on its subject before it runs.

DeepSeek-V4.1 P.84 (2026-11-19): the overseer queued "expert replication, static or per-chunk" although the replica
branch (bringup/<model>/moe_replication, P.45-P.62b) had already measured per-chunk replication and FIX.33 had removed
it from this branch. Nothing in the queue asked for that history.

- ``validate`` refuses a perf task that is not finished and has no ``brief.prior_work``: the related task ids (this
  ledger, other branches' ledgers, known issues) and what they settled, or "none found: <terms searched>".
- ``python -m models.demos.common.bringup prior-work <task> [--terms a,b]`` proposes the candidates: every task of
  this ledger and of every other local branch's copy of it (bringup/<model>/*) whose title or brief mentions the
  terms, with its status, and the known issues that do. The overseer reads them and writes prior_work.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

STOP = set(
    "owner perf with from that this the and for into over each when then than what which while about after before "
    "chunk ms at on in of a to is be or as by it its are not".split()
)


def needs_prior_work(task: dict, status: str | None) -> list[str]:
    if task.get("step") != "perf" or status in ("PASS", "REJECTED") or not task.get("role"):
        return []
    if str((task.get("brief") or {}).get("prior_work") or "").strip():
        return []
    return [
        "no brief.prior_work: name what the ledger already settled on this subject (python -m "
        f"models.demos.common.bringup prior-work {task['id']}), or 'none found: <terms>' (F278)"
    ]


def terms_of(task: dict) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z_]{4,}", str(task.get("title", "")))
    return sorted({w.lower() for w in words} - STOP)


def _branch_ledgers(repo: Path, rel: str) -> dict[str, str]:
    """{branch: tasks.yaml text} of every local branch holding this ledger path, except the checked-out one."""
    out = {}
    refs = subprocess.run(
        ["git", "-C", str(repo), "for-each-ref", "--format=%(refname:short)", "refs/heads"],
        capture_output=True,
        text=True,
    ).stdout.split()
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    for ref in refs:
        if ref == head:
            continue
        r = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{rel}"], capture_output=True, text=True)
        if r.returncode == 0:
            out[ref] = r.stdout
    return out


def _hits(tasks: list, terms: list[str], status_of) -> list[str]:
    rows = []
    for t in tasks or []:
        text = (str(t.get("title", "")) + " " + str((t.get("brief") or {}).get("details", ""))).lower()
        found = [w for w in terms if w in text]
        if len(found) >= max(1, min(2, len(terms))):
            rows.append(f"{t['id']:8s} {status_of(t['id']) or '-':9s} [{', '.join(found)}] {str(t.get('title'))[:110]}")
    return rows


def search(spec, led, tid: str, terms: list[str] | None = None) -> list[str]:
    tasks = led.tasks()
    terms = [t.lower() for t in (terms or terms_of(tasks[tid]))]
    done = led.state() if led.state_store.exists() else {}
    out = [f"terms: {', '.join(terms)}", "", "this ledger:"]
    own = [t for i, t in tasks.items() if i != tid]
    out += _hits(own, terms, lambda i: (done.get(i) or {}).get("status")) or ["  -"]
    repo = Path(spec.repo)
    rel = str(led.tasks_path.resolve().relative_to(repo.resolve()))
    for ref, text in _branch_ledgers(repo, rel).items():
        data = yaml.safe_load(text) or []
        data = data if isinstance(data, list) else data.get("tasks", [])
        hits = _hits(data, terms, lambda i: None)
        if hits:
            out += ["", f"branch {ref}:"] + hits
    ki = Path(spec.bringup_dir) / "known_issues.md"
    if ki.exists():
        rows = [ln[:160] for ln in ki.read_text().splitlines() if sum(w in ln.lower() for w in terms) >= 2]
        if rows:
            out += ["", "known issues:"] + rows
    return out
