# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up framework CLI. Every command takes --spec <model spec.yaml> (or BRINGUP_SPEC).

    python -m models.demos.common.bringup status --spec S       ledger with verdicts and commits
    python -m models.demos.common.bringup next --spec S         tasks whose deps are all PASS
    python -m models.demos.common.bringup gate P2.3 --spec S [--commit] [--force]
    python -m models.demos.common.bringup sweep [prefix] --spec S [--commit]
    python -m models.demos.common.bringup validate --spec S     spec + ledger schema
"""

from __future__ import annotations

import argparse
import os
import sys

from models.demos.common.bringup.core.gate import run_gate, task_commit
from models.demos.common.bringup.core.ledger import Ledger
from models.demos.common.bringup.core.spec import Spec

COMMANDS = {}


def command(name, help_):
    def deco(fn):
        COMMANDS[name] = (fn, help_)
        return fn

    return deco


def load(a) -> tuple[Spec, Ledger]:
    path = a.spec or os.environ.get("BRINGUP_SPEC")
    if not path:
        sys.exit("no spec: pass --spec or set BRINGUP_SPEC")
    spec = Spec.load(path)
    return spec, Ledger(spec.bringup_dir)


@command("status", "print the ledger")
def cmd_status(a):
    spec, led = load(a)
    state = led.state()
    for tid, t in led.tasks().items():
        s = state.get(tid, {})
        print(f"{tid:8} {s.get('status', 'TODO'):7} {task_commit(spec, tid):11} {t['title']}")
    return 0


@command("next", "runnable tasks")
def cmd_next(a):
    _, led = load(a)
    print("\n".join(led.runnable()))
    return 0


@command("gate", "run one gate")
def cmd_gate(a):
    spec, led = load(a)
    res = run_gate(spec, led, a.task, commit=a.commit, force=a.force)
    print(led.task(a.task)["title"])
    print(res.summary())
    if res.commit:
        print(f"committed {res.commit}")
    return res.exit_code


@command("sweep", "re-run every gate in dependency order (optional id prefix); stops at the first failure")
def cmd_sweep(a):
    spec, led = load(a)
    ids = [t for t in led.topo_order() if t.startswith(a.task or "")]
    out = []
    for tid in ids:
        res = run_gate(spec, led, tid, commit=a.commit)
        print(res.summary())
        out.append((tid, res.verdict))
        if res.verdict != "PASS":
            break
    print("\nSWEEP: " + " ".join(f"{t}={v}" for t, v in out))
    return 0 if len(out) == len(ids) and all(v == "PASS" for _, v in out) else 1


@command("validate", "check the spec and ledger schemas")
def cmd_validate(a):
    spec, led = load(a)
    errs = ([] if a.ledger_only else [f"spec: {e}" for e in spec.validate()]) + [f"ledger: {e}" for e in led.validate()]
    print("\n".join(errs) if errs else "valid")
    return 1 if errs else 0


def build_parser(extra=None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m models.demos.common.bringup")
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("task", nargs="?")
    ap.add_argument("--spec")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--force", action="store_true", help="run even if deps are not PASS")
    ap.add_argument("--ledger-only", action="store_true", help="validate: skip the model-spec schema")
    for fn in extra or []:
        fn(ap)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return COMMANDS[a.command][0](a)


if __name__ == "__main__":
    sys.exit(main())
