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
import json
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


@command("freeze", "validate a task's tests (reference passes, zero stub fails) and freeze their hashes")
def cmd_freeze(a):
    from models.demos.common.bringup.core.runs import FreezeError, freeze_task

    spec, led = load(a)
    try:
        rec = freeze_task(spec, led, a.task, files=a.files or None, commit=not a.no_commit)
    except FreezeError as e:
        print(f"FREEZE FAILED {e}")
        return 1
    print(f"frozen {a.task}: reference={rec.get('reference', 'n/a')} stub={rec.get('stub', 'n/a')}")
    for f in rec["files"]:
        print(f"  {f}")
    return 0


@command("init-run", "name the run in this branch (task id argument = run name)")
def cmd_init_run(a):
    from models.demos.common.bringup.core.runs import init_run

    spec, led = load(a)
    print(json.dumps(init_run(spec, led, a.task or "default"), indent=1))
    return 0


@command("rerun", "mark --from <id> and everything downstream TODO, then sweep those gates")
def cmd_rerun(a):
    from models.demos.common.bringup.core.runs import rerun_from

    spec, led = load(a)
    tids = rerun_from(led, a.from_)
    print("reset: " + " ".join(tids))
    if a.no_run:
        return 0
    for tid in tids:
        res = run_gate(spec, led, tid, commit=a.commit)
        print(res.summary())
        if res.verdict != "PASS":
            return res.exit_code
    return 0


@command("fork", "new branch + worktree at --from <id>'s commit, as run --name")
def cmd_fork(a):
    from models.demos.common.bringup.core.runs import fork

    spec, led = load(a)
    fspec = fork(spec, led, a.from_, a.name)
    print(f"forked run {a.name} at {a.from_}: repo {fspec.repo}\n  use --spec {fspec.path}")
    return 0


@command("compare", "compare this run (--spec) with another (--other spec)")
def cmd_compare(a):
    from models.demos.common.bringup.core.runs import compare

    spec, led = load(a)
    other = Ledger(Spec.load(a.other).bringup_dir)
    print(f"{'task':8} {'status':15} {'attempts':9} {'debug':6} {'wall s':14} changes")
    for r in compare(led, other):
        dur = "/".join("-" if d is None else f"{d:.0f}" for d in r["duration_s"])
        notes = []
        if r["agent_defs_changed"]:
            notes.append("defs: " + ",".join(r["agent_defs_changed"]))
        if r["metric_deltas"]:
            notes.append(" ".join(f"{k}{v:+g}" for k, v in list(r["metric_deltas"].items())[:4]))
        print(
            f"{r['task']:8} {'/'.join(r['status']):15} {'/'.join(map(str, r['attempts'])):9} "
            f"{'/'.join(map(str, r['debugger'])):6} {dur:14} {'; '.join(notes)}"
        )
    return 0


@command("approve", "record a person's approval of an approval point (intake, plan, perf) with file hashes")
def cmd_approve(a):
    from models.demos.common.bringup.plan.approvals import approve

    spec, _ = load(a)
    rec = approve(spec, a.task, note=a.note or "")
    print(f"approved {a.task} by {rec['by']} at {rec['at']}: " + ", ".join(rec["files"]))
    return 0


@command("render-tests", "render the component and swap tests of every implement task that has none yet")
def cmd_render_tests(a):
    from models.demos.common.bringup.testing.templates import render_component_test, render_swap_test

    spec, led = load(a)
    n = 0
    for t in led.tasks().values():
        b = t.get("brief") or {}
        if t.get("step") != "implement" or not b:
            continue
        if "swapped" in b:
            p = render_swap_test(spec, b["block_type"], b["swapped"])
        else:
            p = render_component_test(spec, b["block_type"], b["step"])
        n += 1
        print(p.relative_to(spec.repo))
    print(f"{n} tests present")
    return 0


@command("new", "scaffold models/demos/<--model>/bringup/ (spec template, hooks skeleton, breadcrumbs)")
def cmd_new(a):
    from pathlib import Path
    from string import Template

    from models.demos.common.bringup.core.spec import CODE_ROOT

    if not a.model or not a.hf_id:
        sys.exit("new needs --model <slug> and --hf-id <org/name>")
    here = Path(__file__).resolve().parent / "templates"
    d = CODE_ROOT / "models" / "demos" / a.model
    b = d / "bringup"
    b.mkdir(parents=True, exist_ok=True)
    header = "# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC\n#\n# SPDX-License-Identifier: Apache-2.0\n"
    for p in (d / "__init__.py", b / "__init__.py"):
        if not p.exists():
            p.write_text(header)
    vals = {"model": a.model, "hf_id": a.hf_id}
    for name in ("spec.yaml", "hooks.py"):
        out = b / name
        if out.exists():
            print(f"exists, kept: {out.relative_to(CODE_ROOT)}")
            continue
        out.write_text(Template((here / f"{name}.tmpl").read_text()).safe_substitute(vals))
        print(f"wrote {out.relative_to(CODE_ROOT)}")
    bc = b / "BREADCRUMBS.md"
    if not bc.exists():
        bc.write_text(
            f"# {a.hf_id} bring-up: breadcrumbs\n\nAppend-only log, one section per task attempt: what was "
            "done, decisions and why, gotchas, the re-run command, the verdict.\n"
        )
    return 0


def build_parser(extra=None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m models.demos.common.bringup")
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("task", nargs="?")
    ap.add_argument("--spec")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--force", action="store_true", help="run even if deps are not PASS")
    ap.add_argument("--ledger-only", action="store_true", help="validate: skip the model-spec schema")
    ap.add_argument("--files", nargs="*", help="freeze: files to freeze (default: the task's 'tests')")
    ap.add_argument("--no-commit", action="store_true", help="freeze: do not commit")
    ap.add_argument("--from", dest="from_", help="rerun/fork: task id")
    ap.add_argument("--name", help="fork: new run name")
    ap.add_argument("--no-run", action="store_true", help="rerun: only reset the verdicts")
    ap.add_argument("--other", help="compare: the other run's spec")
    ap.add_argument("--note", help="approve: note stored with the approval")
    ap.add_argument("--model", help="new: model slug (package name under models/demos)")
    ap.add_argument("--hf-id", help="new: Hugging Face id")
    for fn in extra or []:
        fn(ap)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return COMMANDS[a.command][0](a)


if __name__ == "__main__":
    sys.exit(main())
