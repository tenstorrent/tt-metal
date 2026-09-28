# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up framework CLI. Every command takes --spec <model spec.yaml> (or BRINGUP_SPEC).

    python -m models.demos.common.bringup status --spec S       ledger with verdicts and commits
    python -m models.demos.common.bringup next --spec S         tasks whose deps are all PASS
    python -m models.demos.common.bringup gate P2.3 --spec S [--commit] [--force]
    python -m models.demos.common.bringup sweep [prefix] --spec S [--commit]
    python -m models.demos.common.bringup validate --spec S     spec + ledger schema

Op requests (F46; steps deferred to op-gen, plan/op_request.py and plan/op_export.py):
    python -m models.demos.common.bringup op-requests --spec S                      list them with status
    python -m models.demos.common.bringup approve op-request <op> --spec S          the owner's approval
    python -m models.demos.common.bringup op-export <op> --spec S [--codegen-root D]  prompt + golden suite for op-gen
    python -m models.demos.common.bringup op-ready <op> [<op> ...] --from <generated op dir(s)> --spec S
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
        print(f"{tid:8} {s.get('status', 'TODO'):8} {task_commit(spec, tid):11} {t['title']}")
    deferred = led.deferred()
    if deferred:
        print(f"\n{len(deferred)} deferred to op-gen (on the CPU bridge until the op is delivered):")
        for tid in deferred:
            d = state[tid].get("deferred") or {}
            print(f"  {tid}: {d.get('op')} ({d.get('request')})")
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
        if led.status(tid) == "DEFERRED":
            print(f"DEFERRED {tid} (skipped: on the CPU bridge until op-gen delivers its op)")
            continue
        res = run_gate(spec, led, tid, commit=a.commit)
        print(res.summary())
        out.append((tid, res.verdict))
        if res.verdict != "PASS":
            break
    print("\nSWEEP: " + " ".join(f"{t}={v}" for t, v in out))
    done = len(out) + len([t for t in ids if led.status(t) == "DEFERRED"])
    return 0 if done == len(ids) and all(v == "PASS" for _, v in out) else 1


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
    tids = rerun_from(led, a.from_[0])
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
    fspec = fork(spec, led, a.from_[0], a.name)
    print(f"forked run {a.name} at {a.from_[0]}: repo {fspec.repo}\n  use --spec {fspec.path}")
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


@command("approve", "record a person's approval (intake, plan, perf, shared:<task id>, op-request <op>)")
def cmd_approve(a):
    from models.demos.common.bringup.plan.approvals import approve

    spec, _ = load(a)
    if a.task == "op-request":
        from models.demos.common.bringup.plan.approvals import approve_op_request

        if not a.more:
            sys.exit("approve op-request needs the op name")
        for op in a.more:
            rec = approve_op_request(spec, op, note=a.note or "")
            print(f"approved op request {op} by {rec['by']} at {rec['at']} ({len(rec['files'])} files)")
        return 0
    if a.task.startswith("shared:"):
        from models.demos.common.bringup.core.ledger import Ledger
        from models.demos.common.bringup.plan.approvals import approve_shared

        tid = a.task.split(":", 1)[1]
        rec = approve_shared(spec, Ledger(spec.bringup_dir).task(tid), note=a.note or "")
        print(f"approved shared-code changes of {tid} by {rec['by']} at {rec['at']}: " + ", ".join(rec["paths"]))
        return 0
    rec = approve(spec, a.task, note=a.note or "")
    print(f"approved {a.task} by {rec['by']} at {rec['at']}: " + ", ".join(rec["files"]))
    return 0


@command("op-requests", "list the op requests (steps deferred to op-gen) with their status")
def cmd_op_requests(a):
    from models.demos.common.bringup.plan import op_request as OR
    from models.demos.common.bringup.plan.approvals import op_request_approved

    spec, led = load(a)
    reqs = OR.all_requests(spec)
    if not reqs:
        print(f"no op requests in {OR.root(spec)}")
        return 0
    state = led.state()
    for d, req in reqs:
        c = req.get("component") or {}
        tasks = ", ".join(f"{t} {state.get(t, {}).get('status', 'TODO')}" for t in req.get("tasks") or [])
        ok = " (approval current)" if op_request_approved(spec, req["op"]) else ""
        print(f"{req['op']:24} {req.get('status', '?'):9}{ok} {c.get('block_type')}.{c.get('step')} [{tasks}] {d}")
        print(f"  {OR.evidence_summary(req)}")
    return 0


@command(
    "op-export", "write an approved op request into the op-gen tree (prompt + golden suite); prints the next steps"
)
def cmd_op_export(a):
    from models.demos.common.bringup.plan.op_export import export

    spec, _ = load(a)
    written, steps = export(spec, a.task, a.codegen_root and Path(a.codegen_root))
    for p in written:
        print(f"wrote {p}")
    print(f"\nop request {a.task} exported. Nothing was committed, pushed or launched. Next steps (the owner's call):")
    for k, s in enumerate(steps, 1):
        print(f"  {k}. {s}")
    return 0


@command("op-ready", "bring op-gen's delivered op(s) in as ttnn.bringup.<op> and reset the deferred tasks")
def cmd_op_ready(a):
    from models.demos.common.bringup.core.gate import git_commit
    from models.demos.common.bringup.plan.op_export import ready

    spec, led = load(a)
    ops = [a.task] + list(a.more or [])
    if not a.from_:
        sys.exit("op-ready needs --from <generated ttnn/ttnn/operations/<op> folder, or the operations folder>")
    out = ready(spec, ops, [Path(p) for p in a.from_], a.bringup_ops and Path(a.bringup_ops))
    print(f"delivered: {', '.join('ttnn.bringup.' + o for o in out['ops'])}")
    print("reset: " + " ".join(out["reset"]))
    if not a.no_commit:
        paths = out["paths"] + [str(p.relative_to(spec.repo)) for p in (led.tasks_path, led.state_path)]
        sha = git_commit(
            spec,
            paths,
            f"[{spec.tag}][op-ready] {', '.join(out['ops'])} from op-gen",
            "Reset: " + " ".join(out["reset"]),
        )
        print(f"committed {sha}")
    print(f"next: python -m models.demos.common.bringup.orchestrator resume --spec {spec.path}")
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
    from string import Template

    from models.demos.common.bringup.core.spec import CODE_ROOT

    if a.prior:
        return _new_from_prior(a)
    if not a.model or not a.hf_id:
        sys.exit("new needs --model <slug> and --hf-id <org/name> (or --prior <slug> of an earlier bring-up)")
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


PRIOR_HOOKS = '''# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bring-up hooks for {model}: the same checkpoint as the prior bring-up {prior} (spec ``prior``), on another mesh.

CPU side: the prior's hooks (same reference, goldens and HF loader); change them there only for a bug.
Device side (implement role): device_component, device_model, written for this bring-up's plan.
"""

from models.demos.{prior}.bringup import hooks as _prior

reference = _prior.reference
for _name in ("tokenizer", "hf_model", "hf_layers"):
    if hasattr(_prior, _name):
        globals()[_name] = getattr(_prior, _name)


def device_component(mesh, spec, layer, step):
    raise NotImplementedError(f"implement step: no device module for {{step}} yet")


def device_model(mesh, spec, layers, lm_head=True):
    raise NotImplementedError("implement step: no device model yet")
'''


def _new_from_prior(a):
    """new --prior <slug> [--model <new slug>] [--mesh R,C]: a spec copied from the prior's, with ``prior`` set, the
    new mesh, and hooks that reuse the prior's CPU side. The person still reviews and approves the spec."""
    import copy

    import yaml

    from models.demos.common.bringup.core.spec import CODE_ROOT, Spec

    prior = Spec.load(CODE_ROOT / "models" / "demos" / a.prior / "bringup" / "spec.yaml")
    mesh = [int(x) for x in a.mesh.split(",")] if a.mesh else prior.mesh
    model = a.model or f"{a.prior}_{mesh[0]}x{mesh[1]}"
    if model == a.prior:
        sys.exit("--model must differ from --prior")
    d = CODE_ROOT / "models" / "demos" / model
    b = d / "bringup"
    if (b / "spec.yaml").exists():
        sys.exit(f"{b / 'spec.yaml'} exists")
    b.mkdir(parents=True, exist_ok=True)
    header = "# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC\n#\n# SPDX-License-Identifier: Apache-2.0\n"
    for p in (d / "__init__.py", b / "__init__.py"):
        if not p.exists():
            p.write_text(header)
    data = copy.deepcopy(prior.data)
    data.update(model=model, model_dir=f"models/demos/{model}", hooks=f"models.demos.{model}.bringup.hooks")
    data["prior"] = f"models/demos/{a.prior}"
    data.setdefault("box", {})["mesh"] = mesh
    data["box"]["name"] = f"{data['box'].get('name', 'box').split(', mesh')[0]}, mesh {mesh[0]}x{mesh[1]}"
    if isinstance(data.get("contract"), dict):
        data["contract"]["adapter"] = model
    for k in ("bringup_dir", "art"):
        data.get("paths", {}).pop(k, None)
    (b / "spec.yaml").write_text(
        f"# {model}: {data['hf_id']} on mesh {mesh[0]}x{mesh[1]}, from the prior bring-up {a.prior} "
        f"(mesh {prior.mesh[0]}x{prior.mesh[1]}). Written by `new --prior`; review, then approve intake.\n"
        + yaml.safe_dump(data, sort_keys=False, width=120)
    )
    (b / "hooks.py").write_text(PRIOR_HOOKS.format(model=model, prior=a.prior))
    (b / "BREADCRUMBS.md").write_text(
        f"# {data['hf_id']} bring-up on mesh {mesh[0]}x{mesh[1]}: breadcrumbs\n\nPrior bring-up: {a.prior} "
        f"(mesh {prior.mesh[0]}x{prior.mesh[1]}); goldens and CPU reference shared. Append-only log, one section per task "
        "attempt: what was done, decisions and why, gotchas, the re-run command, the verdict.\n"
    )
    print(f"wrote {b.relative_to(CODE_ROOT)}/{{spec.yaml,hooks.py,BREADCRUMBS.md}} (prior {a.prior}, mesh {mesh})")
    return 0


def build_parser(extra=None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m models.demos.common.bringup")
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("task", nargs="?")
    ap.add_argument("more", nargs="*", help="approve op-request <op>...; op-ready <op> <op>...")
    ap.add_argument("--spec")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--force", action="store_true", help="run even if deps are not PASS")
    ap.add_argument("--ledger-only", action="store_true", help="validate: skip the model-spec schema")
    ap.add_argument("--files", nargs="*", help="freeze: files to freeze (default: the task's 'tests')")
    ap.add_argument("--no-commit", action="store_true", help="freeze: do not commit")
    ap.add_argument("--from", dest="from_", nargs="+", help="rerun/fork: task id; op-ready: generated op folder(s)")
    ap.add_argument(
        "--codegen-root", help=f"op-export: the op-gen tree (default {'tt_metal/third_party/tt_ops_code_gen'})"
    )
    ap.add_argument("--bringup-ops", help="op-ready: the bring-up ops folder (default ttnn/ttnn/bringup)")
    ap.add_argument("--name", help="fork: new run name")
    ap.add_argument("--no-run", action="store_true", help="rerun: only reset the verdicts")
    ap.add_argument("--other", help="compare: the other run's spec")
    ap.add_argument("--note", help="approve: note stored with the approval")
    ap.add_argument("--model", help="new: model slug (package name under models/demos)")
    ap.add_argument("--hf-id", help="new: Hugging Face id")
    ap.add_argument("--prior", help="new: slug of an earlier bring-up of the same checkpoint to start from")
    ap.add_argument("--mesh", help="new --prior: the new mesh as R,C (default: the prior's)")
    for fn in extra or []:
        fn(ap)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return COMMANDS[a.command][0](a)


if __name__ == "__main__":
    sys.exit(main())
