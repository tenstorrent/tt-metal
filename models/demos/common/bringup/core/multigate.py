# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.31 (OWNER 2026-11-13): gates that run back to back are ONE run. The quick-accepted picks' full gates
(orchestrator.pending_full_gates) and the milestone gate after them (X.3) used to be a gate each, each with its own
model load (~210 GB, minutes): P.49 / P.50 / P.51 shared the ladder and profile by stamp but loaded the model three
times. ``prerun`` collects every pending unit of every task, drops the units each task's own gate would keep or reuse
(core/gate._lookup: its stamp, a safe-runner record, the same runs in another task's gate), runs each remaining run
ONCE however many tasks name it (the same tests, measuring environment and code: core/runrec.unit_keys) in one batch
process per model configuration (core/batch.model_key, the switches at their test values normalized; one process
when they agree), and returns per task its units' results. Each task's gate then runs as before (core/gate.run_gate
with ``pre``): its own verdict and metrics from its own units, its commit; a failing unit fails only the tasks that
name it (the shared process runs past a failure). A unit that cannot share a process is left to its task's gate."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

from models.demos.common.bringup.core import batch, gate
from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.core import runrec, stamps


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def plan(spec, ledger, tids: list[str]) -> dict:
    """{"todo": [(tid, unit, cmd, form, keys)], "same": {(tid, unit): (tid0, unit0)}, "left": [(tid, unit, why)],
    "skipped": [(tid, unit, why)]}: what a multi-task pre-run would run, once each, and what it leaves alone."""
    tasks = ledger.tasks()
    run_dir = spec.run_dir(gate.run_name(ledger))
    store = stamps.Store(run_dir)
    keys = (stamps.code_key(spec.repo, spec), stamps.goldens_key(spec))
    out = {"todo": [], "same": {}, "left": [], "skipped": []}
    first: dict[tuple, tuple] = {}
    for tid in tids:
        task = tasks[tid]
        env = gate.gate_env(spec, ledger, tid)
        stamps.ensure_aliases(run_dir, spec.repo, spec, env)
        limits = gate.gate_limits(task, tasks)
        units = gate.unit_stamps(spec, task, env, tasks, keys)
        delta = stamps.Delta(run_dir, tid)
        unders = stamps.under_test_key(spec.repo, spec, gate.under_test_paths(task, tasks), with_bytes=True)
        for u, cmd, st, sid in units:
            uenv = dict(env, **{M.TASK_ENV: gate.unit_base(u)})
            whole = len(units) == 1
            if gate._kept(delta.get(u), unders, cmd, uenv, limits, whole, store.aliases()):
                out["skipped"].append((tid, u, "kept: passed in its task's gate on the same code under test"))
                continue
            if gate._lookup(spec, store, u, cmd, st, sid, uenv, tasks, limits, whole):
                out["skipped"].append((tid, u, "reused by its gate (same stamp or the same runs)"))
                continue
            why = batch.why_not(cmd)
            if why:
                out["left"].append((tid, u, why))
                continue
            form = batch.launch_form(cmd)[0]
            ks = runrec.unit_keys(form, uenv, keys, spec) or [stamps.sha16(f"{tid}\0{u}")]
            k = (tuple(ks), st.get("waivers") or "")
            if k in first:
                out["same"][(tid, u)] = first[k]
                continue
            first[k] = (tid, u)
            out["todo"].append((tid, u, cmd, form, ks))
    return out


def prerun(spec, ledger, tids: list[str], echo=print) -> dict[str, dict]:
    """Run the pending units of ``tids`` (see the module doc); {tid: {unit: result}} for core/gate.run_gate(pre=...).
    A result is the shape run_units keeps for a shared-process unit (rc, metrics, files, text, s, loads, first) plus
    ``restore`` [(file, name)] (the gate deletes its outputs before it runs; these are put back) and PRE_RUN."""
    p = plan(spec, ledger, tids)
    for tid, u, why in p["left"]:
        echo(f"  [{tid}] {u}: left to its gate (a process of its own: {why})")
    if not p["todo"]:
        return {}
    stamp_t = time.strftime("%Y%m%d-%H%M%S")
    d = gate.log_dir(spec, ledger) / f"multigate-{stamp_t}"
    d.mkdir(parents=True, exist_ok=True)
    # one process per model configuration (one when the units agree): grouped by key, not by adjacency
    names = set()
    parsed = {}
    for tid, u, cmd, form, _ in p["todo"]:
        parsed[(tid, u)] = batch.parse_all(form)
        names |= {k for r in parsed[(tid, u)] for k in r["env"]}
    groups: dict[str, list] = {}
    for tid, u, cmd, form, ks in p["todo"]:
        env = gate.gate_env(spec, ledger, tid)
        mk = {batch.model_key({**env, **r["env"]}, names) for r in parsed[(tid, u)]}
        key = mk.pop() if len(mk) == 1 else f"own:{tid}:{u}"
        groups.setdefault(key, []).append((tid, u, cmd, form))
    results: dict[tuple, dict] = {}
    total_loads = 0
    for gi, (key, members) in enumerate(groups.items()):
        r, loads = _run(spec, ledger, members, d / f"group{gi}", echo)
        for v in r.values():
            v["proc"], v["proc_loads"] = gi, loads
        results.update(r)
        total_loads += loads
    echo(
        f"  multigate (FIX.31): {len(p['todo'])} units of {len(tids)} tasks in {len(groups)} process(es), "
        f"{total_loads} model load(s); {len(p['same'])} more named by several tasks ran once; "
        f"{len(p['skipped'])} kept or reused; log {d}"
    )
    out: dict[str, dict] = {}
    for (tid, u), res in results.items():
        out.setdefault(tid, {})[u] = res
    for (tid, u), (tid0, u0) in p["same"].items():
        src = results.get((tid0, u0))
        if src is None:
            continue
        mine = dict(src, text=f"# {u}: the same runs as {tid0} {u0} (FIX.31)\n" + src["text"])
        mine["restore"] = [(f, gate.stamps.output_key(n, tid0)) for f, n in src.get("restore") or []]
        mine["restore"] = [(f, f"{tid}{n}" if n.startswith("_") else n) for f, n in mine["restore"]]
        mine["files"] = [n for _, n in mine["restore"]]
        out.setdefault(tid, {})[u] = mine
    for tid, units in out.items():  # each task: every process it used counted once (its gate_model_loads)
        seen = set()
        for u, r in units.items():
            r["first"] = r["proc"] not in seen
            r["loads"] = r["proc_loads"] if r["first"] else 0
            seen.add(r["proc"])
    return out


def _run(spec, ledger, members: list, d: Path, echo) -> tuple[dict, int]:
    """One batch process over ``members`` [(tid, unit, cmd, form)]: each unit runs as its own task records it (its
    BRINGUP_TASK / BRINGUP_RECORD_TASK / results dir), the plan runs past a failure. ({(tid, unit): result}, loads)."""
    d.mkdir(parents=True, exist_ok=True)
    ids = {(tid, u): f"{tid}::{u}" for tid, u, _, _ in members}
    logs = {ids[(tid, u)]: d / f"{_safe(ids[(tid, u)])}.log" for tid, u, _, _ in members}
    group = [(ids[(tid, u)], form) for tid, u, _, form in members]
    for (tid, u, cmd, form), (i, _) in zip(members, group):
        logs[i].write_text(
            f"$ {cmd}\n# FIX.31: in one process with {len(members) - 1} other unit(s), launch form {form}\n"
        )
    uenv = {
        ids[(tid, u)]: {
            M.TASK_ENV: gate.unit_base(u),
            M.RECORD_ENV: tid,
            M.RESULTS_ENV: str(ledger.results_dir),
        }
        for tid, u, _, _ in members
    }
    base = gate.gate_env(spec, ledger, members[0][0])
    entries = batch.gate_entries(group, base, uenv, logs)
    plan_path = batch.write_plan(d / "plan.json", entries, stop_on_fail=False)
    hint = batch.hints([r for _, c in group for r in batch.parse_all(c)], spec)
    cmd = batch.command(plan_path)
    blog = d / "process.log"
    echo(f"  multigate (FIX.31): {len(members)} units in one process: {', '.join(i for i, _ in group)}")
    t0 = time.time()
    with open(blog, "w") as bf:
        bf.write(f"$ {cmd}\n")
        bf.flush()
        mem = gate.hostmem_env(blog, [c for _, c in group])
    penv = dict(base, **uenv[group[0][0]], **hint, **mem)

    def proc(c, k):  # F269: a run needing another model goes to a fresh process (k > 0), its output appended
        with open(blog, "a") as bf:
            if k:
                bf.write(f"\n# F269: a fresh process for the runs needing another model\n$ {c}\n")
                bf.flush()
            return subprocess.run(
                c, shell=True, cwd=spec.repo, env=penv, stdout=bf, stderr=subprocess.STDOUT
            ).returncode

    prc = batch.drive(plan_path, proc)
    got = batch.load_results(plan_path)
    runs, running = got.get("runs") or {}, got.get("running")
    btext = blog.read_text(errors="replace")
    keep = d / "files"
    keep.mkdir(exist_ok=True)
    out, loads = {}, 0
    for k, (tid, u, cmd, form) in enumerate(members):
        i = ids[(tid, u)]
        mine = [e for e in entries if e["unit"] == i]
        rs = [runs.get(e["name"]) for e in mine]
        text = logs[i].read_text(errors="replace")
        metrics, files, secs = {}, [], 0.0
        for r in rs:
            if r and r.get("rc") is not None:
                metrics.update(r.get("metrics") or {})
                files += r.get("files") or []
                secs += float(r.get("s") or 0)
        loads += sum(int((r or {}).get("loads") or 0) for r in rs)
        if all(r and r.get("rc") is not None for r in rs):
            rc = next((r["rc"] for r in rs if r["rc"]), 0)
        elif prc == gate.SAFE_HANG_RC and (any(e["name"] == running for e in mine) or (running is None and not runs)):
            rc = gate.SAFE_HANG_RC
            text += f"\n# FIX.31: the shared process hung (in or before this unit); its output:\n{btext}"
        else:
            rc = 1
            text += f"\n# FIX.31: {u} not reached: the process ended (rc {prc}); its output:\n{btext}"
        restore = []
        for name in sorted(set(files)):
            src = ledger.results_dir / name
            if name == f"{tid}.json" or not src.exists():
                continue
            dst = keep / f"{_safe(i)}__{name}"
            shutil.copy2(src, dst)
            restore.append((dst, name))
        out[(tid, u)] = {
            "rc": rc, "metrics": metrics, "files": [n for _, n in restore], "text": text, "s": secs,
            "loads": 0, "first": False, "restore": restore, gate.PRE_RUN: True,
        }  # fmt: skip
    echo(f"  multigate (FIX.31): process rc {prc}, {time.time() - t0:.0f} s, {loads} model load(s)")
    return out, loads
