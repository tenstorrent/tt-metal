# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Re-verify passed tasks under the current framework, without touching the run (F82).

    python -m models.demos.common.bringup.core.verify --spec S --tasks M.1 'L.*' K.1 'C.*' [--batch] [--status PASS]
        [--out DIR] [--force]

Runs each matching task's gate command as its gate would (same env, the safe runner), but with the metrics going to
``runs/<run>/verify/<stamp>/results`` and the logs next to them: state.json, results/ and the commits are untouched,
so a step that passed under older limits keeps its status while the overseer sees whether it still passes (after a
change of limits, F76-F81). Tasks run in the order of the --tasks patterns (dependency order within one). Prints one
line per task and writes summary.json; exit 1 if any task fails. A failure is the overseer's to act on (a fix job,
overseer.py), never an automatic re-run of the task.

F142 ``--batch``: the C.* / S.* tasks whose gate is one test command over their frozen test files run grouped by
block type (and runner command) in ONE pytest process (the mesh opened and the layer's weights loaded once, not once
per task; ``--no-precompile``); testing/batch_tasks.py points each test's metrics at its own task and records each
task's outcome. Every task is then judged exactly as unbatched (its outcome and its gate metrics). A task whose test
never reported (the batch died: a hang, a crash) fails with the batch's exit code: rerun it unbatched. Anything else
(a compound gate, the ladder, the contract) runs as before.

F150: a task whose every gate unit has a PASS kept on the same stamp (same command, code, goldens: core/stamps.py) is
not run again ("reused: same code"); its kept metrics are judged as a run's. ``--force`` runs it anyway (the overseer's
deliberate re-check). An unbatched run is kept like a gate's.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shlex
import subprocess
import time
from pathlib import Path

from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.core import stamps
from models.demos.common.bringup.core.gate import (
    _merge_metrics,
    check_metrics,
    gate_env,
    reusable,
    run_name,
    run_units,
    unit_stamps,
)
from models.demos.common.bringup.core.ledger import Ledger

BATCH_PLUGIN = "models.demos.common.bringup.testing.batch_tasks"


def selected(led, patterns: list[str], status: str | None) -> list[str]:
    """The tasks matching the patterns, in the patterns' order (dependency order within one)."""
    state, order, out = led.state(), led.topo_order(), []
    for p in patterns:
        out += [
            t
            for t in order
            if fnmatch.fnmatchcase(t, p)
            and t not in out
            and (status is None or state.get(t, {}).get("status") == status)
        ]
    return out


def batch_key(task: dict) -> tuple | None:
    """(block type, the runner's words) for a gate that is one test command over the task's frozen files; else None."""
    cmd, files = task["gate"]["cmd"], list(task.get("tests") or [])
    if not files or not task["id"].startswith(("C.", "S.")) or any(c in cmd for c in ("&&", "||", ";", "|")):
        return None
    words = shlex.split(cmd)
    if not all(f in words for f in files):
        return None
    return ((task.get("brief") or {}).get("block_type"), tuple(w for w in words if w not in files))


def judge(task: dict, rc: int | None, got: dict) -> tuple[str, list[str]]:
    ok, lines = check_metrics(task["gate"].get("metrics", {}), got)
    return ("PASS" if rc == 0 and ok else "FAIL"), lines


def run_batch(spec, led, tids: list[str], out: Path, n: int) -> list[dict]:
    """One pytest process over the tasks' frozen test files (F142); a row per task, judged as unbatched."""
    tasks = [led.task(t) for t in tids]
    runner = list(batch_key(tasks[0])[1])
    if any(Path(w).name == "run_safe_pytest.sh" for w in runner) and "--no-precompile" not in runner:
        i = next(k for k, w in enumerate(runner) if Path(w).name == "run_safe_pytest.sh")
        runner.insert(i + 1, "--no-precompile")
    files = [f for t in tasks for f in t["tests"]]
    owner = {str((spec.repo / f).resolve()): t["id"] for t in tasks for f in t["tests"]}
    outcomes = out / f"batch_{n}.outcomes.json"
    env = gate_env(spec, led, tids[0])
    env.update({M.RESULTS_ENV: str(out / "results"), "BRINGUP_BATCH_TASKS": json.dumps(owner)})
    env["BRINGUP_BATCH_OUTCOMES"] = str(outcomes)
    cmd = " ".join(shlex.quote(w) for w in [*runner, "-p", BATCH_PLUGIN, *files])
    log, t0 = out / f"batch_{n}.log", time.time()
    with open(log, "w") as f:
        f.write(f"$ {cmd}\n")
        f.flush()
        rc = subprocess.run(cmd, shell=True, cwd=spec.repo, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    seen = json.loads(outcomes.read_text()) if outcomes.exists() else {}
    rows, dt = [], round((time.time() - t0) / len(tasks), 1)
    for task in tasks:
        tid = task["id"]
        o = seen.get(tid)
        trc = 0 if o == "passed" else (1 if o == "failed" else rc or 1)
        verdict, lines = judge(task, trc, M.load(tid, out / "results"))
        if o is None:
            lines = [
                f"  no test outcome: the batch exited {rc} before this task's test reported (rerun unbatched)"
            ] + lines
        rows.append({"task": tid, "verdict": verdict, "rc": trc, "s": dt, "lines": lines, "log": str(log), "batch": n})
    return rows


def reuse_kept(spec, led, tid: str, out: Path, store, tasks: dict, keys: tuple) -> dict | None:
    """F150: the row of a task whose every unit has a kept PASS on the same stamp, judged as a run; else None."""
    task = tasks[tid]
    env = dict(gate_env(spec, led, tid), **{M.RESULTS_ENV: str(out / "results")})
    plan = unit_stamps(spec, task, env, tasks, keys)
    recs = [store.get(u, sid) for u, _, _, sid in plan]
    if not all(reusable(r, task["gate"].get("metrics") or {}, len(plan) == 1) for r in recs):
        return None
    for (u, _, _, sid), rec in zip(plan, recs):
        _merge_metrics(out / "results", tid, rec["metrics"])
        store.restore(u, sid, rec, out / "results", tid)
    verdict, lines = judge(task, 0, M.load(tid, out / "results"))
    when = ", ".join(f"{u} {r.get('t')}" for (u, *_), r in zip(plan, recs))
    return {"task": tid, "verdict": verdict, "rc": 0, "s": 0, "lines": lines, "reused": when}


def verify(
    spec,
    patterns: list[str],
    status: str | None = "PASS",
    out: Path | None = None,
    batch: bool = False,
    force: bool = False,
) -> dict:
    led = Ledger(spec.bringup_dir)
    tids = selected(led, patterns, status)
    out = Path(out or spec.run_dir(run_name(led)) / "verify" / time.strftime("%Y%m%dT%H%M%S"))
    (out / "results").mkdir(parents=True, exist_ok=True)
    tasks, store = led.tasks(), stamps.Store(spec.run_dir(run_name(led)))
    keys = (stamps.code_key(spec.repo), stamps.goldens_key(spec))
    kept = {}
    for tid in tids if not force else []:
        row = reuse_kept(spec, led, tid, out, store, tasks, keys)
        if row is not None:
            kept[tid] = row
    groups: dict[tuple, list[str]] = {}
    for tid in [t for t in tids if t not in kept] if batch else []:
        key = batch_key(led.task(tid))
        if key is not None:
            groups.setdefault(key, []).append(tid)
    first = {g[0]: g for g in groups.values()}
    batched = {t for g in groups.values() for t in g}
    rows = []
    for tid in tids:
        if tid in kept:
            r = kept[tid]
            print(f"{r['verdict']} {tid} (reused: same code, kept PASS {r['reused']}; --force runs it)", flush=True)
            for line in r["lines"] if r["verdict"] == "FAIL" else []:
                print(line, flush=True)
            rows.append(r)
            continue
        if tid in first:
            new = run_batch(spec, led, first[tid], out, len({r.get("batch") for r in rows if "batch" in r}))
            for r in new:
                print(f"{r['verdict']} {r['task']} (batched, rc {r['rc']}, log {r['log']})", flush=True)
                for line in r["lines"] if r["verdict"] == "FAIL" else []:
                    print(line, flush=True)
            rows += new
            continue
        if tid in batched:
            continue
        task = led.task(tid)
        env = gate_env(spec, led, tid)
        env[M.RESULTS_ENV] = str(out / "results")
        log = out / f"{tid}.log"
        t0 = time.time()
        rc, _, _ = run_units(spec, task, env, log, out / "results", store, force, tasks)
        verdict, lines = judge(task, rc, M.load(tid, out / "results"))
        rows.append({"task": tid, "verdict": verdict, "rc": rc, "s": round(time.time() - t0, 1), "lines": lines})
        print(f"{verdict} {tid} (rc {rc}, {time.time() - t0:.0f}s, log {log})", flush=True)
        for line in lines if verdict == "FAIL" else []:
            print(line, flush=True)
    summary = {"out": str(out), "tasks": rows, "failed": [r["task"] for r in rows if r["verdict"] != "PASS"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"verify: {len(rows) - len(summary['failed'])}/{len(rows)} pass; {out}/summary.json")
    return summary


def main(argv=None) -> int:
    import os

    from models.demos.common.bringup.core.spec import Spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.environ.get("BRINGUP_SPEC"))
    ap.add_argument("--tasks", nargs="+", required=True, help="task id globs, run in this order")
    ap.add_argument("--status", default="PASS", help="only tasks in this status ('' for any)")
    ap.add_argument("--out")
    ap.add_argument(
        "--batch", action="store_true", help="F142: C.* / S.* frozen tests grouped by block type, one process"
    )
    ap.add_argument("--force", action="store_true", help="F150: run a task even when a PASS on the same code is kept")
    a = ap.parse_args(argv)
    res = verify(Spec.load(a.spec), a.tasks, a.status or None, Path(a.out) if a.out else None, a.batch, a.force)
    return 1 if res["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
