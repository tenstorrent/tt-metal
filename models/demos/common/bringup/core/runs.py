# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Runs: freeze a test, resume, rerun, fork, compare, and the agent-definition hashes each step records.

A run is a git branch plus a run directory $ART/<model>/runs/<run>/ (logs, briefs, agent transcripts, run.json).
Its verdicts live in the branch's state.json, so a fork made at a task's commit starts with exactly the
verdicts that were true at that commit. A fork gets its own git worktree, so the main checkout never
switches branch under someone's feet, and its own spec copy pointing at that worktree. Goldens and weight
caches are shared by path, never copied.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import yaml

from models.demos.common.bringup.core import freeze as F
from models.demos.common.bringup.core.gate import format_paths, git_commit, run_gate, task_commit
from models.demos.common.bringup.core.ledger import Ledger
from models.demos.common.bringup.core.spec import Spec

IMPL_ENV = "BRINGUP_IMPL"  # device (default) | reference | stub; read by the test templates


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def git(repo: Path, *args, check=True) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=check).stdout.strip()


# ---------------------------------------------------------------- freeze
class FreezeError(RuntimeError):
    pass


def freeze_task(spec: Spec, ledger: Ledger, tid: str, files: list[str] | None = None, commit: bool = True) -> dict:
    """Validate a task's tests and freeze them.

    The test files are formatted first (the commit hooks would otherwise change them after hashing). Unless the
    task sets ``stub_check: false``, the gate must PASS with BRINGUP_IMPL=reference (the CPU reference as the
    module) and must FAIL with BRINGUP_IMPL=stub (a module that returns zeros). Then every file is hashed into
    the task's ``frozen`` block, and the files and the ledger are committed.
    """
    task = ledger.task(tid)
    files = files or list(task.get("tests") or [])
    if not files:
        raise FreezeError(f"{tid}: nothing to freeze (no 'tests' in the task and no files given)")
    format_paths(spec.repo, files)
    record = {"at": now()}
    if task.get("stub_check", True):
        ref = run_gate(spec, ledger, tid, force=True, extra_env={IMPL_ENV: "reference"}, record=False)
        stub = run_gate(spec, ledger, tid, force=True, extra_env={IMPL_ENV: "stub"}, record=False)
        record["reference"] = ref.verdict
        record["stub"] = stub.verdict
        if ref.verdict != "PASS":
            raise FreezeError(f"{tid}: test fails with the CPU reference as the module:\n{ref.summary()}")
        if stub.verdict == "PASS":
            raise FreezeError(
                f"{tid}: test passes with a zero stub, so it cannot catch a wrong module:\n{stub.summary()}"
            )
    # Pin the goldens the test reads too (their manifest carries the content hash): a regenerated golden fails the gate.
    record["files"] = F.hash_paths(spec.repo, files + list(task.get("freeze_extra") or []))
    ledger.update_task_def(tid, frozen=record)
    if commit:
        # the packages the frozen tests live in (their __init__.py files) go with them
        inits = sorted(
            {
                str(p.relative_to(spec.repo))
                for f in files
                for p in (spec.repo / f).parents
                if (p / "__init__.py").exists() and p.is_relative_to(spec.model_dir)
                for p in [p / "__init__.py"]
            }
        )
        git_commit(
            spec,
            [str(ledger.tasks_path.relative_to(spec.repo)), *files, *inits],
            f"[{spec.tag}][{tid}][freeze] {task['title']}",
            f"Tests frozen: reference={record.get('reference', 'n/a')} stub={record.get('stub', 'n/a')}\n"
            + "\n".join(f"  {k} {v[:12]}" for k, v in record["files"].items()),
        )
    return record


# ---------------------------------------------------------------- agent definitions
def agent_hashes(repo: Path, paths: list[str]) -> dict[str, dict]:
    """Git blob hash of each agent definition / brief template a step used, and whether it had local edits."""
    out = {}
    for p in paths:
        f = repo / p
        if not f.exists():
            out[p] = {"blob": None}
            continue
        blob = git(repo, "hash-object", str(f))
        head = git(repo, "rev-parse", f"HEAD:{p}", check=False)
        out[p] = {"blob": blob, "dirty": blob != head}
    return out


# ---------------------------------------------------------------- runs
def run_info(ledger: Ledger) -> dict:
    return ledger.state().get("_run", {})


def init_run(spec: Spec, ledger: Ledger, name: str, forked_from: dict | None = None) -> dict:
    branch = git(spec.repo, "rev-parse", "--abbrev-ref", "HEAD")
    info = {"name": name, "branch": branch, "repo": str(spec.repo), "created": now()}
    if forked_from:
        info["forked_from"] = forked_from
    with ledger.locked():
        state = ledger.state()
        state["_run"] = info
        ledger._save_state(state)
    d = spec.run_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({**info, "spec": str(spec.path)}, indent=1) + "\n")
    return info


def resume_point(ledger: Ledger) -> str | None:
    """First task in dependency order that has not passed. A HANG, FAIL or STOPPED task resumes at itself."""
    return ledger.first_unpassed()


def rerun_from(ledger: Ledger, tid: str) -> list[str]:
    """Mark tid and everything downstream TODO; earlier steps keep their verdicts. A downstream task deferred to op-gen
    stays DEFERRED (an upstream change does not deliver its op); name it with --from to redo it."""
    state = ledger.state()
    tids = [t for t in ledger.downstream(tid) if t == tid or state.get(t, {}).get("status") != "DEFERRED"]
    ledger.reset(tids)
    return tids


def fork(spec: Spec, ledger: Ledger, tid: str, name: str) -> Spec:
    """New branch + worktree at tid's passing commit, a spec copy pointing at it, downstream verdicts reset."""
    if ledger.status(tid) not in ("PASS", "DEFERRED"):
        raise RuntimeError(f"can only fork from a passed task (or a deferred one); {tid} is {ledger.status(tid)}")
    sha = task_commit(spec, tid)
    if not sha:
        raise RuntimeError(f"no commit tagged [{spec.tag}][{tid}]")
    parent = run_info(ledger).get("name", "default")
    branch = f"bringup/{spec.model}/{name}"
    wt = spec.run_dir(name) / "worktree"
    wt.parent.mkdir(parents=True, exist_ok=True)
    git(spec.repo, "worktree", "add", "-q", "-b", branch, str(wt), sha)

    data = json.loads(json.dumps(spec.data))
    paths = data.setdefault("paths", {})
    paths["repo"] = str(wt)
    # Same artifact root, so goldens, weights and profiles are shared by path with the parent run.
    paths.setdefault("art", str(spec.art.parent))
    fspec_path = spec.run_dir(name) / "spec.yaml"
    fspec_path.write_text(yaml.safe_dump(data, sort_keys=False))
    fspec = Spec.load(fspec_path)
    fled = Ledger(fspec.bringup_dir)
    downstream = [t for t in fled.downstream(tid) if t != tid]
    fled.reset(downstream)
    init_run(fspec, fled, name, forked_from={"run": parent, "task": tid, "commit": sha})
    return fspec


def compare(led_a: Ledger, led_b: Ledger) -> list[dict]:
    """Per task: status, attempts, debugger hand-overs, wall time, agent-definition changes, metric deltas."""
    sa, sb = led_a.state(), led_b.state()
    rows = []
    for tid in led_b.topo_order():
        a, b = sa.get(tid, {}), sb.get(tid, {})
        ma, mb = a.get("metrics", {}), b.get("metrics", {})
        deltas = {
            k: round(mb[k] - ma[k], 6)
            for k in sorted(set(ma) & set(mb))
            if isinstance(ma[k], (int, float)) and isinstance(mb[k], (int, float)) and mb[k] != ma[k]
        }
        defs_a = {k: v.get("blob") for k, v in (a.get("agent", {}).get("defs") or {}).items()}
        defs_b = {k: v.get("blob") for k, v in (b.get("agent", {}).get("defs") or {}).items()}
        rows.append(
            {
                "task": tid,
                "status": (a.get("status", "TODO"), b.get("status", "TODO")),
                "attempts": (a.get("attempts", 0), b.get("attempts", 0)),
                "debugger": (a.get("debugger_attempts", 0), b.get("debugger_attempts", 0)),
                "duration_s": (a.get("duration_s"), b.get("duration_s")),
                "agent_defs_changed": sorted(k for k in set(defs_a) | set(defs_b) if defs_a.get(k) != defs_b.get(k)),
                "metric_deltas": deltas,
            }
        )
    return rows
