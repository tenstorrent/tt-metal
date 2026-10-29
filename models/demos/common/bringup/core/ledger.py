# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Task ledger (tasks.yaml, written by the plan step and by people) and state (state.json, written only by the runner).

tasks.yaml:
    tasks:
      - id: P2.3                      # unique; the commit tag is [<spec tag>][<id>]
        title: RMSNorm on device
        step: implement               # pipeline step that owns it (intake, reference, goldens, plan, tests,
                                      # implement, integrate, contract, perf, framework)
        deps: [P2.2]
        device: true                  # gate cmd must go through run_safe_pytest.sh / tt-probe.sh
        paths: [models/demos/x/tt/norm.py]   # what a PASS commit stages (plus ledger files)
        gate:
          cmd: scripts/run_safe_pytest.sh --run-all models/demos/x/tests/pcc/test_rms_norm.py
          metrics: {"pcc_*": ">= 0.99"}      # glob -> "<op> <value>"; every glob must match >= 1 metric
        artifacts: [path, ...]        # must exist after the gate
        frozen: {files: {path: sha256}}      # written by `freeze`; any change fails the gate before it runs
        brief: {...}                  # extra fields for the agent brief (component, block type, ...)

state.json: {task id: {status, attempts, last_run, duration_s, rc, metrics, log, history, agent, ...},
             "_run": {name, branch, created, forked_from}}

DEFERRED (F46): a component the implement agent deferred to the op code generator (plan/op_request.py). It stays on
the CPU reference through the bridge (testing/cpu_bridge.py), so for its dependents it counts like PASS.

REJECTED (F57): a perf pick the owner declined on its A/B report; the change was reverted. Done for its dependents.

COMMIT_PENDING (F149): the gate passed but git refused its commit (a hook). Not done for its dependents until
`gate <id> --commit-only` (or the orchestrator on resume) commits it; nothing re-runs, its outputs stay on disk.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import re
import time
from pathlib import Path

import yaml

STATUSES = (
    "TODO",
    "RUNNING",
    "PASS",
    "FAIL",
    "HANG",
    "BLOCKED",
    "STOPPED",
    "DEFERRED",
    "REJECTED",
    "COMMIT_PENDING",
)
DONE_STATUSES = ("PASS", "DEFERRED", "REJECTED")  # a dependent may run after any


def satisfied(status: str | None) -> bool:
    return status in DONE_STATUSES


class LedgerError(ValueError):
    pass


# F138: an accuracy metric of a device check is gated only against the limit its test records, never a number; the
# reference's own checks (CPU vs CPU: HF parity, chunked == one-shot) keep theirs
ACCURACY = re.compile(r"^(pcc_|err_|overlap_|top1|top5)")
CPU_CHECK_STEPS = ("intake", "reference", "goldens")


def fixed_accuracy(task: dict, conds: dict | None = None) -> list[str]:
    """F138: the accuracy conditions of a device check (a device task or a perf pick) that hold a number instead of
    the limit its test records."""
    if task.get("step") in CPU_CHECK_STEPS or not (task.get("device") or task.get("step") == "perf"):
        return []
    out = []
    for glob, cond in (conds if conds is not None else (task.get("gate") or {}).get("metrics") or {}).items():
        parts = str(cond).split()
        if ACCURACY.match(glob) and len(parts) == 2 and parts[1] != "limit":
            out.append(
                f"{glob} {cond}: a fixed accuracy number (F138); gate on '{parts[0]} limit' (the limit the test "
                "records: calib x the reference's own noise floor or drift)"
            )
    return out


def wall_only_device_metrics(task: dict) -> list[str]:
    """F212: a gate that needs device_ms_* but whose every test_profile.py segment is wall-only
    (BRINGUP_PROFILE_WALL=1): the gate then adds no profiled run (F114 with_clean_wall) and the metric is MISSING, an
    agent attempt can never pass it (DeepSeek-V4.1 P.6: device_ms_total missing after a passing change)."""
    gate = task.get("gate") or {}
    need = sorted(g for g in (gate.get("metrics") or {}) if str(g).startswith("device_ms"))
    segs = [s for s in str(gate.get("cmd") or "").split("&&") if "test_profile.py" in s]
    if need and segs and all("BRINGUP_PROFILE_WALL" in s for s in segs):
        return [
            f"{', '.join(need)} gated but every test_profile.py segment is wall-only (BRINGUP_PROFILE_WALL=1): drop the "
            "flag, the gate adds the clean wall run before the profiled run itself (F114, F212)"
        ]
    return []


class Ledger:
    def __init__(self, ledger_dir: Path):
        self.dir = Path(ledger_dir)
        self.tasks_path = self.dir / "tasks.yaml"
        self.state_path = self.dir / "state.json"
        self.results_dir = self.dir / "results"

    # ---- tasks
    def load_spec(self) -> dict:
        return yaml.safe_load(self.tasks_path.read_text()) or {} if self.tasks_path.exists() else {"tasks": []}

    def tasks(self) -> dict[str, dict]:
        out = {}
        for t in self.load_spec().get("tasks", []):
            if t["id"] in out:
                raise LedgerError(f"duplicate task id {t['id']}")
            out[t["id"]] = t
        return out

    def task(self, tid: str) -> dict:
        tasks = self.tasks()
        if tid not in tasks:
            raise LedgerError(f"unknown task {tid}")
        return tasks[tid]

    def write_tasks(self, spec: dict) -> None:
        with self.locked():
            self.tasks_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.tasks_path.with_suffix(".tmp")
            tmp.write_text(yaml.safe_dump(spec, sort_keys=False, width=120))
            tmp.replace(self.tasks_path)

    def update_task_def(self, tid: str, **fields) -> None:
        """Change fields of one task in tasks.yaml (used by freeze)."""
        spec = self.load_spec()
        for t in spec.get("tasks", []):
            if t["id"] == tid:
                t.update(fields)
                break
        else:
            raise LedgerError(f"unknown task {tid}")
        self.write_tasks(spec)

    def validate(self) -> list[str]:
        errs = []
        try:
            tasks = self.tasks()
        except LedgerError as e:
            return [str(e)]
        done = self.state() if self.state_path.exists() else {}
        for tid, t in tasks.items():
            for k in ("title", "gate"):
                if k not in t:
                    errs.append(f"{tid}: missing '{k}'")
            if "gate" in t and "cmd" not in t["gate"]:
                errs.append(f"{tid}: gate has no cmd")
            for d in t.get("deps", []):
                if d not in tasks:
                    errs.append(f"{tid}: unknown dep {d}")
            for glob, cond in (t.get("gate", {}).get("metrics") or {}).items():
                parts = str(cond).split()
                if len(parts) != 2 or parts[0] not in (">=", "<=", ">", "<", "=="):
                    errs.append(f"{tid}: bad threshold {glob}: {cond!r}")
                else:
                    errs += [f"{tid}: {e}" for e in fixed_accuracy(t, {glob: cond})]
            if (done.get(tid) or {}).get("status") not in ("PASS", "REJECTED"):  # a finished gate's cmd is history
                errs += [f"{tid}: {e}" for e in wall_only_device_metrics(t)]
        try:
            self.topo_order()
        except LedgerError as e:
            errs.append(str(e))
        return errs

    def topo_order(self) -> list[str]:
        tasks, seen, order, stack = self.tasks(), set(), [], set()

        def visit(t):
            if t in seen:
                return
            if t in stack:
                raise LedgerError(f"dependency cycle through {t}")
            stack.add(t)
            for d in tasks[t].get("deps", []):
                if d in tasks:
                    visit(d)
            stack.discard(t)
            seen.add(t)
            order.append(t)

        for t in tasks:  # declaration order breaks ties
            visit(t)
        return order

    def upstream(self, tid: str) -> list[str]:
        """tid and every task it depends on, directly or not, in topological order."""
        tasks, need, todo = self.tasks(), set(), [tid]
        while todo:
            t = todo.pop()
            if t in tasks and t not in need:
                need.add(t)
                todo += tasks[t].get("deps") or []
        return [t for t in self.topo_order() if t in need]

    def downstream(self, tid: str) -> list[str]:
        """tid and every task that depends on it, directly or not, in topological order."""
        tasks = self.tasks()
        hit = {tid}
        for t in self.topo_order():
            if any(d in hit for d in tasks[t].get("deps", [])):
                hit.add(t)
        return [t for t in self.topo_order() if t in hit]

    # ---- state
    @contextlib.contextmanager
    def locked(self):
        """Serialize state/ledger read-modify-write and commits across concurrent gate runs."""
        self.dir.mkdir(parents=True, exist_ok=True)
        ignore = self.dir / ".gitignore"
        if not ignore.exists():
            ignore.write_text(".lock\n*.tmp\n")
        with open(self.dir / ".lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    # F180: the per-task histories that grow with every attempt / pick live beside state.json, uncommitted (the repo's
    # 500 KB hook refused DeepSeek-V4.1's state.json at Z.1: agent_runs 137 KB, ab 85 KB); state() merges them back
    SIDE_KEYS = ("agent_runs", "ab")

    @property
    def side_path(self) -> Path:
        return self.dir / "history_state.json"

    def _exclude_side(self) -> None:
        """F180: keep history_state.json out of git (the clone's own info/exclude: no tracked file changes)."""
        import subprocess

        try:
            ex = subprocess.run(
                ["git", "rev-parse", "--git-path", "info/exclude"], cwd=self.dir, capture_output=True, text=True
            ).stdout.strip()
        except OSError:
            return
        if not ex:
            return
        p = Path(ex) if Path(ex).is_absolute() else self.dir / ex
        line = "**/bringup/history_state.json"
        text = p.read_text() if p.exists() else ""
        if line not in text.split("\n"):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text + ("" if text.endswith("\n") or not text else "\n") + line + "\n")

    def state(self) -> dict:
        state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        side = json.loads(self.side_path.read_text()) if self.side_path.exists() else {}
        for tid, extra in side.items():
            if isinstance(state.get(tid), dict):
                for k, v in extra.items():
                    state[tid].setdefault(k, v)
        return state

    def _save_state(self, state: dict) -> None:
        from models.demos.common.bringup.core.gate import state_metrics

        try:
            tasks = self.tasks()
        except (FileNotFoundError, KeyError, TypeError):
            tasks = {}
        for tid, e in state.items():  # F129: older entries kept every metric; trim them on the next save
            if isinstance(e, dict) and e.get("metrics") and tid in tasks:
                e["metrics"] = state_metrics(tasks[tid].get("gate", {}).get("metrics") or {}, e["metrics"])
        side = {}
        for tid, e in state.items():  # F180
            if isinstance(e, dict):
                moved = {k: e.pop(k) for k in self.SIDE_KEYS if k in e}
                if moved:
                    side[tid] = moved
        self._exclude_side()
        stmp = self.side_path.with_suffix(".tmp")
        stmp.write_text(json.dumps(side, sort_keys=True) + "\n")
        stmp.replace(self.side_path)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
        tmp.replace(self.state_path)

    def update(self, tid: str, history_add: dict | None = None, **fields) -> dict:
        with self.locked():
            state = self.state()
            entry = state.setdefault(tid, {})
            entry.update(fields)
            if history_add:
                entry.setdefault("history", []).append(history_add)
            self._save_state(state)
            return entry

    def reset(self, tids: list[str]) -> None:
        """Mark tasks TODO again (rerun). Keeps their history."""
        with self.locked():
            state = self.state()
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            for t in tids:
                e = state.setdefault(t, {})
                if e.get("status") not in (None, "TODO"):
                    e.setdefault("history", []).append({"t": now, "status": "RESET"})
                e["status"] = "TODO"
                e["attempts"] = 0
                e.pop("commit_pending", None)  # F149: a rerun replaces the pending result
            self._save_state(state)

    def status(self, tid: str) -> str:
        return self.state().get(tid, {}).get("status", "TODO")

    def runnable(self) -> list[str]:
        """Tasks not PASS (or DEFERRED) whose deps are all PASS or DEFERRED, in topological order."""
        tasks, state = self.tasks(), self.state()
        ok = lambda t: satisfied(state.get(t, {}).get("status"))  # noqa: E731
        return [t for t in self.topo_order() if not ok(t) and all(ok(d) for d in tasks[t].get("deps", []))]

    def first_unpassed(self) -> str | None:
        state = self.state()
        return next((t for t in self.topo_order() if not satisfied(state.get(t, {}).get("status"))), None)

    def deferred(self) -> list[str]:
        """Tasks deferred to op-gen (on the CPU bridge), in topological order."""
        state = self.state()
        return [t for t in self.topo_order() if state.get(t, {}).get("status") == "DEFERRED"]
