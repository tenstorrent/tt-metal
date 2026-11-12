# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F259 (owner 2026-11-06: "this is taking insanely too much time"): several device-test runs in ONE pytest process,
the model loaded once. CPU side: which safe-runner commands may share a process, in what order, the plan file the
device side reads (testing/batch.py, tests/test_batch.py) and the results it writes back.

A Galaxy 8x4 DeepSeek-V4.1 process spends ~4 min uploading weights and ~1.5-2.5 min on its first chunk for a
measurement of seconds; a perf pick's A/B ran ~5-10 such processes a side. Shared here: the mesh (opened once), the
loaded device model (testing/batch.device_model), the program cache, a model's process-level tables (the resident
engram table). Not shared: everything a run records (its own env, results dir, log, metric suffix) and its state
(KV caches: freed between runs).

A run may join a batch when it is one ``scripts/run_safe_pytest.sh --no-precompile`` call (flags: --no-precompile,
--run-all) over framework tests that take the model from testing/batch.device_model (SHARED_TESTS) with no device
profiler variable (TT_METAL_*: read at process start). Every other run keeps its own process: the contract tests
build the model through the engine's adapter (part of what they check) and the model's own tests load it themselves.
The runs of one batch must agree on every variable but the per-run ones (ITEM_VARS): a model's switches are read at
its load (DeepSeek-V4.1 HOST_GLUE freezes them), so different switch values are different processes (off / on).
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path

PLAN_ENV = "BRINGUP_BATCH"  # the plan file (json) of a batch process
CMD_ENV = "BRINGUP_BATCH_CMD"  # or: the original command, planned by the batch process itself (a gate unit)
KEY_ENV = "BRINGUP_BATCH_KEY"  # set per run by the batch process: the model key the plan gave it
LM_HEAD_ENV = "BRINGUP_BATCH_LM_HEAD"  # load the shared model with its LM head (a ladder run in the batch needs it)
MAX_SEQ_ENV = "BRINGUP_BATCH_MAX_SEQ"  # load it with target.seq at least this (the furthest BRINGUP_POS + chunk)
BATCH_TEST = "models/demos/common/bringup/tests/test_batch.py"
TESTS_DIR = "models/demos/common/bringup/tests/"
SHARED_TESTS = ("test_ladder.py", "test_profile.py", "test_positions.py", "test_smoke.py", "test_determinism.py")
# FIX.25: the model-level determinism check (testing/model_determinism.py) every model-level gate runs (core/gate.py)
DETERMINISM_TEST = TESTS_DIR + "test_determinism.py"
DETERMINISM_CMD = f"scripts/run_safe_pytest.sh --run-all --no-precompile {DETERMINISM_TEST}"
FLAGS = {"--no-precompile", "--run-all"}
# per-run variables: what a run measures and where it records it; anything else is the model's configuration
ITEM_VARS = {
    "BRINGUP_TASK",
    "BRINGUP_RECORD_TASK",
    "BRINGUP_RESULTS_DIR",
    "BRINGUP_METRIC_SUFFIX",
    "BRINGUP_POS",
    "BRINGUP_PROFILE_WALL",
    "BRINGUP_WALL_CLEAN",
    "BRINGUP_PROFILE_HOST",
    "BRINGUP_QUICK",
    "BRINGUP_RUNG",
    "BRINGUP_CHUNK_LAST",
    "BRINGUP_FULL_PREFILL",
    "BRINGUP_PROFILE_OPS",
    "BRINGUP_PROFILE_LAYERS",
    "BRINGUP_AB",
}
SAFE_RUN = re.compile(r"^((?:\w+=\S*\s+)*)scripts/run_safe_pytest\.sh\s+(.+)$", re.S)
ASSIGN = re.compile(r"^(\w+)=(\S*)$")


def parse(seg: str) -> dict | None:
    """One `&&` segment as a batch run {env, args, run_all}, or None when it cannot share a process."""
    m = SAFE_RUN.match(seg.strip())
    if not m:
        return None
    env = dict(ASSIGN.match(w).groups() for w in m.group(1).split())
    words = shlex.split(m.group(2))
    flags = [w for w in words if w.startswith("-")]
    tests = [w for w in words if not w.startswith("-")]
    if not tests or "--no-precompile" not in flags or not set(flags) <= FLAGS:
        return None
    if any(not (t.startswith(TESTS_DIR) and t.split("::")[0].rsplit("/", 1)[-1] in SHARED_TESTS) for t in tests):
        return None
    if any(k.startswith("TT_METAL_") for k in env) or env.get("BRINGUP_PROFILER") == "device":
        return None
    return {"env": env, "args": tests, "run_all": "--run-all" in flags}


def parse_all(cmd: str) -> list[dict] | None:
    """Every segment of ``cmd`` as a batch run, or None if any cannot share a process."""
    runs = [parse(s) for s in cmd.split("&&") if s.strip()]
    return runs if runs and all(runs) else None


def phase(run: dict) -> int:
    """The order inside a batch: the determinism check first (FIX.25: its first sequence is the cold one, the first
    after the load), then ladders (their first chunk and the load time as close to a process of their own as it
    gets), then clean walls (F114: before any real-time-profiled run), then the other runs, the profiled runs last."""
    test = run["args"][0].split("::")[0]
    env = run["env"]
    if test.endswith("test_determinism.py"):
        return -1
    if test.endswith("test_ladder.py"):
        return 0
    if test.endswith("test_profile.py") and env.get("BRINGUP_PROFILE_WALL") == "1":
        return 1
    if test.endswith("test_profile.py"):
        return 3
    return 2


def model_key(env: dict, names: set) -> str:
    """The model configuration of a run: its effective value of every non-per-run variable any run sets."""
    return json.dumps({k: env.get(k) for k in sorted(names - ITEM_VARS)}, sort_keys=True)


def hints(runs: list[dict], spec) -> dict:
    """What the shared model is loaded for, so no run needs another load: the LM head when a ladder or the smoke runs,
    target.seq up to the furthest position a profile (BRINGUP_POS) or test_positions times."""
    out = {}
    tests = [r["args"][0].split("::")[0] for r in runs]
    if any(t.endswith(("test_ladder.py", "test_smoke.py")) for t in tests):
        out[LM_HEAD_ENV] = "1"
    pos = [int(r["env"]["BRINGUP_POS"]) for r in runs if r["env"].get("BRINGUP_POS")]
    need = pos or any(t.endswith("test_positions.py") for t in tests)
    chunk = int(spec.get("target.chunk")) if need else 0  # (only then: FIX.25's determinism run needs no hint)
    seqs = [p + chunk for p in pos]
    if any(t.endswith("test_positions.py") for t in tests):
        from models.demos.common.bringup.testing.positions import positions

        seqs.append(max(positions(spec)) + chunk)
    if seqs:
        out[MAX_SEQ_ENV] = str(max(seqs))
    return out


def command(plan: Path) -> str:
    """The batch process: one safe-runner call over tests/test_batch.py reading ``plan``."""
    return f"{PLAN_ENV}={shlex.quote(str(plan))} scripts/run_safe_pytest.sh --no-precompile --run-all {BATCH_TEST}"


def unit_command(cmd: str, spec=None) -> str | None:
    """F259: a gate unit of two or more runs that may share a process (the clean wall and the profiled run, F114) as
    one batch process, its runs in the unit's order (&&: the first failure stops), metrics into the unit's results as
    before; else None (the unit runs as written)."""
    runs = parse_all(cmd)
    if not runs or len(runs) < 2:
        return None
    names = {k for r in runs for k in r["env"]}
    if len({model_key(r["env"], names) for r in runs}) != 1:
        return None
    extra = " ".join(f"{k}={v}" for k, v in hints(runs, spec).items()) if spec is not None else ""
    pre = f"{extra} " if extra else ""
    return (
        f"{pre}{CMD_ENV}={shlex.quote(cmd)} scripts/run_safe_pytest.sh --no-precompile --run-all {BATCH_TEST}"
    ).strip()


def write_plan(path: Path, entries: list[dict], stop_on_fail: bool = False) -> Path:
    """entries: {name, group, env, args, run_all, log, key}. ``group``: runs that stand or fall together (an A/B item:
    its clean wall and its profiled run); a failed run skips the rest of its group, or of the plan with stop_on_fail."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "stop_on_fail": stop_on_fail, "entries": entries}, indent=1) + "\n")
    results_path(path).unlink(missing_ok=True)
    return path


def results_path(plan: Path) -> Path:
    return Path(str(plan) + ".results.json")


def load_results(plan: Path) -> dict:
    """{"running": name or None, "runs": {name: {rc, s, skipped?}}} as the batch process left it."""
    p = results_path(plan)
    return json.loads(p.read_text()) if p.exists() else {"running": None, "runs": {}}


def plan_from_cmd(cmd: str, spec=None) -> dict:
    """The plan of a gate unit's command (unit_command): its runs in order, output to the unit's log (no file of
    their own), &&: stop at the first failure; with ``spec``, the load hints (a hand-run batch needs no others)."""
    runs = parse_all(cmd)
    if not runs:
        raise ValueError(f"{CMD_ENV}: not a command of runs that may share a process: {cmd!r}")
    names = {k for r in runs for k in r["env"]}
    entries = [
        dict(r, name=f"run{i}", group="unit", log=None, key=model_key(dict(os.environ, **r["env"]), names))
        for i, r in enumerate(runs)
    ]
    return {"version": 1, "stop_on_fail": True, "entries": entries, "hints": hints(runs, spec) if spec else {}}
