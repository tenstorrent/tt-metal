# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Metric sink for gates.

A test or script calls ``record(name, value)``. The gate runner sets ``BRINGUP_TASK`` and
``BRINGUP_RESULTS_DIR`` before it runs the gate command, then reads ``<results>/<task>.json`` and compares
every metric with the thresholds in the task ledger. Outside a gate the metrics go to
``generated/bringup_adhoc/`` under the task id ``adhoc``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

TASK_ENV = "BRINGUP_TASK"
# F79: the task whose results file a metric lands in, when it is not the task whose test runs (an overseer fix job
# reruns C.swa.attention's gate as FIX.2: the test sees BRINGUP_TASK=C.swa.attention, its metrics go to FIX.2)
RECORD_ENV = "BRINGUP_RECORD_TASK"
RESULTS_ENV = "BRINGUP_RESULTS_DIR"
SUFFIX_ENV = "BRINGUP_METRIC_SUFFIX"  # F230: appended to every metric name (a profile at a position: _p<start>)
_REPO = Path(__file__).resolve().parents[5]


def task_id(default: str = "adhoc") -> str:
    return os.environ.get(TASK_ENV) or default


def record_task(default: str = "adhoc") -> str:
    return os.environ.get(RECORD_ENV) or task_id(default)


def results_dir() -> Path:
    return Path(os.environ.get(RESULTS_ENV) or _REPO / "generated/bringup_adhoc")


def _path(task: str, root: Path | None = None) -> Path:
    return (root or results_dir()) / f"{task}.json"


def reset(task: str, root: Path | None = None) -> None:
    _path(task, root).unlink(missing_ok=True)


def in_precompile_collect_pass() -> bool:
    """run_safe_pytest.sh may run each test body once on fake tensors before the real pass (up_front_collect)."""
    for name, mod in list(sys.modules.items()):
        if (
            name.endswith("up_front_collect")
            and getattr(mod, "_INLINE", False)
            and getattr(mod, "_PASS", None) == "collect"
        ):
            return True
    return False


def suffixed(name: str) -> str:
    """F230: the name a metric is recorded under: with BRINGUP_METRIC_SUFFIX, so one gate can hold several positions."""
    return name + os.environ.get(SUFFIX_ENV, "")


def record(name: str, value, task: str | None = None, **extra) -> None:
    """Record one metric for the active task. The last write of a name wins. Ignored in the precompile collect pass."""
    if in_precompile_collect_pass():
        return
    task = task or record_task()
    name = suffixed(name)
    p = _path(task)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text()) if p.exists() else {"task": task, "metrics": {}}
    if hasattr(value, "item"):
        value = value.item()
    data["metrics"][name] = {"value": value, "t": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
    tmp = p.with_suffix(f".{os.getpid()}.tmp")  # two processes on one task (dev proofs) must not share it
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, p)


def record_json(suffix: str, data: dict, task: str | None = None) -> Path | None:
    """Write a small side record ``<results>/<task>_<suffix>.json`` (a gate output: deleted before the gate runs and
    committed with it). Ignored in the precompile collect pass."""
    if in_precompile_collect_pass():
        return None
    task = task or record_task()
    p = results_dir() / f"{task}_{suffix}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"task": task, "t": time.strftime("%Y-%m-%dT%H:%M:%S"), **data}, indent=1) + "\n")
    os.replace(tmp, p)
    return p


def load(task: str, root: Path | None = None) -> dict:
    p = _path(task, root)
    return json.loads(p.read_text())["metrics"] if p.exists() else {}


def pcc(a, b) -> float:
    """Pearson correlation in float64. Tests use this, never comp_pcc: the precompile pass stubs comp_pcc to 0.999999."""
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(bool((a == b).all()))


def _physical_cores() -> int:
    try:
        import psutil

        return psutil.cpu_count(logical=False) or os.cpu_count()
    except ImportError:
        return os.cpu_count()


def device_work_running() -> bool:
    """F170: another process on this box runs a bring-up orchestrator or a device test (run_safe_pytest.sh)."""
    try:
        import psutil
    except ImportError:
        return False
    me = os.getpid()
    for p in psutil.process_iter(["pid", "cmdline"]):
        cl = " ".join(p.info.get("cmdline") or [])
        if p.info["pid"] != me and ("bringup.orchestrator" in cl or "run_safe_pytest.sh" in cl):
            return True
    return False


def cpu_threads() -> int:
    """Torch threads for CPU gates: physical cores, not SMT siblings (a 64-row expert GEMM: 4 ms at 16, 27 ms at 32).
    BRINGUP_CPU_THREADS overrides it (F165: a long CPU job sharing the box with a live bring-up).
    F170 (audit item 19): a CPU job started outside a gate (no BRINGUP_TASK) while device work runs on the box takes a
    quarter of the physical cores at the lowest priority by default, so it cannot starve a device gate's CPU reference
    into its timeout (DeepSeek-V4.1: the full-model goldens next to FIX.10). Call once, at the job's start."""
    if os.environ.get("BRINGUP_CPU_THREADS"):
        return int(os.environ["BRINGUP_CPU_THREADS"])
    cores = _physical_cores()
    if not os.environ.get(TASK_ENV) and device_work_running():
        try:
            os.nice(19 - os.nice(0))
        except OSError:
            pass
        return max(1, cores // 4)
    return cores
