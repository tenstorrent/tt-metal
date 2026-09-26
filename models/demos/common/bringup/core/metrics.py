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
RESULTS_ENV = "BRINGUP_RESULTS_DIR"
_REPO = Path(__file__).resolve().parents[5]


def task_id(default: str = "adhoc") -> str:
    return os.environ.get(TASK_ENV) or default


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


def record(name: str, value, task: str | None = None, **extra) -> None:
    """Record one metric for the active task. The last write of a name wins. Ignored in the precompile collect pass."""
    if in_precompile_collect_pass():
        return
    task = task or task_id()
    p = _path(task)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text()) if p.exists() else {"task": task, "metrics": {}}
    if hasattr(value, "item"):
        value = value.item()
    data["metrics"][name] = {"value": value, "t": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, p)


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


def cpu_threads() -> int:
    """Torch threads for CPU gates: physical cores, not SMT siblings (a 64-row expert GEMM: 4 ms at 16, 27 ms at 32)."""
    try:
        import psutil

        return psutil.cpu_count(logical=False) or os.cpu_count()
    except ImportError:
        return os.cpu_count()
