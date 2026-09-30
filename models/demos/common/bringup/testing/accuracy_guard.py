# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Accuracy first for perf picks: a change that fails a frozen accuracy test of its task is never profiled.

The marker ``<run dir>/ab/<task>/ACCURACY_FAIL`` lists the frozen tests of a perf task (step perf with a role) that
failed in the current attempt, one pytest node id per line. The component and swap tests write it (``note``: a failed
test of the task's gate adds its line, a pass removes it), the orchestrator writes it when a gate fails on an accuracy
test and clears it when the task runs again. ``testing/profile.py`` calls ``check`` first and refuses while the marker
lists anything. The orchestrator's own e2e A/B profile sets ``BRINGUP_AB=1``, which ``check`` lets through.
"""

from __future__ import annotations

import os
from pathlib import Path

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.core.gate import run_name
from models.demos.common.bringup.core.ledger import Ledger

MARKER = "ACCURACY_FAIL"
AB_ENV = "BRINGUP_AB"


class AccuracyFailed(RuntimeError):
    pass


def marker(spec, tid: str) -> Path:
    return spec.run_dir(run_name(Ledger(spec.bringup_dir))) / "ab" / tid / MARKER


def perf_task(spec, tid: str | None) -> dict | None:
    """The task if it is a perf pick (step perf, run by an agent role), else None."""
    if not tid:
        return None
    try:
        t = Ledger(spec.bringup_dir).task(tid)
    except Exception:
        return None
    return t if t.get("step") == "perf" and t.get("role") else None


def failing(spec, tid: str) -> list[str]:
    p = marker(spec, tid)
    return [x for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def write(spec, tid: str, tests: list[str]) -> Path:
    p = marker(spec, tid)
    tests = list(dict.fromkeys(t for t in tests if t))
    if tests:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(tests) + "\n")
    else:
        p.unlink(missing_ok=True)
    return p


def clear(spec, tid: str) -> None:
    marker(spec, tid).unlink(missing_ok=True)


def note(spec, ok: bool) -> None:
    """Called by the component and swap tests with their verdict: under a perf task, a failed test of the task's gate
    is added to the marker and a passed one removed. Anything else (other tasks, the precompile pass) is ignored."""
    tid = metrics.task_id(default="")
    task = perf_task(spec, tid)
    node = os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" ", 1)[0]
    if task is None or not node or metrics.in_precompile_collect_pass() or os.environ.get(AB_ENV) == "1":
        return
    if node.split("::", 1)[0] not in task["gate"]["cmd"]:
        return  # not one of this task's frozen tests
    tests = [t for t in failing(spec, tid) if t != node] + ([] if ok else [node])
    write(spec, tid, tests)


def check(spec) -> None:
    """Refuse to profile a perf task whose frozen accuracy test failed in this attempt (roles.yaml perf: accuracy
    first). The orchestrator's A/B profile (BRINGUP_AB=1) is allowed."""
    tid = metrics.task_id(default="")
    if os.environ.get(AB_ENV) == "1" or perf_task(spec, tid) is None:
        return
    bad = failing(spec, tid)
    if bad:
        raise AccuracyFailed(
            f"refusing to profile {tid}: frozen accuracy test(s) of this task failed in this attempt: {bad} "
            f"(marker {marker(spec, tid)}). Accuracy first: write the failing metrics against their limits into the "
            "report and stop; never profile a change that fails accuracy."
        )
