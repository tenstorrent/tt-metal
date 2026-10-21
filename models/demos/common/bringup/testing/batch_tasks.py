# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F142: pytest plugin for a batched verify (core/verify.py --batch): many tasks' frozen test files in one process.

Each test runs with BRINGUP_TASK and BRINGUP_RECORD_TASK set to the task its file belongs to (BRINGUP_BATCH_TASKS:
{absolute test path: task id}), so its metrics land in that task's results file as in its own gate run; each task's
test outcomes go to BRINGUP_BATCH_OUTCOMES ({task: "passed" | "failed"}, a failure wins).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from models.demos.common.bringup.core import metrics as M

MAP_ENV = "BRINGUP_BATCH_TASKS"
OUT_ENV = "BRINGUP_BATCH_OUTCOMES"


def task_of(path) -> str | None:
    return json.loads(os.environ.get(MAP_ENV) or "{}").get(str(Path(str(path)).resolve()))


@pytest.fixture(autouse=True)
def _batch_task(request):
    tid = task_of(request.node.path)
    if tid is None:
        yield
        return
    old = {k: os.environ.get(k) for k in (M.TASK_ENV, M.RECORD_ENV)}
    os.environ[M.TASK_ENV] = os.environ[M.RECORD_ENV] = tid
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


NODE_TASK: dict[str, str] = {}  # nodeid -> task (from the collected items' absolute paths)


def pytest_collection_modifyitems(items):
    for it in items:
        tid = task_of(it.path)
        if tid is not None:
            NODE_TASK[it.nodeid] = tid


def pytest_runtest_logreport(report):
    out, tid = os.environ.get(OUT_ENV), NODE_TASK.get(report.nodeid)
    if not out or tid is None or (report.when != "call" and not report.failed):
        return
    p = Path(out)
    got = json.loads(p.read_text()) if p.exists() else {}
    if got.get(tid) != "failed":
        got[tid] = "failed" if report.failed else "passed"
    p.write_text(json.dumps(got, indent=1, sort_keys=True))
