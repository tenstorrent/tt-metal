# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Gate floors of the PCC suite and the perf bench, read from ``gates.json`` next to this file.

``gates.json`` holds, per ``<shape>/<precision>`` (or ``*/<precision>`` for every shape), the floor of each gate
metric. A floor is a bound in the metric's direction (higher is better, except the metrics in ``LOWER_IS_BETTER``,
which are upper bounds), or an inclusive ``[lo, hi]`` range.
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

GATES = Path(__file__).resolve().parent / "gates.json"
LOWER_IS_BETTER = {
    "median_err",
    "flat_max_err",
    "inactive_max_err",
    "abs_err_max",
    "spread_ratio_p99",
    "spread_ratio_max",
}


@lru_cache(maxsize=1)
def _table() -> dict:
    return json.loads(GATES.read_text())["floors"]


def floors_for(name: str, precision: str) -> dict:
    """The floors of ``<name>/<precision>`` on top of the ``*/<precision>`` defaults."""
    out = dict(_table().get(f"*/{precision}", {}))
    out.update(_table().get(f"{name}/{precision}", {}))
    return out


def passes(metric: str, v: float, floor) -> bool:
    """``floor`` is a bound in the metric's direction, or a ``[lo, hi]`` range."""
    if isinstance(floor, list):
        return floor[0] <= v <= floor[1]
    return v <= floor if metric in LOWER_IS_BETTER else v >= floor


def git_stamp() -> dict:
    """HEAD and the porcelain status of the model's tt/, tests/ and ops/ (tracked files), for the bench records."""
    model = Path(__file__).resolve().parents[1]
    git = lambda *a: subprocess.run(["git", *a], cwd=model, capture_output=True, text=True, check=True).stdout.strip()
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": git("status", "--porcelain", "-uno", "--", "tt", "tests", "ops"),
    }
