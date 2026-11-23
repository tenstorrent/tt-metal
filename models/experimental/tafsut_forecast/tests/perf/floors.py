# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Gate floors of the bench records (``floors.json``), looked up per case and precision.

A floor is a bound in the metric's direction (``LOWER_IS_BETTER`` metrics are upper bounds, every other metric a lower
bound), or a ``[lo, hi]`` inclusive range. ``f2_<metric>`` is forward 2 (teacher-forced) of ``<metric>``.
"""

from __future__ import annotations

import json
from pathlib import Path

FLOORS_JSON = Path(__file__).with_name("floors.json")
LOWER_IS_BETTER = {
    "median_err",
    "flat_max_err",
    "inactive_max_err",
    "abs_err_max",
    "bias_row_p50",
    "bias_row_max",
    "bias_row_p95",
    "spread_ratio_p99",
    "spread_ratio_max",
}


def floors(name: str, precision: str) -> dict:
    """The floors of case ``name`` at ``precision``: the ``*/<precision>`` entry, overridden by the case's own."""
    table = json.loads(FLOORS_JSON.read_text())["floors"]
    out = dict(table.get(f"*/{precision}", {}))
    out.update(table.get(f"{name}/{precision}", {}))
    return out


def passes(metric: str, value: float, floor) -> bool:
    """``value`` meets ``floor`` (a bound in the metric's direction, or a ``[lo, hi]`` range)."""
    if isinstance(floor, list):
        return floor[0] <= value <= floor[1]
    base = metric[len("f2_") :] if metric.startswith("f2_") else metric
    return value <= floor if base in LOWER_IS_BETTER else value >= floor
