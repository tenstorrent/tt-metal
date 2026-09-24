# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Metric sink for bring-up gates.

Tests/scripts call ``record(task_id, name, value)``; ``gate.py`` reads
``bringup/results/<task_id>.json`` after the gate command exits and compares every
metric against the thresholds declared in ``tasks.yaml``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

BRINGUP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BRINGUP_DIR / "results"


def _path(task_id: str) -> Path:
    return RESULTS_DIR / f"{task_id}.json"


def reset(task_id: str) -> None:
    p = _path(task_id)
    if p.exists():
        p.unlink()


def record(task_id: str, name: str, value, **extra) -> None:
    """Record one metric. Last write wins per name. Safe to call from pytest workers."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = _path(task_id)
    data = json.loads(p.read_text()) if p.exists() else {"task": task_id, "metrics": {}}
    if hasattr(value, "item"):
        value = value.item()
    data["metrics"][name] = {"value": value, "t": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, p)


def load(task_id: str) -> dict:
    p = _path(task_id)
    return json.loads(p.read_text())["metrics"] if p.exists() else {}
