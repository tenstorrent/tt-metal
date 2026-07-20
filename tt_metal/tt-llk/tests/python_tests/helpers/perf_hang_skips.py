# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hang-skip registry for Quasar perf suite.

When the all-modes runner detects a hang, it records
``{test_module_stem: [PerfRunType.name, ...]}`` here. Perf tests call
``filter_run_types`` so the hanging mode is skipped on retry / later runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pytest

from .llk_params import PerfRunType

SKIPS_PATH = Path(__file__).resolve().parent.parent / "quasar" / "perf_hang_skips.json"


def load_skips() -> dict[str, list[str]]:
    if not SKIPS_PATH.exists():
        return {}
    try:
        data = json.loads(SKIPS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {str(k): [str(x) for x in v] for k, v in data.items()}


def save_skips(data: dict[str, list[str]]) -> None:
    SKIPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SKIPS_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def add_skip(test_module: str, run_type_name: str) -> bool:
    """Record a hang skip. Returns True if newly added."""
    stem = Path(test_module).stem
    data = load_skips()
    cur = list(data.get(stem, []))
    if run_type_name in cur:
        return False
    cur.append(run_type_name)
    data[stem] = cur
    save_skips(data)
    return True


def filter_run_types(test_module: str, run_types: Iterable) -> list:
    """Drop PerfRunTypes previously recorded as hanging for this test module."""
    stem = Path(test_module).stem
    banned = set(load_skips().get(stem, []))
    run_types = list(run_types)
    if not banned:
        return run_types

    def _name(rt) -> str:
        return rt.name if isinstance(rt, PerfRunType) else getattr(rt, "name", str(rt))

    filtered = [rt for rt in run_types if _name(rt) not in banned]
    if not filtered:
        pytest.skip(
            f"All PerfRunTypes hung/skipped for {stem}: {sorted(banned)}"
        )
    return filtered
