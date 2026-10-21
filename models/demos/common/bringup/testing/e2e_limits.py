# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F128 (owner): every end-to-end accuracy limit (ladder per layer, state and index overlap, final hidden, top-5; the
contract's KV read-back) is calib x the reference's own drift from full precision on the same rung (reference/drift.py):
limit = 1 - e2e.calib x max(1 - drift, 1 - res). No fixed bar; a rung without its drift.json fails, never falls back."""

from __future__ import annotations

import json
from pathlib import Path

from models.demos.common.bringup.core import defaults, metrics

CALIB = float(defaults.get("e2e.calib"))


def load(golden_dir: Path) -> dict:
    p = Path(golden_dir) / "drift.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p}: no end-to-end margin; generate the rung's official golden and its drift "
            "(reference.generate_golden --numerics official, then reference.drift; F128)"
        )
    return json.loads(p.read_text())


def limit(drift: dict, metric: str) -> float:
    e = drift[metric]
    return 1.0 - CALIB * max(1.0 - e["drift"], 1.0 - e["res"])


def check(drift: dict, metric: str, value: float, failed: list[str]) -> bool:
    """Records ``<metric>_limit`` (the gate's `>= limit`) and appends a failure when ``value`` is below it."""
    lim = limit(drift, metric)
    metrics.record(f"{metric}_limit", lim)
    if value < lim:
        failed.append(f"{metric} {value:.5f} < limit {lim:.5f} (calib {CALIB} x the reference's drift)")
        return False
    return True


def state_limit(drift: dict, name: str) -> float:
    """The limit of a state tensor's worst layer (the contract's KV read-back reports each state's worst layer)."""
    keys = [k for k in drift if k.startswith(f"pcc_state_{name}_L")]
    if not keys:
        raise KeyError(f"no drift for state {name!r} (F128)")
    return min(limit(drift, k) for k in keys)
