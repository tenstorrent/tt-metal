# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel gating + sizing (Increment 1).

TP is applied ONLY when a model does not fit on one chip; otherwise the single-chip ladder runs and
data parallelism is a free deploy-time wrapper. The TP degree is the SMALLEST legal value that makes
the model fit, where legal means it divides num_heads, keeps hidden/TP tile-aligned, and maps to a
mesh axis. The remaining chips become data-parallel replicas."""
from __future__ import annotations

import math

CAPACITY_HEADROOM = 0.8
TILE = 32


def fits_on_one_chip(weight_bytes: int, per_chip_capacity: int) -> bool:
    return weight_bytes < per_chip_capacity * CAPACITY_HEADROOM


def tp_regime(mesh_chips: int, weight_bytes: int, per_chip_capacity: int) -> bool:
    return mesh_chips >= 2 and not fits_on_one_chip(weight_bytes, per_chip_capacity)


def legal_tp_degrees(total_chips: int, num_heads: int, hidden: int) -> list[int]:
    return [
        d
        for d in range(2, total_chips + 1)
        if total_chips % d == 0 and num_heads % d == 0 and (hidden // d) % TILE == 0
    ]


def decide_tp(weight_bytes: int, per_chip_capacity: int, total_chips: int, num_heads: int, hidden: int) -> dict:
    tp_min = math.ceil(weight_bytes / (per_chip_capacity * CAPACITY_HEADROOM))
    if tp_min <= 1:
        return {"tp": 1, "dp": total_chips}
    candidates = [d for d in legal_tp_degrees(total_chips, num_heads, hidden) if d >= tp_min]
    if not candidates:
        return {
            "error": f"model needs >= {tp_min} chips but no legal TP divisor fits "
            f"(total_chips={total_chips}, num_heads={num_heads}, hidden={hidden})"
        }
    tp = min(candidates)
    return {"tp": tp, "dp": total_chips // tp}


LATENCY_METRICS = ("device_ms", "wall_ms")


def tp_latency_eligible(metric: str, allow_tp_latency: bool, total_chips: int) -> bool:
    return bool(allow_tp_latency) and (metric or "").lower() in LATENCY_METRICS and total_chips >= 2


def decide_parallelism(
    weight_bytes: int,
    per_chip_capacity: int,
    total_chips: int,
    num_heads: int,
    hidden: int,
    metric: str = "device_ms",
    allow_tp_latency: bool = False,
) -> dict:
    if fits_on_one_chip(weight_bytes, per_chip_capacity):
        if tp_latency_eligible(metric, allow_tp_latency, total_chips):
            return {
                "route": "single-chip+tp-latency",
                "tp": 1,
                "dp": total_chips,
                "tp_regime": True,
                "reason": "model fits on one chip; --tp-latency under a latency metric -> may sweep TP up to cut latency (spends DP throughput)",
            }
        return {
            "route": "single-chip",
            "tp": 1,
            "dp": total_chips,
            "tp_regime": False,
            "reason": "model fits on one chip -> single-chip optimize; DP across the rest for throughput at deploy",
        }
    sized = decide_tp(weight_bytes, per_chip_capacity, total_chips, num_heads, hidden)
    if "error" in sized:
        return {"route": "infeasible", "tp_regime": False, "reason": sized["error"]}
    return {
        "route": "tensor-parallel",
        "tp": sized["tp"],
        "dp": sized["dp"],
        "tp_regime": True,
        "reason": f"model does not fit on one chip -> TP={sized['tp']}, DP={sized['dp']}",
    }
