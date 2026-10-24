# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel gating + sizing (Increment 1), and how the chips TP leaves over are used.

TP is applied ONLY when a model does not fit on one chip; otherwise the single-chip ladder runs. The TP
degree is the SMALLEST legal value that makes the model fit, where legal means it divides num_heads,
keeps hidden/TP tile-aligned, and maps to a mesh axis.

The chips TP leaves over used to become data-parallel replicas unconditionally. A replica only earns its
keep when there is a request for it to serve: with fewer concurrent requests than spare chips the extra
replicas idle, and one request runs no faster than it does on its TP group alone. So the spare chips are
split by the workload the run states (split_spare_chips): replicas for the requests there are, and
SEQUENCE PARALLELISM -- one request's tokens cut into equal, tile-aligned slices, one slice per chip
group, the groups exchanging only attention K/V -- for the chips that would otherwise idle. A run that
states no workload keeps the old answer, so nothing changes until a run says how much it serves.
"""

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


def seq_parallel_legal(seq_len, sp) -> bool:
    """Whether `seq_len` tokens cut into `sp` equal, tile-aligned slices (one slice per chip group).

    A slice is one group's share of the sequence, so it has to be whole tiles like any other token axis
    the device sees. sp=1 is always legal: one slice is the whole sequence."""
    try:
        seq_len, sp = int(seq_len or 0), int(sp or 0)
    except (TypeError, ValueError):
        return False
    if sp < 1:
        return False
    if sp == 1:
        return True
    return seq_len > 0 and seq_len % sp == 0 and (seq_len // sp) % TILE == 0


def _divisors_desc(n: int) -> list[int]:
    return [d for d in range(n, 0, -1) if n % d == 0]


def split_spare_chips(spare, requests=None, seq_len=None) -> tuple[int, int]:
    """(dp, sp) for the `spare` chips a TP group leaves over, from the workload the run states.

    An unknown workload (requests or seq_len missing or non-positive) returns (spare, 1): every spare chip
    is a replica, exactly as before. Otherwise the replicas are the largest divisor of `spare` the requests
    can keep busy, and the chips left in each replica split one request's tokens at the largest degree
    the sequence allows (seq_parallel_legal); a degree the tokens cannot be cut to falls back towards
    replicas rather than to an illegal slice. dp * sp == spare always holds."""
    spare = max(1, int(spare or 1))
    try:
        req = int(requests) if requests is not None else 0
        seq = int(seq_len) if seq_len is not None else 0
    except (TypeError, ValueError):
        req = seq = 0
    if req <= 0 or seq <= 0:
        return spare, 1
    dp = next(d for d in _divisors_desc(spare) if d <= req)  # 1 always divides, so this never fails
    for sp in _divisors_desc(spare // dp):  # sp=1 is legal, so this always returns
        if seq_parallel_legal(seq, sp):
            return spare // sp, sp
    return spare, 1


LATENCY_METRICS = ("device_ms", "wall_ms")


def tp_latency_eligible(metric: str, total_chips: int) -> bool:
    return (metric or "").lower() in LATENCY_METRICS and total_chips >= 2


def decide_parallelism(
    weight_bytes: int,
    per_chip_capacity: int,
    total_chips: int,
    num_heads: int,
    hidden: int,
    metric: str = "device_ms",
    sp: int = 1,
) -> dict:
    """The route for this model on this mesh. `sp` is the sequence-parallel degree the run PLANNED and
    exported for its mesh (perf_adapter.resolve_seq_parallel) -- the planner split the spare chips by
    the stated workload with split_spare_chips; the route honours that answer rather than deriving its
    own from facts that may have changed since (the engine re-pins the sequence length on a shape
    retry), so the route and the mesh the run opens cannot disagree. Unplanned (1), the routes are
    exactly what they were."""
    if fits_on_one_chip(weight_bytes, per_chip_capacity):
        try:
            sp = int(sp or 1)
        except (TypeError, ValueError):
            sp = 1
        if sp > 1 and total_chips % sp == 0:
            # The groups splitting tokens ARE the chips the per-matmul TP sweep would use, so the sweep
            # stays off: one decision about what the spare chips do, not two competing ones.
            dp = total_chips // sp
            return {
                "route": "single-chip+seq-parallel",
                "tp": 1,
                "dp": dp,
                "sp": sp,
                "tp_regime": False,
                "floor": 1,
                "reason": (
                    f"model fits on one chip; the run planned SP={sp}: each request's tokens split over "
                    f"{sp} chip groups, DP={dp} replica(s) on {total_chips} chips"
                ),
            }
        if tp_latency_eligible(metric, total_chips):
            return {
                "route": "single-chip+tp-latency",
                "tp": 1,
                "dp": total_chips,
                "sp": 1,
                "tp_regime": True,
                "floor": 1,
                "reason": "model fits on one chip; latency metric on a mesh -> sweep TP per matmul, keep fastest",
            }
        return {
            "route": "single-chip",
            "tp": 1,
            "dp": total_chips,
            "sp": 1,
            "tp_regime": False,
            "floor": 1,
            "reason": "model fits on one chip -> single-chip optimize; DP across the rest for throughput at deploy",
        }
    sized = decide_tp(weight_bytes, per_chip_capacity, total_chips, num_heads, hidden)
    if "error" in sized:
        return {"route": "infeasible", "tp_regime": False, "reason": sized["error"]}
    return {
        "route": "tensor-parallel",
        "tp": sized["tp"],
        "dp": sized["dp"],
        "sp": 1,
        "tp_regime": True,
        "floor": sized["tp"],
        "reason": f"model does not fit on one chip -> TP={sized['tp']}, DP={sized['dp']}",
    }
