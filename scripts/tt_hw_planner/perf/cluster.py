# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cluster joined Tracy ⋈ tracer rows by configuration.

Two clustering keys, in priority order:

  1. (op_code, args_hash)            -- when tracer args are present.
     Identifies "ttnn.matmul at exactly these kwargs"; portable across
     builds because args_hash is over Python kwargs.

  2. (op_code, kernel_hash, shape)   -- fallback, Tracy-only.
     Identifies "ttnn.matmul compiled to this kernel binary at this
     shape"; not portable across builds (kernel hashes change with the
     compiler) but a perfect dedupe key within a single run.

Output: per-row `cluster_id` (mutated on `JoinedRow.cluster_id`) plus a
list of `Cluster` aggregates (median/p99/total time, top-N consumer rank).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median
from typing import Dict, List, Optional, Tuple

from .join import JoinedRow


def _shape_signature(row: JoinedRow) -> str:
    """Lightweight shape string used as part of the fallback cluster key."""
    parts: List[str] = []
    for t in row.inputs:
        s = "x".join(str(t.get(k, "")) for k in ("W", "Z", "Y", "X"))
        layout = t.get("LAYOUT") or ""
        dtype = t.get("DATATYPE") or ""
        parts.append(f"{s}|{layout}|{dtype}")
    return ";".join(parts)


def _cluster_key(row: JoinedRow) -> Tuple[str, ...]:
    if row.args_hash:
        return (row.op_code or "?", row.args_hash)
    if row.compute_kernel_hash:
        return (row.op_code or "?", row.compute_kernel_hash, _shape_signature(row))
    return (row.op_code or "?", "no_hash", _shape_signature(row))


def _short_id(key: Tuple[str, ...]) -> str:
    h = hashlib.sha256("|".join(key).encode()).hexdigest()[:8]
    return f"c_{h}"


@dataclass
class Cluster:
    """Aggregate statistics for one configuration cluster."""

    cluster_id: str
    op_code: str
    args_hash: Optional[str]
    compute_kernel_hash: Optional[str]
    shape_signature: str
    math_fidelity: str
    n_calls: int
    median_device_ns: float
    p99_device_ns: float
    total_device_ns: float
    mean_pm_ideal_ns: Optional[float]
    mean_fpu_util_pct: Optional[float]
    mean_dram_bw_util_pct: Optional[float]
    mean_noc_util_pct: Optional[float]
    mean_eth_bw_util_pct: Optional[float]
    program_cache_hit_rate: Optional[float]
    blocks: List[str] = field(default_factory=list)
    arguments_example: Dict[str, object] = field(default_factory=dict)
    sample_row_index: int = -1

    @property
    def percent_of_peak(self) -> Optional[float]:
        if self.mean_pm_ideal_ns is None or self.median_device_ns <= 0:
            return None
        return 100.0 * self.mean_pm_ideal_ns / self.median_device_ns

    @property
    def slack_ns_total(self) -> Optional[float]:
        if self.mean_pm_ideal_ns is None:
            return None
        return max(0.0, (self.median_device_ns - self.mean_pm_ideal_ns) * self.n_calls)


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((len(s) - 1) * q))
    return s[max(0, min(idx, len(s) - 1))]


def cluster_rows(rows: List[JoinedRow]) -> List[Cluster]:
    """Assign `cluster_id` to every row and return per-cluster aggregates.

    Side effect: mutates `row.cluster_id` in place.
    """
    groups: Dict[Tuple[str, ...], List[JoinedRow]] = defaultdict(list)
    for r in rows:
        key = _cluster_key(r)
        cid = _short_id(key)
        r.cluster_id = cid
        groups[key].append(r)

    clusters: List[Cluster] = []
    for key, members in groups.items():
        cid = _short_id(key)
        device_times = [m.device_kernel_ns for m in members if m.device_kernel_ns is not None]
        if not device_times:
            device_times = [0.0]
        cache_hits = [1.0 if m.program_cache_hit else 0.0 for m in members if m.program_cache_hit is not None]
        sample = members[0]
        blocks = sorted({m.block_path for m in members})
        cluster = Cluster(
            cluster_id=cid,
            op_code=sample.op_code,
            args_hash=sample.args_hash,
            compute_kernel_hash=sample.compute_kernel_hash or None,
            shape_signature=_shape_signature(sample),
            math_fidelity=sample.math_fidelity,
            n_calls=len(members),
            median_device_ns=float(median(device_times)),
            p99_device_ns=_quantile(device_times, 0.99),
            total_device_ns=float(sum(device_times)),
            mean_pm_ideal_ns=_safe_mean([m.pm_ideal_ns for m in members]),
            mean_fpu_util_pct=_safe_mean([m.pm_fpu_util_pct for m in members]),
            mean_dram_bw_util_pct=_safe_mean([m.dram_bw_util_pct for m in members]),
            mean_noc_util_pct=_safe_mean([m.noc_util_pct for m in members]),
            mean_eth_bw_util_pct=_safe_mean([m.eth_bw_util_pct for m in members]),
            program_cache_hit_rate=(sum(cache_hits) / len(cache_hits)) if cache_hits else None,
            blocks=blocks,
            arguments_example=dict(sample.arguments),
            sample_row_index=sample.row_index,
        )
        clusters.append(cluster)

    clusters.sort(key=lambda c: c.total_device_ns, reverse=True)
    return clusters


def cluster_by_id(clusters: List[Cluster]) -> Dict[str, Cluster]:
    return {c.cluster_id: c for c in clusters}
