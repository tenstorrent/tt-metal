# SPDX-License-Identifier: Apache-2.0
"""Adaptive profiler-drain sizing (MODEL-AGNOSTIC).

Sizes TT_PERF_FLUSH_EVERY (ops between profiler drains) so the on-device marker buffer never overflows,
without a fixed guess. Three tiers, all keyed off the real hardware buffer CAP (markers) — no fudge
factors, no hand-picked ratios:

  static_flush_estimate  START, no run: estimate markers/op from the op-type mix (from the op-sig probe)
                         x cores x chips and a one-time per-arch markers-per-op-type table. Returns None
                         if ANY present op-type is missing from the table (a novel / custom kernel) so the
                         caller keeps its default and leans on the backstop -- no regression vs today.
  safe_flush_interval    From a CLEAN run's measured per-op marker counts: the largest op-window whose
                         worst-case cumulative markers still fit CAP. Data-derived, no safety fraction.
  recompute_interval     ON OVERFLOW: jump straight to the interval implied by the overflowing (partial)
                         run's MEASURED density -- N = CAP // density -- not a blind halve. Converges,
                         since a smaller window yields a cleaner next measurement.
"""

from __future__ import annotations

import json
from pathlib import Path


def safe_flush_interval(per_op_markers, cap: int):
    """Largest N such that no N-consecutive-op window's cumulative markers exceed CAP. `per_op_markers`
    is the ordered per-op marker count on the busiest RISC from a clean run. No fudge factor. Returns
    None on no data; 1 when even a single op exceeds CAP (only a bigger buffer can help that)."""
    p = [int(x) for x in per_op_markers if int(x) >= 0]
    if not p or cap <= 0:
        return None
    best = 0
    for n in range(1, len(p) + 1):
        w = sum(p[:n])
        worst = w
        for i in range(n, len(p)):
            w += p[i] - p[i - n]
            if w > worst:
                worst = w
        if worst > cap:
            break
        best = n
    return max(1, best)


def recompute_interval(measured_markers_per_op: float, cap: int) -> int:
    """On overflow: recompute directly from the run's MEASURED marker density (a lower bound from the
    partial CSV) -- N = CAP // density, at least 1. Data-driven correction, converges on real data."""
    d = float(measured_markers_per_op or 0)
    if d <= 0 or cap <= 0:
        return 1
    return max(1, int(cap // d))


def static_flush_estimate(op_type_counts, markers_per_op_type, cores: int, chips: int, cap: int):
    """START estimate from static info: worst markers/op = max over PRESENT op-types of
    table[op_type] x cores x chips; N = CAP // worst. Returns None if any present op-type is absent from
    the table (unseen / custom kernel) so the caller keeps its default (backstop covers it)."""
    if not op_type_counts or cap <= 0 or cores <= 0 or chips <= 0:
        return None
    worst = 0
    for op_type in op_type_counts:
        per_core = markers_per_op_type.get(op_type)
        if per_core is None:
            return None
        worst = max(worst, int(per_core) * int(cores) * int(chips))
    if worst <= 0:
        return None
    return max(1, int(cap // worst))


def _cache_path(root: Path) -> Path:
    return Path(root) / "models/experimental/perf_automation/cc_optimize/.drain_cache.json"


def _fingerprint(node) -> str:
    try:
        base = Path(str(node).split("::", 1)[0])
        mt = max((f.stat().st_mtime for f in base.parent.rglob("*.py")), default=0.0)
        return str(int(mt))
    except Exception:  # noqa: BLE001
        return ""


def flush_cache_get(root: Path, node, case):
    try:
        entry = json.loads(_cache_path(root).read_text()).get(f"{node}|{case}")
        if entry and entry.get("fp") == _fingerprint(node):
            return int(entry["n"])
    except Exception:  # noqa: BLE001
        pass
    return None


def flush_cache_put(root: Path, node, case, n: int) -> None:
    try:
        path = _cache_path(root)
        data = json.loads(path.read_text()) if path.is_file() else {}
        data[f"{node}|{case}"] = {"n": int(n), "fp": _fingerprint(node)}
        path.write_text(json.dumps(data, indent=1))
    except Exception:  # noqa: BLE001
        pass
