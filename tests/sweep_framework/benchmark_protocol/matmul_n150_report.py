# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize results_export JSON (OpTest list) for matmul N150 protocol manifests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _percentile_nearest_rank(sorted_vals: list[float], q: float) -> float | None:
    """q in [0, 100]. Nearest-rank method."""
    if not sorted_vals:
        return None
    if q <= 0:
        return sorted_vals[0]
    if q >= 100:
        return sorted_vals[-1]
    k = max(1, math.ceil(q / 100.0 * len(sorted_vals)))
    return sorted_vals[k - 1]


def metrics_to_dict(metrics: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not metrics:
        return out
    if isinstance(metrics, list):
        for m in metrics:
            if not isinstance(m, dict):
                continue
            name = m.get("metric_name")
            val = m.get("metric_value")
            if name is not None and val is not None:
                try:
                    out[str(name)] = float(val)
                except (TypeError, ValueError):
                    pass
    return out


def _is_pass(rec: dict[str, Any]) -> bool:
    st = rec.get("status")
    if isinstance(st, str) and st == "pass":
        return True
    if rec.get("success") is True:
        return True
    return False


def _is_timeout_hang(rec: dict[str, Any]) -> bool:
    st = rec.get("status")
    if isinstance(st, str) and st == "fail_crash_hang":
        return True
    exc = str(rec.get("exception") or "")
    return "TIMED OUT" in exc.upper() or "timeout" in exc.lower()


def _e2e_ms(rec: dict[str, Any]) -> float | None:
    m = metrics_to_dict(rec.get("metrics"))
    for key in ("e2e_perf_ms", "e2e_perf_uncached_ms"):
        if key in m:
            return m[key]
    return None


def _mem_sample(rec: dict[str, Any]) -> dict[str, float | None]:
    m = metrics_to_dict(rec.get("metrics"))
    keys = (
        "peak_l1_memory_per_core_bytes",
        "peak_l1_memory_aggregate_bytes",
        "peak_l1_memory_device_bytes",
        "peak_cb_per_core_bytes",
    )
    return {k: m.get(k) for k in keys}


def load_result_file(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "tests" in data:
        t = data["tests"]
        if isinstance(t, list):
            return t
    raise ValueError(f"Unrecognized results format: {path}")


def _record_time_key(rec: dict[str, Any]) -> tuple[str, str]:
    """Prefer end timestamp, then start timestamp, as sortable keys."""
    end_ts = str(rec.get("test_end_ts") or rec.get("end_time_ts") or "")
    start_ts = str(rec.get("test_start_ts") or rec.get("start_time_ts") or "")
    return (end_ts, start_ts)


def dedupe_latest_by_input_hash(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """
    Deduplicate records by input_hash, keeping the latest record per hash.

    Records without an input_hash are preserved unchanged.
    """
    latest_by_hash: dict[str, dict[str, Any]] = {}
    without_hash: list[dict[str, Any]] = []
    duplicate_rows_dropped = 0

    for rec in records:
        input_hash = rec.get("input_hash")
        if not isinstance(input_hash, str) or not input_hash:
            without_hash.append(rec)
            continue

        prev = latest_by_hash.get(input_hash)
        if prev is None:
            latest_by_hash[input_hash] = rec
            continue

        duplicate_rows_dropped += 1
        if _record_time_key(rec) >= _record_time_key(prev):
            latest_by_hash[input_hash] = rec

    deduped = list(latest_by_hash.values()) + without_hash
    return deduped, duplicate_rows_dropped


def aggregate_partition(records: list[dict[str, Any]], hashes: set[str]) -> dict[str, Any]:
    subset = [r for r in records if r.get("input_hash") in hashes]
    n = len(hashes)
    found_hashes = {r.get("input_hash") for r in subset if r.get("input_hash")}
    missing = len(hashes - found_hashes)

    passes = sum(1 for r in subset if _is_pass(r))
    timeouts = sum(1 for r in subset if _is_timeout_hang(r))
    e2e_vals = sorted(x for r in subset if (x := _e2e_ms(r)) is not None and _is_pass(r))

    mem_vals: dict[str, list[float]] = {}
    for r in subset:
        if not _is_pass(r):
            continue
        for k, v in _mem_sample(r).items():
            if v is None:
                continue
            mem_vals.setdefault(k, []).append(float(v))

    def mem_p50(key: str) -> float | None:
        vals = sorted(mem_vals.get(key, []))
        return _percentile_nearest_rank(vals, 50) if vals else None

    return {
        "vectors_expected": n,
        "results_matched": len(subset),
        "hashes_missing_in_results": max(0, missing),
        "pass_count": passes,
        "pass_rate": (passes / len(subset)) if subset else None,
        "timeout_or_hang_count": timeouts,
        "e2e_ms_p50": _percentile_nearest_rank(e2e_vals, 50),
        "e2e_ms_p95": _percentile_nearest_rank(e2e_vals, 95),
        "e2e_samples": len(e2e_vals),
        "memory_p50_peak_l1_per_core_bytes": mem_p50("peak_l1_memory_per_core_bytes"),
        "memory_p50_peak_l1_aggregate_bytes": mem_p50("peak_l1_memory_aggregate_bytes"),
    }


def build_report(
    manifest: dict[str, Any],
    result_paths: list[Path],
) -> dict[str, Any]:
    all_records_raw: list[dict[str, Any]] = []
    for p in result_paths:
        all_records_raw.extend(load_result_file(p))
    all_records, duplicate_rows_dropped = dedupe_latest_by_input_hash(all_records_raw)

    smoke_h = set(manifest["smoke"])
    train_h = set(manifest["train"])
    hold_h = set(manifest["holdout"])

    train_agg = aggregate_partition(all_records, train_h)
    hold_agg = aggregate_partition(all_records, hold_h)
    smoke_agg = aggregate_partition(all_records, smoke_h)

    comparison: dict[str, Any] = {}
    te = train_agg.get("e2e_ms_p95")
    he = hold_agg.get("e2e_ms_p95")
    if isinstance(te, (int, float)) and isinstance(he, (int, float)) and he > 0:
        comparison["holdout_p95_vs_train_p95_ratio"] = he / te
    comparison["train_p95_e2e_ms"] = te
    comparison["holdout_p95_e2e_ms"] = he

    return {
        "protocol_version": manifest.get("protocol_version"),
        "module_name": manifest.get("module_name"),
        "result_files": [str(p) for p in result_paths],
        "total_result_rows_raw": len(all_records_raw),
        "total_result_rows_deduped": len(all_records),
        "duplicate_result_rows_dropped": duplicate_rows_dropped,
        "smoke": smoke_agg,
        "train": train_agg,
        "holdout": hold_agg,
        "train_vs_holdout": comparison,
    }
