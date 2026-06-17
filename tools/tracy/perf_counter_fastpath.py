# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Fast-path perf-counter post-process (Phase 2a, also the Phase 2b OOM fix).

The C++ fast path emits ``cpp_perf_counters.csv`` — a compact per-(op, core,
counter) table of only the perf-counter markers. This module computes the
already-validated utilization metrics from it and merges the
PERF_COUNTER_CSV_HEADERS columns into ``cpp_device_perf_report.csv``, keyed by
op. The existing report passthrough then carries them to the final CSV, so
counters no longer force the full per-core device-log load that OOMs at scale.

Only the per-core distribution stats (Min/Median/Max/Avg) from
``compute_perf_counter_metrics`` are merged — the grid-summed columns
(``Avg ... on full grid``) need the per-op kernel-cycle count and are left to
the legacy path.
"""

import csv
import os

import pandas as pd

from tracy.perf_counter_analysis import PERF_COUNTER_CSV_HEADERS, compute_perf_counter_metrics

COMPACT_COUNTER_CSV = "cpp_perf_counters.csv"
CPP_DEVICE_PERF_REPORT = "cpp_device_perf_report.csv"

# Mirror perf_counter_analysis: these aggregate to a raw rate, not a percentage.
_RATE_METRICS = {"T0 Instrn Issue Rate", "T1 Instrn Issue Rate", "T2 Instrn Issue Rate", "Avg HF Cycles Per Instrn"}

_HEADER_SET = set(PERF_COUNTER_CSV_HEADERS)


def _stat_columns(per_op_stats):
    """Flatten per_op_stats into {(run_host_id, trace_id_count): {column: value}}.

    Emits only columns that exist in PERF_COUNTER_CSV_HEADERS so the report
    passthrough recognizes them.
    """
    by_key = {}
    for base_name, stat_dict in per_op_stats.items():
        if not stat_dict:
            continue
        suffix = "" if base_name in _RATE_METRICS else " (%)"
        for stat in ("min", "median", "max", "avg"):
            column = f"{base_name} {stat.capitalize()}{suffix}"
            if column not in _HEADER_SET:
                continue
            for key, value in stat_dict.get(stat, {}).items():
                by_key.setdefault(key, {})[column] = value
    return by_key


def merge_counter_metrics_into_cpp_report(log_folder, device_arch, total_compute_cores):
    """Merge per-op counter utilization columns into cpp_device_perf_report.csv.

    No-op if the compact counter CSV is absent. Matches report rows on
    GLOBAL CALL COUNT (run_host_id), disambiguating by replay session when the
    op is a traced replay.
    """
    log_folder = str(log_folder)
    compact = os.path.join(log_folder, COMPACT_COUNTER_CSV)
    report = os.path.join(log_folder, CPP_DEVICE_PERF_REPORT)
    if not os.path.isfile(compact) or not os.path.isfile(report):
        return

    perf_counter_df = pd.read_csv(compact)
    if perf_counter_df.empty:
        return

    metrics = compute_perf_counter_metrics(perf_counter_df, device_arch, total_compute_cores)
    by_key = _stat_columns(metrics["per_op_stats"])
    if not by_key:
        return

    # Index columns by run_host_id, keeping the trace session for replay disambiguation.
    by_call_count = {}
    for (run_host_id, trace_id_count), cols in by_key.items():
        by_call_count.setdefault(int(run_host_id), {})[int(trace_id_count)] = cols

    with open(report, newline="") as f:
        reader = csv.DictReader(f)
        in_fields = reader.fieldnames or []
        rows = list(reader)

    new_columns = []
    for cols in by_key.values():
        for c in cols:
            if c not in in_fields and c not in new_columns:
                new_columns.append(c)

    def _match(row):
        try:
            call_count = int(row["GLOBAL CALL COUNT"])
        except (KeyError, ValueError):
            return None
        sessions = by_call_count.get(call_count)
        if not sessions:
            return None
        session_raw = str(row.get("METAL TRACE REPLAY SESSION ID", "")).strip()
        if session_raw not in ("", "nan") and int(session_raw) in sessions:
            return sessions[int(session_raw)]
        if len(sessions) == 1:
            return next(iter(sessions.values()))
        return sessions.get(-1)

    for row in rows:
        cols = _match(row)
        if cols:
            for c, v in cols.items():
                row[c] = v

    out_fields = in_fields + new_columns
    with open(report, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
