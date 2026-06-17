# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Phase 2a of the Nsight-counters plan: counters on the C++ fast post-process path.

The C++ path emits a compact per-(op, core, counter) CSV (cpp_perf_counters.csv);
this step computes the already-validated utilization metrics from it and merges
the PERF_COUNTER_CSV_HEADERS columns into cpp_device_perf_report.csv, so the
existing passthrough carries them to the final report -- without the full
per-core device-log load that OOMs at mesh scale.
"""

import csv
from pathlib import Path

import pytest

from tracy.perf_counter_fastpath import merge_counter_metrics_into_cpp_report


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _compact_counter_csv(path, run_host_id, fpu_values, ref=1000):
    # One FPU_COUNTER row per core; util = value/ref*100.
    header = ["run_host_id", "trace_id_count", "core_x", "core_y", "risc_type", "counter type", "value", "ref cnt"]
    rows = [[run_host_id, -1, i, 0, "BRISC", "FPU_COUNTER", v, ref] for i, v in enumerate(fpu_values)]
    _write_csv(path, header, rows)


def test_merge_adds_fpu_util_column(tmp_path):
    log = tmp_path
    # 4 cores at 20/40/60/80% FPU util -> median 50%.
    _compact_counter_csv(log / "cpp_perf_counters.csv", run_host_id=7, fpu_values=[200, 400, 600, 800])
    _write_csv(
        log / "cpp_device_perf_report.csv",
        ["GLOBAL CALL COUNT", "METAL TRACE ID", "METAL TRACE REPLAY SESSION ID", "DEVICE ID"],
        [[7, "", "", 0]],
    )

    merge_counter_metrics_into_cpp_report(log, device_arch="blackhole", total_compute_cores=140)

    with (log / "cpp_device_perf_report.csv").open() as f:
        row = next(csv.DictReader(f))
    assert "FPU Util Median (%)" in row
    assert abs(float(row["FPU Util Median (%)"]) - 50.0) < 0.01


def test_merge_is_noop_without_compact_csv(tmp_path):
    log = tmp_path
    _write_csv(
        log / "cpp_device_perf_report.csv",
        ["GLOBAL CALL COUNT", "METAL TRACE ID", "METAL TRACE REPLAY SESSION ID"],
        [[1, "", ""]],
    )
    # No cpp_perf_counters.csv -> must not raise and must not add columns.
    merge_counter_metrics_into_cpp_report(log, device_arch="blackhole", total_compute_cores=140)
    with (log / "cpp_device_perf_report.csv").open() as f:
        fields = csv.DictReader(f).fieldnames
    assert fields == ["GLOBAL CALL COUNT", "METAL TRACE ID", "METAL TRACE REPLAY SESSION ID"]


def test_merge_only_emits_known_headers(tmp_path):
    log = tmp_path
    _compact_counter_csv(log / "cpp_perf_counters.csv", run_host_id=1, fpu_values=[500, 500])
    _write_csv(log / "cpp_device_perf_report.csv", ["GLOBAL CALL COUNT"], [[1]])
    merge_counter_metrics_into_cpp_report(log, device_arch="blackhole", total_compute_cores=140)
    with (log / "cpp_device_perf_report.csv").open() as f:
        fields = csv.DictReader(f).fieldnames
    from tracy.perf_counter_analysis import PERF_COUNTER_CSV_HEADERS

    known = set(PERF_COUNTER_CSV_HEADERS) | {"GLOBAL CALL COUNT"}
    assert all(h in known for h in fields), f"emitted unknown header: {set(fields) - known}"
