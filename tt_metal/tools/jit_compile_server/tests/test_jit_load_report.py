# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_report_module():
    script_path = Path(__file__).resolve().parents[1] / "jit_load_report.py"
    spec = importlib.util.spec_from_file_location("jit_load_report", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_report_module()


def test_parse_jit_server_line_extracts_fields():
    line = (
        "[jit_server addr=localhost:6000 ts=1711111111111] "
        "count=42 total_compile_time_ms=900 queued=2 inflight=3 "
        "peak_inflight=7 bytes_in=12345 bytes_out=67890"
    )

    sample = MODULE.parse_jit_server_line(line)
    assert sample is not None
    assert sample.addr == "localhost:6000"
    assert sample.ts == 1711111111111
    assert sample.count == 42
    assert sample.total_compile_time_ms == 900
    assert sample.queued == 2
    assert sample.inflight == 3
    assert sample.peak_inflight == 7
    assert sample.bytes_in == 12345
    assert sample.bytes_out == 67890


def test_parse_jit_server_samples_ignores_non_matching_lines():
    lines = [
        "random non-matching line",
        "[jit_server addr=a:1 ts=1000] count=1 total_compile_time_ms=2 queued=0 inflight=0 peak_inflight=1 bytes_in=10 bytes_out=20",
        "another non-matching line",
    ]

    samples = MODULE.parse_jit_server_samples(lines)
    assert len(samples) == 1
    assert samples[0].addr == "a:1"


def test_build_report_uses_last_sample_without_window():
    lines = [
        "[jit_server addr=server-a:6000 ts=1000] count=10 total_compile_time_ms=100 queued=0 inflight=0 peak_inflight=2 bytes_in=1000 bytes_out=500",
        "[jit_server addr=server-a:6000 ts=3000] count=20 total_compile_time_ms=170 queued=1 inflight=1 peak_inflight=4 bytes_in=1800 bytes_out=900",
        "[jit_server addr=server-a:6000 ts=4000] count=25 total_compile_time_ms=220 queued=0 inflight=0 peak_inflight=4 bytes_in=2200 bytes_out=1200",
        "[jit_server addr=server-b:6001 ts=1500] count=8 total_compile_time_ms=80 queued=0 inflight=0 peak_inflight=1 bytes_in=900 bytes_out=400",
        "[jit_server addr=server-b:6001 ts=3900] count=12 total_compile_time_ms=140 queued=0 inflight=0 peak_inflight=2 bytes_in=1300 bytes_out=700",
    ]

    report = MODULE.build_report(MODULE.parse_jit_server_samples(lines), window_seconds=None)
    servers = {entry["addr"]: entry for entry in report["servers"]}

    assert servers["server-a:6000"]["count"] == 25
    assert servers["server-a:6000"]["total_compile_time_ms"] == 220
    assert servers["server-b:6001"]["count"] == 12
    assert servers["server-b:6001"]["total_compile_time_ms"] == 140
    assert servers["server-a:6000"]["throughput_compiles_per_sec"] is None


def test_build_report_window_computes_deltas_and_throughput():
    lines = [
        "[jit_server addr=server-a:6000 ts=1000] count=10 total_compile_time_ms=100 queued=0 inflight=0 peak_inflight=2 bytes_in=1000 bytes_out=500",
        "[jit_server addr=server-a:6000 ts=3000] count=20 total_compile_time_ms=170 queued=1 inflight=1 peak_inflight=4 bytes_in=1800 bytes_out=900",
        "[jit_server addr=server-a:6000 ts=4000] count=25 total_compile_time_ms=220 queued=0 inflight=0 peak_inflight=4 bytes_in=2200 bytes_out=1200",
        "[jit_server addr=server-b:6001 ts=1500] count=8 total_compile_time_ms=80 queued=0 inflight=0 peak_inflight=1 bytes_in=900 bytes_out=400",
        "[jit_server addr=server-b:6001 ts=2500] count=9 total_compile_time_ms=100 queued=1 inflight=1 peak_inflight=2 bytes_in=1000 bytes_out=500",
        "[jit_server addr=server-b:6001 ts=3900] count=12 total_compile_time_ms=140 queued=0 inflight=0 peak_inflight=2 bytes_in=1300 bytes_out=700",
    ]

    report = MODULE.build_report(MODULE.parse_jit_server_samples(lines), window_seconds=2.0)
    servers = {entry["addr"]: entry for entry in report["servers"]}

    assert servers["server-a:6000"]["count"] == 5
    assert servers["server-a:6000"]["total_compile_time_ms"] == 50
    assert servers["server-a:6000"]["peak_inflight"] == 4
    assert servers["server-a:6000"]["bytes_in"] == 400
    assert servers["server-a:6000"]["bytes_out"] == 300
    assert servers["server-a:6000"]["throughput_compiles_per_sec"] == pytest.approx(5.0)

    assert servers["server-b:6001"]["count"] == 3
    assert servers["server-b:6001"]["total_compile_time_ms"] == 40
    assert servers["server-b:6001"]["peak_inflight"] == 2
    assert servers["server-b:6001"]["bytes_in"] == 300
    assert servers["server-b:6001"]["bytes_out"] == 200
    assert servers["server-b:6001"]["throughput_compiles_per_sec"] == pytest.approx(3 / 1.4)


def test_imbalance_summary_statistics():
    lines = [
        "[jit_server addr=server-a:6000 ts=3000] count=20 total_compile_time_ms=170 queued=1 inflight=1 peak_inflight=4 bytes_in=1800 bytes_out=900",
        "[jit_server addr=server-a:6000 ts=4000] count=25 total_compile_time_ms=220 queued=0 inflight=0 peak_inflight=4 bytes_in=2200 bytes_out=1200",
        "[jit_server addr=server-b:6001 ts=2500] count=9 total_compile_time_ms=100 queued=1 inflight=1 peak_inflight=2 bytes_in=1000 bytes_out=500",
        "[jit_server addr=server-b:6001 ts=3900] count=12 total_compile_time_ms=140 queued=0 inflight=0 peak_inflight=2 bytes_in=1300 bytes_out=700",
    ]

    report = MODULE.build_report(MODULE.parse_jit_server_samples(lines), window_seconds=2.0)
    count_stats = report["imbalance_summary"]["count"]
    compile_time_stats = report["imbalance_summary"]["total_compile_time_ms"]

    assert count_stats["max"] == pytest.approx(5.0)
    assert count_stats["min"] == pytest.approx(3.0)
    assert count_stats["mean"] == pytest.approx(4.0)
    assert count_stats["stddev"] == pytest.approx(1.0)
    assert count_stats["max_min_ratio"] == pytest.approx(5.0 / 3.0)
    assert count_stats["coefficient_of_variation"] == pytest.approx(0.25)

    assert compile_time_stats["max"] == pytest.approx(50.0)
    assert compile_time_stats["min"] == pytest.approx(40.0)
    assert compile_time_stats["mean"] == pytest.approx(45.0)
    assert compile_time_stats["stddev"] == pytest.approx(5.0)
    assert compile_time_stats["max_min_ratio"] == pytest.approx(1.25)
    assert compile_time_stats["coefficient_of_variation"] == pytest.approx(5.0 / 45.0)
