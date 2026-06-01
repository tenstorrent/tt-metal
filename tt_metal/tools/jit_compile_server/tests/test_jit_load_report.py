# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import io
import json
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
        "count=42 dedup_hits=9 total_compile_time_ms=900 queued=2 inflight=3 "
        "peak_inflight=7 bytes_in=12345 bytes_out=67890"
    )

    sample = MODULE.parse_jit_server_line(line)
    assert sample is not None
    assert sample.addr == "localhost:6000"
    assert sample.ts == 1711111111111
    assert sample.count == 42
    assert sample.dedup_hits == 9
    assert sample.total_compile_time_ms == 900
    assert sample.queued == 2
    assert sample.inflight == 3
    assert sample.peak_inflight == 7
    assert sample.bytes_in == 12345
    assert sample.bytes_out == 67890


def test_parse_jit_server_samples_ignores_non_matching_lines():
    lines = [
        "random non-matching line",
        "[jit_server addr=a:1 ts=1000] count=1 dedup_hits=0 total_compile_time_ms=2 queued=0 inflight=0 peak_inflight=1 bytes_in=10 bytes_out=20",
        "another non-matching line",
    ]

    samples = MODULE.parse_jit_server_samples(lines)
    assert len(samples) == 1
    assert samples[0].addr == "a:1"
    assert samples[0].dedup_hits == 0


def test_parse_kernel_compile_line_extracts_kernel_hash():
    line = "compile kernels/reader/abc1234567890def: targets=2 genfiles=1 outstanding=3"
    sample = MODULE.parse_kernel_compile_line(line)
    assert sample is not None
    assert sample.kernel_name == "kernels/reader/abc1234567890def"
    assert sample.kernel_hash == "abc1234567890def"


def test_build_report_uses_last_sample_without_window():
    lines = [
        "[jit_server addr=server-a:6000 ts=1000] count=10 dedup_hits=2 total_compile_time_ms=100 queued=0 inflight=0 peak_inflight=2 bytes_in=1000 bytes_out=500",
        "[jit_server addr=server-a:6000 ts=3000] count=20 dedup_hits=4 total_compile_time_ms=170 queued=1 inflight=1 peak_inflight=4 bytes_in=1800 bytes_out=900",
        "[jit_server addr=server-a:6000 ts=4000] count=25 dedup_hits=5 total_compile_time_ms=220 queued=0 inflight=0 peak_inflight=4 bytes_in=2200 bytes_out=1200",
        "[jit_server addr=server-b:6001 ts=1500] count=8 dedup_hits=1 total_compile_time_ms=80 queued=0 inflight=0 peak_inflight=1 bytes_in=900 bytes_out=400",
        "[jit_server addr=server-b:6001 ts=3900] count=12 dedup_hits=3 total_compile_time_ms=140 queued=0 inflight=0 peak_inflight=2 bytes_in=1300 bytes_out=700",
    ]

    report = MODULE.build_report(MODULE.parse_jit_server_samples(lines), window_seconds=None)
    servers = {entry["addr"]: entry for entry in report["servers"]}

    assert servers["server-a:6000"]["count"] == 25
    assert servers["server-a:6000"]["dedup_hits"] == 5
    assert servers["server-a:6000"]["total_compile_time_ms"] == 220
    assert servers["server-a:6000"]["unique_kernel_hashes"] == 0
    assert servers["server-b:6001"]["count"] == 12
    assert servers["server-b:6001"]["dedup_hits"] == 3
    assert servers["server-b:6001"]["total_compile_time_ms"] == 140
    assert servers["server-b:6001"]["unique_kernel_hashes"] == 0
    assert servers["server-a:6000"]["throughput_compiles_per_sec"] is None


def test_build_report_window_computes_deltas_and_throughput():
    lines = [
        "[jit_server addr=server-a:6000 ts=1000] count=10 dedup_hits=2 total_compile_time_ms=100 queued=0 inflight=0 peak_inflight=2 bytes_in=1000 bytes_out=500",
        "[jit_server addr=server-a:6000 ts=3000] count=20 dedup_hits=4 total_compile_time_ms=170 queued=1 inflight=1 peak_inflight=4 bytes_in=1800 bytes_out=900",
        "[jit_server addr=server-a:6000 ts=4000] count=25 dedup_hits=5 total_compile_time_ms=220 queued=0 inflight=0 peak_inflight=4 bytes_in=2200 bytes_out=1200",
        "[jit_server addr=server-b:6001 ts=1500] count=8 dedup_hits=1 total_compile_time_ms=80 queued=0 inflight=0 peak_inflight=1 bytes_in=900 bytes_out=400",
        "[jit_server addr=server-b:6001 ts=2500] count=9 dedup_hits=2 total_compile_time_ms=100 queued=1 inflight=1 peak_inflight=2 bytes_in=1000 bytes_out=500",
        "[jit_server addr=server-b:6001 ts=3900] count=12 dedup_hits=4 total_compile_time_ms=140 queued=0 inflight=0 peak_inflight=2 bytes_in=1300 bytes_out=700",
    ]

    report = MODULE.build_report(MODULE.parse_jit_server_samples(lines), window_seconds=2.0)
    servers = {entry["addr"]: entry for entry in report["servers"]}

    assert servers["server-a:6000"]["count"] == 5
    assert servers["server-a:6000"]["dedup_hits"] == 1
    assert servers["server-a:6000"]["total_compile_time_ms"] == 50
    assert servers["server-a:6000"]["peak_inflight"] == 4
    assert servers["server-a:6000"]["bytes_in"] == 400
    assert servers["server-a:6000"]["bytes_out"] == 300
    assert servers["server-a:6000"]["throughput_compiles_per_sec"] == pytest.approx(5.0)

    assert servers["server-b:6001"]["count"] == 3
    assert servers["server-b:6001"]["dedup_hits"] == 2
    assert servers["server-b:6001"]["total_compile_time_ms"] == 40
    assert servers["server-b:6001"]["peak_inflight"] == 2
    assert servers["server-b:6001"]["bytes_in"] == 300
    assert servers["server-b:6001"]["bytes_out"] == 200
    assert servers["server-b:6001"]["throughput_compiles_per_sec"] == pytest.approx(3 / 1.4)


def test_imbalance_summary_statistics():
    lines = [
        "[jit_server addr=server-a:6000 ts=3000] count=20 dedup_hits=4 total_compile_time_ms=170 queued=1 inflight=1 peak_inflight=4 bytes_in=1800 bytes_out=900",
        "[jit_server addr=server-a:6000 ts=4000] count=25 dedup_hits=5 total_compile_time_ms=220 queued=0 inflight=0 peak_inflight=4 bytes_in=2200 bytes_out=1200",
        "[jit_server addr=server-b:6001 ts=2500] count=9 dedup_hits=2 total_compile_time_ms=100 queued=1 inflight=1 peak_inflight=2 bytes_in=1000 bytes_out=500",
        "[jit_server addr=server-b:6001 ts=3900] count=12 dedup_hits=4 total_compile_time_ms=140 queued=0 inflight=0 peak_inflight=2 bytes_in=1300 bytes_out=700",
    ]

    report = MODULE.build_report(MODULE.parse_jit_server_samples(lines), window_seconds=2.0)
    count_stats = report["imbalance_summary"]["count"]
    dedup_stats = report["imbalance_summary"]["dedup_hits"]
    compile_time_stats = report["imbalance_summary"]["total_compile_time_ms"]

    assert count_stats["max"] == pytest.approx(5.0)
    assert count_stats["min"] == pytest.approx(3.0)
    assert count_stats["mean"] == pytest.approx(4.0)
    assert count_stats["stddev"] == pytest.approx(1.0)
    assert count_stats["max_min_ratio"] == pytest.approx(5.0 / 3.0)
    assert count_stats["coefficient_of_variation"] == pytest.approx(0.25)

    assert dedup_stats["max"] == pytest.approx(2.0)
    assert dedup_stats["min"] == pytest.approx(1.0)
    assert dedup_stats["mean"] == pytest.approx(1.5)
    assert dedup_stats["stddev"] == pytest.approx(0.5)
    assert dedup_stats["max_min_ratio"] == pytest.approx(2.0)
    assert dedup_stats["coefficient_of_variation"] == pytest.approx(1 / 3)

    assert compile_time_stats["max"] == pytest.approx(50.0)
    assert compile_time_stats["min"] == pytest.approx(40.0)
    assert compile_time_stats["mean"] == pytest.approx(45.0)
    assert compile_time_stats["stddev"] == pytest.approx(5.0)
    assert compile_time_stats["max_min_ratio"] == pytest.approx(1.25)
    assert compile_time_stats["coefficient_of_variation"] == pytest.approx(5.0 / 45.0)


def test_build_report_includes_unique_kernel_hash_counts():
    lines = [
        "[jit_server addr=server-a:6000 ts=3000] count=20 dedup_hits=4 total_compile_time_ms=170 queued=1 inflight=1 peak_inflight=4 bytes_in=1800 bytes_out=900",
        "[jit_server addr=server-b:6001 ts=3900] count=12 dedup_hits=4 total_compile_time_ms=140 queued=0 inflight=0 peak_inflight=2 bytes_in=1300 bytes_out=700",
    ]
    kernel_lines = [
        "compile kernels/a/aaaaaaaaaaaaaaaa: targets=1 genfiles=0 outstanding=1",
        "compile kernels/a/aaaaaaaaaaaaaaaa: targets=1 genfiles=0 outstanding=1",
        "compile kernels/a/bbbbbbbbbbbbbbbb: targets=1 genfiles=0 outstanding=1",
        "compile kernels/b/cccccccccccccccc: targets=1 genfiles=0 outstanding=1",
    ]
    kernel_samples = MODULE.parse_kernel_compile_samples(kernel_lines, source_id="server-a:6000")
    kernel_samples.extend(MODULE.parse_kernel_compile_samples(kernel_lines[3:], source_id="server-b:6001"))

    report = MODULE.build_report(
        MODULE.parse_jit_server_samples(lines), window_seconds=None, kernel_compile_samples=kernel_samples
    )
    servers = {entry["addr"]: entry for entry in report["servers"]}

    assert servers["server-a:6000"]["unique_kernel_hashes"] == 3
    assert servers["server-b:6001"]["unique_kernel_hashes"] == 1
    assert report["unique_kernel_hashes_total"] == 3


def test_build_report_groups_by_source_not_addr():
    server_a_lines = [
        "[jit_server addr=0.0.0.0:9876 ts=1000] count=5 dedup_hits=1 total_compile_time_ms=20 queued=0 inflight=0 peak_inflight=1 bytes_in=100 bytes_out=40",
    ]
    server_b_lines = [
        "[jit_server addr=0.0.0.0:9876 ts=1000] count=7 dedup_hits=3 total_compile_time_ms=25 queued=0 inflight=0 peak_inflight=1 bytes_in=120 bytes_out=45",
    ]
    samples = MODULE.parse_jit_server_samples(server_a_lines, source_id="server-a.log")
    samples.extend(MODULE.parse_jit_server_samples(server_b_lines, source_id="server-b.log"))

    report = MODULE.build_report(samples, window_seconds=None)
    servers = {entry["server"]: entry for entry in report["servers"]}

    assert set(servers) == {"server-a.log", "server-b.log"}
    assert servers["server-a.log"]["addr"] == "0.0.0.0:9876"
    assert servers["server-a.log"]["count"] == 5
    assert servers["server-b.log"]["addr"] == "0.0.0.0:9876"
    assert servers["server-b.log"]["count"] == 7


def test_main_stdin_collapses_prefixed_lines_into_single_server(monkeypatch, capsys):
    # tt-logger style per-line prefixes (timestamp + level) would make prefix-based server
    # inference assign a different server_id per line. Reading from stdin must instead
    # aggregate everything into one coherent server and attribute kernel hashes to it.
    stdin_text = "\n".join(
        [
            "2026-05-29 19:45:26.001 | INFO     | compile kernels/a/aaaaaaaaaaaaaaaa: targets=1 genfiles=0 outstanding=1",
            "2026-05-29 19:45:26.050 | INFO     | compile kernels/a/bbbbbbbbbbbbbbbb: targets=1 genfiles=0 outstanding=1",
            "2026-05-29 19:45:26.100 | INFO     | [jit_server addr=localhost:9876 ts=1000] count=2 dedup_hits=0 total_compile_time_ms=40 queued=0 inflight=0 peak_inflight=1 bytes_in=100 bytes_out=40",
            "2026-05-29 19:45:27.200 | INFO     | [jit_server addr=localhost:9876 ts=2000] count=5 dedup_hits=1 total_compile_time_ms=90 queued=0 inflight=0 peak_inflight=2 bytes_in=300 bytes_out=120",
        ]
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))

    assert MODULE.main(["--json"]) == 0

    report = json.loads(capsys.readouterr().out)

    assert len(report["servers"]) == 1
    server = report["servers"][0]
    assert server["server"] == MODULE.STDIN_SOURCE_ID
    assert server["addr"] == "localhost:9876"
    assert server["count"] == 5
    assert server["unique_kernel_hashes"] == 2
    assert report["unique_kernel_hashes_total"] == 2
