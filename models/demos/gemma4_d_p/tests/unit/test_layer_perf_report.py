# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import os

import pytest

from models.demos.gemma4_d_p.demo import layer_perf_report as lpr

PERF_REPORT_HEADERS = [
    "ID",
    "Total %",
    "Bound",
    "OP Code",
    "Device",
    "Device Time",
    "Op-to-Op Gap",
    "Global Call Count",
]


def _write_cell_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PERF_REPORT_HEADERS)
        writer.writeheader()
        for op, device_us, gap_us, *call_count in rows:
            row = {"OP Code": op, "Device Time": device_us, "Op-to-Op Gap": gap_us}
            if call_count:
                row["Global Call Count"] = call_count[0]
            writer.writerow(row)


def _cell(layer_type, chunk_idx, layer_idx, measured_ms):
    start, stop = (
        f"gemma4-layer-{layer_type}-chunk{chunk_idx}-start",
        f"gemma4-layer-{layer_type}-chunk{chunk_idx}-stop",
    )
    return {
        "chunk_idx": chunk_idx,
        "layer_type": layer_type,
        "layer_idx": layer_idx,
        "chunk_start": chunk_idx * 8192,
        "measured_ms": measured_ms,
        "start_signpost": start,
        "stop_signpost": stop,
    }


def _write_ops_csv(profiler_dir, cells):
    reports = profiler_dir / "reports" / "r"
    reports.mkdir(parents=True)
    path = reports / "ops_perf_results_r.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["OP CODE", "OP TYPE"])
        for c in cells:
            writer.writerows([[c["start_signpost"], "signpost"], ["MatmulDeviceOperation", "tt_dnn_device"]])
            writer.writerow([c["stop_signpost"], "signpost"])
    return path


def test_summarize_ignores_non_device_rows(tmp_path):
    path = tmp_path / "cell.csv"
    _write_cell_csv(
        path,
        [
            ("gemma4-layer-global-chunk0-start (signpost)", "", "", ""),
            ("MatmulDeviceOperation", "300.0", "5.0", "11"),
            ("RingJointSDPADeviceOperation", "500.0", "10.0", "12"),
            ("MatmulDeviceOperation", "200.0", "1000000.0", "10"),
            ("to_torch (torch)", "", "", ""),
        ],
    )
    summary = lpr.summarize_cell_csv(path, top_n=1)
    assert summary["n_ops"] == 3
    assert summary["kernel_us"] == 1000.0
    assert summary["span_us"] == 1015.0
    assert summary["top_ops"] == [{"op": "MatmulDeviceOperation", "device_us": 500.0, "count": 2}]


def test_summarize_includes_all_ops(tmp_path):
    path = tmp_path / "cell.csv"
    _write_cell_csv(
        path,
        [
            ("MatmulDeviceOperation", "300.0", "5.0"),
            ("RingJointSDPADeviceOperation", "500.0", "10.0"),
            ("GatherCodegenDeviceOperation", "200.0", "2.0"),
        ],
    )

    summary = lpr.summarize_cell_csv(path)

    assert [op["op"] for op in summary["top_ops"]] == [
        "RingJointSDPADeviceOperation",
        "MatmulDeviceOperation",
        "GatherCodegenDeviceOperation",
    ]


def test_report_rejects_unsafe_signposts(tmp_path, monkeypatch, expect_error):
    called = False

    def unexpected_import():
        nonlocal called
        called = True

    monkeypatch.setattr(lpr, "_perf_report", unexpected_import)
    with expect_error(ValueError, "invalid start signpost"):
        lpr.run_tt_perf_report(
            tmp_path / "ops.csv",
            "gemma4-layer-global-chunk0-start; touch /tmp/pwned",
            "gemma4-layer-global-chunk0-stop",
            tmp_path / "out.csv",
        )
    assert not called


def test_main_reports_all_cells(tmp_path, monkeypatch):
    root = tmp_path / "summaries"
    monkeypatch.setenv("PREFILL_SUMMARIES", str(root))
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
    cells = [_cell("global", 0, 5, 9.0), _cell("local", 0, 0, 4.0), _cell("global", 31, 5, 12.0)]
    lpr.write_manifest(
        "blackhole-chunkci-both-sz8192-ctx_256k-8x4", cells, context_len=262144, chunk_size=8192, mesh_shape=(8, 4)
    )

    _write_ops_csv(tmp_path / "profiler", cells)

    sliced = []

    def fake_report(ops_csv, start, stop, out_csv):
        sliced.append((start, stop))
        assert ops_csv.name == f"{out_csv.stem}_ops.csv"
        _write_cell_csv(out_csv, [("MatmulDeviceOperation", "1500.0", "100.0")])
        out_csv.with_suffix(".txt").write_text(f"Performance Report for {start}\nStacked report\n")
        return True

    monkeypatch.setattr(lpr, "run_tt_perf_report", fake_report)
    assert lpr.main(["--profiler-dir", str(tmp_path / "profiler")]) == 0

    assert sliced == [(c["start_signpost"], c["stop_signpost"]) for c in cells]
    out_dir = root / lpr.MANIFEST_DIR
    assert (out_dir / "ops_perf_results_r.csv").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert all(c["report"]["kernel_us"] == 1500.0 for c in summary[0]["cells"])

    md = (root / "perf" / lpr.SUMMARY_NAME).read_text()
    assert "| Layer | Chunk 0 | Chunk 31 |" in md
    assert "| local (layer 0) | **1.50**<br>span 1.50<br>host 4.00 | – |" in md
    assert "```text\nPerformance Report for gemma4-layer-global-chunk31-start\nStacked report\n```" in md


def test_cell_ops_csv_drops_gap_before_each_devices_first_replayed_op(tmp_path):
    fieldnames = ["OP CODE", "OP TYPE", "DEVICE ID", "GLOBAL CALL COUNT", "OP TO OP LATENCY [ns]"]
    rows = [
        {"OP CODE": "before", "OP TYPE": "tt_dnn_device", "DEVICE ID": "0", "GLOBAL CALL COUNT": "1"},
        {"OP CODE": "gemma4-layer-global-chunk0-start", "OP TYPE": "signpost"},
        {"OP CODE": "matmul", "OP TYPE": "tt_dnn_device", "DEVICE ID": "0", "GLOBAL CALL COUNT": "12"},
        {"OP CODE": "embedding", "OP TYPE": "tt_dnn_device", "DEVICE ID": "0", "GLOBAL CALL COUNT": "10"},
        {"OP CODE": "embedding", "OP TYPE": "tt_dnn_device", "DEVICE ID": "1", "GLOBAL CALL COUNT": "10"},
        {"OP CODE": "matmul", "OP TYPE": "tt_dnn_device", "DEVICE ID": "1", "GLOBAL CALL COUNT": "12"},
        {"OP CODE": "gemma4-layer-global-chunk0-stop", "OP TYPE": "signpost"},
        {"OP CODE": "after", "OP TYPE": "tt_dnn_device", "DEVICE ID": "0", "GLOBAL CALL COUNT": "20"},
    ]
    for r in rows:
        if r["OP TYPE"] != "signpost":
            r["OP TO OP LATENCY [ns]"] = "1000000" if r["OP CODE"] == "embedding" else "500"
    path = tmp_path / "cell_ops.csv"
    assert lpr.write_cell_ops_csv(
        fieldnames, rows, "gemma4-layer-global-chunk0-start", "gemma4-layer-global-chunk0-stop", path
    )

    _, sliced = lpr.read_ops_csv(path)
    assert [r["OP CODE"] for r in sliced] == [r["OP CODE"] for r in rows[1:7]]
    gaps = {(r["OP CODE"], r["DEVICE ID"]): r["OP TO OP LATENCY [ns]"] for r in sliced if r["OP TYPE"] != "signpost"}
    assert gaps == {("matmul", "0"): "500", ("embedding", "0"): "", ("embedding", "1"): "", ("matmul", "1"): "500"}
    assert rows[3]["OP TO OP LATENCY [ns]"] == "1000000"


def test_summary_omits_report_text_over_the_size_limit(monkeypatch):
    cell = {**_cell("global", 0, 5, 9.0), "report": {"n_ops": 1, "kernel_us": 1.0, "span_us": 1.0}}
    cell["report_text"] = "x" * 20
    monkeypatch.setattr(lpr, "SUMMARY_TEXT_LIMIT", 10)
    md = lpr.render_markdown([{"run_id": "run", "mesh_shape": (8, 4), "cells": [cell]}])
    assert "x" * 20 not in md
    assert "see the layer-perf artifact" in md


def test_main_fails_on_empty_report(tmp_path, monkeypatch):
    monkeypatch.setenv("PREFILL_SUMMARIES", str(tmp_path))
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
    cells = [_cell("global", 0, 5, 9.0)]
    lpr.write_manifest("run", cells, context_len=262144, chunk_size=8192, mesh_shape=(8, 4))
    _write_ops_csv(tmp_path / "profiler", cells)
    monkeypatch.setattr(lpr, "run_tt_perf_report", lambda *_: False)
    assert lpr.main(["--profiler-dir", str(tmp_path / "profiler")]) == 1
    assert "report failed" in (tmp_path / "perf" / lpr.SUMMARY_NAME).read_text()


def test_main_rejects_missing_signposts(tmp_path, monkeypatch):
    monkeypatch.setenv("PREFILL_SUMMARIES", str(tmp_path))
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
    measured, stale = _cell("global", 0, 5, 9.0), _cell("global", 7, 5, 11.0)
    lpr.write_manifest("current", [measured], context_len=262144, chunk_size=8192, mesh_shape=(8, 4))
    lpr.write_manifest("stale", [stale], context_len=262144, chunk_size=8192, mesh_shape=(8, 4))
    _write_ops_csv(tmp_path / "profiler", [measured])

    sliced = []

    def fake_report(ops_csv, start, stop, out_csv):
        sliced.append(start)
        _write_cell_csv(out_csv, [("MatmulDeviceOperation", "1500.0", "100.0")])
        return True

    monkeypatch.setattr(lpr, "run_tt_perf_report", fake_report)
    assert lpr.main(["--profiler-dir", str(tmp_path / "profiler")]) == 1
    assert sliced == [measured["start_signpost"]]
    summary = {m["run_id"]: m for m in json.loads((tmp_path / lpr.MANIFEST_DIR / "summary.json").read_text())}
    assert summary["current"]["cells"][0]["report"]["kernel_us"] == 1500.0
    assert summary["stale"]["cells"][0]["report"] is None


def test_manifest_rank_zero_only(tmp_path, monkeypatch):
    monkeypatch.setenv("PREFILL_SUMMARIES", str(tmp_path))
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    assert lpr.write_manifest("run", []) is None
    assert not (tmp_path / lpr.MANIFEST_DIR).exists()


def test_main_writes_nothing_on_non_zero_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("PREFILL_SUMMARIES", str(tmp_path))
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
    cells = [_cell("global", 0, 5, 9.0)]
    lpr.write_manifest("run", cells, context_len=262144, chunk_size=8192, mesh_shape=(8, 4))
    _write_ops_csv(tmp_path / "profiler", cells)
    before = sorted(p.name for p in (tmp_path / lpr.MANIFEST_DIR).iterdir())
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    monkeypatch.setattr(lpr, "run_tt_perf_report", lambda *_: pytest.fail("non-zero rank ran a report"))
    assert lpr.main(["--profiler-dir", str(tmp_path / "profiler")]) == 0
    assert sorted(p.name for p in (tmp_path / lpr.MANIFEST_DIR).iterdir()) == before
    assert not (tmp_path / "perf").exists()


def test_main_rejects_ops_csv_older_than_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("PREFILL_SUMMARIES", str(tmp_path))
    monkeypatch.delenv("OMPI_COMM_WORLD_RANK", raising=False)
    cells = [_cell("global", 0, 5, 9.0)]
    stale = _write_ops_csv(tmp_path / "profiler", cells)
    manifest = lpr.write_manifest("run", cells, context_len=262144, chunk_size=8192, mesh_shape=(8, 4))
    os.utime(stale, (manifest.stat().st_mtime - 60,) * 2)
    monkeypatch.setattr(lpr, "run_tt_perf_report", lambda *_: pytest.fail("reported a stale ops CSV"))
    assert lpr.main(["--profiler-dir", str(tmp_path / "profiler")]) == 1
    assert not (tmp_path / "perf").exists()
