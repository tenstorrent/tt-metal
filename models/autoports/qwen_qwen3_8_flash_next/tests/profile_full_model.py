"""Capture only the three-layer wrapper and validate all split-trace windows."""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import statistics
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", default="reduced_profile01")
    ap.add_argument("--stage", default="full_model")
    ap.add_argument("--prefill-trace-max", type=int, default=512)
    ap.add_argument("--trace-region-size", type=int, default=768000000)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = root / "doc" / a.stage / "tracy" / a.tag
    raw = Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next") / ("full_model_" + a.tag)
    if raw.exists():
        raise FileExistsError(raw)
    evidence.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("TT_METAL_WATCHER")}
    env.update(
        TT_METAL_TRACE_ALLOC_TRACKING="0",
        TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="8192",
        TT_METAL_PROFILER_DIR=str(raw),
        OMP_NUM_THREADS="8",
    )
    command = [
        sys.executable,
        "-m",
        "tracy",
        "-r",
        "-p",
        "-v",
        "--no-web-server",
        "--disable-device-data-push-to-tracy",
        "-o",
        str(raw),
        "-m",
        "models.autoports.qwen_qwen3_8_flash_next.tests.full_model_perf",
        "--profile",
        "--checkpoint",
        a.checkpoint,
        "--tag",
        a.tag,
    ]
    command += [
        "--stage",
        a.stage,
        "--prefill-trace-max",
        str(a.prefill_trace_max),
        "--trace-region-size",
        str(a.trace_region_size),
    ]
    provenance = dict(
        command=command,
        environment={
            k: env[k]
            for k in (
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT",
                "TT_METAL_PROFILER_DIR",
                "OMP_NUM_THREADS",
            )
        },
    )
    (evidence / "command.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with (evidence / "capture.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (evidence / "capture.log").read_text()
    for failure in ("TT_FATAL:", "markers were dropped", "Traceback (most recent call last)"):
        if failure in log:
            raise RuntimeError(failure)
    source = sorted(raw.glob("reports/*/ops_perf_results_*.csv"), key=lambda p: p.stat().st_mtime)[-1]
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    window_csv = evidence / "measured_windows_raw.csv"
    with window_csv.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        inside = False
        for row in rows:
            if row.get("OP TYPE") == "signpost" and row.get("OP CODE", "").startswith("PERF_"):
                writer.writerow(row)
                inside = not row["OP CODE"].endswith("_END")
            elif inside:
                writer.writerow(row)
    metadata = (raw / ".logs/tracy_ops_data.csv").read_text()
    counts = Counter(
        (device, trace)
        for device, trace, _ in re.findall(r"TT_METAL_TRACE_ENQUEUE_PROGRAM: (\d+), (\d+), (\d+)", metadata)
    )
    assert len(counts) == 4 * (5 + min(10, a.prefill_trace_max.bit_length())), counts
    results = {}
    executable = str(Path(sys.executable).with_name("tt-perf-report"))
    for mode, trace in (
        ("prefill", ["11", "1"] if a.prefill_trace_max >= 128 else None),
        ("decode", "0"),
        ("greedy", "1"),
        ("argmax", "3"),
        ("sample", "2"),
        ("token_out", ["0", "1"]),
    ):
        markers = ["PERF_" + mode.upper(), "PERF_" + mode.upper() + "_END"]
        found = [
            [i for i, row in enumerate(rows) if row.get("OP TYPE") == "signpost" and row.get("OP CODE") == marker]
            for marker in markers
        ]
        assert all(len(x) == 1 for x in found), (mode, found)
        start, end = (x[0] for x in found)
        assert start < end
        selected = [r for r in rows[start + 1 : end] if r.get("OP TYPE") == "tt_dnn_device"]
        devices = {r["DEVICE ID"].strip() for r in selected}
        assert len(devices) == 4
        if trace is not None:
            expected_traces = trace if isinstance(trace, list) else [trace]
            for device in devices:
                device_rows = [r for r in selected if r["DEVICE ID"].strip() == device]
                actual_traces = {r.get("METAL TRACE ID", "").strip() for r in device_rows}
                if mode == "prefill":
                    # Scheduler input/position updates are eager around the
                    # prefill and greedy traces in this complete API window.
                    actual_traces -= {"", "-", "N/A"}
                assert actual_traces == set(expected_traces), (mode, actual_traces)
                for trace_id in expected_traces:
                    trace_rows = [r for r in device_rows if r.get("METAL TRACE ID", "").strip() == trace_id]
                    assert len({r.get("METAL TRACE REPLAY SESSION ID", "").strip() for r in trace_rows}) == 1
                    assert len(trace_rows) == counts[device, trace_id], (mode, device, len(trace_rows), counts)
        base = [executable, str(source), "--start-signpost", markers[0], "--end-signpost", markers[1], "--no-color"]
        if trace is not None and mode != "prefill":
            base += ["--tracing-mode"]
        table = evidence / f"{mode}_perf_report.csv"
        commands = [base + ["--csv", str(table)], base + ["--no-summary"]]
        for cmd, suffix in zip(commands, ("console.log", "txt")):
            with (evidence / f"{mode}_perf_report.{suffix}").open("w") as handle:
                subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True)
        with table.open() as handle:
            report_rows = list(csv.DictReader(handle))
        durations = [
            float(r["Device Time"].replace(",", ""))
            for r in report_rows
            if r["Device Time"].strip() not in ("", "-", "N/A")
        ]
        assert len(durations) in (len(selected), len(selected) // 4)
        rank_windows = {}
        for device in sorted(devices):
            device_rows = [r for r in selected if r["DEVICE ID"].strip() == device]
            clock_rates = [
                (float(r["DEVICE FW END CYCLE"]) - float(r["DEVICE FW START CYCLE"]))
                / float(r["DEVICE FW DURATION [ns]"])
                for r in device_rows
                if r.get("DEVICE FW DURATION [ns]", "").strip() and float(r["DEVICE FW DURATION [ns]"]) > 0
            ]
            cycles_per_ns = statistics.median(clock_rates)
            rank_windows[device] = dict(
                cycles_per_ns=cycles_per_ns,
                firmware_span_ms=(
                    max(float(r["DEVICE FW END CYCLE"]) for r in device_rows)
                    - min(float(r["DEVICE FW START CYCLE"]) for r in device_rows)
                )
                / cycles_per_ns
                / 1e6,
                summed_kernel_ms=sum(float(r["DEVICE KERNEL DURATION [ns]"]) for r in device_rows) / 1e6,
            )
        results[mode] = dict(
            commands=commands,
            raw_rows=len(selected),
            trace_id=trace,
            device_ids=sorted(devices),
            report_rows=len(durations),
            summed_device_us=sum(durations),
            time_unit="microseconds",
            rank_windows=rank_windows,
        )
    provenance.update(
        source_csv=str(source),
        csv_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        retained_window_csv=dict(
            path=str(window_csv),
            sha256=hashlib.sha256(window_csv.read_bytes()).hexdigest(),
            selection="All rows between the six PERF_* / PERF_*_END signpost pairs, inclusive; unmodified raw fields",
        ),
        tool_version=version("tt-perf-report"),
        windows=results,
        aggregation="tt-perf-report mixed-rank op aggregation; sums are not a single-rank critical-path timeline",
    )
    (evidence / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("PROFILE_FULL_MODEL_PASS", json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
