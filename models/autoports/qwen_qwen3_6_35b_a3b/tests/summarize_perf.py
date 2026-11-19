# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Summarize actual signposted tt-perf-report 1.5 CSVs; never supply estimates."""

import csv
import gzip
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

DOC = (
    Path(__file__).parents[1]
    / "doc"
    / ("fused_decoder" if os.environ.get("QWEN_DECODER") == "fused" else "functional_decoder")
)

if os.environ.get("QWEN_FUSED_ARTIFACT_DIR"):
    DOC = Path(os.environ["QWEN_FUSED_ARTIFACT_DIR"])


def number(row, column):
    value = row.get(column, "")
    if value in (None, "", "nan", "None"):
        return 0.0
    result = float(value)
    assert math.isfinite(result) and result >= 0, (column, value)
    return result


def read_bytes(path):
    if path.exists():
        return path.read_bytes()
    archive = path.with_suffix(path.suffix + ".gz")
    if archive.exists():
        return gzip.decompress(archive.read_bytes())
    parts = sorted(path.parent.glob(archive.name + ".part*"))
    assert parts, path
    return gzip.decompress(b"".join(p.read_bytes() for p in parts))


def main():
    records = []
    completeness = []
    for idx, kind in ((0, "linear_attention"), (3, "full_attention")):
        directory = DOC / f"tracy/layer_{idx}"
        correctness = json.loads((directory / "correctness.json").read_text())
        assert correctness["prefill_pcc"] >= 0.995 and correctness["decode_traced_pcc"] >= 0.995
        assert correctness["deterministic"] and correctness["runtime_fallback_audit"] == "pass"
        provenance = json.loads((directory / "correctness.json.provenance.json").read_text())
        assert provenance["regime"] == "serial"
        assert not provenance["env"]["TT_METAL_WATCHER"]
        assert "profile_drain" in correctness, "Reject the initial overflowing profiler run"
        ops_path = directory / "ops.csv"
        ops_bytes = read_bytes(ops_path)
        raw_ops = list(csv.DictReader(ops_bytes.decode().splitlines()))
        all_sessions = defaultdict(list)
        for row in raw_ops:
            if row["METAL TRACE REPLAY SESSION ID"]:
                all_sessions[row["METAL TRACE REPLAY SESSION ID"]].append(row["OP CODE"])
        assert len(all_sessions) == 20 and len({tuple(ops) for ops in all_sessions.values()}) == 1
        log_path = DOC / f"logs/profile_{idx}.log"
        log = read_bytes(log_path).decode()
        assert "markers were dropped" not in log and "Profiler DRAM buffers were full" not in log
        completeness.append(
            dict(
                layer=idx,
                replays_including_correctness_warmup=20,
                measured_replays=16,
                operations_per_replay=len(next(iter(all_sessions.values()))),
                identical_operation_sequence=True,
                no_dropped_markers=True,
            )
        )
        start = next(i for i, r in enumerate(raw_ops) if r["OP CODE"] == "PERF_DECODE")
        end = next(i for i, r in enumerate(raw_ops) if r["OP CODE"] == "PERF_DECODE_END")
        sessions = defaultdict(list)
        for row in raw_ops[start + 1 : end]:
            assert row["METAL TRACE ID"] and row["METAL TRACE REPLAY SESSION ID"]
            assert float(row["DEVICE KERNEL DURATION [ns]"]) > 0
            sessions[row["METAL TRACE REPLAY SESSION ID"]].append(row["OP CODE"])
        assert len(sessions) == 16
        counts = [Counter(ops) for ops in sessions.values()]
        assert all(c == counts[0] for c in counts), "Missing operations in a trace replay"
        for phase in ("prefill", "decode"):
            source = directory / f"{phase}_perf_report.csv"
            raw = read_bytes(source)
            rows = list(csv.DictReader(raw.decode().splitlines()))
            assert rows and "Device Time [μs]" in rows[0], "Expected tt-perf-report 1.5 microsecond column"
            timed = [row for row in rows if number(row, "Device Time [μs]") > 0]
            assert timed and {float(row["Device"]) for row in timed} == {0.0}, "Expected physical chip 0 only"
            assert not any(number(row, "Overlap [μs]") for row in timed), "Overlapped ops need timeline analysis"
            repeats = 1 if phase == "prefill" else correctness["profile_decode_replays"]
            assert repeats == (1 if phase == "prefill" else 16)
            kernel_us = sum(number(row, "Device Time [μs]") for row in timed)
            gap_us = sum(number(row, "Op-to-Op Gap [μs]") for row in timed)
            records.append(
                dict(
                    layer=idx,
                    kind=kind,
                    phase=phase,
                    length=correctness["length"] if phase == "prefill" else 1,
                    decode_position=None if phase == "prefill" else correctness["length"],
                    allocated_context=correctness["context"],
                    latency_us=kernel_us / repeats,
                    metric="sum of measured device operation durations per pass; excludes dispatch gaps",
                    op_to_op_gap_us=gap_us / repeats,
                    measured_passes=repeats,
                    trace_session_operation_counts=(
                        {k: len(v) for k, v in sessions.items()} if phase == "decode" else None
                    ),
                    timed_operation_rows=len(timed),
                    time_column="Device Time [μs]",
                    regime="serial; physical chip 0; no watcher",
                    report=str(source.relative_to(DOC)),
                    report_sha256=hashlib.sha256(raw).hexdigest(),
                    provenance=str((directory / "correctness.json.provenance.json").relative_to(DOC)),
                )
            )
    (DOC / "perf_summary.json").write_text(json.dumps({"measurements": records}, indent=2) + "\n")
    (DOC / "profiler_completeness.json").write_text(json.dumps(dict(valid=True, runs=completeness), indent=2) + "\n")
    lines = [
        "# Decoder device performance",
        "",
        "Measured signposted warmed passes on physical Blackhole chip 0, serialized, without watcher.",
        "Prefill uses one warmed 256-token pass; decode uses 16 warmed trace replays at position 256.",
        "The CSV column `Device Time [μs]` is already in microseconds. Each operation row is counted once;",
        "tt-perf-report's CSV does not include the terminal table's total row. Decode totals are divided by 16.",
        "Device time below excludes dispatch gaps and is not a host-wall latency or full-model throughput claim.",
        "Profiler buffers are drained between complete decode replays; gaps include this measurement overhead.",
        "The gap column records the report's separate op-to-op gaps; no overlap is permitted by this summarizer.",
        "",
        "| Layer kind | Phase | Device time/pass (μs) | Op-to-op gap/pass (μs) | Passes | Human table |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in records:
        table = row["report"].replace(".csv", ".txt")
        lines.append(
            f"| {row['kind']} | {row['phase']} | {row['latency_us']:.3f} | {row['op_to_op_gap_us']:.3f} "
            f"| {row['measured_passes']} | [report]({table}) |"
        )
    collector = "collect_fused_perf.sh" if os.environ.get("QWEN_DECODER") == "fused" else "collect_perf.sh"
    lines += ["", f"Source commands: `tests/{collector}`; CSV hashes and run provenance: `perf_summary.json`."]
    (DOC / ("fused_decoder.md" if os.environ.get("QWEN_DECODER") == "fused" else "functional_decoder.md")).write_text(
        "\n".join(lines) + "\n"
    )
    print("PERF_SUMMARY_WRITTEN", len(records), "measured windows")


if __name__ == "__main__":
    main()
