# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate complete signposted profiler windows and report per-pass kernel time."""

import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder"


def artifact(path):
    """Large raw artifacts are losslessly archived for the repository size gate."""
    archive = path.with_name(path.name + ".gz")
    return gzip.decompress(archive.read_bytes()) if archive.exists() else path.read_bytes()


def main():
    decoder_sha = hashlib.sha256((DOC.parents[1] / "tt/fused_decoder.py").read_bytes()).hexdigest()
    result = {
        "decoder_sha256": decoder_sha,
        "metric": "sum of tt-perf-report Device Time (microseconds), divided by iterations and 1000",
        "regime": "real weights, B1, prefill S1024, traced decode position1024, BF16/HiFi4",
        "excludes": "host dispatch, input copies, setup, and op-to-op gaps; wall time is separately measured",
        "layers": {},
    }
    for layer, kind in [(0, "sliding_attention"), (3, "full_attention")]:
        directory = DOC / f"tracy/layer_{layer}"
        log = artifact(directory / "collection.log").decode()
        assert "ALL_PASS" in log and "markers were dropped" not in log and "Traceback" not in log
        provenance = json.loads((directory / "perf_run.provenance.json").read_text())
        assert provenance["source_sha256"]["tt/fused_decoder.py"] == decoder_sha
        assert provenance["watcher"] is None and provenance["trace_alloc_tracking"] == "0"
        assert provenance["device_profiler"] == "1" and provenance["profiler_program_support_count"] == "10000"
        run = json.loads((directory / "perf_run.json").read_text())[0]
        assert run["prefill_pcc"] >= 0.995 and run["decode_pcc"] >= 0.995
        assert run["repeated_replay_equal"] and run["trace_capture_count"] == 1
        phases = {}
        for phase, iterations in zip(("prefill", "decode"), (3, 20)):
            raw = list(csv.DictReader(io.StringIO(artifact(directory / f"{phase}_ops.csv").decode())))
            start = next(i for i, row in enumerate(raw) if row["OP CODE"] == f"PERF_{phase.upper()}")
            end = next(i for i, row in enumerate(raw) if row["OP CODE"] == f"PERF_{phase.upper()}_END")
            window = raw[start + 1 : end]
            rows = list(csv.DictReader((directory / f"{phase}_perf_report.csv").open()))
            assert len(window) == len(rows) and len(window) % iterations == 0
            ops_per_pass = len(window) // iterations
            assert ops_per_pass > 0
            sequence = [row["OP CODE"] for row in window[:ops_per_pass]]
            assert [row["OP CODE"] for row in window] == sequence * iterations
            assert all(row["DEVICE KERNEL DURATION [ns]"] for row in window)
            assert all(row["Device Time"] for row in rows)
            microseconds = sum(float(row["Device Time"]) for row in rows)
            raw_microseconds = sum(float(row["DEVICE KERNEL DURATION [ns]"]) for row in window) / 1000
            assert abs(microseconds - raw_microseconds) < 0.01
            matmuls = [row for row in rows if row["OP Code"].startswith("Matmul")]
            assert len(matmuls) == 6 * iterations
            assert all(row["Input 1 Datatype"] == "BFLOAT16" and "HiFi4" in row["Math Fidelity"] for row in matmuls)
            assert "μs" in (directory / f"{phase}_perf_report.txt").read_text()
            sessions = Counter(row["METAL TRACE REPLAY SESSION ID"] for row in window)
            if phase == "decode":
                assert sessions == {str(i): ops_per_pass for i in range(4, 24)}, sessions
                assert {row["METAL TRACE ID"] for row in window} == {"0"}
            artifacts = {}
            for suffix in ("ops.csv", "perf_report.csv", "perf_report.txt"):
                path = directory / f"{phase}_{suffix}"
                if path.with_name(path.name + ".gz").exists():
                    path = path.with_name(path.name + ".gz")
                artifacts[str(path.relative_to(DOC))] = hashlib.sha256(path.read_bytes()).hexdigest()
            phases[phase] = {
                "iterations": iterations,
                "op_sequence": sequence,
                "device_ops_per_pass": ops_per_pass,
                "total_device_rows": len(rows),
                "device_kernel_ms_per_pass": microseconds / iterations / 1000,
                "wall_ms_per_pass": run[
                    "prefill_warmed_wall_ms" if phase == "prefill" else "decode_warmed_trace_wall_ms"
                ],
                "trace_replay_sessions": sorted(sessions, key=int) if phase == "decode" else [],
                "artifact_sha256": artifacts,
                "raw_csv_decompressed_sha256": hashlib.sha256(artifact(directory / f"{phase}_ops.csv")).hexdigest(),
            }
        result["layers"][kind] = phases
    before = json.loads((DOC.parent / "functional_decoder/performance_summary.json").read_text())
    result["comparison"] = {}
    for layer, kind in [(0, "sliding_attention"), (3, "full_attention")]:
        selection = json.loads((DOC / f"final_measured_{layer}.json").read_text())
        assert selection["source_sha256"]["tt/fused_decoder.py"] == decoder_sha
        assert all(r["status"] == "pass" for r in selection["rows"])
        baseline = next(r for r in selection["rows"] if r["candidate"] == "baseline")
        final = next(r for r in selection["rows"] if r["candidate"] == "final")
        assert final["decode_ms"] < baseline["decode_ms"]
        assert final["prefill_ms"] < baseline["prefill_ms"]
        assert final["decode_ms"] <= min(r["decode_ms"] for r in selection["rows"])
        result["comparison"][kind] = {
            "same_process_selection_artifact": f"final_measured_{layer}.json",
            "prefill_wall_ms_before_after": [baseline["prefill_ms"], final["prefill_ms"]],
            "decode_wall_ms_before_after": [baseline["decode_ms"], final["decode_ms"]],
            "decode_reduction_percent": 100 * (1 - final["decode_ms"] / baseline["decode_ms"]),
            "final_has_fastest_measured_median": True,
            "candidate_decode_ms": {r["candidate"]: r["decode_ms"] for r in selection["rows"]},
            "profiler_kernel_ms_before_after": {
                phase: [
                    before["layers"][kind][phase]["device_kernel_ms_per_pass"],
                    result["layers"][kind][phase]["device_kernel_ms_per_pass"],
                ]
                for phase in ("prefill", "decode")
            },
        }
    (DOC / "performance_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
