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

DOC = Path(__file__).resolve().parents[1] / "doc/functional_decoder"


def artifact(path):
    """Large raw artifacts are losslessly archived for the repository size gate."""
    archive = path.with_name(path.name + ".gz")
    return gzip.decompress(archive.read_bytes()) if archive.exists() else path.read_bytes()


def main():
    result = {
        "metric": "sum of tt-perf-report Device Time (microseconds), divided by iterations and 1000",
        "regime": "real weights, B1, prefill S1024, traced decode position1024, BF16/HiFi4",
        "excludes": "host dispatch, input copies, setup, and op-to-op gaps; wall time is separately measured",
        "layers": {},
    }
    for layer, kind, op_counts in [(0, "sliding_attention", (49, 44)), (3, "full_attention", (33, 30))]:
        directory = DOC / f"tracy/layer_{layer}"
        log = artifact(directory / "collection.log").decode()
        assert "ALL_PASS" in log and "markers were dropped" not in log and "Traceback" not in log
        provenance = json.loads((directory / "perf_run.provenance.json").read_text())
        assert provenance["watcher"] is None and provenance["trace_alloc_tracking"] == "0"
        assert provenance["device_profiler"] == "1" and provenance["profiler_program_support_count"] == "10000"
        run = json.loads((directory / "perf_run.json").read_text())[0]
        assert run["prefill_pcc"] >= 0.995 and run["decode_pcc"] >= 0.995
        assert run["repeated_replay_equal"] and run["trace_capture_count"] == 1
        phases = {}
        for phase, iterations, ops_per_pass in zip(("prefill", "decode"), (3, 20), op_counts):
            raw = list(csv.DictReader(io.StringIO(artifact(directory / f"{phase}_ops.csv").decode())))
            start = next(i for i, row in enumerate(raw) if row["OP CODE"] == f"PERF_{phase.upper()}")
            end = next(i for i, row in enumerate(raw) if row["OP CODE"] == f"PERF_{phase.upper()}_END")
            window = raw[start + 1 : end]
            rows = list(csv.DictReader((directory / f"{phase}_perf_report.csv").open()))
            assert len(window) == len(rows) == iterations * ops_per_pass
            assert all(row["DEVICE KERNEL DURATION [ns]"] for row in window)
            assert all(row["Device Time"] for row in rows)
            microseconds = sum(float(row["Device Time"]) for row in rows)
            raw_microseconds = sum(float(row["DEVICE KERNEL DURATION [ns]"]) for row in window) / 1000
            assert abs(microseconds - raw_microseconds) < 0.01
            matmuls = [row for row in rows if row["OP Code"].startswith("Matmul")]
            assert len(matmuls) == 8 * iterations
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
    (DOC / "performance_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
