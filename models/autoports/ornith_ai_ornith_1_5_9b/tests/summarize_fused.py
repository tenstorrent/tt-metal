# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Derive compact stage evidence from runner JSON and signposted profiler CSVs."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "doc/fused_decoder"
rows = []
for path in sorted(ART.glob("*.json")):
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        continue
    for row in report.get("records", []):
        if row.get("benchmark"):
            positives = [r["pcc"] for r in report["records"] if "pcc" in r]
            rows.append(
                {
                    "artifact": path.name,
                    "layer": report["layer"],
                    "implementation": row["implementation"],
                    "length": report["lengths"][-1],
                    "prefill_ms": row["prefill_ms"],
                    "decode_ms": row["decode_ms"],
                    "min_pcc": min(positives),
                    "implementation_sha256": report["implementation_sha256"],
                }
            )
with (ART / "candidate_measurements.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "precision": "BF16 weights/activations, HiFi4, FP32 accumulation and recurrence",
    "regime": "serial chip 0, Blackhole P300c",
    "peak_dram_GB_s": 512,
    "implementation_sha256": hashlib.sha256((ROOT / "tt/fused_decoder.py").read_bytes()).hexdigest(),
    "layer_kinds": {},
}
for kind, layer, packed_n in [("linear", 0, 12352), ("full", 3, 10240)]:
    baseline = json.loads((ART / f"baseline_{kind}_s1024.json").read_text())
    final = json.loads((ART / f"default_{kind}_s1024.json").read_text())
    before = next(r for r in baseline["records"] if r.get("benchmark"))
    after = next(r for r in final["records"] if r.get("benchmark"))
    prof = json.loads((ART / f"{kind}_profile_final.json").read_text())
    profile_wall = next(r["traced_host_wall_ms"] for r in prof["records"] if "traced_host_wall_ms" in r)
    # Only the four weight tensors actually read by decode, including bank padding.
    projection_shapes = [(4096, packed_n), (4096, 24576), (12288, 4096), (4096, 4096)]
    dense_weight_bytes = sum(k * n * 2 for k, n in projection_shapes)
    dram_weight_bytes = sum(k * ((n + 255) // 256 * 256) * 2 for k, n in projection_shapes)
    state_traffic = 2 * 32 * 128 * 128 * 4 if kind == "linear" else 2 * 1025 * 4 * 256 * 2
    roofline_ms = (dram_weight_bytes + state_traffic) / (512 * 1e9) * 1000
    phases = {}
    for phase in ["prefill", "decode"]:
        path = ART / f"tracy_final/{kind}/{phase}_perf_report.csv"
        operations = list(csv.DictReader(path.open()))
        device_us = sum(float(row["Device Time [μs]"] or 0) for row in operations)
        gap_us = sum(float(row["Op-to-Op Gap [μs]"] or 0) for row in operations)
        phases[phase] = {
            "artifact": str(path.relative_to(ROOT)),
            "ops": len(operations),
            "device_ms": device_us / 1000,
            "op_gap_ms": gap_us / 1000,
            "matmuls": [
                {
                    key: row[key]
                    for key in [
                        "OP Code",
                        "Math Fidelity",
                        "Input 1 Datatype",
                        "Inner Dim Block Size",
                        "Device Time [μs]",
                        "DRAM Sharded",
                    ]
                }
                for row in operations
                if "Matmul" in row["OP Code"]
            ],
        }
    correctness = json.loads((ART / f"{kind}_final.json").read_text())
    positive = [r for r in correctness["records"] if "pcc" in r]
    record = {
        "baseline_prefill_ms": before["prefill_ms"],
        "baseline_traced_decode_ms": before["decode_ms"],
        "final_prefill_ms": after["prefill_ms"],
        "final_traced_decode_ms": after["decode_ms"],
        "prefill_speedup": before["prefill_ms"] / after["prefill_ms"],
        "decode_speedup": before["decode_ms"] / after["decode_ms"],
        "decode_active_weight_bytes": dram_weight_bytes,
        "decode_state_read_write_or_kv_read_bytes": state_traffic,
        "roofline_ms": roofline_ms,
        "roofline_fraction_unprofiled_warmed_trace": roofline_ms / after["decode_ms"],
        "profile_single_replay_host_wall_ms": profile_wall,
        "profile_single_replay_host_residual_ms": profile_wall
        - phases["decode"]["device_ms"]
        - phases["decode"]["op_gap_ms"],
        "profile_roofline_fraction_single_replay_wall": roofline_ms / profile_wall,
        "profile": phases,
        "min_positive_pcc": min(r["pcc"] for r in positive),
        "min_positive_contribution_pcc": min(r["contribution_pcc"] for r in positive if "contribution_pcc" in r),
        "additional_persistent_projection_bytes": dram_weight_bytes,
        "additional_rope_lookup_bytes": 0 if kind == "linear" else 262144 * 128 * 2,
        "baseline_dense_projection_bytes": dense_weight_bytes,
    }
    summary["layer_kinds"][kind] = record
summary["measurement_note"] = (
    "Headline decode is the median of seven batches of 50 nonblocking warmed trace replays with one sync per batch, "
    "without profiler/watcher/tracker. The same-run profiler reconciliation is a single instrumented replay with "
    "device_only guards and synchronization; its host residual includes instrumentation and per-call overhead. "
    "Do not substitute the profiler single-call wall for the matched steady-state benchmark. Setup, input refresh, "
    "cache reset and result readback are outside all timed windows."
)
(ART / "performance_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("FUSED_SUMMARY_PASS")
