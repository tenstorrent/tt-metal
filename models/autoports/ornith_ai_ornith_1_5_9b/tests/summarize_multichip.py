# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Classify final merged-device reports and verify actual precision rows."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "doc/multichip_decoder"
summary = []
for layer in [0, 3]:
    measured = json.loads((D / f"final_perf_{layer}.json").read_text())
    timing = next(r for r in measured["records"] if r["label"] == "timing_1024")
    for phase in ["prefill", "decode"]:
        path = D / "tracy" / str(layer) / f"{phase}_perf_report.csv"
        rows = [r for r in csv.DictReader(path.open()) if r["OP Code"] and r["Device Time [μs]"] and r["ID"].strip()]
        groups = {k: 0.0 for k in ["matmul", "collective", "data_movement", "other_compute"]}
        gap = 0.0
        matmuls = []
        for r in rows:
            op = r["OP Code"]
            if "Matmul" in op:
                group = "matmul"
                assert r["Input 1 Datatype"] == "BFLOAT8_B", r
                assert r["Input 0 Datatype"] == "BFLOAT16", r
                expected = "LoFi" if any(s in op for s in ["4096 x 3072", "3072 x 4096", "4096 x 6144"]) else "HiFi2"
                assert r["Math Fidelity"].startswith(expected), r
                matmuls.append(
                    {
                        k: r[k]
                        for k in [
                            "OP Code",
                            "Device Time [μs]",
                            "DRAM [%]",
                            "Math Fidelity",
                            "Input 0 Memory",
                            "Advice",
                        ]
                    }
                )
            elif any(s in op for s in ["AllReduce", "AllGather", "ReduceScatter"]):
                group = "collective"
            elif any(
                s in op
                for s in [
                    "Interleaved",
                    "Sharded",
                    "Reshard",
                    "Slice",
                    "Pad",
                    "Copy",
                    "Permute",
                    "Untilize",
                    "Tilize",
                    "Concat",
                    "Reshape",
                    "Typecast",
                ]
            ):
                group = "data_movement"
            else:
                group = "other_compute"
            groups[group] += float(r["Device Time [μs]"])
            gap += float(r["Op-to-Op Gap [μs]"] or 0)
        summary.append(
            {
                "layer": layer,
                "phase": phase,
                "ops": len(rows),
                "category_us": groups,
                "sum_kernel_us": sum(groups.values()),
                "sum_gaps_us": gap,
                "unprofiled_warmed_ms": timing[f"{phase}_ms"],
                "matmuls": matmuls,
                "report": str(path.relative_to(ROOT)),
            }
        )
(D / "perf_summary.json").write_text(
    json.dumps(
        {
            "source_sha256": hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
            "method": "tt-perf-report merged device duration: maximum for noncollectives, mean for collectives; category sums include instrumentation, not an unprofiled critical-path estimate",
            "precision_verified": "BF16 activation, BFP8 weights, attention HiFi2 and MLP LoFi in all measured matmul rows",
            "rows": summary,
        },
        indent=2,
    )
    + "\n"
)
print("PERF_SUMMARY_PASS")
