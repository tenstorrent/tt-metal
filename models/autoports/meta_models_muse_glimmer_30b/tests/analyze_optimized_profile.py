# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Account for all signposted default-path replays; retain dtype/config proof."""

import argparse
import collections
import csv
import gzip
import json
import re
import statistics
from pathlib import Path


def analyze(root):
    raw = list(csv.DictReader(gzip.open(root / "decode_ops.csv.gz", "rt")))
    windows = collections.defaultdict(list)
    phase = None
    for r in raw:
        if r["OP TYPE"] == "signpost":
            code = r["OP CODE"]
            phase = code.removeprefix("PERF_").lower() if code in ("PERF_PREFILL", "PERF_DECODE") else None
        elif phase and r["DEVICE KERNEL DURATION [ns]"]:
            windows[phase].append(r)
    profile = next((root / "raw").rglob("profile_log_device.csv"))
    header = profile.open().readline().strip()
    mhz = int(re.search(r"CHIP_FREQ\[MHz\]: (\d+)", header).group(1))
    run = json.loads((root / "perf_run.json").read_text())[0]
    report = {
        "source": str(root),
        "clock_header": header,
        "scope": "B1 S1024, input DRAM; 3 complete prefills / 20 nonblocking decode replays",
        "phases": {},
    }
    table = []
    for phase, rows in windows.items():
        if phase == "decode":
            grouped = collections.defaultdict(list)
            for r in rows:
                grouped[r["METAL TRACE REPLAY SESSION ID"]].append(r)
            assert len(grouped) == 20 and "" not in grouped
            groups = list(grouped.values())
        else:
            assert len(rows) % 3 == 0
            count = len(rows) // 3
            groups = [rows[i * count : (i + 1) * count] for i in range(3)]
        signature = [r["OP CODE"] for r in groups[0]]
        assert all([r["OP CODE"] for r in g] == signature for g in groups)
        kernels = [sum(float(r["DEVICE KERNEL DURATION [ns]"]) for r in g) / 1e6 for g in groups]
        spans = [
            (max(int(r["DEVICE FW END CYCLE"]) for r in g) - min(int(r["DEVICE FW START CYCLE"]) for r in g))
            / mhz
            / 1000
            for g in groups
        ]
        wall = run["decode_warmed_trace_wall_ms" if phase == "decode" else "prefill_warmed_wall_ms"]
        report["phases"][phase] = {
            "classification": "measured",
            "invocations": len(groups),
            "ops_per_invocation": len(signature),
            "kernel_sum_ms_median": statistics.median(kernels),
            "kernel_sum_ms_samples": kernels,
            "firmware_span_ms_median": statistics.median(spans),
            "firmware_span_ms_samples": spans,
            "wall_ms_same_run": wall,
            "wall_minus_kernel_sum_ms": wall - statistics.median(kernels),
            "wall_minus_firmware_span_ms": wall - statistics.median(spans),
        }
        for index, first in enumerate(groups[0]):
            peers = [g[index] for g in groups]
            row = dict(
                phase=phase,
                index=index,
                op=first["OP CODE"],
                kernel_median_ns=statistics.median(float(r["DEVICE KERNEL DURATION [ns]"]) for r in peers),
                op_gap_median_ns=statistics.median(
                    float(r["OP TO OP LATENCY [ns]"] or 0) for r in (peers[1:] if index == 0 else peers)
                ),
                input_dtype=first["INPUT_0_DATATYPE"],
                weight_dtype=first["INPUT_1_DATATYPE"],
                output_dtype=first["OUTPUT_0_DATATYPE"],
                math_fidelity=first["MATH FIDELITY"],
                core_count=first["CORE COUNT"],
                input_memory=first["INPUT_0_MEMORY"],
                weight_memory=first["INPUT_1_MEMORY"],
                output_memory=first["OUTPUT_0_MEMORY"],
                attributes=first["ATTRIBUTES"],
            )
            for inp in ("INPUT_0", "INPUT_1"):
                row[inp.lower() + "_shape"] = "x".join(
                    first[f"{inp}_{dim}_PAD[LOGICAL]"] for dim in ("W", "Z", "Y", "X")
                )
            table.append(row)
    # Physical BFP4 tile is576B (including exponent/header); BFP8 KV tile is1088B.
    weight_bytes = sum(k // 32 * n // 32 * 576 for k, n in [(6656, 8704), (4096, 6656), (6656, 39936), (19968, 6656)])
    kv_bytes = 2 * 2 * ((1024 + 1 + 31) // 32) * (128 // 32) * 1088
    report["roofline"] = {
        "classification": "modeled",
        "physical_weight_bytes": weight_bytes,
        "physical_kv_read_bytes": kv_bytes,
        "kv_read_rule": "Both K/V,2 KV heads,1025 visible positions rounded to32-token pages; excludes implementation rereads",
        "bandwidth_GBps": 512,
        "bandwidth_source": "ttnn/core/operation.cpp:36-42",
        "decode_lower_bound_ms": (weight_bytes + kv_bytes) / 512e6,
        "limitations": "Weights+KV DRAM traffic lower bound, excludes activation movement, norms, attention compute, and dispatch",
    }
    (root / "accounting.json").write_text(json.dumps(report, indent=2) + "\n")
    with (root / "runtime_rows.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    (root / "runtime_rows.json").write_text(json.dumps(table, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("profile_root", type=Path)
    a = p.parse_args()
    analyze(a.profile_root)
