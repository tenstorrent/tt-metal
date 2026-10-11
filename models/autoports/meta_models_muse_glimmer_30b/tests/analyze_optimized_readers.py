# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join alternating-reader measurements to their exact profiler signposts."""

import argparse
import csv
import gzip
import json
import statistics
from pathlib import Path


def analyze(evidence, raw_root, output):
    measured = json.loads(evidence.read_text())
    raw = next(raw_root.rglob("ops_perf_results*.csv"))
    groups, active = {}, None
    for r in csv.DictReader(raw.open()):
        code = r["OP CODE"]
        if r["OP TYPE"] == "signpost":
            active = None if code.endswith("_END") else code
        elif active and code == "MatmulDeviceOperation" and r["METAL TRACE REPLAY SESSION ID"]:
            groups.setdefault(active, []).append(r)
    rows = []
    for row in measured["rows"]:
        result = dict(row)
        if row["status"] == "pass":
            tag = f"READER_{row['role']}_{row['order']}_{row['readers']}"
            ops = groups[tag]
            assert len(ops) == 90, (tag, len(ops))
            assert all(r["INPUT_1_DATATYPE"] == "BFLOAT4_B" and r["MATH FIDELITY"] == "LoFi" for r in ops)
            assert all(f"num_workers_per_dram_bank={row['readers']}" in r["ATTRIBUTES"] for r in ops)
            times = [float(r["DEVICE KERNEL DURATION [ns]"]) for r in ops]
            ns = statistics.median(times)
            _, k, n = row["shape"]
            physical = k // 32 * row["physical_n"] // 32 * 576
            logical = k * n / 2
            result.update(
                device_kernel_median_ns=ns,
                device_kernel_min_ns=min(times),
                device_kernel_max_ns=max(times),
                replay_count=len(ops),
                physical_weight_bytes_modeled=physical,
                logical_weight_bytes_modeled=logical,
                physical_GBps_modeled_bytes_measured_time=physical / ns,
                logical_GBps_modeled_bytes_measured_time=logical / ns,
                percent_512_GBps_peak=physical / ns / 512 * 100,
                trisc1_median_ns=statistics.median(float(r["DEVICE TRISC1 KERNEL DURATION [ns]"]) for r in ops),
                brisc_median_ns=statistics.median(float(r["DEVICE BRISC KERNEL DURATION [ns]"]) for r in ops),
                ncrisc_median_ns=statistics.median(float(r["DEVICE NCRISC KERNEL DURATION [ns]"]) for r in ops),
                runtime_attributes=ops[0]["ATTRIBUTES"],
                observed_core_count=ops[0]["CORE COUNT"],
            )
        rows.append(result)
    report = dict(
        source=str(evidence),
        profiler=str(raw),
        peak_source="ttnn/core/operation.cpp:36-42 (Blackhole 8-bank 512 GB/s)",
        rows=rows,
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    keys = [
        "role",
        "order",
        "readers",
        "block",
        "status",
        "ms",
        "pcc",
        "device_kernel_median_ns",
        "physical_weight_bytes_modeled",
        "physical_GBps_modeled_bytes_measured_time",
        "logical_GBps_modeled_bytes_measured_time",
        "percent_512_GBps_peak",
        "device_kernel_min_ns",
        "device_kernel_max_ns",
    ]
    with output.with_suffix(".csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with raw.open("rb") as f:
        output.with_suffix(".raw.csv.gz").write_bytes(gzip.compress(f.read(), mtime=0))
    for r in rows:
        print(r["role"], r["readers"], r["status"], r.get("device_kernel_median_ns"), r.get("percent_512_GBps_peak"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("evidence", type=Path)
    p.add_argument("raw_root", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    analyze(a.evidence, a.raw_root, a.output)
