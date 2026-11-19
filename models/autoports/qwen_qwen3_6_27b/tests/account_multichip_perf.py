# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reconcile one signposted mesh profile with its warmed host measurement."""

import argparse
import collections
import csv
import json
from pathlib import Path

from models.autoports.qwen_qwen3_6_27b.tests.multichip_capacity import plan

p = argparse.ArgumentParser()
p.add_argument("--result", type=Path, required=True)
p.add_argument("--perf", type=Path, required=True)
a = p.parse_args()
r = json.loads(a.result.read_text())
m = json.loads((a.perf / "metrics.json").read_text())
capacity = plan()
rows = list(csv.DictReader((a.perf / "decode_perf_report.csv").open()))
times = collections.Counter()
counts = collections.Counter()
categories = collections.Counter()
matmuls = []
advice = []
for row in rows:
    if not row["ID"].isdigit() or not row["Device Time [μs]"]:
        continue
    t = float(row["Device Time [μs]"])
    op = row["OP Code"].split(" ")[0]
    times[op] += t
    counts[op] += 1
    categories[row["Op Category"]] += t
    if op == "MatmulDeviceOperation":
        matmuls.append(
            {
                k: row[k]
                for k in [
                    "OP Code",
                    "Device Time [μs]",
                    "DRAM [GB/s]",
                    "Math Fidelity",
                    "DRAM Sharded",
                    "Input 0 Memory",
                    "Cores",
                    "Inner Dim Block Size",
                ]
            }
        )
    if row.get("Advice"):
        advice.append({"op": row["OP Code"], "advice": row["Advice"]})
# Keep prefill accounting separate: native MMRS combines matmul and CCL,
# so its Other bucket must never be presented as isolated non-math time.
prefill_times = collections.Counter()
prefill_counts = collections.Counter()
prefill_categories = collections.Counter()
prefill_matmuls = []
prefill_advice = []
for row in csv.DictReader((a.perf / "prefill_perf_report.csv").open()):
    if not row["ID"].isdigit() or not row["Device Time [μs]"]:
        continue
    op = row["OP Code"].split(" ")[0]
    duration = float(row["Device Time [μs]"])
    prefill_times[op] += duration
    prefill_counts[op] += 1
    prefill_categories[row["Op Category"]] += duration
    if "Matmul" in op:
        prefill_matmuls.append(
            {
                key: row[key]
                for key in (
                    "OP Code",
                    "Device Time [μs]",
                    "DRAM [GB/s]",
                    "Math Fidelity",
                    "Cores",
                    "Input 0 Memory",
                    "Inner Dim Block Size",
                    "Bound",
                    "Advice",
                )
            }
        )
    if row.get("Advice"):
        prefill_advice.append({"op": row["OP Code"], "advice": row["Advice"]})
prefill = {
    "device_kernel_ms": m["prefill"]["device_kernel_sum_us"] / 1000,
    "device_kernel_plus_gap_ms": m["prefill"]["kernel_plus_gap_us"] / 1000,
    "same_run_host_ms": r["timing"]["prefill_ms"],
    "category_totals_us": dict(prefill_categories),
    "operator_totals_us": [
        {"op": op, "us": value, "count": prefill_counts[op]} for op, value in prefill_times.most_common()
    ],
    "fused_matmul_and_ccl_us": prefill_times["MatmulReduceScatterPrefillDeviceOperation"],
    "matmuls": prefill_matmuls,
    "advice": prefill_advice,
    "scope": "Fused MMRS includes communication and reduction; its measured time is not pure compute. Explicit CCL/category totals exclude that inseparable portion.",
    "timing_comparison": "Report selects the first of three warmed signposted prefill calls; host timing is their mean. Instrumented dispatch gaps and the different statistic can make kernel-plus-gap exceed the host mean. Neither replaces unprofiled latency.",
}
kind = "linear_attention" if r["layer"] == 0 else "full_attention"
weight = capacity["layer_weights_bytes"][kind]
kv = 0 if kind == "linear_attention" else r["batch"] * ((r["length"] + 32) // 32) * 32 * 256 * 2 * 1088 // 1024
# One shard's bytes / one chip's bandwidth equals all shards / aggregate bandwidth.
roofline_ms = (weight + kv) / 512e9 * 1000
host = r["timing"]["decode_trace_ms"]
device = m["decode"]["kernel_plus_gap_us"] / 1000
out = {
    "prefill": prefill,
    "kind": kind,
    "batch": r["batch"],
    "length": r["length"],
    "weight_read_bytes_per_device": weight,
    "kv_read_bytes_per_device": kv,
    "kv_traffic_scope": "Lower bound from visible KV tiles; SDPA key-block padding may cause additional transfers.",
    "dram_bandwidth_bytes_per_second_per_chip": 512e9,
    "roofline_ms": roofline_ms,
    "device_kernel_ms": m["decode"]["device_kernel_sum_us"] / 1000,
    "device_kernel_plus_gap_ms": device,
    "same_run_host_ms": host,
    "host_minus_device_ms": host - device,
    "roofline_fraction_of_host": roofline_ms / host,
    "operator_totals_us": [{"op": k, "us": v, "count": counts[k]} for k, v in times.most_common()],
    "category_totals_us": dict(categories),
    "matmuls": matmuls,
    "advice": advice,
    "limitations": [
        "Roofline is a lower bound: omits intermediate movement and recurrent state traffic.",
        "Merged device report takes the slowest chip per op; host and signposted replay are adjacent warmed executions, so small negative gaps are measurement variation.",
        "Blackhole tool roofline models 1.35 GHz and 512 GB/s per chip, consistent with this P300c source configuration. Compute peaks depend on available cores and fidelity; modeled peaks are not measured sustained throughput.",
    ],
    "source": str(a.result),
    "profile": str(a.perf / "metrics.json"),
}
(a.perf / "accounting.json").write_text(json.dumps(out, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in out.items() if k not in ["operator_totals_us", "matmuls", "advice", "prefill"]}, indent=2
    )
)
