# SPDX-License-Identifier: Apache-2.0
"""Summarize final per-device tt-perf-report CSVs without merging devices."""

import collections
import csv
import json

from .multichip_sweep import OUT


def main():
    result = {}
    for name in ("profile_final_0_128", "profile_final_4_128", "profile_final_0_8193"):
        folder = OUT / name
        modes = {}
        for mode in ("prefill", "decode"):
            devices = {}
            for dev in range(4):
                with (folder / f"{mode}_device{dev}_report.csv").open() as f:
                    rows = list(csv.DictReader(f))
                category = collections.defaultdict(float)
                fabric = local = 0.0
                for row in rows:
                    us = float(row["Device Time"] or 0)
                    op = row["OP Code"]
                    category[row["Op Category"]] += us
                    if any(n in op for n in ("AllReduce", "ReduceScatter", "AllGather")):
                        fabric += us
                    if any(n in op for n in ("Dispatch", "Combine")):
                        local += us
                devices[str(dev)] = dict(
                    rows=len(rows),
                    kernel_us=sum(category.values()),
                    category_us=dict(category),
                    fabric_collective_us=fabric,
                    local_dispatch_combine_us=local,
                    gap_us=sum(float(r["Op-to-Op Gap"] or 0) for r in rows),
                    largest_ops=[
                        dict(op=r["OP Code"], us=float(r["Device Time"] or 0))
                        for r in sorted(rows, key=lambda r: float(r["Device Time"] or 0), reverse=True)[:8]
                    ],
                    matmuls=[
                        {
                            k: r[k]
                            for k in (
                                "OP Code",
                                "Device Time",
                                "Cores",
                                "DRAM",
                                "DRAM %",
                                "Math Fidelity",
                                "Inner Dim Block Size",
                                "Output Subblock H",
                                "Output Subblock W",
                                "Advice",
                            )
                        }
                        for r in rows
                        if "Matmul" in r["OP Code"]
                    ],
                )
            modes[mode] = devices
        result[name] = modes
    (OUT / "profile_analysis.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
