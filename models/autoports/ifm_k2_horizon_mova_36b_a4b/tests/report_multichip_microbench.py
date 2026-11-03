# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only four-rank microbenchmark profiler accounting."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

D = Path("models/autoports/ifm_k2_horizon_mova_36b_a4b/doc/optimized_multichip_decoder")
result = []
provenance = []
for layer in (0, 3):
    raw = max(
        (D / "local" / f"dram_microbench_profile_l{layer}").glob("reports/*/ops_perf_results_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    with raw.open() as f:
        rows = list(csv.DictReader(f))
    provenance.append(dict(layer=layer, raw=str(raw), sha256=hashlib.sha256(raw.read_bytes()).hexdigest()))
    wall = json.loads((D / f"dram_microbench_wall_l{layer}.json").read_text())
    for row in wall["rows"]:
        marker = "MICRO_" + row["role"].replace(".", "_") + "_R" + str(row["readers"])
        a = next(i for i, r in enumerate(rows) if r["OP CODE"] == marker)
        b = next(i for i, r in enumerate(rows) if r["OP CODE"] == marker + "_END")
        rr = [r for r in rows[a + 1 : b] if r["OP CODE"] == "MatmulDeviceOperation"]
        assert sorted(int(r["DEVICE ID"]) for r in rr) == [0, 1, 2, 3], (marker, len(rr))
        kernels = [float(r["DEVICE KERNEL DURATION [ns]"]) for r in rr]
        worst = max(kernels) / 1e6
        result.append(
            dict(
                layer=layer,
                **row,
                median_ms=statistics.median(row["times_ms"]),
                host_samples_note="All four alternating repeats retained; median is robust to first-host-loop startup.",
                device_kernel_ms_per_rank=[
                    dict(
                        rank=int(r["DEVICE ID"]),
                        kernel_ms=float(r["DEVICE KERNEL DURATION [ns]"]) / 1e6,
                        reader_brisc_ms=float(r["DEVICE BRISC KERNEL DURATION [ns]"] or 0) / 1e6,
                        math_trisc0_ms=float(r["DEVICE TRISC0 KERNEL DURATION [ns]"] or 0) / 1e6,
                    )
                    for r in rr
                ],
                worst_rank_kernel_ms=worst,
                physical_GBs_per_chip_device=row["physical_weight_bytes"] / worst / 1e6,
                logical_GBs_per_chip_device=row["logical_weight_bytes"] / worst / 1e6,
                device_peak_fraction=row["physical_weight_bytes"] / worst / 1e6 / 512,
                attributes=rr[0]["ATTRIBUTES"],
                compute_kernel_source=rr[0]["COMPUTE KERNEL SOURCE"],
                reader_kernel_source=rr[0]["DATA MOVEMENT KERNEL SOURCE"],
            )
        )
(D / "dram_microbench_summary.json").write_text(
    json.dumps(
        dict(
            provenance=provenance,
            rows=result,
            logical_bytes_definition="TTNN logical weight shape before DRAM-bank N padding, including model-internal inert shared-expert padding. Physical bytes include stored tile headers and bank N padding.",
            reader_analysis="Per-reader rows0.56-6KiB over8 input cores. Shared-down2/3worker device math/reader times nearly tie; extra fanout adds no DRAM channels. Whole-layer repeats decide. Attributes/kernel paths identify actual multicore execution.",
            peak="512 GB/s per chip (tt-perf-report Blackhole model); four chips have4xtraffic and4xbandwidth",
        ),
        indent=2,
    )
    + "\n"
)
fields = [
    "layer",
    "role",
    "readers",
    "block",
    "mean_ms",
    "worst_rank_kernel_ms",
    "physical_GBs_per_chip_device",
    "logical_GBs_per_chip_device",
    "device_peak_fraction",
    "per_bank_tiles",
    "per_reader_row_bytes_min",
    "per_reader_row_bytes_max",
]
with (D / "dram_microbench_summary.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows({k: r[k] for k in fields} for r in result)
for r in result:
    print(
        r["layer"], r["role"], r["readers"], round(r["mean_ms"] * 1000, 3), round(r["worst_rank_kernel_ms"] * 1000, 3)
    )
