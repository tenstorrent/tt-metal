# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the operator inventory and encoded-weight bandwidth estimates.

Input CSV is reconstructed from p0-priority-v1/completed/capture.json's ordered
gzip parts. No hardware is opened. Timings are profiled kernel sums per chip,
not active hardware counters or an additive end-to-end critical path.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

BANDWIDTH = 512e9
NAMES = {
    (5120, 4608): "GDN packed projection",
    (1536, 5120): "Attention/GDN output projection",
    (5120, 9216): "MLP gate/up",
    (4352, 5120): "MLP down",
    (5120, 3584): "Full-attention packed projection",
    (5120, 16384): "Vocabulary head full chunk",
    (5120, 12928): "Vocabulary head tail chunk",
}


def run(args):
    analysis_path = args.capture / "analysis.json"
    profile_path = args.capture / "profile.json"
    analysis = json.loads(analysis_path.read_text())
    profile = json.loads(profile_path.read_text())
    assert analysis["full_trace_reconciliation_passed"] is True
    assert len(analysis["ranks"]) == 12 and all(r["device_op_rows"] == 5001 for r in analysis["ranks"])
    with args.csv.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == analysis["csv_sha256"]
    inventory = []
    names = sorted({name for rank in analysis["ranks"] for name in rank["operations"]})
    for name in names:
        rows = [rank["operations"].get(name, {"calls": 0, "kernel_ns": 0}) for rank in analysis["ranks"]]
        inventory.append(
            dict(
                operation=name,
                median_calls=statistics.median(r["calls"] for r in rows),
                median_kernel_ms=statistics.median(r["kernel_ns"] for r in rows) / 1e6,
                min_kernel_ms=min(r["kernel_ns"] for r in rows) / 1e6,
                max_kernel_ms=max(r["kernel_ns"] for r in rows) / 1e6,
            )
        )
    inventory.sort(key=lambda row: -row["median_kernel_ms"])
    assert sum(r["median_calls"] for r in inventory) == 5001
    # Collect only executed model trace rows; exclude eager/capture metadata.
    grouped = defaultdict(lambda: defaultdict(list))
    attributes, seen = defaultdict(set), set()
    with args.csv.open() as stream:
        for row in csv.DictReader(stream):
            if row.get("METAL TRACE ID") != str(profile["model_trace_id"]):
                continue
            session = row.get("METAL TRACE REPLAY SESSION ID")
            if session in (None, "", "-") or row["OP CODE"] != "MatmulDeviceOperation":
                continue
            identity = (int(row["DEVICE ID"]), int(session), int(row["GLOBAL CALL COUNT"]))
            assert identity not in seen
            seen.add(identity)
            assert row["INPUT_1_DATATYPE"] == "BFLOAT8_B" and row["MATH FIDELITY"] == "HiFi2"
            shape = tuple(int(row[f"INPUT_1_{c}_PAD[LOGICAL]"].split("[")[0]) for c in "WZYX")
            assert shape[:2] == (1, 1) and shape[-2:] in NAMES
            assert shape[-1] % 32 == 0 and shape[-2] % 32 == 0
            grouped[identity[:2]][shape[-2:]].append(float(row["DEVICE KERNEL DURATION [ns]"]))
            attributes[shape[-2:]].add(row["ATTRIBUTES"])
    assert len(grouped) == 12
    # Match the independently reconciled summary before deriving bandwidth.
    for device in profile["device_ids"]:
        sessions = sorted(session for rank, session in grouped if rank == device)
        assert len(sessions) == 3
        for replay, session in enumerate(sessions):
            data = grouped[device, session]
            total = sum(sum(values) for values in data.values())
            reference = next(r for r in analysis["ranks"] if r["device"] == device and r["replay"] == replay)
            assert total == reference["operations"]["MatmulDeviceOperation"]["kernel_ns"]
            assert sum(len(values) for values in data.values()) == 260
    projections = []
    total_bytes = 0
    for shape, name in NAMES.items():
        times = [data[shape] for data in grouped.values()]
        counts = {len(values) for values in times}
        assert len(counts) == 1
        count = counts.pop()
        encoded_bytes = math.prod(shape) // 1024 * 1088
        total_bytes += encoded_bytes * count
        kernel_ms = statistics.median(sum(values) for values in times) / 1e6
        bw_gbs = encoded_bytes * count / (kernel_ms * 1e6)
        projections.append(
            dict(
                projection=name,
                stored_kn=list(shape),
                calls_per_step=count,
                encoded_bytes_per_call=encoded_bytes,
                median_call_us=statistics.median(v for values in times for v in values) / 1e3,
                median_sum_ms=kernel_ms,
                encoded_weight_gbs=bw_gbs,
                fraction_of_512_gbs=bw_gbs / 512,
                weight_only_floor_ms=encoded_bytes * count / BANDWIDTH * 1e3,
                program_attributes=sorted(attributes[shape]),
            )
        )
    matmul_ms = next(row["median_kernel_ms"] for row in inventory if row["operation"] == "MatmulDeviceOperation")
    report = dict(
        scope="B16/32K TP4, BFP8 weights/KV, FP32 recurrent state; 64 layers, 4 ranks, 3 replays",
        boundaries="Medians of per-rank/replay kernel sums; do not add family medians as a wall-time budget",
        bandwidth_assumptions="One read of every declared padded BFP8 weight tile (1088 B/32x32); assumed 512 GB/s/chip; excludes extra transactions, activations, and compute constraints",
        compute_roofline_calibrated=False,
        physical_dram_counters=False,
        source_sha256={"analysis.json": hashlib.sha256(analysis_path.read_bytes()).hexdigest(), "ops.csv": digest},
        inventory=inventory,
        projections=projections,
        matmul_totals=dict(
            calls=260,
            median_kernel_ms=matmul_ms,
            encoded_weight_bytes=total_bytes,
            encoded_weight_gbs=total_bytes / (matmul_ms * 1e6),
            fraction_of_512_gbs=total_bytes / (matmul_ms * 1e6) / 512,
            weight_only_floor_ms=total_bytes / BANDWIDTH * 1e3,
            weight_time_at_90_percent_ms=total_bytes / (0.9 * BANDWIDTH) * 1e3,
        ),
    )
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# B16/32K measured operator inventory",
        "",
        report["scope"],
        "",
        "All 38 operation types and 5001 device-operation records per rank/replay are included.",
        "These are not program counts. Profiler overhead is about 7.46%; do not subtract it uniformly.",
        "Family medians are not a disjoint wall-time budget; RISC durations include waiting.",
        "",
        "| Operation | Calls/step | Median kernel sum, ms | Rank/replay range, ms |",
        "|---|---:|---:|---:|",
    ]
    for row in inventory:
        lines.append(
            f"| {row['operation']} | {row['median_calls']:g} | {row['median_kernel_ms']:.4f} | "
            f"{row['min_kernel_ms']:.4f}-{row['max_kernel_ms']:.4f} |"
        )
    lines += [
        "",
        "## Matmul weight-streaming estimates",
        "",
        report["bandwidth_assumptions"],
        "This is not measured physical DRAM utilization or a calibrated compute roofline.",
        "Different call shapes have different overheads; padding is included in encoded bytes.",
        "",
        "| Projection | Stored K x N | Calls | Kernel sum, ms | Weight GB/s | Fraction of assumed peak |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in projections:
        lines.append(
            f"| {row['projection']} | {row['stored_kn'][0]} x {row['stored_kn'][1]} | "
            f"{row['calls_per_step']} | {row['median_sum_ms']:.3f} | {row['encoded_weight_gbs']:.1f} | "
            f"{row['fraction_of_512_gbs']:.1%} |"
        )
    total = report["matmul_totals"]
    lines += [
        "",
        f"Aggregate: {total_bytes / 1e9:.6f} GB / {matmul_ms:.3f} ms = "
        f"{total['encoded_weight_gbs']:.1f} GB/s ({total['fraction_of_512_gbs']:.1%}).",
        f"The encoded-weight-only floor is {total['weight_only_floor_ms']:.3f} ms at peak, "
        f"or {total['weight_time_at_90_percent_ms']:.3f} ms at 90%.",
        "Those lower bounds omit necessary work and are not promised kernel timings.",
        "",
    ]
    assert len(inventory) == 38
    (args.output / "INVENTORY.md").write_text("\n".join(lines))
    print(json.dumps(report["matmul_totals"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
