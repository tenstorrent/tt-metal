# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Join TP4 reader microbenchmarks, runtime rows and recorded-rank agreement."""

import argparse
import csv
import gzip
import json
import math
import statistics
from pathlib import Path

import torch


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float(a.dot(b) / (a.norm() * b.norm()))


def analyze(root):
    torch.set_num_threads(8)
    shapes = {
        "qkv": (6656, 2304),
        "self_attn.o_proj": (1024, 6656),
        "mlp.gate_up": (6656, 9984),
        "mlp.down_proj": (4992, 6656),
    }
    result = []
    for layer in (0, 3):
        for readers in (1, 2, 3):
            directory = root / f"tracy_readers_{readers}/layer_{layer}"
            groups, active = {}, None
            for row in csv.DictReader(gzip.open(directory / "ops_perf_results.csv.gz", "rt")):
                code = row["OP CODE"]
                if row["OP TYPE"] == "signpost":
                    active = code if code.startswith("PERF_MICRO_") and not code.endswith("_END") else None
                elif active and code == "MatmulDeviceOperation" and row["METAL TRACE REPLAY SESSION ID"]:
                    groups.setdefault(active, []).append(row)
            for role, (k, n) in shapes.items():
                ops = groups["PERF_MICRO_" + role.replace(".", "_").upper()]
                assert len(ops) == 12  # Three replays on each of four chips.
                assert all(r["INPUT_1_DATATYPE"] == "BFLOAT4_B" and r["MATH FIDELITY"] == "LoFi" for r in ops)
                assert all(f"num_workers_per_dram_bank={readers}" in r["ATTRIBUTES"] for r in ops)
                per_chip = {str(d): [r for r in ops if r["DEVICE ID"] == str(d)] for d in range(4)}
                assert all(len(rows) == 3 for rows in per_chip.values())
                ns = max(
                    statistics.median(float(r["DEVICE KERNEL DURATION [ns]"]) for r in rows)
                    for rows in per_chip.values()
                )
                physical_n = math.ceil(n / (8 * readers * 32)) * (8 * readers * 32)
                physical = (k // 32) * (physical_n // 32) * 576
                tiles_per_bank = physical_n // 32 // 8
                row_bytes = tiles_per_bank // readers * 576
                tensors = torch.load(directory / f"perf_run.micro_{role}.pt", weights_only=True)
                control = torch.load(
                    root / f"tracy_readers_2/layer_{layer}/perf_run.micro_{role}.pt", weights_only=True
                )
                agreement = {
                    kind: [pcc(a, b) for a, b in zip(tensors[kind], control[kind])] for kind in ("inputs", "outputs")
                }
                assert min(agreement["inputs"] + agreement["outputs"]) >= 0.995, agreement
                host = []
                for path in sorted(root.glob(f"reader_final_l{layer}_o*_r{readers}.json")):
                    row = json.loads(path.read_text())
                    host.extend(v["samples_ms"] for v in row["projection_microbench"] if v["name"] == role)
                entry = dict(
                    layer=layer,
                    role=role,
                    readers=readers,
                    logical_shape=[32, k, n],
                    physical_n=physical_n,
                    device_kernel_slowest_chip_median_ns=ns,
                    device_kernel_min_ns=min(float(r["DEVICE KERNEL DURATION [ns]"]) for r in ops),
                    device_kernel_max_ns=max(float(r["DEVICE KERNEL DURATION [ns]"]) for r in ops),
                    per_bank_output_tiles=tiles_per_bank,
                    per_reader_physical_row_bytes=row_bytes,
                    noc_bursts_per_row=math.ceil(row_bytes / 16384),
                    modeled_physical_weight_bytes=physical,
                    modeled_logical_value_bytes=k * n / 2,
                    physical_GBps_modeled_bytes_measured_time=physical / ns,
                    logical_GBps_modeled_bytes_measured_time=(k * n / 2) / ns,
                    percent_512_GBps_peak=physical / ns / 512 * 100,
                    input_multicast_cores=32 if role == "self_attn.o_proj" else 52,
                    alternating_order_host_ms_samples=host,
                    recorded_rank_pcc_vs_two_readers=agreement,
                    kernel_durations={
                        key: statistics.median(float(r[key]) for r in ops)
                        for key in (
                            "DEVICE TRISC1 KERNEL DURATION [ns]",
                            "DEVICE BRISC KERNEL DURATION [ns]",
                            "DEVICE NCRISC KERNEL DURATION [ns]",
                        )
                    },
                    runtime_attributes=ops[0]["ATTRIBUTES"],
                    runtime_source=str(directory),
                    selected=readers == (3 if role in ("qkv", "self_attn.o_proj") else 2),
                )
                result.append(entry)
    report = dict(
        rows=result,
        peak_source="ttnn/core/operation.cpp: Blackhole eight-bank 512 GB/s per chip",
        bandwidth_classification="modeled bytes divided by measured device kernel time; not measured bus transactions",
        agreement_note="Every rank compared to the two-reader run. Later projection inputs can differ slightly because preceding reader configurations change floating-point rounding; input and output PCC are both retained.",
        timing_note="Host samples use alternating 1/2/3/3/2/1 order, three groups of100 traced replays per run, including input reshard. Isolated device rows use three profiled replays per rank, and exclude that reshard.",
        burst_source="ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.cpp: Blackhole NOC_MAX_BURST_SIZE=16384; modeled ceil(row bytes/16384).",
    )
    (root / "reader_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    keys = [k for k, v in result[0].items() if not isinstance(v, (dict, list))]
    with (root / "reader_analysis.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(result)
    print("READER_ANALYSIS_PASS", len(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    analyze(parser.parse_args().root)
