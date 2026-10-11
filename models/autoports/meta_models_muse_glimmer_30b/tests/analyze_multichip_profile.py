# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-chip accounting for every signposted mesh invocation, with raw runtime rows."""
import argparse
import collections
import csv
import gzip
import json
import re
import statistics
from pathlib import Path


def analyze(root):
    raw = list(csv.DictReader(gzip.open(root / "ops_perf_results.csv.gz", "rt")))
    windows = collections.defaultdict(lambda: collections.defaultdict(list))
    phase = None
    for row in raw:
        if row["OP TYPE"] == "signpost":
            phase = (
                row["OP CODE"].removeprefix("PERF_").lower()
                if row["OP CODE"] in ("PERF_PREFILL", "PERF_DECODE")
                else None
            )
        elif phase and row["DEVICE KERNEL DURATION [ns]"]:
            windows[phase][row["DEVICE ID"]].append(row)
    profile = next((root / "raw").rglob("profile_log_device.csv"))
    header = profile.open().readline().strip()
    mhz = int(re.search(r"CHIP_FREQ\[MHz\]: (\d+)", header).group(1))
    run = json.loads((root / "perf_run.json").read_text())
    report = dict(
        source=str(root),
        clock_header=header,
        policy=run["policy"],
        prefill_captured_policies=run.get("prefill_captured_policies"),
        phases={},
    )
    table = []
    for phase, devices in windows.items():
        assert len(devices) == 4, (phase, devices.keys())
        per_device = {}
        for device, rows in devices.items():
            if phase == "decode":
                grouped = collections.defaultdict(list)
                for row in rows:
                    grouped[row["METAL TRACE REPLAY SESSION ID"]].append(row)
                assert len(grouped) == 20 and "" not in grouped, (len(grouped), device)
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
            per_device[device] = dict(
                classification="measured",
                invocations=len(groups),
                ops_per_invocation=len(signature),
                kernel_sum_ms_median=statistics.median(kernels),
                firmware_span_ms_median=statistics.median(spans),
                kernel_sum_samples=kernels,
                firmware_span_samples=spans,
            )
            for index, first in enumerate(groups[0]):
                peers = [g[index] for g in groups]
                out = dict(
                    phase=phase,
                    device=device,
                    index=index,
                    op=first["OP CODE"],
                    kernel_median_ns=statistics.median(float(r["DEVICE KERNEL DURATION [ns]"]) for r in peers),
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
                    out[inp.lower() + "_shape"] = "x".join(
                        first[f"{inp}_{dim}_PAD[LOGICAL]"] for dim in ("W", "Z", "Y", "X")
                    )
                table.append(out)
        wall = run[phase + "_ms"]
        max_kernel = max(v["kernel_sum_ms_median"] for v in per_device.values())
        max_span = max(v["firmware_span_ms_median"] for v in per_device.values())
        report["phases"][phase] = dict(
            classification="measured",
            per_device=per_device,
            wall_ms_same_run=wall,
            slowest_chip_kernel_sum_ms=max_kernel,
            slowest_chip_firmware_span_ms=max_span,
            wall_minus_slowest_firmware_ms=wall - max_span,
        )
    weights = 0
    for role, k, n in (
        ("qkv", 6656, 2304),
        ("self_attn.o_proj", 1024, 6656),
        ("mlp.gate_up", 6656, 9984),
        ("mlp.down_proj", 4992, 6656),
    ):
        multiple = 8 * run["policy"].get(role + ".readers", run["policy"]["readers"]) * 32
        weights += k // 32 * ((n + multiple - 1) // multiple * multiple) // 32 * 576
    logical_weights = sum(k * n // 1024 * 576 for k, n in ((6656, 2304), (1024, 6656), (6656, 9984), (4992, 6656)))
    visible_tokens = min(run["length"], 131071) + 1
    kv_pages = (visible_tokens + 31) // 32
    kv = 2 * kv_pages * 4 * 1088
    report["roofline"] = dict(
        classification="modeled",
        per_device_weight_bytes=weights,
        per_device_logical_weight_bytes=logical_weights,
        per_device_kv_bytes=kv,
        visible_decode_tokens=visible_tokens,
        visible_decode_pages=kv_pages,
        per_device_peak_GBps=512,
        aggregate_peak_GBps=2048,
        decode_lower_bound_ms=(logical_weights + kv) / 512e6,
        decode_full_storage_read_model_ms=(weights + kv) / 512e6,
        source="ttnn/core/operation.cpp; BFP4=576B/tile, BFP8=1088B/tile",
        limits="Modeled one read of logical weights and cache pages; bank-padded storage read model also shown. Actual DRAM transactions are not measured. Excludes norms, CCL, activation movement, computation and launch gaps.",
    )
    (root / "accounting.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "runtime_rows.json").write_text(json.dumps(table, indent=2) + "\n")
    with (root / "runtime_rows.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    analyze(p.parse_args().root)
