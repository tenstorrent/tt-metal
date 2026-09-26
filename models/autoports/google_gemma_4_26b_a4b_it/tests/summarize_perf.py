# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize complete signposted layer windows, with explicit roofline estimates."""

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


def number(row, key):
    value = row.get(key, "")
    return float(value) if value not in (None, "", "-", "nan") else None


def tensor_elements(row, prefix, logical=False):
    values = []
    for axis in "WZYX":
        text = row.get(f"{prefix}_{axis}_PAD[LOGICAL]", row.get(f"{prefix}_{axis}", ""))
        parts = re.findall(r"\d+", text)
        if not parts:
            return 0
        values.append(int(parts[-1] if logical else parts[0]))
    return math.prod(values)


def element_bytes(dtype):
    dtype = dtype.upper()
    if "BFLOAT16" in dtype or "FLOAT16" in dtype or "UINT16" in dtype:
        return 2
    if "FLOAT32" in dtype or "INT32" in dtype:
        return 4
    if "BFLOAT8" in dtype:
        return 1.0625
    if "BFLOAT4" in dtype:
        return 0.5625
    raise ValueError(f"Unaccounted DRAM dtype: {dtype}")


def traffic(row):
    """One operand read/output write per native op, with sparse/gather corrections.

    This estimates DRAM traffic, not controller transactions. It excludes extra
    per-core rereads, NOC multicast, profiler writes, and metadata. Device gaps
    remain in the latency denominator. Slice/embedding inputs use selected rows;
    the preceding whole-cache layout conversion still counts its full traffic.
    """
    total = 0
    op = row["OP CODE"].lower()
    for key, memory in row.items():
        if not key.endswith("_MEMORY") or "DRAM" not in memory:
            continue
        prefix = key.removesuffix("_MEMORY")
        count = tensor_elements(row, prefix)
        if "sparsematmul" in op and prefix == "INPUT_1":
            nnz = re.search(r"['\"]nnz['\"]\s*:\s*['\"]?(\d+)", row["ATTRIBUTES"])
            assert nnz, "Sparse traffic requires the actual nnz metadata"
            count *= int(nnz[1]) / 128
        if "pagedupdatecache" in op and (prefix == "INPUT_0" or prefix.startswith("OUTPUT_")):
            # One active decode lane updates one tile row in each KV head.
            # Reader and writer transfer num_heads * width_tiles cache tiles.
            cache_pages = int(re.findall(r"\d+", row["INPUT_0_W_PAD[LOGICAL]"])[0])
            count = tensor_elements(row, "INPUT_0") / cache_pages
        if "pagedfusedupdatecache" in op and (prefix in ("INPUT_0", "INPUT_2") or prefix.startswith("OUTPUT_")):
            # The fused operation updates K and V together; each cache transfers
            # one page, not its entire allocated pool.
            cache_prefix = "INPUT_2" if prefix in ("INPUT_2", "OUTPUT_1") else "INPUT_0"
            cache_pages = int(re.findall(r"\d+", row[f"{cache_prefix}_W_PAD[LOGICAL]"])[0])
            count = tensor_elements(row, cache_prefix) / cache_pages
        if "embedding" in op and prefix == "INPUT_1":
            count = tensor_elements(row, "OUTPUT_0", logical=True)
        if ("slice" in op or "unpad" in op) and prefix == "INPUT_0":
            count = tensor_elements(row, "OUTPUT_0")
        total += count * element_bytes(row[f"{prefix}_DATATYPE"])
    return total


def useful_prefill_flops(kind):
    s, h, q, kv, d = (
        4096,
        2816,
        16,
        (8 if kind == "sliding_attention" else 2),
        (256 if kind == "sliding_attention" else 512),
    )
    pairs = s * 1024 - 1024 * 1023 // 2 if kind == "sliding_attention" else s * (s + 1) // 2
    terms = {
        "qkv_and_output_projections": 2 * s * h * (2 * q * d + kv * d * (2 if kind == "sliding_attention" else 1)),
        "shared_mlp": 6 * s * h * 2112,
        "active_experts_top8": 6 * s * h * 704 * 8,
        "router_projection": 2 * s * h * 128,
        "causal_attention_qk_and_pv": 4 * q * d * pairs,
    }
    return terms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--layer-type", choices=["sliding_attention", "full_attention"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.csv.open()))
    windows = {"prefill": [], "decode": []}
    phase = None
    for row in rows:
        code = row["OP CODE"]
        if code in ("PERF_PREFILL", "PERF_DECODE"):
            phase = code.removeprefix("PERF_").lower()
        elif code in ("PERF_PREFILL_END", "PERF_DECODE_END"):
            phase = None
        elif phase and row["OP TYPE"] == "tt_dnn_device":
            assert number(row, "DEVICE FW START CYCLE") is not None, "Missing device timing inside measured layer"
            windows[phase].append(row)
    assert all(windows.values()), "Both complete signposted windows are required"
    summary = {}
    per_replay = []
    for phase, selected in windows.items():
        devices = {r["DEVICE ID"] for r in selected}
        assert len(devices) == 1, devices
        ratios = [
            number(r, "DEVICE FW DURATION [ns]")
            / (number(r, "DEVICE FW END CYCLE") - number(r, "DEVICE FW START CYCLE"))
            for r in selected
            if number(r, "DEVICE FW END CYCLE") > number(r, "DEVICE FW START CYCLE")
        ]
        ns_per_cycle = statistics.median(ratios)
        groups = {}
        for row in selected:
            key = row["METAL TRACE REPLAY SESSION ID"] if phase == "decode" else "prefill"
            groups.setdefault(key, []).append(row)
        if phase == "decode":
            assert len(groups) == 128 and all(k not in ("", "-", "nan") for k in groups), list(groups)
            assert len({len(group) for group in groups.values()}) == 1, "Incomplete replay program records"
        durations, transfers = [], []
        for session, group in groups.items():
            start = min(number(r, "DEVICE FW START CYCLE") for r in group)
            end = max(number(r, "DEVICE FW END CYCLE") for r in group)
            us = (end - start) * ns_per_cycle / 1000
            durations.append(us)
            transfers.append(sum(traffic(r) for r in group) if phase == "decode" else 0)
            if phase == "decode":
                per_replay.append(
                    dict(
                        session=session,
                        first_cycle=start,
                        last_cycle=end,
                        device_us=us,
                        native_ops=len(group),
                        estimated_dram_bytes=transfers[-1],
                    )
                )
        summary[phase] = dict(
            device_us=statistics.mean(durations),
            min_device_us=min(durations),
            max_device_us=max(durations),
            ns_per_cycle=ns_per_cycle,
            samples=len(groups),
            native_ops=len(selected),
            summed_kernel_device_us=sum(number(r, "DEVICE KERNEL DURATION [ns]") or 0 for r in selected)
            / 1000
            / len(groups),
        )
        if phase == "decode":
            summary[phase]["estimated_dram_bytes"] = statistics.mean(transfers)
            example = next(iter(groups.values()))
            with args.output.with_name("decode_example_ops.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(example)
    terms = useful_prefill_flops(args.layer_type)
    peak_flops = 120 * 4096 * 1.35e9 / 4
    peak_dram = 512e9
    result = dict(
        layer_type=args.layer_type,
        workload=dict(input_tokens=4096, output_tokens=128, batch=1, concurrency=1),
        source=str(args.csv),
        whole_layer_windows=summary,
        prefill_device_us=summary["prefill"]["device_us"],
        decode_device_us=summary["decode"]["device_us"],
        prefill_useful_flops=sum(terms.values()),
        useful_flops_terms=terms,
        decode_dram_bytes=summary["decode"]["estimated_dram_bytes"],
        peak_flops_per_s=peak_flops,
        peak_dram_bytes_per_s=peak_dram,
        prefill_flops_pct=100 * sum(terms.values()) / (peak_flops * summary["prefill"]["device_us"] / 1e6),
        decode_dram_pct=100
        * summary["decode"]["estimated_dram_bytes"]
        / (peak_dram * summary["decode"]["device_us"] / 1e6),
        peak_basis="One P300 Blackhole ASIC: theoretical120 Tensix cores at1.35GHz,4096 FLOP/core/cycle divided by4 for BF16 HiFi4;512GB/s DRAM. Mixed fidelity/SFPU code, so this is a common useful-work roofline basis, not measured FPU utilization.",
        assumptions=[
            "Time is first native firmware start to last native firmware end, including all intervening layer operations and gaps. Decode is mean of128 individual trace replay windows; inter-replay input-copy/host gaps are excluded.",
            "Useful FLOPs count logical projection/MLP/router and causal attention work, with only8 active experts. Padding, masked attention, extra prefill expert computation, scalar normalization and transcendental work are excluded from numerator; their time remains in denominator.",
            "Estimated DRAM bytes sum one padded DRAM input read/output write per native operation. Sparse weights use the recorded nnz/128 fraction (8/128 in decode); cache update reads/writes one 32-token page across KV heads; embedding/slices count selected inputs; whole-cache layout conversions count full pool. Op metadata caching is disabled to preserve invocation-specific shapes. Extra per-core rereads, metadata and profiler traffic are excluded. This is an estimate, not hardware counters.",
            "BF16 weights/cache, mixed BF16/FP32 activations, HiFi4 attention/QKV and expert prefill, shared gate-up/prefill HiFi2, expert sparse decode LoFi, and policy-specific shared-down decode fidelity (selected fused LoFi; functional HiFi2), SFPU decode QKV/router with policy-specific native or SFPU normalization/rotary (see measured decoder config). No percentages are clamped.",
        ],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with args.output.with_suffix(".replays.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_replay[0]))
        writer.writeheader()
        writer.writerows(per_replay)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
