"""Whole-layer device-window timing; deliberately never sums matmul durations."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


def window(rows, start, end, clock_mhz):
    starts = [i for i, r in enumerate(rows) if r.get("OP CODE") == start and r.get("OP TYPE") == "signpost"]
    ends = [i for i, r in enumerate(rows) if r.get("OP CODE") == end and r.get("OP TYPE") == "signpost"]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError(f"Missing, repeated or inverted window: {start}/{end}")
    ops = [r for r in rows[starts[0] + 1 : ends[0]] if r.get("DEVICE FW START CYCLE", "").strip()]
    if not ops:
        raise ValueError(f"No device timestamps for {start}")
    devices = {r["DEVICE ID"] for r in ops}
    if len(devices) != 1:
        raise ValueError(f"Expected one measured chip: {devices}")
    begin = min(int(r["DEVICE FW START CYCLE"]) for r in ops)
    finish = max(int(r["DEVICE FW END CYCLE"]) for r in ops)
    if finish <= begin:
        raise ValueError("Non-positive device span")
    return {
        "device_us": (finish - begin) / clock_mhz,
        "device_ops": len(ops),
        "start_cycle": begin,
        "end_cycle": finish,
        "kernel_sum_us": sum(float(r.get("DEVICE KERNEL DURATION [ns]") or 0) for r in ops) / 1000,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ops_csv", type=Path)
    parser.add_argument("--device-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    with args.device_log.open() as f:
        header = "".join(next(f) for _ in range(3))
    match = re.search(r"CHIP_FREQ\[MHz\]\s*:\s*([0-9.]+)", header)
    if not match:
        raise ValueError("Device log does not identify profiler clock frequency")
    frequency = float(match.group(1))
    with args.ops_csv.open() as f:
        rows = list(csv.DictReader(f))
    prefill = window(rows, "PERF_PREFILL", "PERF_PREFILL_END", frequency)
    decode = [window(rows, f"PERF_DECODE_{i:03d}", f"PERF_DECODE_{i:03d}_END", frequency) for i in range(128)]
    # Dense model useful work, excluding padding. One multiply+add is two FLOPs.
    hidden, intermediate, heads, kv_heads, dim, seq = 4096, 12288, 32, 8, 128, 4096
    params = hidden * ((heads + 2 * kv_heads) * dim + hidden) + 3 * hidden * intermediate
    linear_flops = 2 * seq * params
    attention_flops = 4 * heads * dim * (seq * (seq + 1) // 2)
    softmax_flops = 5 * heads * (seq * (seq + 1) // 2)
    # Approximate scalar work: two group norms, two residuals, Q/K RoPE,
    # SiLU plus gate multiply; count each transcendental as one operation.
    elementwise_flops = seq * (8 * hidden + 2 * hidden + 4 * (heads + kv_heads) * dim + 6 * intermediate)
    useful = linear_flops + attention_flops + softmax_flops + elementwise_flops
    mean_read_context = statistics.mean((p // 128 + 1) * 128 for p in range(4096, 4224))
    # Stock GQA decode partitions each KV head across cores and reuses it for
    # its four query heads. Conservatively count the full final K128 block.
    weight_bytes = (2 if args.baseline else 576 / 1024) * params  # Norm affine is folded into projection weights.
    kv_bytes = 2 * (2 if args.baseline else 1088 / 1024) * kv_heads * dim * (mean_read_context + 32)
    # Conservative materialized DRAM activation estimate: group norm reshapes,
    # RMS, dynamic residual biases, gate/up split and fused activation, linear
    # inputs/outputs, attention output/reshape. L1 QKV/RoPE/cache excluded.
    # 32H+10I is a padded-tile graph estimate, not a transaction counter.
    activation_bytes = 2 * 32 * (32 * hidden + 10 * intermediate) if args.baseline else 2 * 32 * (3 * hidden)
    table_bytes = (64 if args.baseline else 110) * 132 * 4
    dram_bytes = weight_bytes + kv_bytes + activation_bytes + table_bytes
    peak_flops = 120 * 4096 * 1.35e9 / (4 if args.baseline else 1)
    peak_dram = 512e9
    decode_us = statistics.mean(r["device_us"] for r in decode)
    result = {
        "workload": {"input_tokens": 4096, "output_tokens": 128, "batch": 1, "concurrency": 1},
        "clock_mhz": frequency,
        "prefill": prefill,
        "decode": decode,
        "layer_types": {
            "dense": {
                "prefill_device_us": prefill["device_us"],
                "decode_device_us": decode_us,
                "prefill_useful_flops": useful,
                "decode_dram_bytes": dram_bytes,
                "peak_flops_per_s": peak_flops,
                "peak_dram_bytes_per_s": peak_dram,
                "prefill_flops_pct": 100 * useful / (peak_flops * prefill["device_us"] / 1e6),
                "decode_dram_pct": 100 * dram_bytes / (peak_dram * decode_us / 1e6),
                "peak_basis": "One p300c: nominal 120 Tensix at 1.35GHz, 4096 FLOP/cycle/core; 512GB/s. Whole-chip theoretical peak includes reserved cores (110 runtime workers). "
                + (
                    "Baseline HiFi4 divisor 4, BF16 weights/activations/KV."
                    if args.baseline
                    else "Optimized LoFi divisor 1, BF16 activations, BFP4 weights (576 bytes/tile), BFP8 KV (1088 bytes/tile)."
                ),
                "source": "layer",
                "evidence": [str(args.ops_csv), str(args.device_log), "tests/analyze_optimized_profile.py"],
            }
        },
        "timing_basis": "(max DEVICE FW END CYCLE - min DEVICE FW START CYCLE)/CHIP_FREQ[MHz] in each signposted whole-layer window; includes device gaps and all ops. Decode is arithmetic mean of all 128 step windows.",
        "estimate_assumptions": [
            "Causal useful QK+AV work counts only valid query/key pairs, not padded tiles.",
            "Scalar/transcendental FLOPs are approximate, counted as one operation each; causal softmax includes five operations per valid score (scale, subtract, exp, sum, normalization).",
            "Decode DRAM estimates each weight once and each KV head once across its split-K core group, conservatively counting the final K128 block; all four query heads reuse a KV head. Includes cache writes and per-core page-table reads. See sdpa_decode_program_factory.cpp and reader_decode_all.cpp.",
            "Baseline DRAM activation traffic estimates 32 physical BF16 rows times 32H+10I; optimized estimates input read plus attention output write/read (3H); residual/norm/MLP remain in L1. Assumes each weight tile read once; cache writes count a full tile per token; includes block-float headers. These are traffic estimates, not hardware transaction counters.",
            "No utilization value is clamped; per-op tt-perf-report percentages are not averaged.",
        ],
        "peak_sources": [
            "https://docs.tenstorrent.com/aibs/blackhole/p300.html",
            "tt-perf-report 1.3.0 ArchitectureSpec blackhole fidelity formula",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["layer_types"], indent=2))


if __name__ == "__main__":
    main()
