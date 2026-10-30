"""Whole-layer device windows and explicit useful-work roofline estimates."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def window(rows):
    device = [r for r in rows if r.get("DEVICE FW START CYCLE") and r.get("DEVICE FW END CYCLE")]
    if not device:
        raise ValueError("No device timestamps; host time cannot replace device time")
    assert len({r["DEVICE ID"] for r in device}) == 1
    scales = [
        float(r["DEVICE FW DURATION [ns]"]) / (float(r["DEVICE FW END CYCLE"]) - float(r["DEVICE FW START CYCLE"]))
        for r in device
        if float(r["DEVICE FW END CYCLE"]) > float(r["DEVICE FW START CYCLE"])
    ]
    ns_per_cycle = statistics.median(scales)
    elapsed = (
        (max(float(r["DEVICE FW END CYCLE"]) for r in device) - min(float(r["DEVICE FW START CYCLE"]) for r in device))
        * ns_per_cycle
        / 1000
    )
    return elapsed, ns_per_cycle, len(device)


def extract(rows, start, end):
    begin = next(i for i, r in enumerate(rows) if r["OP CODE"] == start)
    finish = next(i for i, r in enumerate(rows[begin + 1 :], begin + 1) if r["OP CODE"] == end)
    return rows[begin + 1 : finish]


def main(args):
    rows = list(csv.DictReader(Path(args.csv).open()))
    pre = extract(rows, "PERF_PREFILL", "PERF_PREFILL_END")
    dec = extract(rows, "PERF_DECODE", "PERF_DECODE_END")
    pre_us, cycle_ns, pre_ops = window(pre)
    sessions = defaultdict(list)
    for row in dec:
        if row.get("DEVICE FW START CYCLE"):
            key = (row["METAL TRACE ID"], row["METAL TRACE REPLAY SESSION ID"])
            assert all(key), "Decode window contains untraced device work"
            sessions[key].append(row)
    assert len(sessions) == 128, f"Expected 128 decoded steps, found {len(sessions)}"
    windows = [window(group)[0] for group in sessions.values()]
    decode_us = statistics.mean(windows)
    assert len({len(group) for group in sessions.values()}) == 1
    kind = args.layer_type
    sliding = kind == "sliding_attention"
    h, i, nq, d, nk = 5376, 21504, 32, 256 if sliding else 512, 16 if sliding else 4
    q, k = nq * d, nk * d
    b, s = args.batch, 4096
    projection_params = h * (2 * q + (2 if sliding else 1) * k + 3 * i)
    pairs = s * 1024 - 1024 * 1023 // 2 if sliding else s * (s + 1) // 2
    useful_flops = b * (2 * s * projection_params + 4 * q * pairs)
    # Norm/RoPE/activation FLOPs are omitted from useful compute (a small lower bound).
    weight_bytes = 2 * (projection_params + 4 * h + 2 * d)
    live_tokens = 1024 if sliding else statistics.mean(range(4097, 4225))
    kv_read_bytes = 4 * b * k * live_tokens
    # Count logical intermediate I/O across norms, projections, RoPE, GELU,
    # multiply/add and attention. Excludes padding and redundant per-core reads.
    activation_bytes = 2 * b * ((23 if sliding else 22) * h + 20 * q + (22 if sliding else 21) * k + 8 * i)
    cache_update_bytes = 2 * b * 4 * k * 32  # two caches, tile read+write for each update
    page_bytes = b * (132 * 4 + 8)
    dram_bytes = weight_bytes + kv_read_bytes + activation_bytes + cache_update_bytes + page_bytes
    peak_flops = 120 * 4096 * 1.35e9 / 4
    peak_dram = 512e9
    result = dict(
        layer_type=kind,
        batch=b,
        workload=dict(input_tokens=4096, output_tokens=128, concurrency=1 if b == 1 else b),
        prefill_device_us=pre_us,
        decode_device_us=decode_us,
        prefill_useful_flops=useful_flops,
        decode_dram_bytes=dram_bytes,
        peak_flops_per_s=peak_flops,
        peak_dram_bytes_per_s=peak_dram,
        prefill_flops_pct=100 * useful_flops / (peak_flops * pre_us / 1e6),
        decode_dram_pct=100 * dram_bytes / (peak_dram * decode_us / 1e6),
        peak_basis="One p300c ASIC: 120 cores, 4096 ops/cycle, 1.35GHz, HiFi4 /4; 512GB/s. Official p300/QuietBox2 specs and tt-perf-report ArchitectureSpec.",
        source="warmed_layer_tracy",
        evidence=str(Path(args.csv)),
        device_window_basis="FW end minus FW start across ALL layer ops, including gaps. Decode is mean of 128 complete per-replay windows; gaps between separate replays excluded.",
        ns_per_cycle=cycle_ns,
        prefill_device_ops=pre_ops,
        decode_replay_count=len(sessions),
        decode_ops_per_replay=len(next(iter(sessions.values()))),
        decode_min_device_us=min(windows),
        decode_max_device_us=max(windows),
        assumptions="BF16 weights/activations/cache, HiFi4; dense active work. Useful FLOPs count all linear projections and causal QK/PV, omit small elementwise work. Estimated DRAM: each weight once, active KV once, logical intermediate I/O and tile cache updates; excludes tile-padding and redundant core reads. Percentages are estimates, not counters.",
        dram_components=dict(
            weights=weight_bytes,
            active_kv=kv_read_bytes,
            logical_intermediates=activation_bytes,
            cache_tile_update=cache_update_bytes,
            page_table=page_bytes,
        ),
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--layer-type", required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--output", required=True)
    main(p.parse_args())
