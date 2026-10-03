# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Roofline model of one DeepSeek-V4.1-Flash decode step on the 4x8 Blackhole Galaxy (CPU only).

Per chip bytes of weights streamed from DRAM per step + KV reads, FLOPs, against DRAM bandwidth and a compute peak, for batch
sizes 16/32/64/128. The expert term depends on the batch: B tokens x 6 routed experts activate a subset of the 384 experts and
the step is bound by the BUSIEST device (12 experts per device, 32 devices): expected distinct active experts on the busiest
device is simulated with uniform random routing (real routing is skewed; calibrated against the measured 3.8 at batch 16).
Usage: python tests/roofline_model.py
Assumptions (edit below): effective DRAM bandwidth 450 GB/s per chip (measured: moe_compute streams an expert in 83 us), bfp8 =
1.0625 B/element, compute peak 100 TFLOP/s effective per chip (an assumption: the conclusion does not depend on it).
"""

import random

DRAM_GBPS = 450.0
COMPUTE_TFLOPS = 100.0
BFP8 = 1.0625
BF16 = 2.0
FP32 = 4.0
MESH_ROWS, MESH_COLS = 4, 8
N_EXPERTS, EXPERTS_PER_DEV, TOPK = 384, 12, 6
LAYERS = 40
# measured device time of the traced step (ms): batch -> ms
MEASURED_MS = {16: 60.0, 32: 69.0, 64: 85.0}


def expert_load(batch, trials=400):
    """(mean, expected max over devices) of distinct active experts per device."""
    rng = random.Random(0)
    mean_sum, max_sum = 0.0, 0.0
    for _ in range(trials):
        active = set()
        for _tok in range(batch):
            active.update(rng.sample(range(N_EXPERTS), TOPK))
        per = [0] * (N_EXPERTS // EXPERTS_PER_DEV)
        for e in active:
            per[e // EXPERTS_PER_DEV] += 1
        mean_sum += sum(per) / len(per)
        max_sum += max(per)
    return mean_sum / trials, max_sum / trials


def per_layer_bytes(batch, e_max):
    T = batch // MESH_ROWS  # users per mesh row = tokens per device
    B = {}
    # attention (per chip): wqkv replicated, wq_b / wo_a / wo_b sharded over the 8 columns, pair-swap matrix read ~3x, KV reads
    B["attention weights"] = (5120 * 1792 + 1280 * 4096 + 4608 * 1024 + 1024 * 5120) * BFP8 + 3 * 512 * 512 * BF16
    B["attention KV reads"] = T * (128 + 128) * 512 * BF16  # window ring + compressed slots per user (SDPA reads both)
    B["mHC (fn weights x2 + streams)"] = 2 * 24 * 20480 * FP32 + T * 4 * 5120 * FP32 * 10
    B["router weights"] = 5120 * 384 * BF16
    B["routed experts (busiest device)"] = e_max * 3 * 5120 * 2304 * BFP8
    B["shared expert"] = (5120 * 4608 + 2304 * 5120) * BFP8
    return B


def flops_per_layer_per_chip(batch):
    """Total MACs of one layer for `batch` tokens (6 routed + 1 shared expert, attention projections), x2 for FLOPs, / 32 chips."""
    attn = 5120 * 1792 + 1280 * 64 * 512 + 64 * 512 * 1024 * 8 // 8 * 8 // 8 + 8 * 1024 * 5120
    attn = 5120 * 1792 + 1280 * 32768 + 8 * 4096 * 1024 + 8192 * 5120
    moe = (TOPK + 1) * 3 * 5120 * 2304
    return 2.0 * batch * (attn + moe) / (MESH_ROWS * MESH_COLS)


def main():
    print(
        f"assumptions: DRAM {DRAM_GBPS:.0f} GB/s per chip, compute {COMPUTE_TFLOPS:.0f} TFLOP/s per chip, bfp8 {BFP8} B/elem\n"
    )
    for batch in (16, 32, 64, 128):
        e_mean, e_max = expert_load(batch)
        layer = per_layer_bytes(batch, e_max)
        layer_bytes = sum(layer.values())
        t_dram_layer = layer_bytes / (DRAM_GBPS * 1e9) * 1e3  # ms
        t_cmp_layer = flops_per_layer_per_chip(batch) / (COMPUTE_TFLOPS * 1e12) * 1e3
        extra_bytes = (
            2 * (6144 * 25600 * BFP8) / MESH_COLS + 5120 * (129280 // MESH_COLS) * BFP8
        )  # 2 Engram wkv (TP over cols) + head
        extra_ms = extra_bytes / (DRAM_GBPS * 1e9) * 1e3
        floor_ms = LAYERS * max(t_dram_layer, t_cmp_layer) + extra_ms
        meas = MEASURED_MS.get(batch)
        print(
            f"== batch {batch} ({batch // MESH_ROWS} users per mesh row)  expected active experts per device: mean {e_mean:.2f}, busiest {e_max:.2f} of {EXPERTS_PER_DEV}"
        )
        for k, v in layer.items():
            print(f"   {k:34s} {v / 1e6:8.1f} MB   {v / (DRAM_GBPS * 1e9) * 1e6:7.1f} us")
        print(
            f"   {'layer total (DRAM floor)':34s} {layer_bytes / 1e6:8.1f} MB   {t_dram_layer * 1e3:7.1f} us   (compute floor {t_cmp_layer * 1e3:.1f} us -> {'memory' if t_dram_layer > t_cmp_layer else 'compute'}-bound)"
        )
        print(
            f"   token step floor: {LAYERS} layers x {t_dram_layer:.3f} ms + Engram/head {extra_ms:.2f} ms = {floor_ms:.1f} ms  -> {1e3 / floor_ms:.1f} tok/s/user, {batch * 1e3 / floor_ms:.0f} tok/s aggregate"
        )
        if meas:
            print(
                f"   measured device step: {meas:.1f} ms  -> {floor_ms / meas * 100:.0f}% of roofline ({meas / floor_ms:.1f}x the floor), {1e3 / meas:.1f} tok/s/user"
            )
        print()
    # bfp4 experts (bfp4_b = 4-bit mantissa + shared exponent per 16 = 0.5625 B/element), batch 16
    e_mean, e_max = expert_load(16)
    layer = per_layer_bytes(16, e_max)
    layer["routed experts (busiest device)"] *= 0.5625 / BFP8
    t = sum(layer.values()) / (DRAM_GBPS * 1e9) * 1e3
    print(f"batch 16 with bfp4 experts: layer floor {t * 1e3:.0f} us -> token floor {LAYERS * t + 0.25:.1f} ms\n")

    # per-section: measured (batch 16, in-trace, layer 2) against the DRAM floor of the same section
    layer = per_layer_bytes(
        16, 3.8
    )  # 3.8 = busiest-device expert count implied by the measured moe_compute time (375 us = 60 + 83 x 3.8)
    floor = lambda b: b / (DRAM_GBPS * 1e9) * 1e6
    sections = [
        ("attention (weights + KV)", 315, floor(layer["attention weights"] + layer["attention KV reads"])),
        ("mHC (2 sub-blocks)", 250, floor(layer["mHC (fn weights x2 + streams)"])),
        ("router", 171, floor(layer["router weights"])),
        ("MoE core (routed experts)", 438, floor(layer["routed experts (busiest device)"])),
        ("shared expert", 112, floor(layer["shared expert"])),
    ]
    print("per section at batch 16 (measured in-trace us vs DRAM floor us):")
    for name, meas, fl in sections:
        print(
            f"   {name:28s} measured {meas:6.0f}  floor {fl:6.0f}  -> {meas / fl:5.1f}x the floor  (gap {meas - fl:5.0f} us per layer = {(meas - fl) * LAYERS / 1e3:5.1f} ms per token)"
        )
    print("   (glue ops + collectives ~100 us per layer have no weight traffic: pure latency)")
    print()
    print(
        "UPDATE (tests/test_decode_distinct_batch.py, 128 distinct seeded random tokens, wall ms/token steady, 40 layers): tiled -> distinct"
    )
    print(
        "   16: 65.0 -> 62.2 | 32: 73.7 -> 74.7 | 64: 95.8 -> 96.3 | 128: 116.6 -> 127.5. Only batch 128 moves materially (+9%); the tiled"
    )
    print(
        "   numbers already represent the real step time at 16-64. Measured expert activation is below the uniform-random model (real routing is skewed)."
    )


if __name__ == "__main__":
    main()
