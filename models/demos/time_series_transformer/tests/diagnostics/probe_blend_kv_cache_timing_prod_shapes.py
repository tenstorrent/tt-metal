# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
"""
Extends probe_blend_kv_cache_timing.py: same blend-write mechanism, but at
the model's REAL shapes (NUM_HEADS=2, HEAD_DIM_PADDED=32, T_max=24) and
swept across BS in {1,4,16,32} instead of only BS=1. This directly answers
the open question from council review: does the O(T_max) full-tensor
multiply/add cost of the blend technique stay favorable vs slice_write_kv
once BS grows into the range the throughput target (>=100 seq/s) needs?

Not committed to the PR. Diagnostic only.

Run:
  tt-smi -r
  ARCH_NAME=wormhole_b0 timeout 180s python3 tests/diagnostics/probe_blend_kv_cache_timing_prod_shapes.py
"""
import statistics
import time

import torch

import ttnn

NUM_HEADS = 2
HEAD_DIM_PADDED = 32
T_MAX = 24
N_WARMUP = 5
N_TIMED = 30
BS_SWEEP = [1, 4, 16, 32]

# Reference cost from test_tst_perf.py's own profiling (README "Current
# Measured Baseline"): slice_write_kv ~20-22ms/step at BS=1 on this hardware.
SLICE_WRITE_KV_BASELINE_MS_BS1 = 21.0


def blend_step(k_cache, pos_onehot, new_k_buf):
    keep_mask = ttnn.rsub(pos_onehot, 1.0)
    ttnn.multiply(k_cache, keep_mask, output_tensor=k_cache)
    new_k_bcast = ttnn.multiply(new_k_buf, pos_onehot)
    ttnn.add(k_cache, new_k_bcast, output_tensor=k_cache)
    return k_cache


def run_for_bs(device, BS):
    k_cache = ttnn.from_torch(
        torch.zeros(BS, NUM_HEADS, HEAD_DIM_PADDED, T_MAX),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    pos_onehot = ttnn.from_torch(
        torch.zeros(1, 1, 1, T_MAX),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    new_k_buf = ttnn.from_torch(
        torch.zeros(BS, NUM_HEADS, HEAD_DIM_PADDED, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # warm-up (untraced), then reset cache, then capture
    blend_step(k_cache, pos_onehot, new_k_buf)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            torch.zeros(BS, NUM_HEADS, HEAD_DIM_PADDED, T_MAX), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        ),
        k_cache,
    )
    ttnn.synchronize_device(device)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    blend_step(k_cache, pos_onehot, new_k_buf)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    onehot_host = []
    newk_host = []
    for t in range(T_MAX):
        oh = torch.zeros(1, 1, 1, T_MAX)
        oh[..., t] = 1.0
        onehot_host.append(ttnn.from_torch(oh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
        newk_host.append(
            ttnn.from_torch(
                torch.randn(BS, NUM_HEADS, HEAD_DIM_PADDED, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
        )

    def one_step(idx, write_times):
        t_idx = idx % T_MAX
        t0 = time.perf_counter()
        ttnn.copy_host_to_device_tensor(onehot_host[t_idx], pos_onehot)
        ttnn.copy_host_to_device_tensor(newk_host[t_idx], new_k_buf)
        t1 = time.perf_counter()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        t2 = time.perf_counter()
        if write_times is not None:
            write_times.append((t1 - t0) * 1000.0)
        return (t2 - t1) * 1000.0, (t2 - t0) * 1000.0

    for i in range(N_WARMUP):
        one_step(i, None)

    exec_ms, write_ms, full_ms = [], [], []
    for i in range(N_TIMED):
        e, f = one_step(i, write_ms)
        exec_ms.append(e)
        full_ms.append(f)

    ttnn.release_trace(device, trace_id)
    return exec_ms, write_ms, full_ms


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=2_000_000)
    try:
        print(
            f"{'BS':>4} {'exec_median_ms':>16} {'write_median_ms':>17} "
            f"{'full_median_ms':>16} {'proj_24step_ms':>16} {'vs_slice_write_BS1':>20}"
        )
        for BS in BS_SWEEP:
            exec_ms, write_ms, full_ms = run_for_bs(device, BS)
            full_med = statistics.median(full_ms)
            proj = full_med * T_MAX
            print(
                f"{BS:>4} {statistics.median(exec_ms):>16.3f} {statistics.median(write_ms):>17.3f} "
                f"{full_med:>16.3f} {proj:>16.2f} "
                f"{proj / (SLICE_WRITE_KV_BASELINE_MS_BS1 * T_MAX):>19.2f}x"
            )
        print()
        print("Interpretation: 'vs_slice_write_BS1' is projected-24-step-blend-cost")
        print("divided by the CURRENT slice_write_kv-only cost at BS=1 (~21ms/step x24")
        print("from test_tst_perf.py's own profiling). <1.0 means the blend write is")
        print("already cheaper than JUST the current slice_write_kv sub-cost -- before")
        print("even counting the qkv_linear/qkv_split/to_layout_kv overhead the blend")
        print("approach also removes by not needing the untraced Phase-1 split at all.")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
