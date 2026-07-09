# SPDX-License-Identifier: Apache-2.0
"""
Timing harness for the verified blend-based KV-cache write (see
probe_blend_kv_cache.py, which PASSED correctness: max abs error 0.007792).

Measures two numbers per decode step:
  1. execute_trace-only time -- pure device blend compute
  2. full step time -- host writes (pos_onehot, new_k_buf) + execute_trace,
     the number directly comparable to the current slice_write_kv cost
     (~30-45ms of the ~90-120ms total single-sequence latency)

ttnn.execute_trace(..., blocking=True) blocks the host until device
completion, so a host wall-clock wrap around it is a valid measurement --
no separate device profiler needed for this comparison.

Methodology: 5 untimed warm-up replays (JIT/cache settling), then 50 timed
replays. Report min/mean/median/max in ms, matching the style of
test_tst_perf.py's existing latency reporting for direct comparability.

Run with a hard timeout:
  tt-smi -r
  ARCH_NAME=wormhole_b0 timeout 60s python3 tests/diagnostics/probe_blend_kv_cache_timing.py
  echo "exit code: $?"
"""
import statistics
import time

import torch

import ttnn

BS = 1
H = 2
D = 32
T_max = 24
N_WARMUP_REPLAYS = 5
N_TIMED_REPLAYS = 50


def blend_step(k_cache, pos_onehot, new_k_buf):
    keep_mask = ttnn.rsub(pos_onehot, 1.0)
    ttnn.multiply(k_cache, keep_mask, output_tensor=k_cache)
    new_k_bcast = ttnn.multiply(new_k_buf, pos_onehot)
    ttnn.add(k_cache, new_k_bcast, output_tensor=k_cache)
    return k_cache


def summarize(label, samples_ms):
    print(
        f"{label:32s} min={min(samples_ms):7.3f}ms  "
        f"mean={statistics.mean(samples_ms):7.3f}ms  "
        f"median={statistics.median(samples_ms):7.3f}ms  "
        f"max={max(samples_ms):7.3f}ms"
    )


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=200000)
    try:
        k_cache = ttnn.from_torch(
            torch.zeros(BS, H, D, T_max),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        pos_onehot = ttnn.from_torch(
            torch.zeros(1, 1, 1, T_max),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        new_k_buf = ttnn.from_torch(
            torch.zeros(BS, H, D, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        print("Compiling kernels...")
        blend_step(k_cache, pos_onehot, new_k_buf)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.zeros(BS, H, D, T_max), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            k_cache,
        )
        print("Compile complete, cache reset.")

        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        blend_step(k_cache, pos_onehot, new_k_buf)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

        # Pre-build all host tensors up front so the timed loop only measures
        # the write + replay, not torch/ttnn.from_torch construction overhead.
        onehot_host = []
        newk_host = []
        for t in range(T_max):
            oh = torch.zeros(1, 1, 1, T_max)
            oh[..., t] = 1.0
            onehot_host.append(ttnn.from_torch(oh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
            newk_host.append(ttnn.from_torch(torch.randn(BS, H, D, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))

        def one_step(idx, timed_writes):
            t_idx = idx % T_max
            t0 = time.perf_counter()
            ttnn.copy_host_to_device_tensor(onehot_host[t_idx], pos_onehot)
            ttnn.copy_host_to_device_tensor(newk_host[t_idx], new_k_buf)
            t1 = time.perf_counter()
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            t2 = time.perf_counter()
            if timed_writes is not None:
                timed_writes.append((t1 - t0) * 1000.0)
            return (t2 - t1) * 1000.0, (t2 - t0) * 1000.0

        print(f"Warm-up replays ({N_WARMUP_REPLAYS}, untimed)...")
        for i in range(N_WARMUP_REPLAYS):
            one_step(i, None)

        print(f"Timed replays ({N_TIMED_REPLAYS})...")
        exec_only_ms = []
        write_ms = []
        full_step_ms = []
        for i in range(N_TIMED_REPLAYS):
            exec_ms, full_ms = one_step(i, write_ms)
            exec_only_ms.append(exec_ms)
            full_step_ms.append(full_ms)

        ttnn.release_trace(device, trace_id)

        print()
        summarize("execute_trace only", exec_only_ms)
        summarize("host writes only", write_ms)
        summarize("full step (writes+exec)", full_step_ms)
        print()
        print(f"Projected 24-step decode cost (full step * 24): " f"{statistics.mean(full_step_ms) * 24:.2f}ms")
        print("Compare against current slice_write_kv cost: ~30-45ms of ~90-120ms total.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
