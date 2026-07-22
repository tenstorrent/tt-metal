# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Micro-benchmark ONLY: does transpose+concat+one-write beat two separate
writes for K/V? Not a correctness test -- pure timing, on real shapes,
real ops, real hardware. No production code modified.

Arm A (current): slice_write(K) + update_cache(V) -- 2 dispatches, matches
                 _extract_and_write_kv exactly for BS==1.
Arm B (candidate): transpose(V) -> concat(K, V_transposed) -> slice_write
                 once into a merged cache -- 3 ops, 1 of them a write.

Decision rule: if Arm B's median time < Arm A's, the K+V fusion idea is
worth pursuing further. If not, it's dead -- don't build it into
production regardless of the dispatch-count argument, since wall-clock
time is what actually matters for the 50ms target, not call count itself.
"""
import statistics
import sys
import time
from pathlib import Path

import torch

import ttnn

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from models.demos.time_series_transformer.tt.tst_attention import (  # noqa: E402
    HEAD_DIM_PADDED,
    NUM_HEADS,
    allocate_kv_cache,
)

BS = 1
T_MAX = 24  # real context length, not the probe's T_MAX=8 -- matters for write cost at realistic size
WARMUP = 5
REPLAYS = 30


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24_576)
    try:
        key_rm = ttnn.from_torch(
            torch.randn(BS, NUM_HEADS, HEAD_DIM_PADDED, 1, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        value_tile = ttnn.from_torch(
            torch.randn(BS, NUM_HEADS, 1, HEAD_DIM_PADDED, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        k_cache, v_cache = allocate_kv_cache(device, BS, T_max=T_MAX)
        merged_cache = ttnn.from_torch(
            torch.zeros(BS, NUM_HEADS, HEAD_DIM_PADDED, 2 * T_MAX, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        def arm_a(step):
            ttnn.experimental.slice_write(
                key_rm, k_cache, [0, 0, 0, step], [BS, NUM_HEADS, HEAD_DIM_PADDED, step + 1], [1, 1, 1, 1]
            )
            ttnn.update_cache(v_cache, value_tile, update_idx=step, batch_offset=0)

        def arm_b(step):
            v_t = ttnn.transpose(value_tile, -2, -1)  # [BS,H,1,D] -> [BS,H,D,1], matches key_rm
            v_rm = ttnn.to_layout(v_t, ttnn.ROW_MAJOR_LAYOUT)
            merged = ttnn.concat([key_rm, v_rm], dim=-1)  # [BS,H,D,2]
            ttnn.experimental.slice_write(
                merged, merged_cache, [0, 0, 0, 2 * step], [BS, NUM_HEADS, HEAD_DIM_PADDED, 2 * step + 2], [1, 1, 1, 1]
            )

        # Warm up EVERY distinct step value for BOTH arms -- slice_start/end
        # are baked into the compiled program per compute_program_hash, so each
        # unique step is its own compile-cache entry. Must warm all T_MAX of them.
        for step in range(T_MAX):
            arm_a(step)
            arm_b(step)
        ttnn.synchronize_device(device)

        times_a, times_b = [], []
        for step in range(REPLAYS):
            t0 = time.perf_counter()
            arm_a(step % T_MAX)
            ttnn.synchronize_device(device)
            times_a.append((time.perf_counter() - t0) * 1000.0)

        for step in range(REPLAYS):
            t0 = time.perf_counter()
            arm_b(step % T_MAX)
            ttnn.synchronize_device(device)
            times_b.append((time.perf_counter() - t0) * 1000.0)

        med_a, med_b = statistics.median(times_a), statistics.median(times_b)
        print(f"[RESULT] Arm A (2 separate writes):        median={med_a:.4f}ms")
        print(f"[RESULT] Arm B (transpose+concat+1 write): median={med_b:.4f}ms")
        print(
            f"[RESULT] {'Arm B wins' if med_b < med_a else 'Arm A wins -- fusion via concat is NOT worth it'} "
            f"(delta={med_a - med_b:+.4f}ms per layer per step)"
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
