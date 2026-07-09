# SPDX-License-Identifier: Apache-2.0
"""
Extends probe_v_cache_update_cache.py: multi-step correctness (steps 0..7,
not just step 0) + same-session timing A/B (slice_write+to_layout vs
update_cache) post-warmup. Still untraced -- trace compatibility is a
separate follow-up once this passes.
"""
import time

import torch

import ttnn


def main():
    device = ttnn.open_device(device_id=0)
    try:
        BS, H, T_max, D = 1, 2, 24, 32
        N_STEPS = 8

        # ---- Path A: existing ROW_MAJOR + slice_write + to_layout ----
        v_cache_rm = ttnn.from_torch(
            torch.zeros(BS, H, T_max, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        torch.manual_seed(0)
        steps_input = [torch.randn(BS, H, 1, D) for _ in range(N_STEPS)]

        for step, v_step in enumerate(steps_input):
            v_new = ttnn.from_torch(v_step, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            ttnn.experimental.slice_write(
                v_new,
                v_cache_rm,
                [0, 0, step, 0],
                [BS, H, step + 1, D],
                [1, 1, 1, 1],
            )
        ref = ttnn.to_torch(v_cache_rm).float()

        # ---- Path B: TILE-native + update_cache ----
        v_cache_tile = ttnn.from_torch(
            torch.zeros(BS, H, T_max, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        for step, v_step in enumerate(steps_input):
            v_new_tile = ttnn.from_torch(v_step, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            ttnn.update_cache(v_cache_tile, v_new_tile, update_idx=step, batch_offset=0)
        cand = ttnn.to_torch(v_cache_tile).float()

        diff = (ref - cand).abs().max().item()
        print(f"[MULTISTEP] max abs diff over {N_STEPS} steps: {diff}")
        print("MULTISTEP PASS" if diff < 1e-3 else "MULTISTEP FAIL")

        # ---- Timing A/B, same session, post-warmup ----
        WARMUP, TIMED = 5, 20

        def time_slice_write_path():
            v_cache = ttnn.from_torch(
                torch.zeros(BS, H, T_max, D), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            v_raw = ttnn.from_torch(
                torch.randn(BS, H, 1, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )  # mimic QKV-split output
            step = 0
            t0 = time.perf_counter()
            v_rm = ttnn.to_layout(v_raw, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.experimental.slice_write(v_rm, v_cache, [0, 0, step, 0], [BS, H, step + 1, D], [1, 1, 1, 1])
            ttnn.synchronize_device(device)
            return (time.perf_counter() - t0) * 1000

        def time_update_cache_path():
            v_cache = ttnn.from_torch(
                torch.zeros(BS, H, T_max, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            v_raw = ttnn.from_torch(
                torch.randn(BS, H, 1, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            step = 0
            t0 = time.perf_counter()
            ttnn.update_cache(v_cache, v_raw, update_idx=step, batch_offset=0)
            ttnn.synchronize_device(device)
            return (time.perf_counter() - t0) * 1000

        for _ in range(WARMUP):
            time_slice_write_path()
            time_update_cache_path()

        sw_times = [time_slice_write_path() for _ in range(TIMED)]
        uc_times = [time_update_cache_path() for _ in range(TIMED)]

        import statistics

        print(f"[TIMING] slice_write+to_layout median: {statistics.median(sw_times):.3f} ms")
        print(f"[TIMING] update_cache median:          {statistics.median(uc_times):.3f} ms")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
