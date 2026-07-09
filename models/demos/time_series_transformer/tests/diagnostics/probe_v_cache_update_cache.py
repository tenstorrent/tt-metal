# SPDX-License-Identifier: Apache-2.0
"""
Isolated probe: can update_cache/fill_cache replace slice_write for V only,
given V's cache shape [BS, H, T_max, D] already matches update_cache's
expected [batch, H, seq, D] contract (unlike K's transposed [BS,H,D,T_max])?

Not committed to the PR. Correctness-only check, single decode step,
compares against existing slice_write_kv path as ground truth.
"""
import torch

import ttnn


def main():
    device = ttnn.open_device(device_id=0)
    try:
        BS, H, T_max, D = 1, 2, 24, 32

        # Existing path: ROW_MAJOR cache + slice_write
        v_cache_rm = ttnn.from_torch(
            torch.zeros(BS, H, T_max, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        v_new = ttnn.from_torch(
            torch.randn(BS, H, 1, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        step = 0
        ttnn.experimental.slice_write(
            v_new,
            v_cache_rm,
            [0, 0, step, 0],
            [BS, H, step + 1, D],
            [1, 1, 1, 1],
        )
        ref = ttnn.to_torch(v_cache_rm).float()

        # Candidate path: TILE cache + update_cache
        v_cache_tile = ttnn.from_torch(
            torch.zeros(BS, H, T_max, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        v_new_tile = ttnn.from_torch(
            ttnn.to_torch(v_new).float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        try:
            ttnn.update_cache(v_cache_tile, v_new_tile, update_idx=step, batch_offset=0)
            cand = ttnn.to_torch(v_cache_tile).float()
            diff = (ref - cand).abs().max().item()
            print(f"[RESULT] max abs diff vs slice_write ground truth: {diff}")
            print("PASS" if diff < 1e-3 else "FAIL - values diverge")
        except Exception as e:
            print(f"[RESULT] update_cache call failed: {e}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
