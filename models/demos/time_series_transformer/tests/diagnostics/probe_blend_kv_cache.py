# SPDX-License-Identifier: Apache-2.0
"""
Standalone probe: verify blend-based KV-cache write can be captured in a
trace and replayed correctly across multiple steps with different values.

Fix vs. prior run: previous version's max error was 4.1 -- root cause was
that ttnn.multiply/ttnn.add are NOT in-place; each call allocated a fresh
output tensor, so k_cache was read every replay but never actually
accumulated (only the last of 24 writes ever "stuck"). Fixed by passing
output_tensor=k_cache to both ops (documented param on both ttnn.add and
ttnn.multiply: "preallocated output tensor"), so each replay writes the
blended result back into k_cache's own buffer at a fixed address -- which
is also required for a traced op to be replayable at all.

Warm-up-before-capture and rsub/plain-multiply-broadcast fixes retained
from prior runs.

Run with a hard timeout; a failure mid-capture can leave the device
needing tt-smi -r:
  tt-smi -r
  ARCH_NAME=wormhole_b0 timeout 60s python3 tests/diagnostics/probe_blend_kv_cache.py
  echo "exit code: $?"
"""
import torch

import ttnn

BS = 1
H = 2
D = 32  # padded head dim
T_max = 24


def blend_step(k_cache, pos_onehot, new_k_buf):
    keep_mask = ttnn.rsub(pos_onehot, 1.0)  # 1.0 - pos_onehot
    ttnn.multiply(k_cache, keep_mask, output_tensor=k_cache)  # zero slot t, in place
    new_k_bcast = ttnn.multiply(new_k_buf, pos_onehot)  # broadcast write value into slot t
    ttnn.add(k_cache, new_k_bcast, output_tensor=k_cache)  # accumulate, in place
    return k_cache


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=200000)
    try:
        torch.manual_seed(0)
        new_k_per_step = [torch.randn(BS, H, D, 1) for _ in range(T_max)]
        expected_cache = torch.zeros(BS, H, D, T_max)
        for t in range(T_max):
            expected_cache[:, :, :, t : t + 1] = new_k_per_step[t]

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

        print("Warm-up pass (compiling kernels, no capture)...")
        blend_step(k_cache, pos_onehot, new_k_buf)
        # reset k_cache to zero after warm-up so the real replay loop starts clean
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.zeros(BS, H, D, T_max), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            k_cache,
        )
        print("Warm-up complete, cache reset.")

        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        blend_step(k_cache, pos_onehot, new_k_buf)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

        for t in range(T_max):
            onehot_t = torch.zeros(1, 1, 1, T_max)
            onehot_t[..., t] = 1.0
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(onehot_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                pos_onehot,
            )
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(new_k_per_step[t], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                new_k_buf,
            )
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)

        ttnn.release_trace(device, trace_id)

        result_cache = ttnn.to_torch(k_cache).float()
        max_err = (result_cache - expected_cache).abs().max().item()
        print(f"[RESULT] max abs error vs ground truth: {max_err:.6f}")
        assert max_err < 1e-2, "Blend-based trace write does NOT match ground truth"
        print("[PASS] Blend-based KV write matches slice_write ground truth across all T_max steps.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
