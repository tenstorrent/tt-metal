# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Isolates one question: does a trace-captured ttnn.add(ttnn.to_layout(x, TILE), y)
re-read x's LIVE device content on each execute_trace, or does it bake in
x's content as it was during capture?

Never reimplements production code -- pure primitive-level check.
"""
import torch

import ttnn


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24_576)
    try:
        shape = (1, 2, 32, 8)
        cache = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        zeros_tile = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        # warmup (compile kernels, program cache)
        _ = ttnn.add(ttnn.to_layout(cache, ttnn.TILE_LAYOUT), zeros_tile)
        ttnn.synchronize_device(device)

        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        cache_tile = ttnn.to_layout(cache, ttnn.TILE_LAYOUT)
        out = ttnn.add(cache_tile, zeros_tile)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

        for step, fill_val in enumerate([1.0, 5.0, 9.0]):
            host_new = ttnn.from_torch(
                torch.full(shape, fill_val, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
            )
            ttnn.copy_host_to_device_tensor(host_new, cache)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            result = ttnn.to_torch(out).float()
            print(
                f"[step={step}] wrote {fill_val} into cache -> trace output mean={result.mean().item():.4f} "
                f"(expected {fill_val:.4f})"
            )

        ttnn.release_trace(device, trace_id)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
