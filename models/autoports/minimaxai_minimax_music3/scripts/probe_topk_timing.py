#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Timing probe for the on-device CFG + top-k sampling ops (stage 07): eager and traced ``topk(k=64)`` + ``gather``
over the 16416-wide logits window and the 8192-wide padded depth-head logits, against the read-back they replace.

    with_hw_lock timeout 240 $MM3_PY -u $MM3_MODEL_DIR/scripts/probe_topk_timing.py
"""

import time

import torch

import ttnn

TILE = 32


def log(*a):
    print(*a, flush=True)


def main():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=20_000_000)
    dev.enable_program_cache()
    try:
        for width in (16416, 8192):
            x = ttnn.from_torch(
                (torch.randn(1, 1, TILE, width) * 4).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
            )
            log("start width", width)
            for it in range(6):
                if it == 2:
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                vals, idx = ttnn.topk(x, 64, dim=-1, largest=True, sorted=True)
            ttnn.synchronize_device(dev)
            log(f"topk width {width} k 64: {(time.perf_counter() - t0) / 4 * 1e3:.2f} ms/op eager")
            for it in range(6):
                if it == 2:
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                g = ttnn.gather(x, -1, idx)
            ttnn.synchronize_device(dev)
            log(f"gather width {width}: {(time.perf_counter() - t0) / 4 * 1e3:.2f} ms/op eager")
            for it in range(6):
                if it == 2:
                    t0 = time.perf_counter()
                ttnn.to_torch(x)
            log(f"readback [32,{width}] bf16 tile: {(time.perf_counter() - t0) / 4 * 1e3:.2f} ms")
            u = ttnn.untilize(x, use_multicore=True)
            for it in range(6):
                if it == 2:
                    t0 = time.perf_counter()
                ttnn.to_torch(u)
            log(f"readback [32,{width}] bf16 row-major: {(time.perf_counter() - t0) / 4 * 1e3:.2f} ms")
            for it in range(6):
                if it == 2:
                    t0 = time.perf_counter()
                ttnn.to_torch(vals)
                ttnn.to_torch(g)
            log(f"readback topk vals+gathered [32,64]: {(time.perf_counter() - t0) / 4 * 1e3:.2f} ms")
            log("capturing trace")
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            vals, idx = ttnn.topk(x, 64, dim=-1, largest=True, sorted=True)
            g = ttnn.gather(x, -1, idx)
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(10):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            log(f"traced topk64+gather width {width}: {(time.perf_counter() - t0) / 10 * 1e3:.2f} ms")
            ttnn.release_trace(dev, tid)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
