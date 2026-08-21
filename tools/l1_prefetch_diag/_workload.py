#!/usr/bin/env python3
"""
Workload driver for l1_repro.py stage 2. Exercises the DRAM->prefetcher-L1 relay.

Runs on ONE device -- no mesh, no model. The point is not the arithmetic; it is to compile and
dispatch many DISTINCT programs so the prefetcher's kernel-binary cache fills and wraps,
walking the whole ring buffer. Every fill is a CQ_PREFETCH_CMD_PAGED_TO_RINGBUFFER read from
the DRAM kernels_buffer into the prefetch core's L1 -- the leg where the corruption enters.

Exits when done so the audit can read L1 without sharing the device. L1 is SRAM: its contents
survive process exit, so the cached binaries are still there to inspect.
"""
import argparse, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=40)
    a = ap.parse_args()

    import ttnn
    import torch

    dev = ttnn.open_device(device_id=a.device)
    try:
        n = 0
        # Vary op AND shape: each distinct (op, shape, dtype, layout) compiles its own kernels
        # and therefore claims its own slot in the prefetcher cache.
        shapes = [
            (1, 1, 32, 32),
            (1, 1, 64, 64),
            (1, 1, 32, 128),
            (1, 1, 128, 32),
            (1, 1, 96, 64),
            (1, 1, 160, 32),
            (1, 1, 32, 224),
            (1, 1, 192, 64),
        ]
        ops = ["add", "mul", "sub", "gelu", "relu", "exp", "sqrt", "sigmoid"]
        for it in range(a.iters):
            shape = shapes[it % len(shapes)]
            op = ops[it % len(ops)]
            x = torch.rand(shape, dtype=torch.bfloat16).abs() + 0.5
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            try:
                if op in ("add", "mul", "sub"):
                    ty = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                    r = getattr(ttnn, op)(tx, ty)
                    ttnn.deallocate(ty)
                else:
                    r = getattr(ttnn, op)(tx)
                ttnn.to_torch(r)
                ttnn.deallocate(r)
                n += 1
            except Exception as e:
                print(f"  (skipped {op}{shape}: {str(e)[:60]})", file=sys.stderr)
            ttnn.deallocate(tx)
        print(f"dispatched {n} programs on device {a.device}")
    finally:
        ttnn.close_device(dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
