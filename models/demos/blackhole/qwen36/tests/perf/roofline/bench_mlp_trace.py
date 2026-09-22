"""Traced: how fast can ONE chip do one layer's MLP for a token shard, weights in L1?

Sweeps the shard size, because the 32-way token split (128 tok/chip) may be too small to fill
the core grid.  Budget: a layer is 83 us for 2 ms; MLP is ~2/3 of a layer's FLOPs -> ~55 us.
"""
import time

import torch

import ttnn

dim, inter = 2048, 6144
dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=90000000)
try:

    def w(a, b, dt):
        return ttnn.from_torch(
            torch.randn(a, b, dtype=torch.bfloat16),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    wg, wu = w(dim, inter, ttnn.bfloat4_b), w(dim, inter, ttnn.bfloat4_b)
    wd = w(inter, dim, ttnn.bfloat8_b)
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    print(f"  MLP weights in L1: {(2*dim*inter*0.5625 + inter*dim*1.0625)/1e6:.1f} MB of 189 MB\n")
    print(f"  {'shard':>6} {'traced':>9} {'TFLOPS':>8} {'us/4096tok':>11}  (32 chips x shard = 4096 needs shard=128)")
    for TOK in (128, 256, 512, 1024, 2048):
        x = ttnn.from_torch(
            torch.randn(1, 1, TOK, dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        def once():
            g = ttnn.linear(x, wg, activation="silu", compute_kernel_config=ck, memory_config=ttnn.L1_MEMORY_CONFIG)
            u = ttnn.linear(x, wu, compute_kernel_config=ck, memory_config=ttnn.L1_MEMORY_CONFIG)
            h = ttnn.mul(g, u, memory_config=ttnn.L1_MEMORY_CONFIG)
            o = ttnn.linear(h, wd, compute_kernel_config=ck, memory_config=ttnn.L1_MEMORY_CONFIG)
            for t in (g, u, h):
                ttnn.deallocate(t)
            return o

        try:
            ttnn.deallocate(once())
            ttnn.synchronize_device(dev)
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            out = once()
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ttnn.synchronize_device(dev)
            N = 100
            best = 1e9
            for _ in range(5):
                t0 = time.time()
                for _ in range(N):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(dev)
                best = min(best, (time.time() - t0) / N * 1e6)
            gflop = 3 * 2 * TOK * dim * inter / 1e9
            print(f"  {TOK:>6} {best:8.1f}us {gflop/best*1e6/1e3:8.1f} {best*4096/TOK:10.1f}us")
            ttnn.release_trace(dev, tid)
            ttnn.deallocate(out)
        except Exception as e:
            print(f"  {TOK:>6}  FAILED: {str(e).splitlines()[0][:70]}")
        ttnn.deallocate(x)
finally:
    ttnn.close_device(dev)
