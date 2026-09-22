"""For every (SP,TP) split of 32 chips, how long does ONE chip take for its share of one
layer's MLP?  All chips run concurrently, so this is the per-layer MLP wall-clock -- a hard
LOWER BOUND on the layer (it ignores GDN, attention and every collective)."""
import time

import torch

import ttnn

dim, inter, T = 2048, 6144, 4096
dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=100000000)
try:
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    print(f"  {'SP':>3}x{'TP':>2} {'M':>5} {'N':>5} {'MLP/layer':>10} {'TFLOPS':>7} {'x24 layers':>11}")
    for sp in (4, 8, 16, 32):
        tp = 32 // sp
        M, N = T // sp, inter // tp
        try:
            x = ttnn.from_torch(
                torch.randn(1, 1, M, dim, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            mk = lambda a, b, dt: ttnn.from_torch(
                torch.randn(a, b, dtype=torch.bfloat16),
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            wg, wu, wd = mk(dim, N, ttnn.bfloat4_b), mk(dim, N, ttnn.bfloat4_b), mk(N, dim, ttnn.bfloat8_b)

            def once():
                g = ttnn.linear(
                    x, wg, activation="silu", compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                u = ttnn.linear(x, wu, compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                h = ttnn.mul(g, u, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                o = ttnn.linear(h, wd, compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for t_ in (g, u, h):
                    ttnn.deallocate(t_)
                return o

            ttnn.deallocate(once())
            ttnn.synchronize_device(dev)
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            o = once()
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ttnn.synchronize_device(dev)
            best = 1e9
            for _ in range(4):
                t0 = time.time()
                for _ in range(50):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(dev)
                best = min(best, (time.time() - t0) / 50 * 1e6)
            gf = 3 * 2 * M * dim * N / 1e9
            print(f"  {sp:>3}x{tp:>2} {M:>5} {N:>5} {best:9.1f}us {gf/best*1e6/1e3:7.1f} {best*24/1000:10.2f}ms")
            ttnn.release_trace(dev, tid)
            for t_ in (x, wg, wu, wd, o):
                ttnn.deallocate(t_)
        except Exception as e:
            print(f"  {sp:>3}x{tp:>2} {M:>5} {N:>5}  FAIL {str(e).splitlines()[0][:55]}")
finally:
    ttnn.close_device(dev)
