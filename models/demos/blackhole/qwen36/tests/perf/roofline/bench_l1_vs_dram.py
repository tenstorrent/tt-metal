"""Direct A/B at the winning shape (SP=8 x TP=4): do L1-resident weights actually beat DRAM?
This is the core premise of the 'keep the whole 2B in SRAM' plan, so test it head-on."""
import time

import torch

import ttnn

dim, N, M = 2048, 1536, 512
dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=100000000)
try:
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    for tag, wmc in (("weights DRAM", ttnn.DRAM_MEMORY_CONFIG), ("weights L1", ttnn.L1_MEMORY_CONFIG)):
        mk = lambda a, b, dt: ttnn.from_torch(
            torch.randn(a, b, dtype=torch.bfloat16), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=wmc
        )
        wg, wu, wd = mk(dim, N, ttnn.bfloat4_b), mk(dim, N, ttnn.bfloat4_b), mk(N, dim, ttnn.bfloat8_b)
        x = ttnn.from_torch(
            torch.randn(1, 1, M, dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        amc = ttnn.L1_MEMORY_CONFIG

        def once():
            g = ttnn.linear(x, wg, activation="silu", compute_kernel_config=ck, memory_config=amc)
            u = ttnn.linear(x, wu, compute_kernel_config=ck, memory_config=amc)
            h = ttnn.mul(g, u, memory_config=amc)
            o = ttnn.linear(h, wd, compute_kernel_config=ck, memory_config=amc)
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
        for _ in range(5):
            t0 = time.time()
            for _ in range(100):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            best = min(best, (time.time() - t0) / 100 * 1e6)
        gf = 3 * 2 * M * dim * N / 1e9
        print(f"  {tag:>13}: {best:7.1f}us  {gf/best*1e6/1e3:6.1f} TFLOPS   -> 24 layers {best*24/1000:.2f}ms")
        ttnn.release_trace(dev, tid)
        for t_ in (x, wg, wu, wd, o):
            ttnn.deallocate(t_)
finally:
    ttnn.close_device(dev)
