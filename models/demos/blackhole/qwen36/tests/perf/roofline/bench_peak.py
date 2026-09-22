"""What is ONE Blackhole chip's actual achievable matmul TFLOPS? Everything else is judged
against this, so measure it rather than trusting a spec number."""
import time

import torch

import ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=90000000)
try:
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    print(f"  {'M':>5} {'K':>5} {'N':>5} {'wdt':>4} {'mem':>4} {'traced':>9} {'TFLOPS':>8}")
    cases = [
        (m, k, n, wdt, mc)
        for (m, k, n) in [
            (512, 2048, 6144),
            (1024, 2048, 6144),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (2048, 4096, 4096),
        ]
        for wdt in (ttnn.bfloat8_b, ttnn.bfloat4_b)
        for mc in ("DRAM",)
    ]
    for M, K, N, wdt, mcn in cases:
        mc = ttnn.DRAM_MEMORY_CONFIG
        try:
            a = ttnn.from_torch(
                torch.randn(1, 1, M, K, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=mc,
            )
            b = ttnn.from_torch(
                torch.randn(K, N, dtype=torch.bfloat16),
                dtype=wdt,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=mc,
            )
            ttnn.deallocate(ttnn.linear(a, b, compute_kernel_config=ck, memory_config=mc))
            ttnn.synchronize_device(dev)
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            o = ttnn.linear(a, b, compute_kernel_config=ck, memory_config=mc)
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            ttnn.synchronize_device(dev)
            best = 1e9
            for _ in range(4):
                t0 = time.time()
                for _ in range(50):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(dev)
                best = min(best, (time.time() - t0) / 50 * 1e6)
            print(
                f"  {M:>5} {K:>5} {N:>5} {str(wdt).split('.')[-1][:4]:>4} {mcn:>4} {best:8.1f}us "
                f"{2*M*K*N/1e9/best*1e6/1e3:8.1f}"
            )
            ttnn.release_trace(dev, tid)
            for t in (a, b, o):
                ttnn.deallocate(t)
        except Exception as e:
            print(
                f"  {M:>5} {K:>5} {N:>5} {str(wdt).split('.')[-1][:4]:>4} {mcn:>4}  FAIL {str(e).splitlines()[0][:50]}"
            )
finally:
    ttnn.close_device(dev)
