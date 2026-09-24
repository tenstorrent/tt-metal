# add + rms_norm at batched shapes (DRAM bfp8): stock two ops vs fused add+RMSNorm (row-granular) vs row-split R=2/4; PCC + traced timing
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common_traced import make_traced

from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_add_rmsnorm import (
    fused_add_rmsnorm,
    fused_add_rmsnorm_split,
    make_add_norm_constants,
)

W, EPS = 2560, 1e-6
B8 = ttnn.bfloat8_b


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
traced = make_traced(D)


def traced_multi(fn, n=3):
    for _ in range(2):
        [ttnn.deallocate(t) for t in fn()]
    ttnn.synchronize_device(D)
    tid = ttnn.begin_trace_capture(D, cq_id=0)
    outs = [fn() for _ in range(n)]
    ttnn.end_trace_capture(D, tid, cq_id=0)
    ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(6):
        ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
    us = (time.perf_counter() - t0) / 6 / n * 1e6
    ttnn.release_trace(D, tid)
    [ttnn.deallocate(t) for o in outs for t in o]
    return us


ckc = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
try:
    torch.manual_seed(0)
    gamma = torch.rand(W) * 0.5 + 0.75
    G, SC, EP = make_add_norm_constants(gamma, EPS, D)
    g_rm = ttnn.from_torch(
        gamma.reshape(1, 1, W // 32, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=D
    )
    mc = ttnn.DRAM_MEMORY_CONFIG
    for M in (4096, 8192, 16384):
        a = ttnn.from_torch(torch.randn(1, 1, M, W), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=mc)
        b = ttnn.from_torch(
            torch.randn(1, 1, M, W) * 0.3, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=mc
        )

        def ref():
            s = ttnn.add(a, b, memory_config=mc, dtype=B8)
            n = ttnn.rms_norm(s, epsilon=EPS, weight=g_rm, compute_kernel_config=ckc, memory_config=mc)
            return (s, n)

        rs, rn = ref()
        fs, fn_ = fused_add_rmsnorm(a, b, G, SC, EP, memory_config=mc)
        p_fused = pcc(ttnn.to_torch(fn_), ttnn.to_torch(rn))
        [ttnn.deallocate(z) for z in (rs, rn, fs, fn_)]
        ur = traced_multi(ref)
        ug = traced_multi(lambda: fused_add_rmsnorm(a, b, G, SC, EP, memory_config=mc))
        line = f"[M={M:5d}] stock add+rms_norm {ur:7.1f} us | fused R=1 {ug:7.1f} ({100*(ug/ur-1):+.1f}%, pcc {p_fused:.5f})"
        for R in (2, 4, 5, 8):
            try:
                fs, fn_ = fused_add_rmsnorm_split(a, b, G, SC, EP, R=R, memory_config=mc)
                rn2 = ref()[1]
                p = pcc(ttnn.to_torch(fn_), ttnn.to_torch(rn2))
                [ttnn.deallocate(z) for z in (fs, fn_, rn2)]
                us = traced_multi(lambda: fused_add_rmsnorm_split(a, b, G, SC, EP, R=R, memory_config=mc))
                line += f" | split R={R} {us:7.1f} ({100*(us/ur-1):+.1f}%, pcc {p:.5f})"
            except Exception as e:
                line += f" | split R={R} FAIL {str(e).splitlines()[0][:60]}"
        print(line, flush=True)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    print("[done]")
finally:
    ttnn.close_device(D)
