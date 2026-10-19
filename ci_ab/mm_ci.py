#!/usr/bin/env python3
"""Scratch, not for merge. 2048x4096x4096 matmul (in0 bf16) on an 11x10 core grid (the p100a grid, so every per-core
block and subblock matches the local card): LoFi, HiFi2, HiFi4 with bfp8_b and bf16 outputs, plus bfp8_b weights at LoFi
and HiFi2. Each case warmed up twice, then signposted and run MM_N times; outputs hashed per case (bit identity)."""
import hashlib, os
import torch
import ttnn
from tracy import signpost

N = int(os.environ.get("MM_N", "12"))
dev = ttnn.open_device(device_id=0, l1_small_size=32768)
D = ttnn.DRAM_MEMORY_CONFIG
F = ttnn.MathFidelity
CG = ttnn.CoreGrid(y=10, x=11)
torch.manual_seed(0)


def ckc(fid):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True, dst_full_sync_en=False)


def tt(shape, dt):
    return ttnn.from_torch(torch.randn(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)


def bmm(name, wdt, fid, odt):
    a, w = tt((1, 1, 2048, 4096), ttnn.bfloat16), tt((1, 1, 4096, 4096), wdt)
    f = lambda: ttnn.matmul(a, w, compute_kernel_config=ckc(fid), core_grid=CG, memory_config=D, dtype=odt)
    try:
        for _ in range(2):
            o = f()
        ttnn.synchronize_device(dev)
        h = hashlib.sha1(ttnn.to_torch(o).float().numpy().tobytes()).hexdigest()[:12]
        signpost(name)
        for _ in range(N):
            f()
        ttnn.synchronize_device(dev)
        print("case done", name, h, flush=True)
    except Exception as e:
        print("case FAIL", name, str(e).splitlines()[0][:200] if str(e) else type(e).__name__, flush=True)


for fn, fid in (("lofi", F.LoFi), ("hifi2", F.HiFi2), ("hifi4", F.HiFi4)):
    for on, odt in (("outbfp8", ttnn.bfloat8_b), ("outbf16", ttnn.bfloat16)):
        bmm(f"n_{fn}_{on}", ttnn.bfloat16, fid, odt)
for fn, fid in (("lofi", F.LoFi), ("hifi2", F.HiFi2)):
    bmm(f"w8_{fn}_outbfp8", ttnn.bfloat8_b, fid, ttnn.bfloat8_b)
ttnn.close_device(dev)
print("ALL DONE", flush=True)
