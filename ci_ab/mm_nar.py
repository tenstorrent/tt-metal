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
def ckc2(fid, fp32=False, l1=False):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=l1, dst_full_sync_en=False)


def tt2(shape, dt=ttnn.bfloat16):
    return ttnn.from_torch(torch.randn(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)


def case(name, f):
    try:
        for _ in range(2):
            f()
        ttnn.synchronize_device(dev)
        signpost(name)
        for _ in range(N):
            f()
        ttnn.synchronize_device(dev)
        print("case done", name, flush=True)
    except Exception as e:
        print("case FAIL", name, str(e).splitlines()[0][:200] if str(e) else type(e).__name__, flush=True)


for m, k, n in ((2048, 4096, 4096), (4096, 4096, 4096), (2048, 2048, 8192), (1024, 1024, 1024), (256, 1024, 4096), (384, 1024, 1024)):
    a, b = tt2((1, 1, m, k)), tt2((1, 1, k, n))
    case(f"a_{m}x{k}x{n}_l11", lambda: ttnn.matmul(a, b, compute_kernel_config=ckc2(F.HiFi2, l1=True), memory_config=D))
# explicit 2D mcast with one-tile sub blocks, packer l1 acc on and off, fp32 dest on
grid = ttnn.CoreGrid(y=8, x=8)
for pm, pn, ibw in ((4, 4, 2), (8, 4, 4), (4, 8, 4), (2, 8, 2), (8, 8, 4)):
    M, K, Nn = pm * 32 * 8, 1024, pn * 32 * 8
    a, b = tt2((1, 1, M, K)), tt2((1, 1, K, Nn))
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(8, 8), in0_block_w=ibw, out_subblock_h=1, out_subblock_w=1,
        per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None)
    for fp32 in (False, True):
        case(f"p2d_{pm}x{pn}_ibw{ibw}_fp32{int(fp32)}", lambda: ttnn.matmul(a, b, program_config=pc, compute_kernel_config=ckc2(F.HiFi2, fp32=fp32, l1=True), memory_config=D))
ttnn.close_device(dev)
print("ALL DONE", flush=True)
