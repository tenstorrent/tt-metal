#!/usr/bin/env python3
"""Scratch, not for merge. bmm op-level cases for the pack stack: 2048x4096x4096 bf16 on an 11x10 grid at LoFi, HiFi2,
HiFi4 with half and full DEST sync and bf16 / bfp8_b outputs, fp32 DEST, L1 accumulation off, and four smaller shapes.
Each case is warmed up twice, then signposted and run MM_N times; the output of the first run is hashed (bit identity)."""
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


def ckc(fid, full=False, fp32=False, l1=True):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=l1, dst_full_sync_en=full)


def tt(shape, dt=ttnn.bfloat16):
    return ttnn.from_torch(torch.randn(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)


def case(name, f):
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


a, w = tt((1, 1, 2048, 4096)), tt((1, 1, 4096, 4096))
for fn, fid in (("lofi", F.LoFi), ("hifi2", F.HiFi2), ("hifi4", F.HiFi4)):
    for sn, full in (("half", False), ("full", True)):
        for on, odt in (("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b)):
            case(f"big_{fn}_{sn}_{on}", lambda: ttnn.matmul(a, w, compute_kernel_config=ckc(fid, full=full), core_grid=CG, memory_config=D, dtype=odt))
for sn, full in (("half", False), ("full", True)):
    case(f"big_hifi4_{sn}_fp32", lambda: ttnn.matmul(a, w, compute_kernel_config=ckc(F.HiFi4, full=full, fp32=True), core_grid=CG, memory_config=D, dtype=ttnn.bfloat16))
case("big_hifi2_half_nol1", lambda: ttnn.matmul(a, w, compute_kernel_config=ckc(F.HiFi2, l1=False), core_grid=CG, memory_config=D, dtype=ttnn.bfloat16))
for m, k, n in ((1024, 1024, 1024), (256, 1024, 4096), (4096, 4096, 4096), (384, 1024, 1024)):
    x, y = tt((1, 1, m, k)), tt((1, 1, k, n))
    for sn, full in (("half", False), ("full", True)):
        case(f"s_{m}x{k}x{n}_hifi2_{sn}", lambda: ttnn.matmul(x, y, compute_kernel_config=ckc(F.HiFi2, full=full), memory_config=D))
        case(f"s_{m}x{k}x{n}_lofi_{sn}", lambda: ttnn.matmul(x, y, compute_kernel_config=ckc(F.LoFi, full=full), memory_config=D))
grid = ttnn.CoreGrid(y=8, x=8)
for pm, pn, ibw in ((4, 4, 2), (8, 4, 4), (4, 8, 4), (8, 8, 4)):
    M, K, Nn = pm * 32 * 8, 1024, pn * 32 * 8
    x, y = tt((1, 1, M, K)), tt((1, 1, K, Nn))
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(8, 8), in0_block_w=ibw, out_subblock_h=1, out_subblock_w=1,
        per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None)
    for sn, full in (("half", False), ("full", True)):
        case(f"p2d_{pm}x{pn}_ibw{ibw}_{sn}", lambda: ttnn.matmul(x, y, program_config=pc, compute_kernel_config=ckc(F.HiFi2, full=full), memory_config=D))
ttnn.close_device(dev)
print("ALL DONE", flush=True)
