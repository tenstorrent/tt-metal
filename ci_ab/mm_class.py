#!/usr/bin/env python3
"""Scratch, not for merge. ttnn matmuls of the class bmm metal2, HiFi2, bf16 operands, auto or explicit program configs,
sizes of the model and unit-test shapes (BERT, ViT, attention, FFN, decode, batched). Each case warmed up twice, signposted,
run MM_N times; every case is timed whether or not its ELF changes."""
import os
import torch
import ttnn
from tracy import signpost

N = int(os.environ.get("MM_N", "10"))
dev = ttnn.open_device(device_id=0, l1_small_size=32768)
D = ttnn.DRAM_MEMORY_CONFIG
F = ttnn.MathFidelity
torch.manual_seed(0)


def ckc(fid, fp32=False, l1=False):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=l1, dst_full_sync_en=False)


def tt(shape, dt=ttnn.bfloat16):
    return ttnn.from_torch(torch.randn(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)


def case(name, f):
    try:
        for _ in range(2):
            o = f()
        ttnn.synchronize_device(dev)
        signpost(name)
        for _ in range(N):
            f()
        ttnn.synchronize_device(dev)
        print("case done", name, flush=True)
    except Exception as e:
        print("case FAIL", name, str(e).splitlines()[0][:200] if str(e) else type(e).__name__, flush=True)


SH = [(1024, 1024, 1024), (4096, 4096, 4096), (512, 2048, 2048), (2048, 2048, 8192), (8192, 1024, 1024), (128, 4096, 4096),
      (32, 4096, 4096), (2048, 1024, 4096), (1024, 4096, 1024), (384, 1024, 1024), (2048, 8192, 2048), (256, 1024, 4096),
      (3072, 1024, 1024), (1024, 1280, 5120), (4608, 1536, 1536), (2048, 4096, 4096)]
for m, k, n in SH:
    for l1 in (False, True):
        a, b = tt((1, 1, m, k)), tt((1, 1, k, n))
        case(f"a_{m}x{k}x{n}_l1{int(l1)}", lambda: ttnn.matmul(a, b, compute_kernel_config=ckc(F.HiFi2, l1=l1), memory_config=D))
# batched attention style
for bsz, m, k, n in ((16, 384, 64, 384), (16, 384, 384, 64), (12, 1024, 64, 1024), (24, 512, 64, 512)):
    a, b = tt((1, bsz, m, k)), tt((1, bsz, k, n))
    case(f"att_{bsz}x{m}x{k}x{n}", lambda: ttnn.matmul(a, b, compute_kernel_config=ckc(F.HiFi2), memory_config=D))
# explicit 2D mcast with one-tile sub blocks, packer l1 acc on and off, fp32 dest on
grid = ttnn.CoreGrid(y=8, x=8)
for pm, pn, ibw in ((4, 4, 2), (8, 4, 4), (4, 8, 4), (2, 8, 2), (8, 8, 4)):
    M, K, Nn = pm * 32 * 8, 1024, pn * 32 * 8
    a, b = tt((1, 1, M, K)), tt((1, 1, K, Nn))
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(8, 8), in0_block_w=ibw, out_subblock_h=1, out_subblock_w=1,
        per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None)
    for fp32 in (False, True):
        case(f"p2d_{pm}x{pn}_ibw{ibw}_fp32{int(fp32)}", lambda: ttnn.matmul(a, b, program_config=pc, compute_kernel_config=ckc(F.HiFi2, fp32=fp32, l1=True), memory_config=D))
ttnn.close_device(dev)
print("ALL DONE", flush=True)
