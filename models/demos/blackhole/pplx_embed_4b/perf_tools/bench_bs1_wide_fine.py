# 12x8 legacy 2D-mcast at M=512 (sharded bfp4 weights, fixed factory): in0_block_w x subblock fine sweep per projection
import math
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common_traced import make_traced

B4 = ttnn.bfloat4_b
B8 = ttnn.bfloat8_b
T = 32
M = 512
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
traced = make_traced(D)
ckc = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
)
dram_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])


def sharded_w(wt, K, N):
    pad = math.ceil(N / (T * 8)) * (T * 8)
    spec = ttnn.ShardSpec(dram_grid, (K, pad // 8), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.from_torch(
        wt,
        dtype=B4,
        layout=ttnn.TILE_LAYOUT,
        device=D,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec),
    )


def pc(grid, bw, sh, sw, pm, pn):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=bw,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


GX = int(os.getenv("GX", "12"))
try:
    for name, K, N, cur in (
        ("QKV", 2560, 6144, (10, 1, 4)),
        ("WO", 4096, 2560, (16, 2, 1)),
        ("FF1", 2560, 9728, (10, 2, 2)),
        ("FF2", 9728, 2560, (38, 2, 1)),
    ):
        xt = torch.randn(1, 1, M, K)
        wt = torch.randn(1, 1, K, N) * 0.02
        x = ttnn.from_torch(xt, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.L1_MEMORY_CONFIG)
        w = sharded_w(wt, K, N)
        Nt = N // T
        Kt = K // T
        kpr = Kt // 8
        pn = math.ceil(Nt / GX)
        ref = ttnn.to_torch(x).float() @ ttnn.to_torch(w).float()
        res = []
        for bw in sorted({d for d in (1, 2, 4, 5, 8, 10, 16, 19, 20, 32, 38) if kpr % d == 0} | {cur[0]}):
            for sh in (1, 2):
                for sw in (1, 2, 3, 4, 5, 6, 7, 8, 13):
                    if pn % sw or sh * sw > 8:
                        continue
                    try:
                        c = pc((GX, 8), bw, sh, sw, 2, pn)
                        us = traced(
                            lambda: ttnn.matmul(
                                x,
                                w,
                                program_config=c,
                                compute_kernel_config=ckc,
                                memory_config=ttnn.L1_MEMORY_CONFIG,
                                dtype=B8,
                            )
                        )
                        res.append((us, bw, sh, sw))
                    except Exception as e:
                        res.append((float("inf"), bw, sh, sw))
        base = [r for r in res if (r[1], r[2], r[3]) == cur]
        base = base[0][0] if base else float("nan")
        print(
            f"[bs1 {name} {GX}x8 pn={pn}] current bw={cur[0]} sb={cur[1]}x{cur[2]}: {base:7.1f} us  ({sum(1 for r in res if r[0]==float('inf'))} failed)",
            flush=True,
        )
        for us, bw, sh, sw in sorted(res)[:6]:
            c = pc((GX, 8), bw, sh, sw, 2, pn)
            o = ttnn.to_torch(
                ttnn.matmul(
                    x, w, program_config=c, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=B8
                )
            )
            print(
                f"    bw={bw:2d} sb={sh}x{sw}: {us:7.1f} us ({100*(us/base-1):+5.1f}%) pcc={pcc(o,ref):.5f}", flush=True
            )
        for t in (x, w):
            ttnn.deallocate(t)
    print("[done]")
finally:
    ttnn.close_device(D)
