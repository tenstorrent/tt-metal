# bs1 legacy 2D-mcast matmul on wider grids WITH the model's DRAM width-sharded bfp4 weights (8 banks):
# numerics (finite? PCC vs 8x8 and vs torch) + traced time, for exact-fit and padded per_core_N cases.
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


try:
    for name, K, N, pn8, bw, sw8 in (
        ("QKV", 2560, 6144, 24, 10, 4),
        ("WO", 4096, 2560, 10, 16, 2),
        ("FF1", 2560, 9728, 38, 10, 2),
        ("FF2", 9728, 2560, 10, 38, 2),
    ):
        print(f"[bs1 {name} K={K} N={N}]", flush=True)
        xt = torch.randn(1, 1, M, K)
        wt = torch.randn(1, 1, K, N) * 0.02
        x = ttnn.from_torch(xt, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.L1_MEMORY_CONFIG)
        w = sharded_w(wt, K, N)
        Nt = N // T
        ref_torch = ttnn.to_torch(x).float() @ ttnn.to_torch(w).float()
        c8 = pc((8, 8), bw, 1, sw8, 2, pn8)
        out8 = ttnn.to_torch(
            ttnn.matmul(
                x, w, program_config=c8, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=B8
            )
        )
        t8 = traced(
            lambda: ttnn.matmul(
                x, w, program_config=c8, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=B8
            )
        )
        print(
            f"    8x8 pn={pn8} (model): {t8:7.1f} us  finite={bool(torch.isfinite(out8).all())} pcc_vs_torch={pcc(out8,ref_torch):.5f}",
            flush=True,
        )
        for gx in (9, 10, 11, 12):
            pn = math.ceil(Nt / gx)
            fit = "exact" if pn * gx == Nt else f"pad {pn*gx-Nt}t"
            for sh, sw in ((1, 1), (1, 2), (2, 2), (2, 1)):
                if pn % sw:
                    continue
                try:
                    c = pc((gx, 8), bw, sh, sw, 2, pn)
                    o = ttnn.to_torch(
                        ttnn.matmul(
                            x,
                            w,
                            program_config=c,
                            compute_kernel_config=ckc,
                            memory_config=ttnn.L1_MEMORY_CONFIG,
                            dtype=B8,
                        )
                    )
                    fin = bool(torch.isfinite(o).all())
                    p = pcc(o, ref_torch) if fin else float("nan")
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
                    print(
                        f"    {gx}x8 pn={pn:2d} ({fit:8s}) bw={bw} sb={sh}x{sw}: {us:7.1f} us ({100*(us/t8-1):+5.1f}%) finite={fin} pcc_vs_torch={p:.5f}",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"    {gx}x8 pn={pn:2d} ({fit:8s}) bw={bw} sb={sh}x{sw}: FAIL {str(e).splitlines()[0][:100]}",
                        flush=True,
                    )
        for t in (x, w):
            ttnn.deallocate(t)
    print("[done]")
finally:
    ttnn.close_device(D)
