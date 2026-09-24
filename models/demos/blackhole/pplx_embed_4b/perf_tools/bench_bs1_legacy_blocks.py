# bs1 legacy 2D-mcast matmul (8x8, sharded bfp4 weights, L1 activations): in0_block_w x out_subblock sweep
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
ONLY = os.getenv("ONLY", "").split(",") if os.getenv("ONLY") else None


def sharded_w(K, N):
    pad = math.ceil(N / (T * 8)) * (T * 8)
    spec = ttnn.ShardSpec(dram_grid, (K, pad // 8), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.from_torch(
        torch.randn(1, 1, K, N) * 0.02,
        dtype=B4,
        layout=ttnn.TILE_LAYOUT,
        device=D,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec),
    )


def pc(bw, sh, sw, pn, tm=False):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=bw,
        out_subblock_h=sh,
        out_subblock_w=sw,
        per_core_M=2,
        per_core_N=pn,
        transpose_mcast=tm,
        fused_activation=None,
        fuse_batch=True,
    )


try:
    x2560 = ttnn.from_torch(
        torch.randn(1, 1, M, 2560), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    x4096 = ttnn.from_torch(
        torch.randn(1, 1, M, 4096), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    x9728 = ttnn.from_torch(
        torch.randn(1, 1, M, 9728), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    cases = {
        "QKV": (x2560, 2560, 6144, 24, 10, (1, 4)),
        "WO": (x4096, 4096, 2560, 10, 16, (1, 2)),
        "FF1": (x2560, 2560, 9728, 38, 10, (1, 2)),
        "FF2": (x9728, 9728, 2560, 10, 38, (1, 2)),
    }
    for name, (x, K, N, pn, cur_bw, cur_sb) in cases.items():
        if ONLY and name not in ONLY:
            continue
        w = sharded_w(K, N)
        Kt = K // T
        bws = sorted(
            {cur_bw} | {d for d in (8, 10, 16, 19, 20, 32, 38, 40, 64, 76, 80, 152) if Kt % d == 0 and d <= Kt}
        )
        sbs = sorted(
            {cur_sb} | {(h, wd) for h in (1, 2) for wd in (1, 2, 3, 4, 5, 6, 8, 19) if pn % wd == 0 and h * wd <= 8}
        )
        res = []
        for bw in bws:
            for sh, sw in sbs:
                try:
                    c = pc(bw, sh, sw, pn)
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
        base = [r for r in res if r[1] == cur_bw and (r[2], r[3]) == cur_sb][0][0]
        print(
            f"[bs1 {name} K={K} N={N}] current in0_bw={cur_bw} sb={cur_sb[0]}x{cur_sb[1]}: {base:7.1f} us   ({sum(1 for r in res if r[0]==float('inf'))} configs failed)",
            flush=True,
        )
        for us, bw, sh, sw in sorted(res)[:6]:
            print(f"    in0_bw={bw:3d} sb={sh}x{sw}: {us:7.1f} us ({100*(us/base-1):+5.1f}%)", flush=True)
        ttnn.deallocate(w)
    print("[done]")
finally:
    ttnn.close_device(D)
