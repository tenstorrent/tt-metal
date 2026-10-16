# What bounds the bs1 (M=512) legacy 2D-mcast matmul? Vary in1 placement (DRAM width-sharded / DRAM interleaved / L1 interleaved),
# fidelity (LoFi/HiFi2), in0 dtype (bfp8/bf16), and grid (8x8 vs 12x8 with L1 or interleaved weights).
import math
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common_traced import make_traced

B4 = ttnn.bfloat4_b
B8 = ttnn.bfloat8_b
BF = ttnn.bfloat16
T = 32
M = 512
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=64 * 1024 * 1024)
traced = make_traced(D)


def CK(fid):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )


LO, HI = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2
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


def run(label, x, w, c, ck):
    try:
        us = traced(
            lambda: ttnn.matmul(
                x, w, program_config=c, compute_kernel_config=ck, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=B8
            )
        )
        print(f"    {label:58s} {us:7.1f} us", flush=True)
    except Exception as e:
        print(f"    {label:58s}    FAIL {str(e).splitlines()[0][:90]}", flush=True)


try:
    for name, K, N, pn, bw, sw in (
        ("QKV", 2560, 6144, 24, 10, 4),
        ("FF1", 2560, 9728, 38, 10, 2),
        ("WO", 4096, 2560, 10, 16, 2),
    ):
        print(f"[bs1 {name} K={K} N={N}]", flush=True)
        xt = torch.randn(1, 1, M, K)
        wt = torch.randn(1, 1, K, N) * 0.02
        x8 = ttnn.from_torch(xt, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.L1_MEMORY_CONFIG)
        w_sh = sharded_w(wt, K, N)
        w_il = ttnn.from_torch(wt, dtype=B4, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        c88 = pc((8, 8), bw, 1, sw, 2, pn)
        run("8x8 in1 DRAM width-sharded bfp4 (model)", x8, w_sh, c88, CK(LO))
        run("8x8 in1 DRAM width-sharded bfp4, HiFi2", x8, w_sh, c88, CK(HI))
        run("8x8 in1 DRAM interleaved bfp4", x8, w_il, c88, CK(LO))
        try:
            w_l1 = ttnn.from_torch(wt, dtype=B4, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.L1_MEMORY_CONFIG)
            run("8x8 in1 L1 interleaved bfp4", x8, w_l1, c88, CK(LO))
            run("8x8 in1 L1 interleaved bfp4, HiFi2", x8, w_l1, c88, CK(HI))
            Nt = N // T
            pn12 = math.ceil(Nt / 12)
            run(
                f"12x8 in1 L1 interleaved bfp4 (pn={pn12})",
                x8,
                w_l1,
                pc((12, 8), bw, 1, 1 if pn12 % 2 else 2, 2, pn12),
                CK(LO),
            )
            run(
                f"12x8 in1 DRAM interleaved bfp4 (pn={pn12})",
                x8,
                w_il,
                pc((12, 8), bw, 1, 1 if pn12 % 2 else 2, 2, pn12),
                CK(LO),
            )
            pn10 = math.ceil(Nt / 10)
            run(
                f"10x8 in1 L1 interleaved bfp4 (pn={pn10})",
                x8,
                w_l1,
                pc((10, 8), bw, 1, 1 if pn10 % 2 else 2, 2, pn10),
                CK(LO),
            )
            ttnn.deallocate(w_l1)
        except Exception as e:
            print("    L1 weights:", str(e).splitlines()[0][:100])
        x16 = ttnn.from_torch(xt, dtype=BF, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.L1_MEMORY_CONFIG)
        run("8x8 in1 DRAM width-sharded bfp4, in0 bf16", x16, w_sh, c88, CK(LO))
        w8 = sharded_w(wt, K, N) if False else None
        for t in (x8, x16, w_sh, w_il):
            ttnn.deallocate(t)
    print("[done]")
finally:
    ttnn.close_device(D)
