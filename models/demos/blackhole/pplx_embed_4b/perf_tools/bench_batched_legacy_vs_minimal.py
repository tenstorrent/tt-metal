# Batched shapes (M=4096 bs8, M=8192 bs16): legacy 2D-mcast on 12x8 (sharded bfp4 weights, coalesced reads) vs the model's minimal_matmul configs
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


MIN = {
    "bs8": {"FF1": (8, 8, 8, 1, 8), "FF2": (16, 8, 8, 1, 8), "QKV": (8, 4, 8, 1, 8), "WO": (16, 8, 8, 1, 8)},
    "bs16": {"FF1": (8, 8, 8, 1, 8), "FF2": (8, 8, 8, 1, 8), "QKV": (8, 8, 8, 1, 8), "WO": (8, 8, 8, 1, 8)},
}
MS = [int(v) for v in os.getenv("MS", "4096,8192").split(",")]
try:
    for M in MS:
        label = "bs8" if M == 4096 else "bs16"
        for name, K, N in (("FF1", 2560, 9728), ("FF2", 9728, 2560), ("QKV", 2560, 6144), ("WO", 4096, 2560)):
            xt = torch.randn(1, 1, M, K)
            wt = torch.randn(1, 1, K, N) * 0.02
            x = ttnn.from_torch(xt, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            w = sharded_w(wt, K, N)
            Kt = K // T
            Nt = N // T
            Mt = M // T
            mb, kb, nb, sh, sw = MIN[label][name]
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sh,
                subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(12, 10),
            )
            try:
                base = traced(
                    lambda: ttnn.experimental.minimal_matmul(
                        x, w, compute_kernel_config=ckc, config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=B8
                    ),
                    n=2,
                )
            except Exception as e:
                base = float("nan")
                print("minimal failed", str(e).splitlines()[0][:100])
            gflop = 2 * M * K * N / 1e9
            print(
                f"[{label} {name} K={K} N={N}] minimal 12x10 blk {mb},{kb},{nb} sb {sh}x{sw}: {base:8.1f} us ({gflop/base*1e3:4.0f} TFLOP/s)",
                flush=True,
            )
            res = []
            for gx, gy in ((12, 8), (8, 8), (12, 10)):
                pm = math.ceil(Mt / gy)
                pn = math.ceil(Nt / gx)
                kpr = Kt // gy if Kt % gy == 0 else None
                if kpr is None:
                    continue
                for bw in sorted({d for d in (2, 4, 5, 8, 10, 16, 19, 38) if kpr % d == 0}):
                    # L1: in0 2*pm*bw + in1 2*bw*pn + out pm*pn tiles (bfp8 1088 / bfp4 576)
                    l1 = (2 * pm * bw * 1088) + (2 * bw * pn * 576) + (pm * pn * 1088)
                    if l1 > 1_350_000:
                        continue
                    for sh2, sw2 in ((1, 1), (2, 1), (4, 1), (8, 1), (1, 2), (2, 2), (4, 2), (1, 4), (2, 4), (1, 8)):
                        if pn % sw2 or pm % sh2 or sh2 * sw2 > 8:
                            continue
                        try:
                            c = pc((gx, gy), bw, sh2, sw2, pm, pn)
                            us = traced(
                                lambda: ttnn.matmul(
                                    x,
                                    w,
                                    program_config=c,
                                    compute_kernel_config=ckc,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                    dtype=B8,
                                ),
                                n=2,
                            )
                            res.append((us, gx, gy, bw, sh2, sw2, pm, pn))
                        except Exception as e:
                            res.append((float("inf"), gx, gy, bw, sh2, sw2, pm, pn))
            ok = [r for r in res if r[0] < float("inf")]
            print(f"    legacy: {len(ok)}/{len(res)} configs ran; best:", flush=True)
            for us, gx, gy, bw, sh2, sw2, pm, pn in sorted(ok)[:5]:
                print(
                    f"      {gx}x{gy} pm={pm} pn={pn} bw={bw:2d} sb={sh2}x{sw2}: {us:8.1f} us ({100*(us/base-1):+5.1f}% vs minimal, {gflop/us*1e3:4.0f} TFLOP/s)",
                    flush=True,
                )
            ttnn.deallocate(x)
            ttnn.deallocate(w)
    print("[done]")
finally:
    ttnn.close_device(D)
