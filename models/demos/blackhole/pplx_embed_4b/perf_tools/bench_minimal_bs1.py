# minimal_matmul at bs1 (M=512) on 120/96 cores vs the legacy 12x8 numbers; model-faithful weights (bfp4 DRAM width-sharded) and interleaved
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


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


LEGACY = {"QKV": 49.8, "WO": 40.1, "FF1": 79.5, "FF2": 78.8}
try:
    for name, K, N in (("FF1", 2560, 9728), ("FF2", 9728, 2560), ("QKV", 2560, 6144), ("WO", 4096, 2560)):
        xt = torch.randn(1, 1, M, K)
        wt = torch.randn(1, 1, K, N) * 0.02
        x = ttnn.from_torch(xt, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.L1_MEMORY_CONFIG)
        ws = {
            "sharded": sharded_w(wt, K, N),
            "interleaved": ttnn.from_torch(
                wt, dtype=B4, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
        }
        ref = ttnn.to_torch(x).float() @ wt.float()
        res = []
        for gx, gy in ((12, 10), (12, 8)):
            for mb in (4, 8, 16):
                for kb in (5, 8, 10):
                    for nb in (4, 8):
                        for sh, sw in ((1, 8), (2, 4), (1, 4)):
                            if sw > nb:
                                continue
                            for wname, w in ws.items():
                                try:
                                    cfg = ttnn.MinimalMatmulConfig(
                                        M_block_size=mb,
                                        K_block_size=kb,
                                        N_block_size=nb,
                                        subblock_h=sh,
                                        subblock_w=sw,
                                        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                                    )
                                    us = traced(
                                        lambda: ttnn.experimental.minimal_matmul(
                                            x,
                                            w,
                                            compute_kernel_config=ckc,
                                            config=cfg,
                                            memory_config=ttnn.L1_MEMORY_CONFIG,
                                            dtype=B8,
                                        ),
                                        n=3,
                                    )
                                    res.append((us, gx, gy, mb, kb, nb, sh, sw, wname))
                                except Exception as e:
                                    res.append((float("inf"), gx, gy, mb, kb, nb, sh, sw, wname))
        ok = [r for r in res if r[0] < float("inf")]
        print(
            f"[bs1 minimal {name} K={K} N={N}] legacy 12x8 {LEGACY[name]:.1f} us; {len(ok)}/{len(res)} configs ran",
            flush=True,
        )
        for r in sorted(ok)[:6]:
            us, gx, gy, mb, kb, nb, sh, sw, wname = r
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sh,
                subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            )
            o = ttnn.to_torch(
                ttnn.experimental.minimal_matmul(
                    x, ws[wname], compute_kernel_config=ckc, config=cfg, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=B8
                )
            )
            print(
                f"    {gx}x{gy} blk {mb},{kb},{nb} sb {sh}x{sw} {wname:11s}: {us:7.1f} us ({100*(us/LEGACY[name]-1):+5.1f}% vs legacy 12x8) pcc={pcc(o,ref):.5f}",
                flush=True,
            )
        ttnn.deallocate(x)
        [ttnn.deallocate(w) for w in ws.values()]
    print("[done]")
finally:
    ttnn.close_device(D)
