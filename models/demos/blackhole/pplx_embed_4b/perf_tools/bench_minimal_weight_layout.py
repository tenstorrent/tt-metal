# minimal_matmul at batched shapes: DRAM width-sharded bfp4 weights (model default) vs DRAM interleaved, shipped blocks
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


BLK = {
    4096: {"FF1": (8, 8, 8, 1, 8), "FF2": (16, 8, 8, 1, 8), "QKV": (8, 4, 8, 1, 8), "WO": (16, 8, 8, 1, 8)},
    8192: {"FF1": (8, 8, 8, 1, 8), "FF2": (8, 8, 8, 1, 8), "QKV": (8, 8, 8, 1, 8), "WO": (8, 8, 8, 1, 8)},
    16384: {"FF1": (8, 8, 8, 1, 8), "FF2": (8, 8, 8, 1, 8), "QKV": (8, 8, 8, 1, 8), "WO": (8, 8, 8, 1, 8)},
}
try:
    for M in (4096, 8192, 16384):
        for name, K, N in (("FF1", 2560, 9728), ("FF2", 9728, 2560), ("QKV", 2560, 6144), ("WO", 4096, 2560)):
            wt = torch.randn(1, 1, K, N) * 0.02
            x = ttnn.from_torch(
                torch.randn(1, 1, M, K),
                dtype=B8,
                layout=ttnn.TILE_LAYOUT,
                device=D,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            mb, kb, nb, sh, sw = BLK[M][name]
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sh,
                subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(12, 10),
            )
            out = {}
            for wname, w in (
                ("sharded", sharded_w(wt, K, N)),
                (
                    "interleaved",
                    ttnn.from_torch(
                        wt, dtype=B4, layout=ttnn.TILE_LAYOUT, device=D, memory_config=ttnn.DRAM_MEMORY_CONFIG
                    ),
                ),
            ):
                try:
                    out[wname] = traced(
                        lambda: ttnn.experimental.minimal_matmul(
                            x, w, compute_kernel_config=ckc, config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=B8
                        ),
                        n=2,
                    )
                except Exception as e:
                    out[wname] = float("inf")
                ttnn.deallocate(w)
            print(
                f"[M={M:5d} {name}] sharded {out['sharded']:8.1f} us   interleaved {out['interleaved']:8.1f} us  ({100*(out['interleaved']/out['sharded']-1):+5.1f}%)",
                flush=True,
            )
            ttnn.deallocate(x)
    print("[done]")
finally:
    ttnn.close_device(D)
