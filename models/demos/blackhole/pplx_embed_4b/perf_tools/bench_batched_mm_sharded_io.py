# bs16 / bs32: can the fused add+RMSNorm's neighbours take its operands resident in L1, ND-sharded in the op's unit
# layout (shard = 1 tile row x W/R tiles, dealt round-robin over the 120 cores), without slowing down?
#   producers (write the norm's b operand): WO, FF2 -> output in DRAM / L1 interleaved / L1 unit-sharded
#   consumers (read the norm's output):     QKV, FF1(+FF3) -> in0 in DRAM / L1 interleaved / L1 unit-sharded
# Shapes, weight placement and MinimalMatmulConfig are the 09-24 profiles' (LoFi, bfp8 activations, bfp4 weights).
# Outputs are checked bit-identical to the DRAM arm. Usage: TT_VISIBLE_DEVICES=<chip> bench_batched_mm_sharded_io.py
import math
import os
import statistics
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common_traced import make_traced

B4, B8, T = ttnn.bfloat4_b, ttnn.bfloat8_b, 32
D = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=128 * 1024 * 1024)
traced = make_traced(D)
CK = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
)
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
GRID = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 9))])


def unit_sharded(shape, R):
    """The add+RMSNorm unit layout: shard [1 tile row, W/R tiles], round-robin over the 12x10 grid, y fastest."""
    shard = [1] * (len(shape) - 2) + [T, shape[-1] // R]
    spec = ttnn.NdShardSpec(
        ttnn.Shape(shard), GRID, ttnn.ShardOrientation.COL_MAJOR, ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D
    )
    return ttnn.MemoryConfig(ttnn.BufferType.L1, spec)


def weight(K, N, sharded):
    w = torch.randn(1, 1, K, N) * 0.02
    if not sharded:
        return ttnn.from_torch(w, dtype=B4, layout=ttnn.TILE_LAYOUT, device=D, memory_config=DRAM)
    pad = math.ceil(N / (T * 8)) * (T * 8)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))])
    spec = ttnn.ShardSpec(grid, (K, pad // 8), ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)
    return ttnn.from_torch(w, dtype=B4, layout=ttnn.TILE_LAYOUT, device=D, memory_config=mc)


def cfg(m, k, n, sw):
    return ttnn.MinimalMatmulConfig(
        M_block_size=m,
        K_block_size=k,
        N_block_size=n,
        subblock_h=1,
        subblock_w=sw,
        compute_with_storage_grid_size=ttnn.CoreCoord(12, 10),
    )


def mm(x, w, c, mc):
    return ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=CK, config=c, memory_config=mc, dtype=B8)


# (batch, name, role, in0 shape, N, sharded weight, config, R of the norm at that batch)
CASES = [
    (16, "QKV", "consumer", [1, 4, 2048, 2560], 6144, True, cfg(8, 8, 8, 8), 5),
    (16, "FF1+FF3 (fused)", "consumer", [1, 1, 8192, 2560], 19456, False, cfg(4, 20, 8, 4), 5),
    (16, "WO", "producer", [1, 8, 1024, 4096], 2560, True, cfg(8, 8, 8, 8), 5),
    (16, "FF2", "producer", [1, 1, 8192, 9728], 2560, True, cfg(8, 8, 8, 8), 5),
    (32, "QKV", "consumer", [1, 8, 2048, 2560], 6144, True, cfg(8, 8, 8, 8), 4),
    (32, "FF1 (FF3 same)", "consumer", [1, 32, 512, 2560], 9728, True, cfg(8, 8, 8, 8), 4),
    (32, "WO", "producer", [1, 16, 1024, 4096], 2560, True, cfg(8, 8, 8, 8), 4),
    (32, "FF2", "producer", [1, 32, 512, 9728], 2560, True, cfg(8, 8, 8, 8), 4),
]

if __name__ == "__main__":
    try:
        torch.manual_seed(0)
        for bs, name, role, shape, N, wsh, c, R in CASES:
            K = shape[-1]
            w = weight(K, N, wsh)
            x_host = torch.randn(*shape)
            x_dram = ttnn.from_torch(x_host, dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=DRAM)
            if role == "consumer":
                arms = [("in0 DRAM", lambda: x_dram, DRAM)]
                arms.append(("in0 L1 interleaved", lambda: ttnn.to_memory_config(x_dram, L1), DRAM))
                arms.append(
                    ("in0 L1 unit-sharded", lambda: ttnn.to_memory_config(x_dram, unit_sharded(shape, R)), DRAM)
                )
            else:
                out_shape = shape[:-1] + [N]
                arms = [("out DRAM", lambda: x_dram, DRAM), ("out L1 interleaved", lambda: x_dram, L1)]
                arms.append(("out L1 unit-sharded", lambda: x_dram, unit_sharded(out_shape, R)))
            print(
                f"[bs{bs} {name}: in0 {'x'.join(map(str, shape))}, N={N}, weights {'DRAM WS' if wsh else 'DRAM I'}]",
                flush=True,
            )
            ref, base = None, None
            for tag, make_x, out_mc in arms:
                try:
                    x = make_x()
                    out = mm(x, w, c, out_mc)
                    got = ttnn.to_torch(out)
                    ttnn.deallocate(out)
                    ref = got if ref is None else ref
                    same = "bit-identical" if torch.equal(got, ref) else "DIFFERS"
                    # one call per trace: n=4 keeps four outputs alive, which an L1 output at bs32 cannot fit
                    us = statistics.median(traced(lambda: mm(x, w, c, out_mc), n=1) for _ in range(7))
                    base = us if base is None else base
                    print(f"    {tag:22s} {us:8.1f} us ({100 * (us / base - 1):+5.1f}%)  {same}", flush=True)
                    if x is not x_dram:
                        ttnn.deallocate(x)
                except Exception as e:
                    print(f"    {tag:22s} FAILED: {str(e).splitlines()[0][:160]}", flush=True)
            ttnn.deallocate(x_dram)
            ttnn.deallocate(w)
        print("[done]")
    finally:
        ttnn.close_device(D)
