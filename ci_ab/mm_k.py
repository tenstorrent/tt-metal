#!/usr/bin/env python3
"""Scratch, not for merge. 2048x4096x4096 matmul (in0 bf16) on an 11x10 core grid (the p100a grid, so every per-core
block and subblock matches the local card): LoFi, HiFi2, HiFi4 with bfp8_b and bf16 outputs, plus bfp8_b weights at LoFi
and HiFi2. Each case warmed up twice, then signposted and run MM_N times; outputs hashed per case (bit identity)."""
import hashlib, math, os
import torch
import ttnn
from tracy import signpost

N = int(os.environ.get("MM_N", "12"))
dev = ttnn.open_device(device_id=0, l1_small_size=32768)
D = ttnn.DRAM_MEMORY_CONFIG
F = ttnn.MathFidelity
CG = ttnn.CoreGrid(y=10, x=11)
torch.manual_seed(0)


def ckc(fid):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True, dst_full_sync_en=False)


def tt(shape, dt):
    return ttnn.from_torch(torch.randn(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=D)



def case(name, f):
    try:
        for _ in range(2):
            o = f()
        ttnn.synchronize_device(dev)
        h = "-" if os.environ.get("MM_NOHASH") else hashlib.sha1(ttnn.to_torch(o).float().numpy().tobytes()).hexdigest()[:12]
        signpost(name)
        for _ in range(N):
            f()
        ttnn.synchronize_device(dev)
        print("case done", name, h, flush=True)
    except Exception as e:
        print("case FAIL", name, str(e).splitlines()[0][:200] if str(e) else type(e).__name__, flush=True)


def ckc2(fp32=False, l1=True):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=F.HiFi2, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=l1, dst_full_sync_en=False)


def tt2(shape, dt=ttnn.bfloat16, mc=D):
    return ttnn.from_torch(torch.randn(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)


# 1. operand format: 2k4k4k on 11x10 (32 inner blocks), HiFi2 1x1
for dn, dt in (("bf16", ttnn.bfloat16), ("f32", ttnn.float32)):
    a, w = tt2((1, 1, 2048, 4096), dt), tt2((1, 1, 4096, 4096), dt)
    for fp32 in (False, True):
        for on, odt in (("outbf16", ttnn.bfloat16), ("outsame", dt)):
            case(f"k_{dn}_fp32d{int(fp32)}_{on}", lambda: ttnn.matmul(a, w, compute_kernel_config=ckc2(fp32), core_grid=CG, memory_config=D, dtype=odt))

# 2a. DRAM-sharded decode, large K, 1x1 sub blocks, 64 inner blocks
dram_x = dev.dram_grid_size().x
for sh_tiles, ibw in ((32, 4), (32, 8)):
    cores = dram_x
    k, n = cores * 32 * sh_tiles, 32 * dram_x
    grid = ttnn.CoreGrid(x=cores, y=1)
    in0_mc = ttnn.create_sharded_memory_config(shape=(32, k // cores), core_grid=grid, strategy=ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True)
    in0 = tt2((1, 1, 32, k), mc=in0_mc)
    padded_n = math.ceil(n / (32 * dram_x)) * 32 * dram_x
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_x - 1, 0))})
    in1_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, ttnn.ShardSpec(dram_grid, (k, padded_n // dram_x), ttnn.ShardOrientation.ROW_MAJOR))
    in1 = tt2((k, n), mc=in1_mc)
    out_mc = ttnn.create_sharded_memory_config(shape=(32, n // cores), core_grid=grid, strategy=ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True)
    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=ibw, per_core_M=1, per_core_N=math.ceil(n / (32 * cores)), fused_activation=None)
    for fp32 in (False, True):
        case(f"dec_k{k}_ibw{ibw}_fp32d{int(fp32)}", lambda: ttnn.linear(in0, in1, program_config=pc, memory_config=out_mc, dtype=ttnn.bfloat16, compute_kernel_config=ckc2(fp32)))

# 2b. mcast 1d and 2d, one-tile sub blocks, 64 inner blocks
a, b = tt2((1, 1, 32, 8192)), tt2((1, 1, 8192, 1024))
pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8, 4), in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=1, fuse_batch=True, fused_activation=None, mcast_in0=True)
for fp32 in (False, True):
    case(f"m1d_fp32d{int(fp32)}", lambda: ttnn.matmul(a, b, program_config=pc, compute_kernel_config=ckc2(fp32), memory_config=D))
a, b = tt2((1, 1, 256, 8192)), tt2((1, 1, 8192, 256))
pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(8, 8), in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=1, transpose_mcast=False, fused_activation=None)
for fp32 in (False, True):
    case(f"m2d_fp32d{int(fp32)}", lambda: ttnn.matmul(a, b, program_config=pc, compute_kernel_config=ckc2(fp32), memory_config=D))
ttnn.close_device(dev)
print("ALL DONE", flush=True)
