# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark: DRAM-sharded matmul vs three regular matmul variants
#
# Usage:
#   python matmul_dram_sharded_bench.py
#
# Four variants are timed for each shape:
#   1. dram_sharded  – MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
#                      in0: L1 WIDTH_SHARDED, in1: DRAM WIDTH_SHARDED across banks
#   2. 1d_multicast  – MatmulMultiCoreReuseMultiCast1DProgramConfig (mcast_in0=True)
#                      in0: L1 WIDTH_SHARDED (same shard), in1: DRAM INTERLEAVED
#   3. 2d_multicast  – MatmulMultiCoreReuseMultiCastProgramConfig
#                      in0: L1 BLOCK_SHARDED on grid (8,1), in1: DRAM INTERLEAVED
#                      (grid_y=1 because M=32 is a single tile; multicast along x)
#   4. auto          – ttnn.matmul with no explicit program config
#                      in0: DRAM INTERLEAVED, in1: DRAM INTERLEAVED
#
# For each: one warmup (kernel compile), then N_ITERS dispatches, single sync,
# divided by N_ITERS to get average µs/iter.

import time
import torch
import ttnn
from models.common.utility_functions import torch2tt_tensor


# ── helpers ──────────────────────────────────────────────────────────────────


def pad_to_dram_banks(num, num_banks):
    lcm = 32 * num_banks
    rem = num % lcm
    return num if rem == 0 else num + (lcm - rem)


def find_max_subblock(out_block_h, out_block_w, limit=4):
    """Return (sub_h, sub_w) maximising sub_h*sub_w subject to <= limit.
    limit=4 because fp32_dest_acc_en=True halves available DST registers."""
    best_h, best_w = 1, 1
    for h in range(1, out_block_h + 1):
        if out_block_h % h == 0:
            for w in range(1, out_block_w + 1):
                if out_block_w % w == 0 and h * w <= limit and h * w > best_h * best_w:
                    best_h, best_w = h, w
    return best_h, best_w


# ── benchmark shapes ─────────────────────────────────────────────────────────

SHAPES = [
    # (M,    K,     N,    grid_size)
    (32, 8192, 1280, (8, 1)),
    (32, 8192, 4096, (8, 2)),
    (32, 8192, 1024, (8, 1)),
    (32, 32768, 1024, (8, 2)),
]

N_ITERS = 200
FIDELITY = ttnn.MathFidelity.HiFi2
IN0_DTYPE = ttnn.bfloat16
IN1_DTYPE = ttnn.bfloat8_b
OUT_DTYPE = ttnn.bfloat16
NUM_BANKS = 12  # WH has 12 DRAM banks; BH: query device.dram_grid_size().x


# ── main ─────────────────────────────────────────────────────────────────────


def bench_shape(device, M, K, N, grid_size):
    num_banks = device.dram_grid_size().x  # correct for both WH and BH
    N_padded = pad_to_dram_banks(N, num_banks)
    num_cores = grid_size[0] * grid_size[1]

    in0_block_h = M // 32
    in0_block_w = K // num_cores // 32
    out_block_h = M // 32
    out_block_w = N // num_cores // 32

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    in1_shard_shape = [K, N_padded // num_banks]

    interleaved_dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    sharded_l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)

    compute_cfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=FIDELITY,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    in0_raw = torch.randn(in0_shape).bfloat16().float()
    in1_raw = torch.randn(in1_shape).bfloat16().float()

    # ── tensors shared by both variants ──────────────────────────────────────

    # in0: interleaved DRAM → shard to L1 (used by DRAM-sharded variant)
    in0_dram = torch2tt_tensor(in0_raw, device, tt_memory_config=interleaved_dram, tt_dtype=IN0_DTYPE)
    in0_l1 = ttnn.interleaved_to_sharded(
        in0_dram,
        grid_size,
        [M, in0_block_w * 32],
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    # ── DRAM-sharded in1 ─────────────────────────────────────────────────────
    dram_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    print(f"DRAM grid: {device.dram_grid_size()}  → dram_grid for sharding: {dram_grid}")
    dram_grid_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), dram_grid)})
    in1_shard_spec = ttnn.ShardSpec(dram_grid_set, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_dram_sharded_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec
    )
    in1_dram_sharded = torch2tt_tensor(in1_raw, device, tt_memory_config=in1_dram_sharded_cfg, tt_dtype=IN1_DTYPE)

    dram_sharded_prog_cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w // 4,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fused_activation=None,
    )

    # warmup
    out = ttnn.matmul(
        in0_l1,
        in1_dram_sharded,
        program_config=dram_sharded_prog_cfg,
        memory_config=sharded_l1,
        dtype=OUT_DTYPE,
        compute_kernel_config=compute_cfg,
    )
    out.cpu()

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        out = ttnn.matmul(
            in0_l1,
            in1_dram_sharded,
            program_config=dram_sharded_prog_cfg,
            memory_config=sharded_l1,
            dtype=OUT_DTYPE,
            compute_kernel_config=compute_cfg,
        )
    out.cpu()
    dram_sharded_us = (time.perf_counter() - t0) / N_ITERS * 1e6

    # ── variant 1b: DRAM-sharded via interleaved_to_sharded ─────────────────
    # Same matmul as variant 1, but in1 starts as DRAM interleaved and is
    # resharded to DRAM WIDTH_SHARDED on-device before the matmul.
    in1_interleaved_for_reshard = torch2tt_tensor(
        in1_raw, device, tt_memory_config=interleaved_dram, tt_dtype=IN1_DTYPE
    )
    in1_dram_resharded = ttnn.interleaved_to_sharded(in1_interleaved_for_reshard, in1_dram_sharded_cfg)

    # warmup
    out1b = ttnn.matmul(
        in0_l1,
        in1_dram_resharded,
        program_config=dram_sharded_prog_cfg,
        memory_config=sharded_l1,
        dtype=OUT_DTYPE,
        compute_kernel_config=compute_cfg,
    )
    out1b.cpu()

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        out1b = ttnn.matmul(
            in0_l1,
            in1_dram_resharded,
            program_config=dram_sharded_prog_cfg,
            memory_config=sharded_l1,
            dtype=OUT_DTYPE,
            compute_kernel_config=compute_cfg,
        )
    out1b.cpu()
    dram_resharded_us = (time.perf_counter() - t0) / N_ITERS * 1e6

    # ── variant 1c: bf16 → typecast to bfp8 → DRAM-sharded ─────────────────
    # Load weights as bf16 interleaved, typecast to bfp8 on device (runs on
    # compute cores, interleaved layout), then interleaved_to_sharded to DRAM.
    in1_bf16_interleaved = torch2tt_tensor(in1_raw, device, tt_memory_config=interleaved_dram, tt_dtype=ttnn.bfloat16)
    in1_bfp8_interleaved = ttnn.typecast(in1_bf16_interleaved, IN1_DTYPE)
    in1_dram_from_cast = ttnn.interleaved_to_sharded(in1_bfp8_interleaved, in1_dram_sharded_cfg)

    # warmup
    out1c = ttnn.matmul(
        in0_l1,
        in1_dram_from_cast,
        program_config=dram_sharded_prog_cfg,
        memory_config=sharded_l1,
        dtype=OUT_DTYPE,
        compute_kernel_config=compute_cfg,
    )
    out1c.cpu()

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        out1c = ttnn.matmul(
            in0_l1,
            in1_dram_from_cast,
            program_config=dram_sharded_prog_cfg,
            memory_config=sharded_l1,
            dtype=OUT_DTYPE,
            compute_kernel_config=compute_cfg,
        )
    out1c.cpu()
    dram_typecast_us = (time.perf_counter() - t0) / N_ITERS * 1e6

    in1_interleaved = torch2tt_tensor(in1_raw, device, tt_memory_config=interleaved_dram, tt_dtype=IN1_DTYPE)

    # ── variant 2: 1D multicast (WIDTH_SHARDED in0, interleaved DRAM in1) ─────
    # Closest apples-to-apples: same in0 shard layout, in1 multicast from DRAM
    # instead of each core pulling its own DRAM shard.
    sub_h_1d, sub_w_1d = find_max_subblock(out_block_h, out_block_w)
    multicast_1d_prog_cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w // 2,
        out_subblock_h=sub_h_1d,
        out_subblock_w=sub_w_1d,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    out2 = ttnn.matmul(
        in0_l1,
        in1_interleaved,
        program_config=multicast_1d_prog_cfg,
        memory_config=sharded_l1,
        dtype=OUT_DTYPE,
        compute_kernel_config=compute_cfg,
    )
    out2.cpu()

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        out2 = ttnn.matmul(
            in0_l1,
            in1_interleaved,
            program_config=multicast_1d_prog_cfg,
            memory_config=sharded_l1,
            dtype=OUT_DTYPE,
            compute_kernel_config=compute_cfg,
        )
    out2.cpu()
    multicast_1d_us = (time.perf_counter() - t0) / N_ITERS * 1e6

    # ── variant 3: 2D multicast (BLOCK_SHARDED in0, interleaved DRAM in1) ──────
    # MatmulMultiCoreReuseMultiCastProgramConfig requires BLOCK or HEIGHT sharded
    # in0. For M=32 (one tile) grid_y must be 1; we use grid (8,1) so each core
    # owns the full M but a K/8 slice, which is a valid BLOCK_SHARDED layout.
    grid_2d = (8, 1)
    num_cores_2d = grid_2d[0] * grid_2d[1]
    per_core_M_2d = M // 32
    per_core_N_2d = N // num_cores_2d // 32
    in0_block_w_2d = K // num_cores_2d // 32 // 2  # halved loop iterations vs //4
    _, out_sub_w_2d = find_max_subblock(per_core_M_2d, per_core_N_2d)

    in0_dram2 = torch2tt_tensor(in0_raw, device, tt_memory_config=interleaved_dram, tt_dtype=IN0_DTYPE)
    in0_block = ttnn.interleaved_to_sharded(
        in0_dram2,
        grid_2d,
        [M, K // num_cores_2d],
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    block_sharded_l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)

    multicast_2d_prog_cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_2d,
        in0_block_w=in0_block_w_2d,
        out_subblock_h=1,
        out_subblock_w=out_sub_w_2d,
        per_core_M=per_core_M_2d,
        per_core_N=per_core_N_2d,
        transpose_mcast=False,
        fuse_batch=True,
    )

    out3 = ttnn.matmul(
        in0_block,
        in1_interleaved,
        program_config=multicast_2d_prog_cfg,
        memory_config=block_sharded_l1,
        dtype=OUT_DTYPE,
        compute_kernel_config=compute_cfg,
    )
    out3.cpu()

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        out3 = ttnn.matmul(
            in0_block,
            in1_interleaved,
            program_config=multicast_2d_prog_cfg,
            memory_config=block_sharded_l1,
            dtype=OUT_DTYPE,
            compute_kernel_config=compute_cfg,
        )
    out3.cpu()
    multicast_2d_us = (time.perf_counter() - t0) / N_ITERS * 1e6

    # ── variant 4: auto-selected (fully interleaved DRAM, no explicit config) ─
    in0_dram3 = torch2tt_tensor(in0_raw, device, tt_memory_config=interleaved_dram, tt_dtype=IN0_DTYPE)

    out4 = ttnn.matmul(in0_dram3, in1_interleaved, dtype=OUT_DTYPE, compute_kernel_config=compute_cfg)
    out4.cpu()

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        out4 = ttnn.matmul(in0_dram3, in1_interleaved, dtype=OUT_DTYPE, compute_kernel_config=compute_cfg)
    out4.cpu()
    auto_us = (time.perf_counter() - t0) / N_ITERS * 1e6

    print(
        f"  M={M:5d} K={K:6d} N={N:5d}  "
        f"dram_sharded={dram_sharded_us:6.1f}  "
        f"dram_reshrd={dram_resharded_us:6.1f}  "
        f"dram_tcast={dram_typecast_us:6.1f}  "
        f"1d_mcast={multicast_1d_us:6.1f}  "
        f"2d_mcast={multicast_2d_us:6.1f}  "
        f"auto={auto_us:6.1f}  "
        f"(vs dram_sharded: reshrd={dram_resharded_us/dram_sharded_us:.2f}x "
        f"tcast={dram_typecast_us/dram_sharded_us:.2f}x "
        f"1d={multicast_1d_us/dram_sharded_us:.2f}x "
        f"2d={multicast_2d_us/dram_sharded_us:.2f}x "
        f"auto={auto_us/dram_sharded_us:.2f}x)"
    )

    ttnn.deallocate(in0_dram)
    ttnn.deallocate(in0_l1)
    ttnn.deallocate(in1_dram_sharded)
    ttnn.deallocate(in1_interleaved_for_reshard)
    ttnn.deallocate(in1_dram_resharded)
    ttnn.deallocate(in1_bf16_interleaved)
    ttnn.deallocate(in1_bfp8_interleaved)
    ttnn.deallocate(in1_dram_from_cast)
    ttnn.deallocate(in1_interleaved)
    ttnn.deallocate(in0_dram2)
    ttnn.deallocate(in0_block)
    ttnn.deallocate(in0_dram3)


def main():
    device = ttnn.open_device(device_id=0)
    try:
        print(f"\nBenchmark: DRAM-sharded vs 1D/2D-multicast vs auto  ({N_ITERS} iters each)")
        print(f"  All times in µs/iter\n")
        print(
            f"  {'M':>5} {'K':>6} {'N':>5}  "
            f"{'dram_shrd':>11} {'dram_reshrd':>12} {'dram_tcast':>11} {'1d_mcast':>10} {'2d_mcast':>10} {'auto':>8}  "
            f"vs dram_sharded"
        )
        print("  " + "-" * 85)
        for M, K, N, grid_size in SHAPES:
            bench_shape(device, M, K, N, grid_size)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

#       M      K     N    dram_shrd  dram_reshrd  dram_tcast   1d_mcast   2d_mcast     auto  vs dram_sharded
#   -------------------------------------------------------------------------------------
# DRAM grid: (x=12,y=1)  → dram_grid for sharding: (x=11,y=0)
# Mapping 12 DRAM shards to 2 compute core ranges: [0,0 - 7,0] [0,1 - 3,1]
#   M=   32 K=  8192 N= 1280  dram_sharded=  61.6  dram_reshrd=  61.5  dram_tcast=  61.5  1d_mcast=  90.4  2d_mcast=  90.2  auto= 153.1  (vs dram_sharded: reshrd=1.00x tcast=1.00x 1d=1.47x 2d=1.46x auto=2.48x)
# DRAM grid: (x=12,y=1)  → dram_grid for sharding: (x=11,y=0)
# Mapping 12 DRAM shards to 2 compute core ranges: [0,0 - 7,0] [0,1 - 3,1]
#   M=   32 K=  8192 N= 4096  dram_sharded= 163.0  dram_reshrd= 162.8  dram_tcast= 162.8  1d_mcast= 208.1  2d_mcast= 244.6  auto= 190.8  (vs dram_sharded: reshrd=1.00x tcast=1.00x 1d=1.28x 2d=1.50x auto=1.17x)
# DRAM grid: (x=12,y=1)  → dram_grid for sharding: (x=11,y=0)
# Mapping 12 DRAM shards to 2 compute core ranges: [0,0 - 7,0] [0,1 - 3,1]
#   M=   32 K=  8192 N= 1024  dram_sharded=  50.7  dram_reshrd=  50.5  dram_tcast=  50.5  1d_mcast=  69.4  2d_mcast=  69.9  auto= 141.8  (vs dram_sharded: reshrd=1.00x tcast=1.00x 1d=1.37x 2d=1.38x auto=2.79x)
# DRAM grid: (x=12,y=1)  → dram_grid for sharding: (x=11,y=0)
# Mapping 12 DRAM shards to 2 compute core ranges: [0,0 - 7,0] [0,1 - 3,1]
#   M=   32 K= 32768 N= 1024  dram_sharded= 157.0  dram_reshrd= 156.9  dram_tcast= 157.0  1d_mcast= 200.2  2d_mcast= 272.6  auto= 556.3  (vs dram_sharded: reshrd=1.00x tcast=1.00x 1d=1.28x 2d=1.74x auto=3.54x)

# Get optimal cores for DRAM:
#       M      K     N    dram_shrd  dram_reshrd  dram_tcast   1d_mcast   2d_mcast     auto  vs dram_sharded
#   -------------------------------------------------------------------------------------
# DRAM grid: (x=12,y=1)  → dram_grid for sharding: (x=11,y=0)
#   M=   32 K=  8192 N= 1280  dram_sharded=  61.6  dram_reshrd=  61.5  dram_tcast=  61.7  1d_mcast=  90.4  2d_mcast=  90.2  auto= 153.2  (vs dram_sharded: reshrd=1.00x tcast=1.00x 1d=1.47x 2d=1.46x auto=2.48x)
# DRAM grid: (x=12,y=1)  → dram_grid for sharding: (x=11,y=0)
#   M=   32 K=  8192 N= 4096  dram_sharded= 163.0  dram_reshrd= 162.8  dram_tcast= 162.9  1d_mcast= 207.7  2d_mcast= 245.2  auto= 190.5  (vs dram_sharded: reshrd=1.00x tcast=1.00x 1d=1.27x 2d=1.50x auto=1.17x)
# DRAM grid: (x=12,y=1)  → dram_grid for sharding: (x=11,y=0)
#   M=   32 K=  8192 N= 1024  dram_sharded=  50.7  dram_reshrd=  50.5  dram_tcast=  50.5  1d_mcast=  69.4  2d_mcast=  69.6  auto= 141.7  (vs dram_sharded: reshrd=1.00x tcast=1.00x 1d=1.37x 2d=1.37x auto=2.79x)
# DRAM grid: (x=12,y=1)  → dram_grid for sharding: (x=11,y=0)
#   M=   32 K= 32768 N= 1024  dram_sharded= 157.1  dram_reshrd= 157.2  dram_tcast= 157.0  1d_mcast= 201.2  2d_mcast= 272.6  auto= 555.9  (vs dram_sharded: reshrd=1.00x tcast=1.00x 1d=1.28x 2d=1.74x auto=3.54x)
