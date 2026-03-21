#!/usr/bin/env python3
"""
Compare MinimalMatmul vs MatmulDeviceOperation for LLaMA 70B prefill shapes.

Run on Galaxy:
    python scripts/compare_matmul_perf.py

This script benchmarks both matmul implementations on the actual tensor shapes
used in LLaMA 70B prefill to determine which is faster.
"""

import torch
import ttnn
import time
from loguru import logger

# LLaMA 70B prefill shapes (per-device after tensor parallelism)
TEST_SHAPES = [
    # (name, M, K, N)
    ("FF1/FF3 4K", 4096, 2048, 3584),
    ("FF1/FF3 8K", 8192, 2048, 3584),
    ("FF2 4K", 4096, 3584, 2048),
    ("FF2 8K", 8192, 3584, 2048),
    ("WO 4K", 4096, 2048, 2048),
    ("WO 8K", 8192, 2048, 2048),
]


def benchmark_minimal_matmul(device, tt_input, tt_weight, compute_config, warmup=3, iters=10):
    """Benchmark MinimalMatmul."""
    compute_grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreCoord(min(7, compute_grid.x), min(8, compute_grid.y))

    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=2,
        subblock_w=2,
        compute_with_storage_grid_size=core_grid,
    )

    # Warmup
    for _ in range(warmup):
        out = ttnn.experimental.minimal_matmul(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            compute_kernel_config=compute_config,
            config=matmul_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out.deallocate(True)

    # Benchmark
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for i in range(iters):
        out = ttnn.experimental.minimal_matmul(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            compute_kernel_config=compute_config,
            config=matmul_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if i < iters - 1:
            out.deallocate(True)
    ttnn.synchronize_device(device)
    end = time.perf_counter()

    avg_ms = (end - start) / iters * 1000
    return avg_ms, out


def benchmark_matmul_dram_sharded_weights(device, tt_input, tt_weight, M, K, N, compute_config, warmup=3, iters=10):
    """Benchmark MatmulDeviceOperation with DRAM-sharded weights (like decode path)."""
    import math

    # DRAM sharding uses 8 or 12 DRAM banks depending on arch
    # For WH, typically 12 DRAM banks (8 in row 0 + 4 in row 6)
    dram_grid = device.dram_grid_size()
    num_dram_banks = dram_grid.x  # typically 8

    # For DRAM sharded matmul:
    # - Input: WIDTH_SHARDED in L1 (small M for decode) or DRAM interleaved (large M for prefill)
    # - Weight: DRAM WIDTH_SHARDED across DRAM banks
    # - Output: WIDTH_SHARDED in L1

    # Since M is large for prefill, we need to use a different approach
    # Use L1 WIDTH_SHARDED for input across compute cores
    compute_grid = device.compute_with_storage_grid_size()
    grid_x = min(8, compute_grid.x)

    # For the prefill case with large M, the input shard would be too large
    # Instead, use interleaved input and DRAM sharded weights

    # Shard weights across DRAM banks
    weight_shard_shape = (K, (N + num_dram_banks - 1) // num_dram_banks)
    weight_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))}),
        weight_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    weight_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        weight_shard_spec,
    )

    # Convert weight to DRAM sharded
    tt_weight_dram_sharded = ttnn.to_memory_config(tt_weight, weight_mem_config)

    # For large M, shard input across L1 cores (WIDTH)
    # Shard: [M, K/grid_x] per core
    input_shard_width = K // grid_x
    input_shard_shape = (M, input_shard_width)

    # Check L1 fit
    shard_size = M * input_shard_width  # bytes for bf8
    if shard_size > 1_300_000:
        # Too large, use interleaved input instead
        tt_input_ready = tt_input
        input_sharded = False
    else:
        input_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, 0))}),
            input_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        input_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            input_shard_spec,
        )
        tt_input_ready = ttnn.to_memory_config(tt_input, input_mem_config)
        input_sharded = True

    # Output WIDTH_SHARDED
    out_shard_width = N // grid_x
    out_shard_shape = (M, out_shard_width)
    out_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, 0))}),
        out_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    out_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        out_shard_spec,
    )

    # DRAM sharded program config
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=K // 32 // grid_x,
        per_core_M=M // 32,
        per_core_N=N // 32 // grid_x,
        fused_activation=None,
    )

    # Warmup
    for _ in range(warmup):
        out = ttnn.matmul(
            tt_input_ready,
            tt_weight_dram_sharded,
            compute_kernel_config=compute_config,
            dtype=ttnn.bfloat8_b,
            program_config=program_config,
            memory_config=out_mem_config,
        )
        out.deallocate(True)

    # Benchmark
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for i in range(iters):
        out = ttnn.matmul(
            tt_input_ready,
            tt_weight_dram_sharded,
            compute_kernel_config=compute_config,
            dtype=ttnn.bfloat8_b,
            program_config=program_config,
            memory_config=out_mem_config,
        )
        if i < iters - 1:
            out.deallocate(True)
    ttnn.synchronize_device(device)
    end = time.perf_counter()

    if input_sharded:
        tt_input_ready.deallocate(True)
    tt_weight_dram_sharded.deallocate(True)

    avg_ms = (end - start) / iters * 1000
    return avg_ms, out


def benchmark_matmul_1d(device, tt_input, tt_weight, M, K, N, compute_config, warmup=3, iters=10):
    """Benchmark MatmulDeviceOperation with 1D multicast for DRAM interleaved tensors."""
    import math

    compute_grid = device.compute_with_storage_grid_size()
    # Use 1D grid along M dimension - better L1 efficiency for DRAM interleaved
    num_cores = min(compute_grid.x * compute_grid.y, M // 32)
    grid_x = min(8, num_cores)
    grid_y = math.ceil(num_cores / grid_x)

    per_core_M = math.ceil(M / 32 / num_cores)
    per_core_N = N // 32

    # For 1D mcast with DRAM interleaved, in0_block_w controls K blocking
    in0_block_w = min(4, K // 32)

    # Find valid subblocks (product <= 4 for fp32 acc)
    out_subblock_w = 1
    for sw in [4, 2, 1]:
        if per_core_N % sw == 0:
            out_subblock_w = sw
            break

    max_subblock_h = 4 // out_subblock_w
    out_subblock_h = 1
    for sh in range(max_subblock_h, 0, -1):
        if per_core_M % sh == 0:
            out_subblock_h = sh
            break

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    # Warmup
    for _ in range(warmup):
        out = ttnn.linear(
            tt_input,
            tt_weight,
            compute_kernel_config=compute_config,
            dtype=ttnn.bfloat8_b,
            program_config=program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out.deallocate(True)

    # Benchmark
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for i in range(iters):
        out = ttnn.linear(
            tt_input,
            tt_weight,
            compute_kernel_config=compute_config,
            dtype=ttnn.bfloat8_b,
            program_config=program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if i < iters - 1:
            out.deallocate(True)
    ttnn.synchronize_device(device)
    end = time.perf_counter()

    avg_ms = (end - start) / iters * 1000
    return avg_ms, out


def main():
    logger.info("Opening device...")
    device = ttnn.open_device(device_id=0)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    results = []

    logger.info("\n" + "=" * 100)
    logger.info("MinimalMatmul vs MatmulDeviceOperation Comparison")
    logger.info("=" * 100)

    for name, M, K, N in TEST_SHAPES:
        logger.info(f"\nTesting {name}: M={M}, K={K}, N={N}")

        # Create tensors
        torch_input = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
        torch_weight = torch.randn((1, 1, K, N), dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat8_b,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_weight = ttnn.from_torch(
            torch_weight,
            dtype=ttnn.bfloat8_b,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Benchmark MinimalMatmul
        try:
            minimal_ms, _ = benchmark_minimal_matmul(device, tt_input, tt_weight, compute_config)
            logger.info(f"  MinimalMatmul: {minimal_ms:.3f} ms")
        except Exception as e:
            logger.error(f"  MinimalMatmul failed: {e}")
            minimal_ms = None

        # Benchmark MatmulDeviceOperation (DRAM sharded weights)
        try:
            matmul_ms, _ = benchmark_matmul_dram_sharded_weights(device, tt_input, tt_weight, M, K, N, compute_config)
            logger.info(f"  MatmulDeviceOp (DRAM-shrd wt): {matmul_ms:.3f} ms")
        except Exception as e:
            logger.error(f"  MatmulDeviceOp (DRAM-shrd wt) failed: {e}")
            matmul_ms = None

        # Calculate speedup
        if minimal_ms and matmul_ms:
            speedup = minimal_ms / matmul_ms
            if speedup > 1.0:
                logger.info(f"  >>> MatmulDeviceOp is {speedup:.2f}x FASTER <<<")
            else:
                logger.info(f"  >>> MinimalMatmul is {1/speedup:.2f}x FASTER <<<")
        else:
            speedup = None

        results.append((name, M, K, N, minimal_ms, matmul_ms, speedup))

        tt_input.deallocate(True)
        tt_weight.deallocate(True)

    # Print summary table
    logger.info("\n" + "=" * 110)
    logger.info(
        f"{'Shape':<15} | {'M':>6} | {'K':>6} | {'N':>6} | {'Minimal (ms)':>12} | {'DRAM Shrd (ms)':>14} | {'Winner':>18}"
    )
    logger.info("-" * 110)

    for name, M, K, N, minimal_ms, matmul_ms, speedup in results:
        minimal_str = f"{minimal_ms:.3f}" if minimal_ms else "-"
        matmul_str = f"{matmul_ms:.3f}" if matmul_ms else "-"

        if speedup:
            winner = "DRAM Sharded" if speedup > 1.0 else "MinimalMatmul"
            winner += f" ({max(speedup, 1/speedup):.2f}x)"
        else:
            winner = "-"

        logger.info(f"{name:<15} | {M:>6} | {K:>6} | {N:>6} | {minimal_str:>12} | {matmul_str:>14} | {winner:>18}")

    logger.info("=" * 100)

    ttnn.close_device(device)
    logger.info("Done!")


if __name__ == "__main__":
    main()
