#!/usr/bin/env python3
"""
Small subset GEMM benchmark runner for GCC vs LLVM comparison
Takes a small subset of configurations and reduced iterations
"""
import os
import sys
import csv
from pathlib import Path

# Add the repo to path
sys.path.insert(0, str(Path(__file__).parent))

import ttnn
import torch
from loguru import logger

# Small subset of configurations as suggested by Borys
# Reduced from full set - just a few key configurations
SMALL_MATMUL_SHAPES = [
    (64, 64, 64, True, True, 1, 1, 1),  # Very small, sharded
    (128, 128, 128, True, True, 1, 1, 1),  # Small, sharded
    (256, 256, 256, True, True, 1, 1, 1),  # Medium, sharded
    (512, 512, 512, False, False, 1, 2, 2),  # Larger, DRAM
    (1024, 1024, 1024, False, False, 2, 4, 4),  # Large, DRAM
]

# Small subset of dtype/fidelity configs
SMALL_MATMUL_CONFIGS = [
    (ttnn.bfloat16, ttnn.MathFidelity.HiFi4, False),  # Standard precision
    (ttnn.bfloat16, ttnn.MathFidelity.HiFi2, False),  # Standard precision, lower fidelity
    (ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, False),  # Lower precision
    (ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, False),  # Lowest precision
]

# Reduced iterations as suggested
NUM_WARMUP_ITERATIONS = 5  # Warmup iterations
NUM_MEASUREMENT_ITERATIONS = 75  # Measurement iterations for better accuracy


def run_gemm_benchmark(device, compiler_name):
    """Run GEMM benchmark with small subset"""
    TT_METAL_HOME = Path(os.environ.get("TT_METAL_HOME", os.getcwd()))
    ARTIFACTS_DIR = TT_METAL_HOME / "generated"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FILE_NAME = ARTIFACTS_DIR / f"matmul_2d_host_perf_report_{compiler_name}.csv"

    tile_h = 32
    tile_w = 32
    compute_grid_size = device.compute_with_storage_grid_size()
    grid_size = (compute_grid_size.x, compute_grid_size.y)

    logger.info(f"Running GEMM benchmark with {compiler_name}")
    logger.info(f"Grid size: {grid_size}, Warmup: {NUM_WARMUP_ITERATIONS}, Measurements: {NUM_MEASUREMENT_ITERATIONS}")

    results = []

    for dtype, math_fidelity, use_trace in SMALL_MATMUL_CONFIGS:
        for (
            m,
            k,
            n,
            in0_sharded,
            out_sharded,
            in0_block_w_div,
            num_out_blocks_h,
            num_out_blocks_w,
        ) in SMALL_MATMUL_SHAPES:
            try:
                # Scale by grid size
                m_scaled = m * grid_size[1]
                k_scaled = k * grid_size[0]
                n_scaled = n * grid_size[0]

                logger.info(
                    f"Testing M*K*N = {m_scaled}*{k_scaled}*{n_scaled}, dtype={dtype}, fidelity={math_fidelity}"
                )

                in0_shape = [1, 1, m_scaled, k_scaled]
                in1_shape = [1, 1, k_scaled, n_scaled]

                in0_block_w = k_scaled // grid_size[0] // 32 // in0_block_w_div
                per_core_M = m_scaled // grid_size[1] // tile_h
                per_core_N = n_scaled // grid_size[0] // tile_w
                out_block_h = per_core_M // num_out_blocks_h
                out_block_w = per_core_N // num_out_blocks_w

                # Use proper subblock calculation from test_benchmark.py
                SUBBLOCK_HW_CHOICES = [
                    (4, 2),
                    (2, 4),
                    (8, 1),
                    (1, 8),
                    (7, 1),
                    (1, 7),
                    (3, 2),
                    (2, 3),
                    (6, 1),
                    (1, 6),
                    (5, 1),
                    (1, 5),
                    (2, 2),
                    (4, 1),
                    (1, 4),
                    (3, 1),
                    (1, 3),
                    (2, 1),
                    (1, 2),
                    (1, 1),
                ]

                def get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, out_sharded=False):
                    for subblock_hw in SUBBLOCK_HW_CHOICES:
                        out_subblock_h = subblock_hw[0]
                        out_subblock_w = subblock_hw[1]
                        if out_sharded:
                            if n_tiles_per_core % out_subblock_w != 0 or out_subblock_h != 1:
                                continue
                        if m_tiles_per_core % out_subblock_h == 0 and n_tiles_per_core % out_subblock_w == 0:
                            return (out_subblock_h, out_subblock_w)
                    return (1, 1)

                out_subblock_h, out_subblock_w = get_subblock_sizes(out_block_h, out_block_w, out_sharded)

                in0 = torch.ones(in0_shape).bfloat16()
                in1 = torch.randn(in1_shape).bfloat16()

                if in0_sharded:
                    in0_memory_config = ttnn.create_sharded_memory_config(
                        (1, 1, m_scaled, k_scaled),
                        core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
                        strategy=ttnn.ShardStrategy.BLOCK,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    )
                    in0_storage_type = "L1"
                else:
                    in0_memory_config = ttnn.DRAM_MEMORY_CONFIG
                    in0_storage_type = "DRAM"

                in1_storage_type = "DRAM"
                out_storage_type = "L1" if out_sharded else "DRAM"

                in0_t = ttnn.from_torch(
                    in0,
                    tile=ttnn.Tile((tile_h, 32)),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=in0_memory_config,
                )

                in1_t = ttnn.from_torch(
                    in1,
                    tile=ttnn.Tile((32, tile_w)),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=grid_size,
                    in0_block_w=in0_block_w,
                    out_subblock_h=out_subblock_h,
                    out_subblock_w=out_subblock_w,
                    out_block_h=out_block_h,
                    out_block_w=out_block_w,
                    per_core_M=per_core_M,
                    per_core_N=per_core_N,
                    transpose_mcast=False,
                    fused_activation=None,
                )

                compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=math_fidelity,
                    math_approx_mode=True,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                    throttle_level=ttnn.ThrottleLevel.NO_THROTTLE,
                )

                if out_sharded:
                    out_mem_config = ttnn.MemoryConfig(
                        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                        buffer_type=ttnn.BufferType.L1,
                    )
                else:
                    out_mem_config = ttnn.DRAM_MEMORY_CONFIG

                output_tile = ttnn.Tile([tile_h, 32]) if tile_h <= 16 else ttnn.Tile([tile_h, tile_w])

                # Warmup
                for _ in range(NUM_WARMUP_ITERATIONS):
                    output_t = ttnn.matmul(
                        in0_t,
                        in1_t,
                        program_config=program_config,
                        memory_config=out_mem_config,
                        dtype=dtype,
                        compute_kernel_config=compute_kernel_config,
                        output_tile=output_tile,
                    )

                # Synchronize after warmup
                ttnn.synchronize_device(device)

                # Measurements
                import time

                times = []
                for _ in range(NUM_MEASUREMENT_ITERATIONS):
                    start = time.perf_counter_ns()
                    output_t = ttnn.matmul(
                        in0_t,
                        in1_t,
                        program_config=program_config,
                        memory_config=out_mem_config,
                        dtype=dtype,
                        compute_kernel_config=compute_kernel_config,
                        output_tile=output_tile,
                    )
                    ttnn.synchronize_device(device)  # Wait for completion
                    end = time.perf_counter_ns()
                    times.append(end - start)

                avg_time_ns = sum(times) / len(times)

                # Calculate TFLOPS: 2 * M * K * N operations per second
                # Formula: (2 * M * K * N) / (time_in_seconds) / 1e12
                # time_in_seconds = avg_time_ns / 1e9
                # So: (2 * M * K * N) / (avg_time_ns / 1e9) / 1e12 = (2 * M * K * N * 1e9) / (avg_time_ns * 1e12)
                # = (2 * M * K * N) / (avg_time_ns * 1e3)
                total_ops = 2 * m_scaled * k_scaled * n_scaled
                tflops = total_ops / (avg_time_ns * 1e3)  # Convert nanoseconds to seconds, then to TFLOPS

                results.append(
                    {
                        "m": m_scaled,
                        "k": k_scaled,
                        "n": n_scaled,
                        "use_trace": use_trace,
                        "grid_size": f"({grid_size[0]}, {grid_size[1]})",
                        "in0_sharded": in0_sharded,
                        "out_sharded": out_sharded,
                        "in0_storage_type": in0_storage_type,
                        "in1_storage_type": in1_storage_type,
                        "out_storage_type": out_storage_type,
                        "dtype": str(dtype),
                        "math_fidelity": str(math_fidelity),
                        "inference_time_avg_ns": avg_time_ns,
                        "TFLOPs": tflops,
                    }
                )

                logger.info(f"  Result: {avg_time_ns/1e6:.2f} ms, {tflops:.2f} TFLOPS")

            except Exception as e:
                logger.error(f"  Failed: {e}")
                continue

    # Write results to CSV
    with open(FILE_NAME, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            logger.info(f"Results written to {FILE_NAME}")
        else:
            logger.warning("No results to write")

    return FILE_NAME


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", choices=["gcc", "llvm"], required=True)
    args = parser.parse_args()

    # Enable device profiler as per manager's requirement
    os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
    os.environ["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"

    # Set compiler
    if args.compiler == "llvm":
        os.environ["TT_METAL_KERNEL_COMPILER"] = "llvm"
    else:
        os.environ.pop("TT_METAL_KERNEL_COMPILER", None)

    # Get device
    device = ttnn.open_device(device_id=0)
    try:
        run_gemm_benchmark(device, args.compiler)
    finally:
        ttnn.close_device(device)
