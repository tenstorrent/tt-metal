# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SFPU Helpers Performance Analysis: Raw LLK vs Helper API

Comprehensive benchmark matrix:
  - 6 workloads (simple → complex)
  - 4 tile counts (8, 32, 64, 256)
  - Raw LLK baseline vs CRTP helper wrapper

Workloads ordered by complexity:
  1. exp           — single unary op, lightweight
  2. sigmoid       — single unary op, moderate
  3. gelu          — single unary op, compute-heavy
  4. exp+recip     — 2-op chain (stride=1)
  5. tanhshrink    — double load + tanh + binary sub (stride=2, 2 compute ops)
  6. hardswish     — double load + hardsigmoid + binary mul (stride=2, 2 compute ops)
  7. mish          — double load + exp + log1p + tanh + binary mul (stride=2, 4 compute ops)

Use `pytest -s` to see the full results table.
"""

import time
import pytest
import torch
import ttnn
from loguru import logger
from models.common.utility_functions import skip_for_blackhole

pytestmark = pytest.mark.use_module_device

KERNEL_DIR = "tests/ttnn/unit_tests/operations/debug/kernels"

# Tile counts to benchmark: small → large (powers of 2)
TILE_COUNTS = [8, 32, 128, 512, 2048, 8192, 32768]

# Workload definitions: (name, description, raw_kernel, helper_kernel, chain_info)
WORKLOADS = [
    ("exp", "1 unary op (light)", "perf_raw_exp", "perf_helper_exp", "Load→Exp"),
    ("sigmoid", "1 unary op (moderate)", "perf_raw_sigmoid", "perf_helper_sigmoid", "Load→Sigmoid"),
    ("gelu", "1 unary op (heavy)", "perf_raw_gelu", "perf_helper_gelu", "Load→Gelu"),
    ("exp+recip", "2-op chain, stride=1", "perf_raw_chain", "perf_helper_chain", "Load→Exp→Recip"),
    ("tanhshrink", "2x load + 2 ops, stride=2", "perf_raw_tanhshrink", "perf_helper_tanhshrink", "Load×2→Tanh→Sub"),
    ("hardswish", "2x load + 2 ops, stride=2", "perf_raw_hardswish", "perf_helper_hardswish", "Load×2→Hardsigmoid→Mul"),
    ("mish", "2x load + 4 ops, stride=2", "perf_raw_mish", "perf_helper_mish", "Load×2→Exp→Log1p→Tanh→Mul"),
]

OVERHEAD_THRESHOLD = 10.0  # percent — allow some measurement noise


def get_page_size(dtype):
    if dtype == ttnn.bfloat16:
        return 2 * 1024
    elif dtype == ttnn.float32:
        return 4 * 1024
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def setup_single_core(device):
    core = ttnn.CoreCoord(0, 0)
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def run_perf_kernel(device, compute_kernel_path, num_tiles, warmup_runs=3, timed_runs=7):
    """Run a compute kernel, return (avg_us, min_us, all_runs_us)."""
    dtype = ttnn.bfloat16
    shape = [1, 1, 32, num_tiles * 32]
    core_grid = setup_single_core(device)
    dram_memory_config = ttnn.DRAM_MEMORY_CONFIG
    page_size = get_page_size(dtype)

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        dram_memory_config,
    )
    io_tensors = [input_tensor, output_tensor]

    in_cb = 0
    out_cb = 16
    in_cb_desc = ttnn.CBDescriptor(
        total_size=2 * page_size,
        core_ranges=core_grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=in_cb, data_format=dtype, page_size=page_size)],
    )
    out_cb_desc = ttnn.CBDescriptor(
        total_size=2 * page_size,
        core_ranges=core_grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=out_cb, data_format=dtype, page_size=page_size)],
    )

    reader_compile_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    writer_compile_args = [out_cb]
    writer_compile_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    compute_compile_args = [1, num_tiles]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [input_tensor.buffer_address(), num_tiles, 0]
    writer_rt[0][0] = [output_tensor.buffer_address(), num_tiles, 0]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_compile_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_compile_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=compute_kernel_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compute_compile_args,
        defines=[],
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    program_desc = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[in_cb_desc, out_cb_desc],
    )

    # Warmup
    for _ in range(warmup_runs):
        ttnn.generic_op(io_tensors, program_desc)
    ttnn.synchronize_device(device)

    # Timed runs
    durations = []
    for _ in range(timed_runs):
        ttnn.synchronize_device(device)
        start = time.perf_counter_ns()
        ttnn.generic_op(io_tensors, program_desc)
        ttnn.synchronize_device(device)
        end = time.perf_counter_ns()
        durations.append((end - start) / 1000.0)

    # Drop highest and lowest for stability
    if len(durations) > 3:
        durations.sort()
        durations = durations[1:-1]

    avg_us = sum(durations) / len(durations)
    min_us = min(durations)
    return avg_us, min_us, durations


@skip_for_blackhole("Not tested / built for Blackhole")
def test_sfpu_perf_analysis(device):
    """
    Comprehensive perf analysis: all workloads x all tile counts.
    Prints a formatted table for use in documentation and future validation.
    """
    results = []

    for wl_name, wl_desc, raw_kern, helper_kern, chain_info in WORKLOADS:
        for num_tiles in TILE_COUNTS:
            raw_avg, raw_min, _ = run_perf_kernel(device, f"{KERNEL_DIR}/compute/{raw_kern}.cpp", num_tiles)
            helper_avg, helper_min, _ = run_perf_kernel(device, f"{KERNEL_DIR}/compute/{helper_kern}.cpp", num_tiles)

            overhead_pct = ((helper_min - raw_min) / raw_min) * 100.0 if raw_min > 0 else 0.0

            results.append(
                {
                    "workload": wl_name,
                    "desc": wl_desc,
                    "chain": chain_info,
                    "tiles": num_tiles,
                    "raw_avg": raw_avg,
                    "raw_min": raw_min,
                    "helper_avg": helper_avg,
                    "helper_min": helper_min,
                    "overhead_pct": overhead_pct,
                }
            )

    # Print formatted table
    logger.info("")
    logger.info("=" * 100)
    logger.info("SFPU HELPERS PERFORMANCE ANALYSIS — Raw LLK vs Helper API")
    logger.info("=" * 100)
    logger.info("")

    header = f"{'Workload':<14} {'Chain':<30} {'Tiles':>5}  {'Raw min':>8} {'Hlpr min':>8} {'Overhead':>8}"
    logger.info(header)
    logger.info("-" * len(header))

    worst_overhead = 0.0
    for r in results:
        line = (
            f"{r['workload']:<14} {r['chain']:<30} {r['tiles']:>5}  "
            f"{r['raw_min']:>7.1f}us {r['helper_min']:>7.1f}us {r['overhead_pct']:>+7.1f}%"
        )
        logger.info(line)
        worst_overhead = max(worst_overhead, r["overhead_pct"])

    logger.info("-" * len(header))
    logger.info(f"Worst overhead: {worst_overhead:+.1f}%  (threshold: {OVERHEAD_THRESHOLD}%)")
    logger.info("")

    # Group summary by workload
    logger.info("Summary by workload (min across tile counts):")
    workload_names_seen = []
    for wl_name, wl_desc, _, _, chain_info in WORKLOADS:
        if wl_name in workload_names_seen:
            continue
        workload_names_seen.append(wl_name)
        wl_results = [r for r in results if r["workload"] == wl_name]
        overheads = [r["overhead_pct"] for r in wl_results]
        avg_overhead = sum(overheads) / len(overheads)
        max_overhead = max(overheads)
        min_overhead = min(overheads)
        logger.info(
            f"  {wl_name:<14} ({wl_desc}): "
            f"avg={avg_overhead:+.1f}%  min={min_overhead:+.1f}%  max={max_overhead:+.1f}%"
        )

    logger.info("")
    logger.info("=" * 100)

    # Assert worst case is within threshold
    assert (
        worst_overhead < OVERHEAD_THRESHOLD
    ), f"Worst overhead {worst_overhead:.1f}% exceeds {OVERHEAD_THRESHOLD}% threshold"
