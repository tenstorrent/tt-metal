# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Unified device performance test for all GPT-OSS fused ops.

This test runs device performance measurements for all fused op unit tests
and collects results into a single benchmark file.

Usage:
    # Run all fused op device perf tests and save unified benchmark:
    pytest test_gpt_oss_all_fused_ops_device_perf.py::test_all_fused_ops_device_perf -v

    # Run a subset of ops:
    pytest test_gpt_oss_all_fused_ops_device_perf.py::test_all_fused_ops_device_perf -v \
        --fused-ops router,experts_mlp

    # Run individual op device perf (still available):
    pytest test_gpt_oss_experts.py::test_gpt_oss_experts_device_perf -v

The unified test:
1. Runs each sub-op's device perf test sequentially
2. Collects all metrics (kernel duration, op-to-op latency)
3. Saves all results to a single BenchmarkData pickle file
4. Prints a summary table of all results

Individual op tests remain runnable standalone for debugging specific ops.
"""

import json
import os
from datetime import datetime

import pytest
from loguru import logger

from models.demos.gpt_oss.tests.fused_op_unit_tests.fused_op_device_perf_utils import (
    DEVICE_PERF_ITERS,
    DevicePerfResult,
    add_result_to_benchmark,
    get_all_fused_op_names,
    get_fused_op_config,
    run_single_op_device_perf,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler


def pytest_addoption(parser):
    """Add custom command-line options for the test."""
    parser.addoption(
        "--fused-ops",
        action="store",
        default=None,
        help="Comma-separated list of fused ops to test (default: all)",
    )


@pytest.fixture
def selected_fused_ops(request):
    """Get the list of fused ops to test based on command-line option."""
    ops_arg = request.config.getoption("--fused-ops")
    if ops_arg is None:
        return get_all_fused_op_names()
    return [op.strip() for op in ops_arg.split(",")]


def _print_summary_table(results: list[DevicePerfResult]) -> None:
    """Print a formatted summary table of all results."""
    logger.info("\n" + "=" * 100)
    logger.info("GPT-OSS FUSED OPS DEVICE PERFORMANCE SUMMARY")
    logger.info("=" * 100)

    # Table header
    header = f"{'Op Name':<30} {'Batch':<8} {'Seq':<6} {'Kernel (µs)':<15} {'Op-to-Op (µs)':<15} {'Avg Kernel':<12} {'Avg O2O':<12}"
    logger.info(header)
    logger.info("-" * 100)

    total_kernel = 0.0
    total_op_to_op = 0.0

    for r in results:
        row = f"{r.op_name:<30} {r.batch_size:<8} {r.seq_len:<6} {r.total_kernel_us:<15.2f} {r.total_op_to_op_us:<15.2f} {r.avg_kernel_us:<12.2f} {r.avg_op_to_op_us:<12.2f}"
        logger.info(row)
        total_kernel += r.avg_kernel_us
        total_op_to_op += r.avg_op_to_op_us

    logger.info("-" * 100)
    logger.info(
        f"{'TOTAL (avg per iter)':<30} {'':<8} {'':<6} {'':<15} {'':<15} {total_kernel:<12.2f} {total_op_to_op:<12.2f}"
    )
    logger.info("=" * 100)

    # Print op-level breakdown for each result
    logger.info("\nDetailed per-op breakdown:")
    for r in results:
        logger.info(f"\n{r.op_name}:")
        for op_code, stats in sorted(r.op_stats.items()):
            avg_kernel_ns = stats["avg_kernel_duration_ns"]
            avg_op_to_op_ns = stats["avg_op_to_op_latency_ns"]
            call_count = stats["call_count"]
            logger.info(
                f"  {op_code}: kernel={avg_kernel_ns/1000:.2f}µs, op2op={avg_op_to_op_ns/1000:.2f}µs, calls={call_count}"
            )


@pytest.mark.timeout(900)  # Allow 15 minutes for all 5 sub-tests with Tracy profiling
@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
    ],
)
def test_all_fused_ops_device_perf(mode: str, seq_len: int):
    """Run device performance tests for all GPT-OSS fused ops.

    This test:
    1. Iterates through all registered fused ops
    2. Runs each op's device perf test with Tracy profiler
    3. Collects metrics (total/avg kernel duration, op-to-op latency)
    4. Saves all results to a single BenchmarkData file
    5. Prints a summary table

    Args:
        mode: "decode" or "prefill"
        seq_len: Sequence length (1 for decode)
    """
    assert mode == "decode", "Currently only decode mode is supported"
    assert seq_len == 1, "Decode mode always has seq_len=1"

    # Get all fused op names
    fused_ops = get_all_fused_op_names()
    logger.info(f"Running device perf tests for {len(fused_ops)} fused ops: {fused_ops}")

    # Create profiler and benchmark data for unified results
    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    perf_profiler.start("run")

    results: list[DevicePerfResult] = []
    failed_ops: list[str] = []

    for op_name in fused_ops:
        config = get_fused_op_config(op_name)
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {op_name}")
        logger.info(f"Description: {config['description']}")
        logger.info(f"{'='*60}")

        try:
            perf_profiler.start(f"gpt_oss_{op_name}_device_perf_{mode}_seq{seq_len}")

            result = run_single_op_device_perf(
                op_name=op_name,
                test_path=config["test_path"],
                test_function=config["test_function"],
                env_var=config["env_var"],
                mode=mode,
                seq_len=seq_len,
                batch_size=config["batch_size"],
                subdir="gpt_oss_all_fused_ops_device_perf",
                use_trace=config["use_trace"],
            )

            perf_profiler.end(f"gpt_oss_{op_name}_device_perf_{mode}_seq{seq_len}")

            # Add to benchmark data
            add_result_to_benchmark(benchmark_data, perf_profiler, result)
            results.append(result)

            logger.info(f"✓ {op_name} completed successfully")

        except Exception as e:
            logger.error(f"✗ {op_name} failed: {e}")
            failed_ops.append(op_name)
            # Continue with other ops

    perf_profiler.end("run")

    # Print summary
    if results:
        _print_summary_table(results)

    # Save unified benchmark data
    # Use max batch size from results for the unified file
    max_batch_size = max(r.batch_size for r in results) if results else 128
    benchmark_data.save_partial_run_json(
        perf_profiler,
        run_type="gpt_oss_all_fused_ops_device_perf",
        ml_model_name="gpt-oss",
        batch_size=max_batch_size,
        input_sequence_length=seq_len,
    )

    # Also save a JSON summary for easy inspection
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "seq_len": seq_len,
        "num_iterations": DEVICE_PERF_ITERS,
        "results": [
            {
                "op_name": r.op_name,
                "batch_size": r.batch_size,
                "seq_len": r.seq_len,
                "total_kernel_us": r.total_kernel_us,
                "total_op_to_op_us": r.total_op_to_op_us,
                "avg_kernel_us": r.avg_kernel_us,
                "avg_op_to_op_us": r.avg_op_to_op_us,
            }
            for r in results
        ],
        "failed_ops": failed_ops,
    }

    summary_path = f"generated/benchmark_data/gpt_oss_fused_ops_summary_{mode}_seq{seq_len}.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to: {summary_path}")

    # Assert all ops passed
    assert not failed_ops, f"The following ops failed: {failed_ops}"
    assert len(results) == len(fused_ops), f"Expected {len(fused_ops)} results, got {len(results)}"

    logger.info(f"\n✓ All {len(results)} fused ops device perf tests completed successfully!")


@pytest.mark.parametrize("op_name", get_all_fused_op_names())
@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 1),
    ],
)
def test_single_fused_op_device_perf(op_name: str, mode: str, seq_len: int):
    """Run device performance test for a single fused op.

    This is a parametrized test that allows running individual ops
    while still collecting results in the unified format.

    Args:
        op_name: Name of the fused op to test
        mode: "decode" or "prefill"
        seq_len: Sequence length
    """
    assert mode == "decode", "Currently only decode mode is supported"

    config = get_fused_op_config(op_name)

    perf_profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    perf_profiler.start("run")
    perf_profiler.start(f"gpt_oss_{op_name}_device_perf_{mode}_seq{seq_len}")

    result = run_single_op_device_perf(
        op_name=op_name,
        test_path=config["test_path"],
        test_function=config["test_function"],
        env_var=config["env_var"],
        mode=mode,
        seq_len=seq_len,
        batch_size=config["batch_size"],
        subdir=f"gpt_oss_{op_name}_device_perf",
        use_trace=config["use_trace"],
    )

    perf_profiler.end(f"gpt_oss_{op_name}_device_perf_{mode}_seq{seq_len}")
    perf_profiler.end("run")

    # Add to benchmark data
    add_result_to_benchmark(benchmark_data, perf_profiler, result)

    # Save individual op benchmark
    benchmark_data.save_partial_run_json(
        perf_profiler,
        run_type=f"gpt_oss_{op_name}_device_perf",
        ml_model_name="gpt-oss",
        batch_size=result.batch_size,
        input_sequence_length=seq_len,
    )

    # Log results
    logger.info(f"\n{op_name} Device Performance Results:")
    logger.info(f"  Total kernel duration: {result.total_kernel_us:.2f} µs")
    logger.info(f"  Total op-to-op latency: {result.total_op_to_op_us:.2f} µs")
    logger.info(f"  Avg kernel duration: {result.avg_kernel_us:.2f} µs")
    logger.info(f"  Avg op-to-op latency: {result.avg_op_to_op_us:.2f} µs")


if __name__ == "__main__":
    """Allow running directly with python for quick testing."""
    import sys

    # Run all fused ops device perf test
    pytest.main([__file__, "-v", "-k", "test_all_fused_ops_device_perf", "-s"] + sys.argv[1:])
