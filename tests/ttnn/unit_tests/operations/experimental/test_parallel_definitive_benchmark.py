# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Definitive benchmark test for ttnn.parallel performance.

This test provides a fair, controlled comparison between:
1. Running N operations sequentially (N separate dispatches)
2. Running N operations in parallel (1 parallel dispatch with N branches)

Run with profiler:
    TT_METAL_DEVICE_PROFILER=1 python -m tracy -r -m pytest <this_file> -v -s
"""

import pytest
import torch
import ttnn
import time


def torch_layer_norm(x, gamma=None, beta=None, eps=1e-5):
    """PyTorch reference implementation of LayerNorm."""
    mean = x.mean(-1, keepdim=True)
    variance = x.var(-1, keepdim=True, unbiased=False)
    x_normed = (x - mean) / torch.sqrt(variance + eps)
    if gamma is not None:
        x_normed = x_normed * gamma
    if beta is not None:
        x_normed = x_normed + beta
    return x_normed


def assert_with_pcc(expected, actual, pcc=0.99):
    """Assert that actual matches expected within PCC tolerance."""
    expected_flat = expected.flatten().float()
    actual_flat = actual.flatten().float()
    correlation = torch.corrcoef(torch.stack([expected_flat, actual_flat]))[0, 1]
    assert correlation >= pcc, f"PCC {correlation} < {pcc}"


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name, num_ops, total_time_ns, times_per_trial):
        self.name = name
        self.num_ops = num_ops
        self.total_time_ns = total_time_ns
        self.times_per_trial = times_per_trial
        self.avg_time_ns = sum(times_per_trial) / len(times_per_trial)
        self.min_time_ns = min(times_per_trial)
        self.max_time_ns = max(times_per_trial)

    def __str__(self):
        return (
            f"{self.name}: avg={self.avg_time_ns/1000:.1f}us, "
            f"min={self.min_time_ns/1000:.1f}us, max={self.max_time_ns/1000:.1f}us"
        )


def benchmark_sequential_layernorms(device, inputs, weights, biases, num_trials=10, warmup=3):
    """
    Benchmark: Run N LayerNorm operations sequentially.
    Each operation is a separate dispatch to the device.
    """
    n = len(inputs)

    # Warmup
    for _ in range(warmup):
        for i in range(n):
            _ = ttnn.layer_norm(inputs[i], epsilon=1e-5, weight=weights[i], bias=biases[i])
        ttnn.synchronize_device(device)

    # Benchmark
    times = []
    for trial in range(num_trials):
        start = time.perf_counter_ns()
        for i in range(n):
            _ = ttnn.layer_norm(inputs[i], epsilon=1e-5, weight=weights[i], bias=biases[i])
        ttnn.synchronize_device(device)
        end = time.perf_counter_ns()
        times.append(end - start)

    return BenchmarkResult(f"Sequential ({n} ops)", n, sum(times), times)


def benchmark_parallel_layernorms(device, inputs, weights, biases, core_ranges, num_trials=10, warmup=3):
    """
    Benchmark: Run N LayerNorm operations in parallel.
    All operations are dispatched together in a single parallel call.
    """
    n = len(inputs)

    # Warmup
    for _ in range(warmup):
        branches = []
        for i in range(n):
            branch = ttnn.parallel.branch(
                ttnn.layer_norm, inputs[i], cores=core_ranges[i], epsilon=1e-5, weight=weights[i], bias=biases[i]
            )
            branches.append(branch)
        _ = ttnn.parallel(branches)
        ttnn.synchronize_device(device)

    # Benchmark
    times = []
    for trial in range(num_trials):
        branches = []
        for i in range(n):
            branch = ttnn.parallel.branch(
                ttnn.layer_norm, inputs[i], cores=core_ranges[i], epsilon=1e-5, weight=weights[i], bias=biases[i]
            )
            branches.append(branch)

        start = time.perf_counter_ns()
        _ = ttnn.parallel(branches)
        ttnn.synchronize_device(device)
        end = time.perf_counter_ns()
        times.append(end - start)

    return BenchmarkResult(f"Parallel ({n} branches)", n, sum(times), times)


def benchmark_single_layernorm(device, input_tensor, weight, bias, num_trials=10, warmup=3):
    """
    Benchmark: Run a single LayerNorm operation.
    This establishes the baseline for one dispatch.
    """
    # Warmup
    for _ in range(warmup):
        _ = ttnn.layer_norm(input_tensor, epsilon=1e-5, weight=weight, bias=bias)
        ttnn.synchronize_device(device)

    # Benchmark
    times = []
    for trial in range(num_trials):
        start = time.perf_counter_ns()
        _ = ttnn.layer_norm(input_tensor, epsilon=1e-5, weight=weight, bias=bias)
        ttnn.synchronize_device(device)
        end = time.perf_counter_ns()
        times.append(end - start)

    return BenchmarkResult("Single LayerNorm", 1, sum(times), times)


@pytest.mark.parametrize("num_branches", [2, 4])
def test_parallel_benchmark_definitive(device, num_branches):
    """
    Definitive benchmark comparing sequential vs parallel execution.

    This test:
    1. Runs a single LayerNorm to establish baseline dispatch overhead
    2. Runs N LayerNorms sequentially (N separate dispatches)
    3. Runs N LayerNorms in parallel (1 dispatch with N branches)
    4. Compares total execution time and calculates speedup

    The parallel version should be faster because:
    - Only 1 dispatch to device instead of N
    - All branches execute simultaneously on disjoint cores
    """
    torch.manual_seed(42)

    # Configuration
    batch_size, h, w = 1, 64, 128
    num_trials = 10
    warmup = 5

    print(f"\n{'='*80}")
    print(f"DEFINITIVE PARALLEL BENCHMARK - {num_branches} branches")
    print(f"{'='*80}")
    print(f"Tensor shape: {batch_size}x{h}x{w}")
    print(f"Trials: {num_trials}, Warmup: {warmup}")
    print()

    # Create torch tensors
    torch_inputs = [torch.rand((batch_size, h, w), dtype=torch.bfloat16) for _ in range(num_branches)]
    torch_weights = [torch.rand((w,), dtype=torch.bfloat16) for _ in range(num_branches)]
    torch_biases = [torch.rand((w,), dtype=torch.bfloat16) for _ in range(num_branches)]

    # Create core ranges - divide the 8x8 grid evenly among branches
    cores_per_branch = 8 // num_branches  # cores in x-dimension per branch
    core_ranges = []
    for i in range(num_branches):
        start_x = i * cores_per_branch
        end_x = start_x + cores_per_branch - 1
        core_range = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(start_x, 0), ttnn.CoreCoord(end_x, 7))])
        core_ranges.append(core_range)

    # Move tensors to device
    inputs = [ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT) for t in torch_inputs]
    weights = [ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT) for t in torch_weights]
    biases = [ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT) for t in torch_biases]

    # Run benchmarks
    print("Running benchmarks...")
    print("-" * 60)

    # 1. Single operation baseline
    single_result = benchmark_single_layernorm(device, inputs[0], weights[0], biases[0], num_trials, warmup)
    print(f"  {single_result}")

    # 2. Sequential execution
    seq_result = benchmark_sequential_layernorms(device, inputs, weights, biases, num_trials, warmup)
    print(f"  {seq_result}")

    # 3. Parallel execution
    par_result = benchmark_parallel_layernorms(device, inputs, weights, biases, core_ranges, num_trials, warmup)
    print(f"  {par_result}")

    # Calculate metrics
    single_time = single_result.avg_time_ns
    seq_time = seq_result.avg_time_ns
    par_time = par_result.avg_time_ns

    # Theoretical sequential time (N × single op time)
    theoretical_seq_time = single_time * num_branches

    # Dispatch overhead analysis
    actual_dispatch_overhead = (
        (seq_time - theoretical_seq_time) / num_branches if seq_time > theoretical_seq_time else 0
    )
    parallel_overhead = par_time - single_time  # Extra time for parallel vs single

    # Speedup calculations
    speedup_vs_sequential = seq_time / par_time
    speedup_vs_theoretical = theoretical_seq_time / par_time

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Single LayerNorm:           {single_time/1000:>10.1f} us")
    print(
        f"  Theoretical {num_branches}x sequential: {theoretical_seq_time/1000:>10.1f} us (= {num_branches} × single)"
    )
    print(f"  Actual sequential:          {seq_time/1000:>10.1f} us")
    print(f"  Parallel ({num_branches} branches):      {par_time/1000:>10.1f} us")
    print()
    print("OVERHEAD ANALYSIS")
    print("-" * 60)
    print(f"  Per-dispatch overhead:      {actual_dispatch_overhead/1000:>10.1f} us")
    print(f"  Parallel overhead vs single:{parallel_overhead/1000:>10.1f} us")
    print()
    print("SPEEDUP")
    print("-" * 60)
    print(f"  Parallel vs Sequential:     {speedup_vs_sequential:>10.2f}x")
    print(f"  Parallel vs Theoretical:    {speedup_vs_theoretical:>10.2f}x")
    print()

    # Time saved
    time_saved = seq_time - par_time
    print(f"TIME SAVED: {time_saved/1000:.1f} us ({time_saved/seq_time*100:.1f}% reduction)")
    print("=" * 60)

    # Verify correctness
    print("\nVerifying correctness...")
    branches = []
    for i in range(num_branches):
        branch = ttnn.parallel.branch(
            ttnn.layer_norm, inputs[i], cores=core_ranges[i], epsilon=1e-5, weight=weights[i], bias=biases[i]
        )
        branches.append(branch)

    results = ttnn.parallel(branches)

    for i in range(num_branches):
        expected = torch_layer_norm(torch_inputs[i], torch_weights[i], torch_biases[i])
        actual = ttnn.to_torch(ttnn.from_device(results[i][0]))
        assert_with_pcc(expected, actual, 0.998)

    print(f"✓ All {num_branches} branch outputs verified correct!")

    # Assert performance benefit
    assert speedup_vs_sequential > 1.0, f"Parallel should be faster than sequential! Got {speedup_vs_sequential:.2f}x"
    print(f"\n✓ BENCHMARK PASSED: Parallel is {speedup_vs_sequential:.2f}x faster than sequential")


@pytest.mark.parametrize("num_branches", [2])
def test_parallel_dispatch_overhead_analysis(device, num_branches):
    """
    Detailed analysis of dispatch overhead.

    Measures:
    1. Time for empty sync (baseline device communication)
    2. Time for single op + sync
    3. Time for N ops + sync (sequential)
    4. Time for parallel(N branches) + sync

    This helps isolate where time is spent.
    """
    torch.manual_seed(42)

    batch_size, h, w = 1, 64, 128
    num_trials = 20
    warmup = 10

    print(f"\n{'='*80}")
    print("DISPATCH OVERHEAD ANALYSIS")
    print(f"{'='*80}")

    # Create tensors
    torch_inputs = [torch.rand((batch_size, h, w), dtype=torch.bfloat16) for _ in range(num_branches)]
    torch_weights = [torch.rand((w,), dtype=torch.bfloat16) for _ in range(num_branches)]
    torch_biases = [torch.rand((w,), dtype=torch.bfloat16) for _ in range(num_branches)]

    inputs = [ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT) for t in torch_inputs]
    weights = [ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT) for t in torch_weights]
    biases = [ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT) for t in torch_biases]

    cores_per_branch = 8 // num_branches
    core_ranges = []
    for i in range(num_branches):
        start_x = i * cores_per_branch
        end_x = start_x + cores_per_branch - 1
        core_range = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(start_x, 0), ttnn.CoreCoord(end_x, 7))])
        core_ranges.append(core_range)

    # Warmup everything
    for _ in range(warmup):
        ttnn.synchronize_device(device)
        _ = ttnn.layer_norm(inputs[0], epsilon=1e-5, weight=weights[0], bias=biases[0])
        ttnn.synchronize_device(device)
        branches = [
            ttnn.parallel.branch(
                ttnn.layer_norm, inputs[i], cores=core_ranges[i], epsilon=1e-5, weight=weights[i], bias=biases[i]
            )
            for i in range(num_branches)
        ]
        _ = ttnn.parallel(branches)
        ttnn.synchronize_device(device)

    # 1. Empty sync baseline
    sync_times = []
    for _ in range(num_trials):
        start = time.perf_counter_ns()
        ttnn.synchronize_device(device)
        end = time.perf_counter_ns()
        sync_times.append(end - start)
    avg_sync = sum(sync_times) / len(sync_times)

    # 2. Single op + sync
    single_times = []
    for _ in range(num_trials):
        start = time.perf_counter_ns()
        _ = ttnn.layer_norm(inputs[0], epsilon=1e-5, weight=weights[0], bias=biases[0])
        ttnn.synchronize_device(device)
        end = time.perf_counter_ns()
        single_times.append(end - start)
    avg_single = sum(single_times) / len(single_times)

    # 3. N sequential ops + sync
    seq_times = []
    for _ in range(num_trials):
        start = time.perf_counter_ns()
        for i in range(num_branches):
            _ = ttnn.layer_norm(inputs[i], epsilon=1e-5, weight=weights[i], bias=biases[i])
        ttnn.synchronize_device(device)
        end = time.perf_counter_ns()
        seq_times.append(end - start)
    avg_seq = sum(seq_times) / len(seq_times)

    # 4. Parallel + sync
    par_times = []
    for _ in range(num_trials):
        branches = [
            ttnn.parallel.branch(
                ttnn.layer_norm, inputs[i], cores=core_ranges[i], epsilon=1e-5, weight=weights[i], bias=biases[i]
            )
            for i in range(num_branches)
        ]
        start = time.perf_counter_ns()
        _ = ttnn.parallel(branches)
        ttnn.synchronize_device(device)
        end = time.perf_counter_ns()
        par_times.append(end - start)
    avg_par = sum(par_times) / len(par_times)

    # Analysis
    single_compute = avg_single - avg_sync  # Single op compute time
    seq_compute = avg_seq - avg_sync  # All sequential ops compute time
    par_compute = avg_par - avg_sync  # Parallel compute time

    per_op_time = seq_compute / num_branches
    parallel_overhead = par_compute - single_compute  # Extra time for parallel vs single

    print(f"Raw timings (avg of {num_trials} trials):")
    print(f"  Sync only:                  {avg_sync/1000:>8.1f} us")
    print(f"  Single op + sync:           {avg_single/1000:>8.1f} us")
    print(f"  {num_branches} sequential ops + sync:    {avg_seq/1000:>8.1f} us")
    print(f"  Parallel({num_branches}) + sync:         {avg_par/1000:>8.1f} us")
    print()
    print("Computed metrics:")
    print(f"  Single op compute time:     {single_compute/1000:>8.1f} us")
    print(f"  Per-op time (sequential):   {per_op_time/1000:>8.1f} us")
    print(f"  Parallel compute time:      {par_compute/1000:>8.1f} us")
    print(f"  Parallel overhead:          {parallel_overhead/1000:>8.1f} us")
    print()
    print(f"Speedup: {avg_seq / avg_par:.2f}x")
    print(f"Time saved: {(avg_seq - avg_par)/1000:.1f} us")

    assert avg_par < avg_seq, "Parallel should be faster than sequential!"
