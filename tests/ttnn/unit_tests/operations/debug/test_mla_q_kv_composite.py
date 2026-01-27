# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Performance benchmark comparing sequential vs parallel RMS norm execution.

This test quantifies the performance gain from running two RMS norm
operations in parallel (merged program) vs sequentially (separate program launches).

Run for host timing only:
    pytest tests/ttnn/unit_tests/operations/debug/test_mla_q_kv_composite.py -v -s

Run with device profiling (Tracy):
    python -m tracy -r -m pytest \
        tests/ttnn/unit_tests/operations/debug/test_mla_q_kv_composite.py -v -s
"""

import pytest
import torch
import ttnn
import time

from tracy import signpost


def torch_rms_norm(x, gamma, eps=1e-5):
    """Reference RMS norm implementation in PyTorch."""
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    x_normed = x.float() / rms
    if gamma is not None:
        x_normed = x_normed * gamma.float()
    return x_normed.to(x.dtype)


def create_tensors(device):
    """
    Create two sets of tensors for q_norm and kv_norm.
    Uses original L1 width sharding configuration:
    - q_norm: L1 width sharded on cores (0,0)-(3,3), shard [32,96], tensor [32,1536]
    - kv_norm: L1 width sharded on cores (5,0)-(6,7), shard [32,32], tensor [32,512]
    """
    torch.manual_seed(42)

    # q_norm: L1 width sharded on cores (0,0) to (3,3) = 16 cores
    q_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    num_q_cores = 16
    q_shard_width = 96
    q_total_width = num_q_cores * q_shard_width  # 1536

    q_shape = (1, 1, 32, q_total_width)
    q_weight_shape = (1, 1, 1, q_total_width)
    torch_q_input = torch.rand(q_shape, dtype=torch.bfloat16)
    torch_q_weight = torch.rand(q_weight_shape, dtype=torch.bfloat16)

    # Create sharded memory config for q input (L1 width sharded)
    q_shard_spec = ttnn.ShardSpec(q_cores, [32, q_shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    q_sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=q_shard_spec,
    )

    q_input = ttnn.from_torch(
        torch_q_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=q_sharded_mem_config,
    )
    # Weights are DRAM interleaved
    q_weight = ttnn.from_torch(
        torch_q_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # kv_norm: L1 width sharded on cores (5,0) to (6,7) = 2x8 = 16 cores
    kv_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 7))])
    num_kv_cores = 16
    kv_shard_width = 32
    kv_total_width = num_kv_cores * kv_shard_width  # 512

    kv_shape = (1, 1, 32, kv_total_width)
    kv_weight_shape = (1, 1, 1, kv_total_width)
    torch_kv_input = torch.rand(kv_shape, dtype=torch.bfloat16)
    torch_kv_weight = torch.rand(kv_weight_shape, dtype=torch.bfloat16)

    # Create sharded memory config for kv input (L1 width sharded)
    kv_shard_spec = ttnn.ShardSpec(kv_cores, [32, kv_shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    kv_sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=kv_shard_spec,
    )

    kv_input = ttnn.from_torch(
        torch_kv_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=kv_sharded_mem_config,
    )
    # Weights are DRAM interleaved
    kv_weight = ttnn.from_torch(
        torch_kv_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return {
        "q": {
            "input": q_input,
            "weight": q_weight,
            "torch_input": torch_q_input,
            "torch_weight": torch_q_weight,
            "cores": q_cores,
            "memory_config": q_sharded_mem_config,
        },
        "kv": {
            "input": kv_input,
            "weight": kv_weight,
            "torch_input": torch_kv_input,
            "torch_weight": torch_kv_weight,
            "cores": kv_cores,
            "memory_config": kv_sharded_mem_config,
        },
    }


def run_sequential(device, tensors, num_iterations=1, use_signpost=False):
    """Run q_norm and kv_norm sequentially using two separate program launches."""
    if use_signpost:
        signpost("sequential_start")

    for _ in range(num_iterations):
        # Run q_norm - launches a program
        q_output = ttnn.rms_norm(
            tensors["q"]["input"],
            epsilon=1e-5,
            weight=tensors["q"]["weight"],
        )

        # Run kv_norm - launches another program
        kv_output = ttnn.rms_norm(
            tensors["kv"]["input"],
            epsilon=1e-5,
            weight=tensors["kv"]["weight"],
        )

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost("sequential_stop")

    return q_output, kv_output


def run_q_norm_only(device, tensors, num_iterations=1, use_signpost=False):
    """Run only q_norm for profiling."""
    if use_signpost:
        signpost("q_norm_start")

    for _ in range(num_iterations):
        q_output = ttnn.rms_norm(
            tensors["q"]["input"],
            epsilon=1e-5,
            weight=tensors["q"]["weight"],
        )

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost("q_norm_stop")

    return q_output


def run_kv_norm_only(device, tensors, num_iterations=1, use_signpost=False):
    """Run only kv_norm for profiling."""
    if use_signpost:
        signpost("kv_norm_start")

    for _ in range(num_iterations):
        kv_output = ttnn.rms_norm(
            tensors["kv"]["input"],
            epsilon=1e-5,
            weight=tensors["kv"]["weight"],
        )

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost("kv_norm_stop")

    return kv_output


def run_parallel_composite(device, tensors, num_iterations=1, use_signpost=False):
    """Run q_norm and kv_norm in parallel using launch_composite with merged descriptors.

    Uses the original L1 width sharded configuration. The sharded layernorm operations
    are merged into a single program and executed via generic_op.

    NOTE: This version creates descriptors/outputs inside the loop every iteration.
    When device.enable_program_cache() is called, the compiled program is still cached
    based on GenericOpDeviceOperation::compute_program_hash, which hashes the descriptor
    structure (kernels, CBs, core ranges, etc). However, descriptor creation and merging
    overhead is paid on every iteration.
    """
    if use_signpost:
        signpost("parallel_composite_start")

    for _ in range(num_iterations):
        # Create program descriptors using the experimental API with original sharded inputs
        # For sharded inputs, core_range_set is optional (validates shard spec is within range)
        q_desc, q_output = ttnn.experimental.programs.rms_norm(
            tensors["q"]["input"],
            epsilon=1e-5,
            weight=tensors["q"]["weight"],
        )

        kv_desc, kv_output = ttnn.experimental.programs.rms_norm(
            tensors["kv"]["input"],
            epsilon=1e-5,
            weight=tensors["kv"]["weight"],
        )

        # Build io_tensors list for all tensors referenced by the programs
        io_tensors = [
            tensors["q"]["input"],
            tensors["q"]["weight"],
            q_output,
            tensors["kv"]["input"],
            tensors["kv"]["weight"],
            kv_output,
        ]

        # Launch composite - merges descriptors and executes via single generic_op call
        ttnn.experimental.launch_composite(
            [(q_desc, q_output), (kv_desc, kv_output)],
            io_tensors,
        )

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost("parallel_composite_stop")

    return q_output, kv_output


def run_parallel_composite_cached(device, tensors, num_iterations=1, use_signpost=False):
    """Run q_norm and kv_norm in parallel using pre-created merged descriptors.

    Uses the original L1 width sharded configuration. The sharded layernorm operations
    are merged into a single program once, then executed repeatedly.

    This version minimizes host overhead by:
    1. Creating descriptors and output tensors ONCE (outside the loop)
    2. Merging descriptors ONCE
    3. Executing the same merged descriptor repeatedly

    With device.enable_program_cache(), GenericOpDeviceOperation::compute_program_hash
    hashes the descriptor structure, so the compiled program is cached and reused on
    subsequent generic_op calls. This version avoids both:
    - Descriptor creation overhead (Python/C++ calls to create descriptors)
    - Descriptor merging overhead (merge_descriptors call)
    """
    # Create program descriptors ONCE (outside the loop)
    q_desc, q_output = ttnn.experimental.programs.rms_norm(
        tensors["q"]["input"],
        epsilon=1e-5,
        weight=tensors["q"]["weight"],
    )

    kv_desc, kv_output = ttnn.experimental.programs.rms_norm(
        tensors["kv"]["input"],
        epsilon=1e-5,
        weight=tensors["kv"]["weight"],
    )

    # Merge descriptors ONCE
    merged_descriptor = ttnn.ProgramDescriptor.merge_descriptors([q_desc, kv_desc])

    # Build io_tensors list ONCE (these tensors are reused)
    io_tensors = [
        tensors["q"]["input"],
        tensors["q"]["weight"],
        q_output,
        tensors["kv"]["input"],
        tensors["kv"]["weight"],
        kv_output,
    ]

    if use_signpost:
        signpost("parallel_composite_cached_start")

    # Execute the cached merged program multiple times
    for _ in range(num_iterations):
        ttnn.generic_op(io_tensors, merged_descriptor)

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost("parallel_composite_cached_stop")

    return q_output, kv_output


def verify_outputs(tensors, q_output, kv_output):
    """Verify outputs against PyTorch reference."""
    # Compute expected outputs
    q_expected = torch_rms_norm(tensors["q"]["torch_input"], tensors["q"]["torch_weight"])
    kv_expected = torch_rms_norm(tensors["kv"]["torch_input"], tensors["kv"]["torch_weight"])

    # Get actual outputs
    q_actual = ttnn.to_torch(q_output)
    kv_actual = ttnn.to_torch(kv_output)

    # Calculate PCC
    def calc_pcc(expected, actual):
        expected_flat = expected.flatten().float()
        actual_flat = actual.flatten().float()
        # Handle edge cases
        if torch.all(expected_flat == 0) or torch.all(actual_flat == 0):
            return 1.0 if torch.allclose(expected_flat, actual_flat, atol=1e-3) else 0.0
        corr = torch.corrcoef(torch.stack([expected_flat, actual_flat]))[0, 1]
        return corr.item() if not torch.isnan(corr) else 0.0

    q_pcc = calc_pcc(q_expected, q_actual)
    kv_pcc = calc_pcc(kv_expected, kv_actual)

    return q_pcc, kv_pcc


@pytest.mark.parametrize("num_iterations", [50])
def test_parallel_rms_performance(device, num_iterations):
    """
    Benchmark sequential vs parallel RMS norm execution.

    Sequential: Two separate ttnn.rms_norm calls (two program launches)
    Parallel: Single merged program via launch_composite (one program launch)

    The parallel version should show reduced host overhead from combining
    two separate program launches into one.

    Program caching is enabled via device.enable_program_cache() which uses
    GenericOpDeviceOperation::compute_program_hash to cache compiled programs.
    """
    print(f"\n{'='*80}")
    print("PARALLEL RMS NORM PERFORMANCE BENCHMARK")
    print(f"{'='*80}")

    # Enable program cache - this uses GenericOpDeviceOperation::compute_program_hash
    # to cache compiled programs based on the ProgramDescriptor structure
    device.enable_program_cache()
    print(f"\nProgram cache enabled (using GenericOpDeviceOperation::compute_program_hash)")

    # Create tensors (both DRAM interleaved for fair comparison)
    tensors = create_tensors(device)

    print(f"\nConfiguration:")
    print(f"  q_norm:  L1 width sharded on cores (0,0)-(3,3), shard [32,96], tensor [32,1536]")
    print(f"  kv_norm: L1 width sharded on cores (5,0)-(6,7), shard [32,32], tensor [32,512]")
    print(f"  Iterations per measurement: {num_iterations}")
    print(f"\nSequential: Two separate program launches")
    print(f"Parallel:   Single merged program via launch_composite")

    # Warmup
    print("\nWarming up...")
    warmup_iters = 5
    for _ in range(warmup_iters):
        _ = run_q_norm_only(device, tensors)
        _ = run_kv_norm_only(device, tensors)
        _ = run_sequential(device, tensors)
    # Warmup for parallel composite (both versions)
    _ = run_parallel_composite(device, tensors)
    _ = run_parallel_composite_cached(device, tensors)

    # Profile q_norm individually
    print("Profiling q_norm...")
    num_trials = 5
    for trial in range(num_trials):
        _ = run_q_norm_only(device, tensors, num_iterations, use_signpost=(trial == 0))

    # Profile kv_norm individually
    print("Profiling kv_norm...")
    for trial in range(num_trials):
        _ = run_kv_norm_only(device, tensors, num_iterations, use_signpost=(trial == 0))

    # Profile sequential (combined)
    print("Profiling sequential (q + kv)...")
    sequential_times = []
    for trial in range(num_trials):
        start = time.perf_counter_ns()
        seq_outputs = run_sequential(device, tensors, num_iterations, use_signpost=(trial == 0))
        end = time.perf_counter_ns()
        sequential_times.append(end - start)

    # Profile parallel composite (merged program - no caching)
    print("Profiling parallel composite (no caching)...")
    parallel_composite_times = []
    for trial in range(num_trials):
        start = time.perf_counter_ns()
        par_composite_outputs = run_parallel_composite(device, tensors, num_iterations, use_signpost=(trial == 0))
        end = time.perf_counter_ns()
        parallel_composite_times.append(end - start)
    print("  ✓ Parallel composite (no cache) succeeded!")

    # Profile parallel composite CACHED (proper program caching)
    print("Profiling parallel composite CACHED...")
    parallel_composite_cached_times = []
    for trial in range(num_trials):
        start = time.perf_counter_ns()
        par_cached_outputs = run_parallel_composite_cached(device, tensors, num_iterations, use_signpost=(trial == 0))
        end = time.perf_counter_ns()
        parallel_composite_cached_times.append(end - start)
    print("  ✓ Parallel composite CACHED succeeded!")

    # Calculate statistics
    def stats(times):
        avg = sum(times) / len(times)
        min_t = min(times)
        max_t = max(times)
        return avg, min_t, max_t

    seq_avg, seq_min, seq_max = stats(sequential_times)
    seq_per_iter = seq_avg / num_iterations

    # Results
    print(f"\n{'='*60}")
    print("HOST TIME RESULTS")
    print(f"{'='*60}")
    print(f"\nSequential (q + kv, {num_iterations} iterations):")
    print(f"  Average: {seq_avg/1e6:.3f} ms  (per iter: {seq_per_iter/1e3:.2f} us)")
    print(f"  Min:     {seq_min/1e6:.3f} ms")
    print(f"  Max:     {seq_max/1e6:.3f} ms")

    par_comp_avg, par_comp_min, par_comp_max = stats(parallel_composite_times)
    par_comp_per_iter = par_comp_avg / num_iterations
    print(f"\nParallel Composite - NO CACHE (merged q + kv, {num_iterations} iterations):")
    print(f"  Average: {par_comp_avg/1e6:.3f} ms  (per iter: {par_comp_per_iter/1e3:.2f} us)")
    print(f"  Min:     {par_comp_min/1e6:.3f} ms")
    print(f"  Max:     {par_comp_max/1e6:.3f} ms")
    speedup_nocache = seq_avg / par_comp_avg
    print(f"  Speedup vs Sequential: {speedup_nocache:.2f}x")

    par_cached_avg, par_cached_min, par_cached_max = stats(parallel_composite_cached_times)
    par_cached_per_iter = par_cached_avg / num_iterations
    print(f"\nParallel Composite - CACHED (merged q + kv, {num_iterations} iterations):")
    print(f"  Average: {par_cached_avg/1e6:.3f} ms  (per iter: {par_cached_per_iter/1e3:.2f} us)")
    print(f"  Min:     {par_cached_min/1e6:.3f} ms")
    print(f"  Max:     {par_cached_max/1e6:.3f} ms")
    speedup_cached = seq_avg / par_cached_avg
    print(f"  Speedup vs Sequential: {speedup_cached:.2f}x")

    # Summary comparison
    print(f"\n{'='*60}")
    print("SPEEDUP SUMMARY")
    print(f"{'='*60}")
    print(f"  Sequential:             baseline ({seq_per_iter/1e3:.2f} us/iter)")
    print(f"  Parallel (no cache):    {speedup_nocache:.2f}x ({par_comp_per_iter/1e3:.2f} us/iter)")
    print(f"  Parallel CACHED:        {speedup_cached:.2f}x ({par_cached_per_iter/1e3:.2f} us/iter)")

    print(f"\nNote: Device kernel duration times are reported separately in Tracy logs.")
    print(f"      Look for 'Device kernel duration perf summary' messages.")

    # Show program cache statistics
    cache_entries = device.num_program_cache_entries()
    print(f"\nProgram Cache Statistics:")
    print(f"  Cached programs: {cache_entries}")
    print(f"  (GenericOpDeviceOperation::compute_program_hash is used to cache compiled programs)")

    # Verify correctness with fresh tensors
    print(f"\n{'='*60}")
    print("CORRECTNESS VERIFICATION (fresh tensors)")
    print(f"{'='*60}")

    # Create fresh tensors for correctness check
    fresh_tensors = create_tensors(device)
    seq_outputs = run_sequential(device, fresh_tensors, num_iterations=1)
    q_pcc, kv_pcc = verify_outputs(fresh_tensors, seq_outputs[0], seq_outputs[1])

    print(f"  q_norm PCC:  {q_pcc:.6f}")
    print(f"  kv_norm PCC: {kv_pcc:.6f}")

    assert q_pcc > 0.99, f"q_norm PCC {q_pcc} below threshold"
    assert kv_pcc > 0.99, f"kv_norm PCC {kv_pcc} below threshold"
    print("  ✓ All outputs verified correct!")

    print(f"\n{'='*60}")
    print("DEVICE KERNEL DURATION")
    print(f"{'='*60}")
    print("Device kernel duration times are reported in Tracy logs.")
    print("Look for 'Device kernel duration perf summary' messages:")
    print("  - 'q_norm': q_norm only kernel times")
    print("  - 'kv_norm': kv_norm only kernel times")
    print("  - 'sequential': sequential q + kv kernel times")
    print("  - 'parallel_composite': merged q + kv (no cache) kernel times")
    print("  - 'parallel_composite_cached': merged q + kv (cached) kernel times")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Allow running directly with pytest
    pytest.main([__file__, "-v", "-s"])
