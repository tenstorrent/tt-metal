# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Performance benchmark comparing sequential vs parallel RMS norm execution.

This test quantifies the performance gain from running two sharded RMS norm
operations in parallel vs sequentially using ttnn.experimental.launch_composite().

Run for host timing only:
    pytest tests/ttnn/unit_tests/operations/debug/parallel_rms_test.py -v -s

Run with device profiling (Tracy):
    python -m tracy -r -m pytest \
        tests/ttnn/unit_tests/operations/debug/parallel_rms_test.py::test_parallel_rms_performance -v -s

The two RMS norm operations:
    - q_norm: L1 width sharded on cores (0,0)-(3,3), shard [32,96], tensor [32,1536]
    - kv_norm: L1 width sharded on cores (5,0)-(6,7), shard [32,32], tensor [32,512]
"""

import pytest
import torch
import ttnn
import time

from tracy import signpost


def torch_rms_norm(x, gamma, eps=1e-5):
    """Reference RMS norm implementation in PyTorch."""
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    x_normed = x / rms
    if gamma is not None:
        x_normed = x_normed * gamma
    return x_normed


def create_q_norm_tensors(device):
    """
    Create q_norm tensors:
    - L1 width sharded on cores (0,0) to (3,3) = 16 cores
    - Shard shape: [32, 96]
    - Total tensor shape: [1, 1, 32, 1536] (16 cores × 96 = 1536)
    - Weights are DRAM interleaved
    """
    torch.manual_seed(42)

    # Core range: (0,0) to (3,3) = 4x4 = 16 cores
    q_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    num_cores = 16
    shard_width = 96
    total_width = num_cores * shard_width  # 1536

    # Tensor shapes
    input_shape = (1, 1, 32, total_width)
    weight_shape = (1, 1, 1, total_width)

    # Create torch tensors
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_weight = torch.rand(weight_shape, dtype=torch.bfloat16)

    # Create sharded memory config for input (L1 width sharded)
    shard_spec = ttnn.ShardSpec(q_cores, [32, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    # Move input to device with sharded memory config
    input_tensor = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_mem_config,
    )

    # Weights are DRAM interleaved
    weight_tensor = ttnn.from_torch(
        torch_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Program config for sharded RMS norm
    # compute_with_storage_grid_size should match the core grid dimensions
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(4, 4),  # 4x4 grid
        subblock_w=shard_width // 32,  # tiles per shard width = 96/32 = 3
        block_h=1,  # 32/32 = 1 tile high
        block_w=shard_width // 32,  # 3 tiles wide per core
        inplace=False,
    )

    return {
        "input": input_tensor,
        "weight": weight_tensor,
        "cores": q_cores,
        "memory_config": sharded_mem_config,
        "program_config": program_config,
        "torch_input": torch_input,
        "torch_weight": torch_weight,
        "name": "q_norm",
    }


def create_kv_norm_tensors(device):
    """
    Create kv_norm tensors:
    - L1 width sharded on cores (5,0) to (6,7) = 2x8 = 16 cores
    - Shard shape: [32, 32]
    - Total tensor shape: [1, 1, 32, 512] (16 cores × 32 = 512)
    - Weights are DRAM interleaved
    """
    torch.manual_seed(123)

    # Core range: (5,0) to (6,7) = 2 cols × 8 rows = 16 cores
    kv_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 7))])
    num_cores = 16
    shard_width = 32
    total_width = num_cores * shard_width  # 512

    # Tensor shapes
    input_shape = (1, 1, 32, total_width)
    weight_shape = (1, 1, 1, total_width)

    # Create torch tensors
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_weight = torch.rand(weight_shape, dtype=torch.bfloat16)

    # Create sharded memory config for input (L1 width sharded)
    shard_spec = ttnn.ShardSpec(kv_cores, [32, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    # Move input to device with sharded memory config
    input_tensor = ttnn.from_torch(
        torch_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_mem_config,
    )

    # Weights are DRAM interleaved
    weight_tensor = ttnn.from_torch(
        torch_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Program config for sharded RMS norm
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(2, 8),  # 2 cols × 8 rows
        subblock_w=shard_width // 32,  # tiles per shard width = 32/32 = 1
        block_h=1,  # 32/32 = 1 tile high
        block_w=shard_width // 32,  # 1 tile wide per core
        inplace=False,
    )

    return {
        "input": input_tensor,
        "weight": weight_tensor,
        "cores": kv_cores,
        "memory_config": sharded_mem_config,
        "program_config": program_config,
        "torch_input": torch_input,
        "torch_weight": torch_weight,
        "name": "kv_norm",
    }


def run_sequential(device, q_tensors, kv_tensors, num_iterations=1, use_signpost=False):
    """Run q_norm and kv_norm sequentially."""
    outputs = []

    if use_signpost:
        signpost("sequential_start")

    for _ in range(num_iterations):
        # Run q_norm
        q_output = ttnn.rms_norm(
            q_tensors["input"],
            epsilon=1e-5,
            weight=q_tensors["weight"],
            memory_config=q_tensors["memory_config"],
            program_config=q_tensors["program_config"],
        )

        # Run kv_norm
        kv_output = ttnn.rms_norm(
            kv_tensors["input"],
            epsilon=1e-5,
            weight=kv_tensors["weight"],
            memory_config=kv_tensors["memory_config"],
            program_config=kv_tensors["program_config"],
        )

        outputs = [q_output, kv_output]

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost("sequential_stop")

    return outputs


def run_parallel(device, q_tensors, kv_tensors, num_iterations=1, use_signpost=False):
    """Run q_norm and kv_norm in parallel using ttnn.experimental.launch_composite."""
    # Create branches ONCE outside the loop (for optimal caching)
    q_branch = ttnn.experimental.programs.rms_norm(
        q_tensors["input"],
        epsilon=1e-5,
        weight=q_tensors["weight"],
        memory_config=q_tensors["memory_config"],
        core_range_set=q_tensors["cores"],  # Optional: validates shard spec is within cores
    )

    kv_branch = ttnn.experimental.programs.rms_norm(
        kv_tensors["input"],
        epsilon=1e-5,
        weight=kv_tensors["weight"],
        memory_config=kv_tensors["memory_config"],
        core_range_set=kv_tensors["cores"],  # Optional: validates shard spec is within cores
    )

    if use_signpost:
        signpost("parallel_start")

    # Execute in parallel - launch_composite handles caching automatically
    for _ in range(num_iterations):
        q_output, kv_output = ttnn.experimental.launch_composite([q_branch, kv_branch])
        outputs = [q_output, kv_output]

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost("parallel_stop")

    return outputs


def verify_outputs(q_tensors, kv_tensors, q_output, kv_output):
    """Verify outputs against PyTorch reference."""
    # Compute expected outputs
    q_expected = torch_rms_norm(q_tensors["torch_input"], q_tensors["torch_weight"])
    kv_expected = torch_rms_norm(kv_tensors["torch_input"], kv_tensors["torch_weight"])

    # Get actual outputs
    q_actual = ttnn.to_torch(ttnn.from_device(q_output))
    kv_actual = ttnn.to_torch(ttnn.from_device(kv_output))

    # Calculate PCC
    def calc_pcc(expected, actual):
        expected_flat = expected.flatten().float()
        actual_flat = actual.flatten().float()
        return torch.corrcoef(torch.stack([expected_flat, actual_flat]))[0, 1].item()

    q_pcc = calc_pcc(q_expected, q_actual)
    kv_pcc = calc_pcc(kv_expected, kv_actual)

    return q_pcc, kv_pcc


@pytest.mark.parametrize("num_iterations", [10])
def test_parallel_rms_performance(device, num_iterations):
    """
    Benchmark sequential vs parallel RMS norm execution.

    Measures:
    - Host time for sequential execution
    - Host time for parallel execution
    - Performance gain (speedup)

    For device FW duration, run with Tracy:
        TT_METAL_DEVICE_PROFILER=1 python -m tracy -r -m pytest <this_file> -v -s
    """
    print(f"\n{'='*80}")
    print("PARALLEL RMS NORM PERFORMANCE BENCHMARK")
    print(f"{'='*80}")

    # Enable program cache for optimal performance
    device.enable_program_cache()

    # Create tensors
    q_tensors = create_q_norm_tensors(device)
    kv_tensors = create_kv_norm_tensors(device)

    print(f"\nConfiguration:")
    print(f"  q_norm:  cores (0,0)-(3,3), shard [32,96], tensor [32,1536]")
    print(f"  kv_norm: cores (5,0)-(6,7), shard [32,32], tensor [32,512]")
    print(f"  Iterations per measurement: {num_iterations}")

    # Warmup
    print("\nWarming up...")
    warmup_iters = 10
    for _ in range(warmup_iters):
        _ = run_sequential(device, q_tensors, kv_tensors)
        _ = run_parallel(device, q_tensors, kv_tensors)

    # Benchmark sequential
    print("Benchmarking sequential execution...")
    num_trials = 5
    sequential_times = []

    for trial in range(num_trials):
        start = time.perf_counter_ns()
        seq_outputs = run_sequential(device, q_tensors, kv_tensors, num_iterations, use_signpost=(trial == 0))
        end = time.perf_counter_ns()
        sequential_times.append(end - start)

    # Benchmark parallel
    print("Benchmarking parallel execution...")
    parallel_times = []

    for trial in range(num_trials):
        start = time.perf_counter_ns()
        par_outputs = run_parallel(device, q_tensors, kv_tensors, num_iterations, use_signpost=(trial == 0))
        end = time.perf_counter_ns()
        parallel_times.append(end - start)

    # Calculate statistics
    def stats(times):
        avg = sum(times) / len(times)
        min_t = min(times)
        max_t = max(times)
        return avg, min_t, max_t

    seq_avg, seq_min, seq_max = stats(sequential_times)
    par_avg, par_min, par_max = stats(parallel_times)

    # Per-iteration times
    seq_per_iter = seq_avg / num_iterations
    par_per_iter = par_avg / num_iterations

    # Speedup
    speedup = seq_avg / par_avg
    time_saved_per_iter = seq_per_iter - par_per_iter
    time_saved_total = seq_avg - par_avg

    # Results
    print(f"\n{'='*60}")
    print("HOST TIME RESULTS")
    print(f"{'='*60}")
    print(f"\nSequential ({num_iterations} iterations):")
    print(f"  Average: {seq_avg/1e6:.3f} ms  (per iter: {seq_per_iter/1e3:.2f} us)")
    print(f"  Min:     {seq_min/1e6:.3f} ms")
    print(f"  Max:     {seq_max/1e6:.3f} ms")

    print(f"\nParallel ({num_iterations} iterations):")
    print(f"  Average: {par_avg/1e6:.3f} ms  (per iter: {par_per_iter/1e3:.2f} us)")
    print(f"  Min:     {par_min/1e6:.3f} ms")
    print(f"  Max:     {par_max/1e6:.3f} ms")

    print(f"\n{'='*60}")
    print("PERFORMANCE GAIN")
    print(f"{'='*60}")
    print(f"  Speedup:           {speedup:.2f}x")
    print(f"  Time saved/iter:   {time_saved_per_iter/1e3:.2f} us")
    print(f"  Total time saved:  {time_saved_total/1e6:.3f} ms ({(time_saved_total/seq_avg)*100:.1f}%)")

    # Verify correctness
    print(f"\n{'='*60}")
    print("CORRECTNESS VERIFICATION")
    print(f"{'='*60}")

    # Create fresh tensors for verification (to avoid any in-place modifications)
    q_tensors_fresh = create_q_norm_tensors(device)
    kv_tensors_fresh = create_kv_norm_tensors(device)

    # Run once more to verify
    par_outputs = run_parallel(device, q_tensors_fresh, kv_tensors_fresh, num_iterations=1)
    q_pcc, kv_pcc = verify_outputs(q_tensors_fresh, kv_tensors_fresh, par_outputs[0], par_outputs[1])

    print(f"  q_norm PCC:  {q_pcc:.6f}")
    print(f"  kv_norm PCC: {kv_pcc:.6f}")

    assert q_pcc > 0.99, f"q_norm PCC {q_pcc} below threshold"
    assert kv_pcc > 0.99, f"kv_norm PCC {kv_pcc} below threshold"
    print("  ✓ All outputs verified correct!")

    print(f"\n{'='*60}")
    print("DEVICE PROFILING INSTRUCTIONS")
    print(f"{'='*60}")
    print("To measure device kernel duration, run:")
    print("  python -m tracy -r -m pytest \\")
    print("      tests/ttnn/unit_tests/operations/debug/parallel_rms_test.py::test_parallel_rms_performance -v -s")
    print()
    print("Look for signposts in Tracy:")
    print("  - 'sequential_start' to 'sequential_stop': Sequential execution")
    print("  - 'parallel_start' to 'parallel_stop': Parallel execution")
    print()
    print("Check the generated CSV for device kernel durations:")
    print("  - Q kernel: LayerNormDeviceOperation entries with INPUT_0_X_PAD=1536")
    print("  - KV kernel: LayerNormDeviceOperation entries with INPUT_0_X_PAD=512")
    print("  - Parallel: GenericOpDeviceOperation entries")
    print(f"{'='*60}\n")

    # Assert performance gain
    assert speedup > 1.0, f"Expected parallel to be faster than sequential! Got {speedup:.2f}x"


@pytest.mark.parametrize("num_iterations", [1])
def test_parallel_rms_correctness(device, num_iterations):
    """Quick correctness test for parallel RMS norm."""
    q_tensors = create_q_norm_tensors(device)
    kv_tensors = create_kv_norm_tensors(device)

    # Run parallel
    par_outputs = run_parallel(device, q_tensors, kv_tensors, num_iterations=1)

    # Verify
    q_pcc, kv_pcc = verify_outputs(q_tensors, kv_tensors, par_outputs[0], par_outputs[1])

    print(f"\nq_norm PCC:  {q_pcc:.6f}")
    print(f"kv_norm PCC: {kv_pcc:.6f}")

    assert q_pcc > 0.99, f"q_norm PCC {q_pcc} below threshold"
    assert kv_pcc > 0.99, f"kv_norm PCC {kv_pcc} below threshold"


if __name__ == "__main__":
    # Allow running directly with pytest
    pytest.main([__file__, "-v", "-s"])
