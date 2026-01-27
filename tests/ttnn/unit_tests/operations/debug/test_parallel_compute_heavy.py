# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Compute-heavy parallel RMS norm benchmark.

This test splits the 8x8 core grid in half and runs identical compute-heavy
RMS norm operations on each half. The goal is to measure device kernel
duration accurately by using operations that take significantly longer.

Run with Tracy profiling:
    python -m tracy -r -m pytest tests/ttnn/unit_tests/operations/debug/test_parallel_compute_heavy.py -v -s
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


def create_compute_heavy_tensors(device):
    """
    Create two sets of compute-heavy tensors for left_half and right_half RMS norm.

    Each half uses 32 cores (4x8 grid) with width sharding.
    Tensor dimensions are chosen to be compute-heavy:
    - Large width (many tiles per core for reduction)
    - Multiple rows (more work per core)

    Left half: cores (0,0)-(3,7) = 32 cores
    Right half: cores (4,0)-(7,7) = 32 cores
    """
    torch.manual_seed(42)

    # Left half: cores (0,0) to (3,7) = 4 columns x 8 rows = 32 cores
    left_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))])
    num_left_cores = 32

    # Right half: cores (4,0) to (7,7) = 4 columns x 8 rows = 32 cores
    right_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 7))])
    num_right_cores = 32

    # Make it compute-heavy: larger shard width and multiple tile rows
    # Each core processes a shard of [num_rows, shard_width]
    # Total width = num_cores * shard_width
    num_tile_rows = 4  # 4 tile rows = 128 rows
    shard_width_tiles = 8  # 8 tiles per core = 256 elements per row per core
    shard_width = shard_width_tiles * 32  # 256 elements
    shard_height = num_tile_rows * 32  # 128 rows

    total_width = num_left_cores * shard_width  # 32 * 256 = 8192 elements
    total_height = shard_height  # 128 rows

    # Create tensors with shape [1, 1, height, width]
    shape = (1, 1, total_height, total_width)
    weight_shape = (1, 1, 1, total_width)

    torch_left_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_left_weight = torch.rand(weight_shape, dtype=torch.bfloat16)

    torch_right_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_right_weight = torch.rand(weight_shape, dtype=torch.bfloat16)

    # Create sharded memory configs
    # Width sharded: each core gets [shard_height, shard_width]
    left_shard_spec = ttnn.ShardSpec(left_cores, [shard_height, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    left_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=left_shard_spec,
    )

    right_shard_spec = ttnn.ShardSpec(right_cores, [shard_height, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    right_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=right_shard_spec,
    )

    # Convert to TTNN tensors
    left_input = ttnn.from_torch(
        torch_left_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=left_mem_config,
    )
    left_weight = ttnn.from_torch(
        torch_left_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    right_input = ttnn.from_torch(
        torch_right_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=right_mem_config,
    )
    right_weight = ttnn.from_torch(
        torch_right_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    return {
        "left": {
            "input": left_input,
            "weight": left_weight,
            "torch_input": torch_left_input,
            "torch_weight": torch_left_weight,
            "cores": left_cores,
            "mem_config": left_mem_config,
        },
        "right": {
            "input": right_input,
            "weight": right_weight,
            "torch_input": torch_right_input,
            "torch_weight": torch_right_weight,
            "cores": right_cores,
            "mem_config": right_mem_config,
        },
        "config": {
            "shape": shape,
            "shard_shape": [shard_height, shard_width],
            "num_cores_per_half": num_left_cores,
        },
    }


def run_left_only(device, tensors, num_iterations=1, use_signpost=False):
    """Run RMS norm on left half only using launch_composite."""
    # Create branch descriptor ONCE
    left_branch = ttnn.experimental.programs.rms_norm(
        tensors["left"]["input"],
        epsilon=1e-5,
        weight=tensors["left"]["weight"],
    )

    if use_signpost:
        signpost("left_only_start")

    for _ in range(num_iterations):
        (left_output,) = ttnn.experimental.launch_composite([left_branch])

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost("left_only_stop")

    return left_output


def run_right_only(device, tensors, num_iterations=1, use_signpost=False):
    """Run RMS norm on right half only using launch_composite."""
    # Create branch descriptor ONCE
    right_branch = ttnn.experimental.programs.rms_norm(
        tensors["right"]["input"],
        epsilon=1e-5,
        weight=tensors["right"]["weight"],
    )

    if use_signpost:
        signpost("right_only_start")

    for _ in range(num_iterations):
        (right_output,) = ttnn.experimental.launch_composite([right_branch])

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost("right_only_stop")

    return right_output


def run_sequential(device, tensors, num_iterations=1, use_signpost=False):
    """Run left and right RMS norms sequentially using launch_composite."""
    # Create branch descriptors ONCE
    left_branch = ttnn.experimental.programs.rms_norm(
        tensors["left"]["input"],
        epsilon=1e-5,
        weight=tensors["left"]["weight"],
    )
    right_branch = ttnn.experimental.programs.rms_norm(
        tensors["right"]["input"],
        epsilon=1e-5,
        weight=tensors["right"]["weight"],
    )

    if use_signpost:
        signpost("sequential_start")

    for _ in range(num_iterations):
        # Run left then right (sequential - 2 separate dispatches)
        (left_output,) = ttnn.experimental.launch_composite([left_branch])
        (right_output,) = ttnn.experimental.launch_composite([right_branch])

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost("sequential_stop")

    return left_output, right_output


def run_parallel(device, tensors, num_iterations=1, use_signpost=False):
    """Run left and right RMS norms in parallel - creates branches each iteration."""
    if use_signpost:
        signpost("parallel_start")

    for _ in range(num_iterations):
        # Create descriptors inside the loop (measures descriptor creation overhead)
        left_branch = ttnn.experimental.programs.rms_norm(
            tensors["left"]["input"],
            epsilon=1e-5,
            weight=tensors["left"]["weight"],
        )
        right_branch = ttnn.experimental.programs.rms_norm(
            tensors["right"]["input"],
            epsilon=1e-5,
            weight=tensors["right"]["weight"],
        )
        # launch_composite always merges; device cache handles compiled program caching
        left_output, right_output = ttnn.experimental.launch_composite([left_branch, right_branch])

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost("parallel_stop")

    return left_output, right_output


def run_parallel_cached(device, tensors, num_iterations=1, use_signpost=False):
    """
    Run parallel using launch_composite() with automatic caching.

    Creates branch descriptors ONCE outside the loop. launch_composite()
    automatically caches the merged descriptor based on branch identity.

    This is the ideal usage pattern:
    - Create branches once (outside loop)
    - Call launch_composite with same branches in loop
    - Merged descriptor is cached automatically
    - Device program cache handles compiled program caching
    """
    # Create branch descriptors ONCE outside the loop
    left_branch = ttnn.experimental.programs.rms_norm(
        tensors["left"]["input"],
        epsilon=1e-5,
        weight=tensors["left"]["weight"],
    )
    right_branch = ttnn.experimental.programs.rms_norm(
        tensors["right"]["input"],
        epsilon=1e-5,
        weight=tensors["right"]["weight"],
    )

    if use_signpost:
        signpost("parallel_cached_start")

    for _ in range(num_iterations):
        # launch_composite caches the merged descriptor automatically
        left_output, right_output = ttnn.experimental.launch_composite([left_branch, right_branch])

    ttnn.synchronize_device(device)

    if use_signpost:
        signpost("parallel_cached_stop")

    return left_output, right_output


def verify_outputs(tensors, left_output, right_output):
    """Verify outputs against PyTorch reference."""
    left_expected = torch_rms_norm(tensors["left"]["torch_input"], tensors["left"]["torch_weight"])
    right_expected = torch_rms_norm(tensors["right"]["torch_input"], tensors["right"]["torch_weight"])

    left_actual = ttnn.to_torch(left_output)
    right_actual = ttnn.to_torch(right_output)

    def calc_pcc(expected, actual):
        expected_flat = expected.flatten().float()
        actual_flat = actual.flatten().float()
        return torch.corrcoef(torch.stack([expected_flat, actual_flat]))[0, 1].item()

    left_pcc = calc_pcc(left_expected, left_actual)
    right_pcc = calc_pcc(right_expected, right_actual)

    return left_pcc, right_pcc


@pytest.mark.parametrize("num_iterations", [20])
def test_parallel_compute_heavy(device, num_iterations):
    """
    Benchmark compute-heavy parallel RMS norm.

    Uses large tensors split across the 8x8 grid:
    - Left half: (0,0)-(3,7) = 32 cores
    - Right half: (4,0)-(7,7) = 32 cores

    Each core processes [128, 256] elements = 32,768 elements
    Total per half: 32 cores * 32,768 = 1,048,576 elements
    """
    print(f"\n{'='*80}")
    print("COMPUTE-HEAVY PARALLEL RMS NORM BENCHMARK")
    print(f"{'='*80}")

    # Enable program cache
    device.enable_program_cache()
    print(f"\nProgram cache enabled")

    # Create tensors
    tensors = create_compute_heavy_tensors(device)
    config = tensors["config"]

    print(f"\nConfiguration:")
    print(f"  Tensor shape: {config['shape']}")
    print(f"  Shard shape per core: {config['shard_shape']}")
    print(f"  Cores per half: {config['num_cores_per_half']}")
    print(f"  Left cores: (0,0)-(3,7)")
    print(f"  Right cores: (4,0)-(7,7)")
    print(f"  Elements per half: {config['shape'][2] * config['shape'][3]:,}")
    print(f"  Iterations: {num_iterations}")

    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        _ = run_left_only(device, tensors)
        _ = run_right_only(device, tensors)
        _ = run_sequential(device, tensors)
        _ = run_parallel(device, tensors)
        _ = run_parallel_cached(device, tensors)

    # Profile left only
    print("Profiling left half...")
    num_trials = 3
    for trial in range(num_trials):
        _ = run_left_only(device, tensors, num_iterations, use_signpost=(trial == 0))

    # Profile right only
    print("Profiling right half...")
    for trial in range(num_trials):
        _ = run_right_only(device, tensors, num_iterations, use_signpost=(trial == 0))

    # Profile sequential
    print("Profiling sequential (left + right)...")
    sequential_times = []
    for trial in range(num_trials):
        start = time.perf_counter_ns()
        _ = run_sequential(device, tensors, num_iterations, use_signpost=(trial == 0))
        end = time.perf_counter_ns()
        sequential_times.append(end - start)

    # Profile parallel (no descriptor cache)
    print("Profiling parallel (no descriptor cache)...")
    parallel_times = []
    for trial in range(num_trials):
        start = time.perf_counter_ns()
        _ = run_parallel(device, tensors, num_iterations, use_signpost=(trial == 0))
        end = time.perf_counter_ns()
        parallel_times.append(end - start)

    # Profile parallel cached
    print("Profiling parallel cached...")
    parallel_cached_times = []
    for trial in range(num_trials):
        start = time.perf_counter_ns()
        _ = run_parallel_cached(device, tensors, num_iterations, use_signpost=(trial == 0))
        end = time.perf_counter_ns()
        parallel_cached_times.append(end - start)

    # Calculate statistics
    def stats(times):
        avg = sum(times) / len(times)
        return avg, min(times), max(times)

    seq_avg, seq_min, seq_max = stats(sequential_times)
    par_avg, par_min, par_max = stats(parallel_times)
    par_cached_avg, par_cached_min, par_cached_max = stats(parallel_cached_times)

    # Results
    print(f"\n{'='*70}")
    print("HOST TIME RESULTS")
    print(f"{'='*70}")

    print(f"\nSequential ({num_iterations} iterations):")
    print(f"  Average: {seq_avg/1e6:.3f} ms  (per iter: {seq_avg/num_iterations/1e3:.2f} us)")

    print(f"\nParallel - no descriptor cache ({num_iterations} iterations):")
    print(f"  Average: {par_avg/1e6:.3f} ms  (per iter: {par_avg/num_iterations/1e3:.2f} us)")
    print(f"  Speedup vs Sequential: {seq_avg/par_avg:.2f}x")

    print(f"\nParallel CACHED ({num_iterations} iterations):")
    print(f"  Average: {par_cached_avg/1e6:.3f} ms  (per iter: {par_cached_avg/num_iterations/1e3:.2f} us)")
    print(f"  Speedup vs Sequential: {seq_avg/par_cached_avg:.2f}x")

    # Program cache stats
    cache_entries = device.num_program_cache_entries()
    print(f"\nProgram Cache: {cache_entries} entries")

    # Verify correctness
    print(f"\n{'='*70}")
    print("CORRECTNESS VERIFICATION")
    print(f"{'='*70}")

    fresh_tensors = create_compute_heavy_tensors(device)
    left_out, right_out = run_parallel_cached(device, fresh_tensors, num_iterations=1)
    left_pcc, right_pcc = verify_outputs(fresh_tensors, left_out, right_out)

    print(f"  Left PCC:  {left_pcc:.6f}")
    print(f"  Right PCC: {right_pcc:.6f}")

    assert left_pcc > 0.99, f"Left PCC {left_pcc} below threshold"
    assert right_pcc > 0.99, f"Right PCC {right_pcc} below threshold"
    print("  ✓ All outputs verified correct!")

    print(f"\n{'='*70}")
    print("DEVICE KERNEL DURATION")
    print(f"{'='*70}")
    print("Run with Tracy to get device kernel durations:")
    print("  python -m tracy -r -m pytest <this_file> -v -s")
    print()
    print("Look for signpost sections:")
    print("  - left_only:       Device time for left half alone")
    print("  - right_only:      Device time for right half alone")
    print("  - sequential:      Device time for left + right sequentially")
    print("  - parallel_cached: Device time for merged parallel program")
    print()
    print("Expected: parallel_cached device time ≈ max(left_only, right_only)")
    print("          This would indicate ~2x device speedup vs sequential")
    print(f"{'='*70}")
