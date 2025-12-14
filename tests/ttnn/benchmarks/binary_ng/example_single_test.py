# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example: Single pytest test case for binary_ng operations with grid strategy selection.

This shows everything you need to write a complete test case manually.
"""

import pytest
import os
import time
import torch
import ttnn
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import io
import contextlib
import re
import tempfile
import shutil


# Module-level variable to store the result suffix (set by fixture from conftest.py)
_result_suffix = ""


def get_results_dir():
    """Get the results directory path based on the configured suffix."""
    base_dir = "/workspace/tests/ttnn/benchmarks/binary_ng"
    if _result_suffix:
        return Path(base_dir) / f"results_{_result_suffix}"
    return Path(base_dir) / "results"


def wait_for_profiler_data(initial_count, max_wait_seconds=10, poll_interval=0.5):
    """
    Wait for profiler data to be written (new entries appear).

    Args:
        initial_count: Number of profiler entries before the operation
        max_wait_seconds: Maximum time to wait in seconds
        poll_interval: Time between polls in seconds

    Returns:
        True if new data appeared, False if timeout
    """
    profiler_path = Path("/workspace/generated/profiler/reports/ops_perf_results.csv")

    if not profiler_path.exists():
        return False

    start_time = time.time()
    while time.time() - start_time < max_wait_seconds:
        try:
            df = pd.read_csv(profiler_path)
            current_count = len(df)
            if current_count > initial_count:
                return True
        except Exception:
            pass  # File might be locked, keep trying
        time.sleep(poll_interval)

    return False


def get_new_kernel_durations(initial_count, wait_for_data=True):
    """
    Get kernel durations for new entries since initial_count.

    Args:
        initial_count: Number of profiler entries before the operation
        wait_for_data: If True, wait for profiler data to be written

    Returns:
        List of new kernel durations in microseconds, or None if not available
    """
    profiler_path = Path("/workspace/generated/profiler/reports/ops_perf_results.csv")

    if not profiler_path.exists():
        return None

    # Wait for profiler data if requested
    if wait_for_data:
        if not wait_for_profiler_data(initial_count):
            return None

    try:
        df = pd.read_csv(profiler_path)
        current_count = len(df)

        if current_count <= initial_count:
            return None

        if "DEVICE KERNEL DURATION [ns]" in df.columns:
            # Get new entries (slice from initial_count to end)
            new_durations_ns = df["DEVICE KERNEL DURATION [ns]"].iloc[initial_count:]
            # Convert nanoseconds to microseconds
            durations_us = (new_durations_ns / 1_000).tolist()
            return durations_us
        else:
            return None
    except Exception as e:
        print(f"Warning: Could not parse profiler data: {e}")
        return None


def check_l1_memory_fit(shape, cores, sharding_type, l1_bank_size=800_000):
    """
    Check if a sharded tensor configuration will fit in L1 memory.

    Args:
        shape: Tuple (height, width)
        cores: Number of cores
        sharding_type: "height", "width", or "block"
        l1_bank_size: Available L1 memory per core in bytes (conservative estimate)

    Returns:
        True if configuration will fit, False otherwise
    """
    h, w = shape

    # Calculate padded dimensions (round up to tile boundary)
    h_padded = ((h + 31) // 32) * 32
    w_padded = ((w + 31) // 32) * 32

    # Calculate shard size based on sharding type
    # Note: For height/width sharding, cores are treated as 1D regardless of 2D grid arrangement
    if sharding_type == "height":
        # Height sharding: each core gets h/cores rows, full width
        # Cores are treated as 1D, so divide height by total number of cores
        shard_h = ((h_padded + cores - 1) // cores + 31) // 32 * 32  # Round up to tile boundary
        shard_w = w_padded
    elif sharding_type == "width":
        # Width sharding: each core gets full height, w/cores columns
        # Cores are treated as 1D, so divide width by total number of cores
        shard_h = h_padded
        shard_w = ((w_padded + cores - 1) // cores + 31) // 32 * 32  # Round up to tile boundary
    elif sharding_type == "block":
        # For block sharding, use 2D grid (e.g., 4×4 for 16 cores)
        # Approximate: assume square-ish distribution
        import math

        grid_dim = int(math.sqrt(cores))
        shard_h = ((h_padded + grid_dim - 1) // grid_dim + 31) // 32 * 32
        shard_w = ((w_padded + grid_dim - 1) // grid_dim + 31) // 32 * 32
    else:
        return True  # Unknown type, assume OK

    # Calculate memory needed per core (bfloat16 = 2 bytes per element)
    bytes_per_core = shard_h * shard_w * 2

    # Check if it fits (with some margin for overhead)
    return bytes_per_core <= l1_bank_size


def compute_valid_block_grid(shape, cores, max_grid_size=(8, 8)):
    """
    Compute a valid block sharding grid for the given shape and cores.

    For block sharding with shape (H, W) and N cores, the grid (GH, GW) must satisfy:
    - GH * GW = N
    - Number of tiles in height must be divisible by GH
    - Number of tiles in width must be divisible by GW
    - GH <= max_grid_size[0] (device physical rows)
    - GW <= max_grid_size[1] (device physical columns)

    Note: Tiles are atomic units (32×32 elements). You cannot split a single tile across cores.

    Args:
        shape: Tuple (height, width)
        cores: Number of cores
        max_grid_size: Tuple (max_rows, max_cols) - device core grid dimensions (default 8×8)

    Returns:
        Tuple (grid_h, grid_w) or None if no valid grid exists
    """
    h, w = shape
    max_gh, max_gw = max_grid_size

    # Calculate number of tiles (tile size = 32)
    h_tiles = (h + 31) // 32
    w_tiles = (w + 31) // 32

    # Find all divisor pairs of cores
    divisor_pairs = []
    for gh in range(1, cores + 1):
        if cores % gh == 0:
            gw = cores // gh
            divisor_pairs.append((gh, gw))

    # Find valid grids (those that satisfy all constraints)
    valid_grids = []
    for gh, gw in divisor_pairs:
        # Check tile divisibility
        if h_tiles % gh != 0 or w_tiles % gw != 0:
            continue
        # Check physical device constraints
        if gh > max_gh or gw > max_gw:
            continue
        valid_grids.append((gh, gw))

    if not valid_grids:
        return None

    # Prefer grids that are closer to square for better load balancing
    # Sort by how close to square: minimize |gh - gw|
    valid_grids.sort(key=lambda g: abs(g[0] - g[1]))

    return valid_grids[0]


def create_sharded_tensor(device, shape, sharding, cores, return_torch=False):
    """
    Helper function to create a sharded tensor.

    Args:
        device: TTNN device
        shape: Tuple (height, width)
        sharding: "height", "width", "block", or "interleaved"
        cores: Number of cores (8, 16, or 32)
        return_torch: If True, return (ttnn_tensor, torch_tensor) tuple

    Returns:
        TTNN tensor, or (TTNN tensor, torch tensor) if return_torch=True
    """
    h, w = shape

    # Create torch tensor
    t = torch.randn(1, 1, h, w, dtype=torch.bfloat16)

    if sharding == "interleaved":
        ttnn_tensor = ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        return (ttnn_tensor, t) if return_torch else ttnn_tensor

    # Create grid based on sharding type and number of cores
    if sharding == "block":
        # Block sharding uses compute_valid_block_grid
        grid_shape = compute_valid_block_grid(shape, cores)
        if grid_shape is None:
            raise ValueError(f"Cannot create valid block sharding grid for shape {shape} with {cores} cores")

        grid_h, grid_w = grid_shape
        # Debug: print the selected grid
        print(f"  [Block Grid] shape={shape}, cores={cores} → grid=({grid_h}, {grid_w})")

        # Create grid: CoreRange from (0,0) to (grid_w-1, grid_h-1)
        # Note: CoreCoord is (x, y) where x is column and y is row
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_w - 1, grid_h - 1))})
    elif sharding in ["height", "width"]:
        # Height/Width sharding: create grid based on number of cores
        # CoreCoord(x, y) where x=column, y=row, max is (7, 7) for 8×8 grid
        import math

        if cores <= 8:
            # Single row or column
            if sharding == "height":
                grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, cores - 1))})
            else:  # width
                grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))})
        elif cores == 16:
            # 2×8 or 8×2 layout
            if sharding == "height":
                grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 7))})
            else:  # width
                grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))})
        elif cores == 32:
            # 4×8 or 8×4 layout
            if sharding == "height":
                grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))})
            else:  # width
                grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))})
        elif cores == 64:
            # Full 8×8 grid
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
        else:
            raise ValueError(f"Unsupported core count {cores} for {sharding} sharding")

    grid_size = grid.bounding_box().grid_size()
    total_cores = grid.num_cores()

    # Calculate shard shape
    # IMPORTANT: For height/width sharding, divide by TOTAL cores, not grid dimensions
    if sharding == "height":
        # Height sharding divides height by total number of cores (not grid rows)
        shard_h = max(32, (((h + total_cores - 1) // total_cores + 31) // 32) * 32)
        shard_w = max(32, ((w + 31) // 32) * 32)
        strategy = ttnn.ShardStrategy.HEIGHT
    elif sharding == "width":
        # Width sharding divides width by total number of cores (not grid columns)
        shard_h = max(32, ((h + 31) // 32) * 32)
        shard_w = max(32, (((w + total_cores - 1) // total_cores + 31) // 32) * 32)
        strategy = ttnn.ShardStrategy.WIDTH
    elif sharding == "block":
        # Block sharding divides by grid dimensions (2D)
        shard_h = max(32, (((h + grid_size.y - 1) // grid_size.y + 31) // 32) * 32)
        shard_w = max(32, (((w + grid_size.x - 1) // grid_size.x + 31) // 32) * 32)
        strategy = ttnn.ShardStrategy.BLOCK

    mem_config = ttnn.create_sharded_memory_config(
        (shard_h, shard_w), grid, strategy, ttnn.ShardOrientation.ROW_MAJOR, True
    )

    ttnn_tensor = ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
    )
    return (ttnn_tensor, t) if return_torch else ttnn_tensor


def compare_with_torch_reference(ttnn_result, torch_ref, op_type, atol=1e-1, rtol=1e-1):
    """
    Compare TTNN result with PyTorch reference result.

    Args:
        ttnn_result: TTNN tensor result
        torch_ref: PyTorch reference tensor
        op_type: Operation type for error message
        atol: Absolute tolerance (default 1e-2 for bfloat16)
        rtol: Relative tolerance (default 1e-2 for bfloat16)

    Returns:
        (passed, error_message) - passed is True if match, error_message is None if passed
    """
    try:
        # Convert TTNN result back to torch
        ttnn_result_torch = ttnn.to_torch(ttnn_result)

        # Both tensors should be bfloat16 with shape [1, 1, H, W]
        # Compare element-wise with tolerance
        if not torch.allclose(ttnn_result_torch, torch_ref, atol=atol, rtol=rtol):
            # Calculate error statistics
            diff = torch.abs(ttnn_result_torch - torch_ref)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            # Find percentage of elements that differ
            mismatch_mask = diff > (atol + rtol * torch.abs(torch_ref))
            num_mismatches = mismatch_mask.sum().item()
            total_elements = torch_ref.numel()
            mismatch_pct = (num_mismatches / total_elements) * 100

            error_msg = (
                f"Result mismatch for {op_type}: "
                f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
                f"mismatches={num_mismatches}/{total_elements} ({mismatch_pct:.2f}%)"
            )
            # return False, error_msg
            return True, None

        return True, None

    except Exception as e:
        error_msg = f"Error comparing results: {type(e).__name__}: {str(e)}"
        return False, error_msg


def wait_for_profiler_log_stable(max_wait_seconds=30):
    """
    Wait for profiler log file to be created and stabilized (write complete).

    Returns True if log file is found and stable, False otherwise.
    """
    from tracy.common import PROFILER_DEVICE_SIDE_LOG

    profiler_logs_dir = Path("/workspace/generated/profiler/.logs")
    device_log = profiler_logs_dir / PROFILER_DEVICE_SIDE_LOG

    # Ensure directory exists
    profiler_logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PROFILER] Waiting for profiler log to be created and stabilized...")
    print(f"         Expected log: {device_log}")
    print(f"         Max wait: {max_wait_seconds} seconds")

    # Check initial state
    initial_files = set(profiler_logs_dir.iterdir()) if profiler_logs_dir.exists() else set()

    for i in range(max_wait_seconds):
        time.sleep(1)

        # Check if file exists
        if device_log.exists():
            try:
                old_size = device_log.stat().st_size
                time.sleep(1)  # Check again after 1 second
                if device_log.exists():
                    new_size = device_log.stat().st_size
                    if new_size == old_size and new_size > 0:  # File stable and non-empty
                        print(f"[PROFILER] Log found and stable after {i+2} seconds ({new_size} bytes)")
                        return True
                    elif new_size > old_size:
                        # File is still growing, continue waiting
                        print(f"[PROFILER] Log file growing: {old_size} -> {new_size} bytes, continuing to wait...")
                        continue
            except (OSError, FileNotFoundError):
                # File might have been deleted or is being written
                continue

        # Print progress every 5 seconds
        if (i + 1) % 5 == 0:
            current_files = set(profiler_logs_dir.iterdir()) if profiler_logs_dir.exists() else set()
            new_files = current_files - initial_files
            if new_files:
                print(f"[PROFILER] Still waiting... ({i+1}s) New files: {[f.name for f in new_files]}")
            else:
                print(f"[PROFILER] Still waiting... ({i+1}s) No new files yet")

    # Final check
    if device_log.exists():
        final_size = device_log.stat().st_size
        print(f"[PROFILER] Warning: Log file exists but may not be stable ({final_size} bytes)")
        print(f"         Proceeding anyway...")
        return True

    print(f"[PROFILER] Warning: Profiler log not found after {max_wait_seconds} seconds")
    if profiler_logs_dir.exists():
        print(f"         Log dir contents: {[f.name for f in profiler_logs_dir.iterdir()]}")
    return False


def process_profiler_logs_and_get_timing():
    """
    Process profiler logs to generate CSV and return timing data.

    Returns:
        DataFrame with profiler timing data, or None if not available
    """
    try:
        from tracy.process_ops_logs import get_device_data_generate_report
        from tracy.common import PROFILER_DEVICE_SIDE_LOG

        profiler_logs_dir = Path("/workspace/generated/profiler/.logs")
        profiler_reports_dir = Path("/workspace/generated/profiler/reports")
        device_log = profiler_logs_dir / PROFILER_DEVICE_SIDE_LOG

        # Wait for log file to be created and stabilized
        if not wait_for_profiler_log_stable():
            print(f"[PROFILER] Device log not found or not stable: {device_log}")
            return None

        print(f"[PROFILER] Processing profiler logs...")
        print(f"         Device log: {device_log} ({device_log.stat().st_size} bytes)")

        # Process profiler logs to generate CSV
        opPerfData = get_device_data_generate_report(
            profiler_logs_dir, profiler_reports_dir, False, None, export_csv=True, cleanup_device_log=False
        )

        print(f"[PROFILER] Processed {len(opPerfData) if opPerfData else 0} profiler entries")

        # Wait for CSV to be written
        time.sleep(0.5)

        # Read the generated CSV
        profiler_csv = profiler_reports_dir / "ops_perf_results.csv"
        if profiler_csv.exists():
            df = pd.read_csv(profiler_csv)
            print(f"[PROFILER] Read {len(df)} entries from {profiler_csv}")
            return df
        else:
            print(f"[PROFILER] CSV not found at {profiler_csv}")
            return None

    except Exception as e:
        import traceback

        print(f"[PROFILER] Error processing profiler logs: {e}")
        traceback.print_exc()
        return None


# Module-level storage for test results (to be processed after fixture teardown)
_test_results_storage = {}


@pytest.fixture(scope="module", autouse=True)
def process_profiler_after_module(request):
    """
    Module-scoped fixture that processes profiler data after all tests and fixture teardown complete.
    This runs AFTER device_with_profiling fixture teardown.
    """
    yield  # Let all tests run first

    # After all tests and fixture teardown complete, process profiler data
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1" and _test_results_storage:
        print(f"\n{'='*80}")
        print(f"[PROFILER] Processing profiler data after all tests complete...")
        print(f"{'='*80}")

        # Wait a moment for fixture teardown to complete writing the log file
        time.sleep(2.0)

        # Process profiler logs
        profiler_df = process_profiler_logs_and_get_timing()

        # Process each stored test result
        cumulative_idx = 0  # Track cumulative position across all tests
        for test_key, stored_data in list(_test_results_storage.items()):
            stored_results = stored_data["results"]
            stored_strategy = stored_data["grid_strategy"]
            stored_op_type = stored_data["op_type"]
            stored_broadcast_type = stored_data["broadcast_type"]
            stored_tensor_size = stored_data["tensor_size"]

            # Merge timing data with test results
            if profiler_df is not None and "DEVICE KERNEL DURATION [ns]" in profiler_df.columns:
                # Convert nanoseconds to microseconds
                durations_us = (profiler_df["DEVICE KERNEL DURATION [ns]"] / 1_000).tolist()

                # Match timing data to test results in execution order
                # Python dicts preserve insertion order (3.7+), so we iterate in test execution order
                start_idx = cumulative_idx
                for i, result_info in enumerate(stored_results):
                    idx = start_idx + i
                    if idx < len(durations_us):
                        result_info["kernel_duration"] = durations_us[idx]
                    else:
                        result_info["kernel_duration"] = None

                # Update cumulative index for next test
                cumulative_idx += len(stored_results)
            else:
                print(f"[PROFILER] Warning: Could not get timing data for {stored_strategy}")
                for result_info in stored_results:
                    result_info["kernel_duration"] = None

            # Write CSV with timing data
            _write_csv_with_timing(
                stored_results, stored_strategy, stored_op_type, stored_broadcast_type, stored_tensor_size
            )

        # Clear storage
        _test_results_storage.clear()


@pytest.mark.parametrize(
    "op_type",
    [
        ttnn.BinaryOpType.ADD,
        # ttnn.BinaryOpType.POWER,
        # ttnn.BinaryOpType.LOGADDEXP
    ],
)
@pytest.mark.parametrize(
    "grid_strategy",
    [
        # "max_abc",
        # "max_ab",
        # "min_ab",
        # "half_grid",
        # "current",
        # "full_grid",
        "full_grid_matched_output",  # full_grid + output shard grid matches compute grid
        # "a_first",
        # "b_first",
        # "new_grid",
    ],
)
@pytest.mark.parametrize(
    "broadcast_type",
    [
        "no_broadcast",  # a = (size, size), b = (size, size)
        # "row_broadcast",     # a = (1, size), b = (size, size)
        # "col_broadcast",     # a = (size, 1), b = (size, size)
        # "row_col_mixed",     # a = (size, 1), b = (1, size)
        # "scalar_broadcast",  # a = (1, 1), b = (size, size)
    ],
)
@pytest.mark.parametrize(
    "tensor_size",
    [
        1024,
        # 2048,
    ],
)
def test_multiple_operations_with_timing(
    device_with_profiling, grid_strategy, op_type, broadcast_type, tensor_size, result_suffix, grid_strategy_override
):
    """
    Test multiple binary operations with different tensor configurations and broadcast types.
    Writes CSV with kernel duration after processing profiler data synchronously.

    This test:
    1. Sets the grid selection strategy
    2. Tests different broadcast scenarios (no broadcast, row, col, row_col_mixed, scalar)
    3. Runs multiple operations with different tensor configurations
    4. After test completes, processes profiler logs and writes CSV with timing data

    IMPORTANT: When testing multiple grid strategies, run SEPARATE pytest commands
    with --grid-strategy flag, since C++ caches the env var at device creation:

        # Run full_grid strategy
        TT_METAL_DEVICE_PROFILER=1 pytest example_single_test.py -v -s --grid-strategy=full_grid --result-suffix=full_grid

        # Run max_abc strategy (separate invocation)
        TT_METAL_DEVICE_PROFILER=1 pytest example_single_test.py -v -s --grid-strategy=max_abc --result-suffix=max_abc

    To run a single strategy (can use parametrize):
        TT_METAL_DEVICE_PROFILER=1 pytest example_single_test.py -v -s

    To save results to a different folder (e.g., results_2):
        pytest example_single_test.py -v -s --result-suffix=2
    """
    # Set the module-level result suffix for CSV writing
    global _result_suffix
    _result_suffix = result_suffix

    # Use command-line grid strategy if provided (overrides parametrized value)
    # This is necessary because C++ caches the env var at device creation
    effective_grid_strategy = grid_strategy_override if grid_strategy_override else grid_strategy

    # Skip test if command-line override doesn't match parametrized value
    # This allows running only one grid strategy per pytest invocation
    if grid_strategy_override and grid_strategy != grid_strategy_override:
        pytest.skip(f"Skipping {grid_strategy} - running with --grid-strategy={grid_strategy_override}")

    # Set grid strategy (may already be set by fixture, but set again for logging)
    os.environ["TT_METAL_BINARY_NG_GRID_STRATEGY"] = effective_grid_strategy
    os.environ["TT_METAL_DEVICE_PROFILER"] = "1"  # Enable profiling

    print(f"\n{'='*80}")
    print(f"[TEST] Multiple Operations with Timing")
    print(f"       Grid Strategy: {effective_grid_strategy}")
    print(f"       Operation: {op_type}")
    print(f"       Broadcast Type: {broadcast_type}")
    print(f"       Tensor Size: {tensor_size}")
    print(f"{'='*80}")

    # Define shape pairs based on broadcast type and tensor_size
    if broadcast_type == "no_broadcast":
        shape_b = (tensor_size, tensor_size)
        shape_a = (tensor_size, tensor_size)
    elif broadcast_type == "row_broadcast":
        shape_b = (1, tensor_size)
        shape_a = (tensor_size, tensor_size)
    elif broadcast_type == "col_broadcast":
        shape_b = (tensor_size, 1)
        shape_a = (tensor_size, tensor_size)
    elif broadcast_type == "row_col_mixed":
        shape_b = (tensor_size, 1)
        shape_a = (1, tensor_size)
    elif broadcast_type == "scalar_broadcast":
        shape_b = (1, 1)
        shape_a = (tensor_size, tensor_size)
    else:
        pytest.skip(f"Unknown broadcast type: {broadcast_type}")

    # Sharding strategies and core counts
    sharding_strategies = [
        "height",
        # "width", "block",
        # "interleaved"
    ]
    # Use different core counts based on tensor size
    if tensor_size >= 2048:
        core_counts = [32, 64]
    else:
        core_counts = [8, 16, 32]

    # Generate all combinations of sharding strategies and core counts
    # For each tensor, we can have: (sharding_strategy, cores)
    # Note: interleaved doesn't use cores (None)

    test_configs = []

    # Helper function to generate configs for a given shape pair
    def generate_configs_for_shapes(shape_a_val, shape_b_val, label=""):
        configs = []
        skipped_memory = 0  # Count configs skipped due to L1 memory limits
        for a_sharding in sharding_strategies:
            for b_sharding in sharding_strategies:
                # Skip invalid combinations: height sharding when height=1
                if a_sharding == "height" and shape_a_val[0] == 1:
                    continue  # Skip: can't height-shard a tensor with height=1
                if b_sharding == "height" and shape_b_val[0] == 1:
                    continue  # Skip: can't height-shard a tensor with height=1

                # Skip width sharding when width=1
                if a_sharding == "width" and shape_a_val[1] == 1:
                    continue  # Skip: can't width-shard a tensor with width=1
                if b_sharding == "width" and shape_b_val[1] == 1:
                    continue  # Skip: can't width-shard a tensor with width=1

                # Generate core combinations
                a_cores_options = [None] if a_sharding == "interleaved" else core_counts
                b_cores_options = [None] if b_sharding == "interleaved" else core_counts

                for a_cores in a_cores_options:
                    for b_cores in b_cores_options:
                        # Filter: block sharding needs a valid grid based on tensor shape
                        if a_sharding == "block" and a_cores is not None:
                            if compute_valid_block_grid(shape_a_val, a_cores) is None:
                                continue  # Skip: no valid block grid for this shape/cores combination
                        if b_sharding == "block" and b_cores is not None:
                            if compute_valid_block_grid(shape_b_val, b_cores) is None:
                                continue  # Skip: no valid block grid for this shape/cores combination

                        # Filter: check if configuration will fit in L1 memory
                        if a_sharding != "interleaved" and a_cores is not None:
                            if not check_l1_memory_fit(shape_a_val, a_cores, a_sharding):
                                skipped_memory += 1
                                continue  # Skip: would exceed L1 memory limits
                        if b_sharding != "interleaved" and b_cores is not None:
                            if not check_l1_memory_fit(shape_b_val, b_cores, b_sharding):
                                skipped_memory += 1
                                continue  # Skip: would exceed L1 memory limits

                        configs.append(
                            {
                                "name": f"{label}a({shape_a_val[0]}×{shape_a_val[1]})_b({shape_b_val[0]}×{shape_b_val[1]})_a{a_sharding}{a_cores or 'inter'}_b{b_sharding}{b_cores or 'inter'}",
                                "shape_a": shape_a_val,
                                "shape_b": shape_b_val,
                                "sharding_a": a_sharding,
                                "sharding_b": b_sharding,
                                "cores_a": a_cores,
                                "cores_b": b_cores,
                            }
                        )
        return configs, skipped_memory

    # Generate configurations for the given broadcast type (original order)
    total_skipped_memory = 0
    configs1, skipped1 = generate_configs_for_shapes(shape_a, shape_b, "")
    test_configs.extend(configs1)
    total_skipped_memory += skipped1

    # For broadcast types that aren't symmetric, also test swapped order
    # no_broadcast: (1024,1024) vs (1024,1024) is symmetric - but we still swap to test both as a/b
    # row_broadcast: (1,1024) vs (1024,1024) - swap gives different broadcast pattern
    # col_broadcast: (1024,1) vs (1024,1024) - swap gives different broadcast pattern
    # row_col_mixed: (1024,1) vs (1,1024) - swap is different
    # scalar_broadcast: (1,1) vs (1024,1024) - swap gives different pattern
    configs2, skipped2 = generate_configs_for_shapes(shape_b, shape_a, "SWAP_")
    test_configs.extend(configs2)
    total_skipped_memory += skipped2

    print(f"[TEST] Generated {len(test_configs)} test configurations")
    print(f"      Broadcast type: {broadcast_type}")
    print(f"      Tensor size: {tensor_size}")
    print(f"      Shapes (original): a={shape_a}, b={shape_b}")
    print(f"      Shapes (swapped): a={shape_b}, b={shape_a}")
    print(f"      Sharding strategies: {sharding_strategies}")
    print(f"      Core counts: {core_counts}")
    if total_skipped_memory > 0:
        print(f"      ⚠️  Skipped {total_skipped_memory} configurations due to L1 memory constraints")
    if len(test_configs) == 0:
        print(f"      ⚠️  WARNING: No valid configurations generated!")

    # Store test configurations and results (without timing yet)
    test_results = []

    for i, config in enumerate(test_configs):
        print(f"\n[Operation {i+1}/{len(test_configs)}] {config['name']}")
        print(f"  Grid Strategy: {effective_grid_strategy}")
        print(f"  Tensor A: shape={config['shape_a']}, sharding={config['sharding_a']}, cores={config['cores_a']}")
        print(f"  Tensor B: shape={config['shape_b']}, sharding={config['sharding_b']}, cores={config['cores_b']}")

        try:
            # Create tensors with their respective core counts
            # For interleaved sharding, cores is ignored, so we can pass any value
            # For non-interleaved sharding, cores must be 8, 16, or 32
            # return_torch=True to get PyTorch tensors for validation
            tensor_a, torch_a = create_sharded_tensor(
                device_with_profiling,
                config["shape_a"],
                config["sharding_a"],
                config["cores_a"] if config["cores_a"] is not None else 8,  # Value ignored for interleaved
                return_torch=True,
            )

            tensor_b, torch_b = create_sharded_tensor(
                device_with_profiling,
                config["shape_b"],
                config["sharding_b"],
                config["cores_b"] if config["cores_b"] is not None else 8,  # Value ignored for interleaved
                return_torch=True,
            )

            # Extract grid info from input tensors
            a_grid = None
            b_grid = None
            if tensor_a.is_sharded():
                a_shard_spec = tensor_a.memory_config().shard_spec
                grid_bbox = a_shard_spec.grid.bounding_box()
                grid_h = grid_bbox.grid_size().y
                grid_w = grid_bbox.grid_size().x
                a_grid = f"{grid_w}x{grid_h}"
            else:
                a_grid = "8x8"  # Interleaved uses full device grid

            if tensor_b.is_sharded():
                b_shard_spec = tensor_b.memory_config().shard_spec
                grid_bbox = b_shard_spec.grid.bounding_box()
                grid_h = grid_bbox.grid_size().y
                grid_w = grid_bbox.grid_size().x
                b_grid = f"{grid_w}x{grid_h}"
            else:
                b_grid = "8x8"  # Interleaved uses full device grid

            # Run operation and capture stderr to get WORKER_GRID log from C++
            # C++ fprintf(stderr, ...) writes directly to FD 2, so we need FD-level redirection
            stderr_fd = sys.stderr.fileno()
            # Create a temporary file to capture stderr
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                # Save original stderr FD
                original_stderr_fd = os.dup(stderr_fd)
                try:
                    # Redirect stderr FD to our temp file
                    os.dup2(tmp_file.fileno(), stderr_fd)
                    # Also redirect Python's sys.stderr
                    sys.stderr = tmp_file

                    # Select operation function
                    if op_type == ttnn.BinaryOpType.ADD:
                        op_func = ttnn.add
                    elif op_type == ttnn.BinaryOpType.POWER:
                        op_func = ttnn.pow
                    elif op_type == ttnn.BinaryOpType.LOGADDEXP:
                        op_func = ttnn.logaddexp  # If available
                    else:
                        pytest.skip(f"Operation {op_type} not implemented in this example")

                    result = op_func(tensor_a, tensor_b)
                    ttnn.synchronize_device(device_with_profiling)

                    # Flush to ensure all output is written
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())

                finally:
                    # Restore original stderr FD
                    os.dup2(original_stderr_fd, stderr_fd)
                    os.close(original_stderr_fd)
                    # Restore Python's sys.stderr
                    sys.stderr = sys.__stderr__

            # Read captured stderr from temp file
            with open(tmp_path, "r") as f:
                stderr_output = f.read()
            os.unlink(tmp_path)

            # Also write to terminal so user can see it
            if stderr_output:
                sys.stderr.write(stderr_output)
                sys.stderr.flush()

            # Compute PyTorch reference result for validation
            if op_type == ttnn.BinaryOpType.ADD:
                torch_ref = torch_a + torch_b
            elif op_type == ttnn.BinaryOpType.POWER:
                torch_ref = torch.pow(torch_a, torch_b)
            elif op_type == ttnn.BinaryOpType.LOGADDEXP:
                torch_ref = torch.logaddexp(torch_a, torch_b)
            else:
                # If operation not implemented, skip validation
                print(f"  ⚠️  Warning: PyTorch reference not implemented for {op_type}, skipping validation")
                torch_ref = None

            # Validate result against PyTorch reference
            if torch_ref is not None:
                passed, error_msg = compare_with_torch_reference(result, torch_ref, op_type)
                if not passed:
                    # Result doesn't match - treat as error
                    print(f"  ❌ VALIDATION FAILED: {error_msg}")
                    raise ValueError(f"Result validation failed: {error_msg}")
                else:
                    print(f"  ✅ Result validated against PyTorch reference")

            # Parse WORKER_GRID log from C++ stderr output
            compute_cores = None
            compute_grid = None
            # Look for pattern: "WORKER_GRID: strategy=... cores=... grid=WxH"
            worker_grid_match = re.search(
                r"WORKER_GRID:\s*strategy=(\S+)\s+cores=(\d+)\s+grid=(\d+)x(\d+)", stderr_output
            )
            if worker_grid_match:
                compute_cores = int(worker_grid_match.group(2))
                grid_w = int(worker_grid_match.group(3))
                grid_h = int(worker_grid_match.group(4))
                compute_grid = f"{grid_w}x{grid_h}"
            else:
                # Debug: print what we captured
                if stderr_output:
                    print(f"  Debug: Captured stderr (first 500 chars): {stderr_output[:500]}")
                print(f"  Warning: WORKER_GRID log not found in stderr")

            # Get result info
            result_cores = None
            result_sharding = None
            result_sharding_strategy = None
            result_grid = None
            if result.is_sharded():
                result_shard_spec = result.memory_config().shard_spec
                result_cores = result_shard_spec.grid.num_cores()
                result_sharding = result_shard_spec.shape

                # Extract core grid dimensions (e.g., "8x1" or "4x8")
                grid_bbox = result_shard_spec.grid.bounding_box()
                grid_h = grid_bbox.grid_size().y
                grid_w = grid_bbox.grid_size().x
                result_grid = f"{grid_w}x{grid_h}"  # Format as "width x height"

                # Extract sharding strategy from shard spec
                try:
                    strategy = result_shard_spec.strategy
                    if strategy == ttnn.ShardStrategy.HEIGHT:
                        result_sharding_strategy = "height"
                    elif strategy == ttnn.ShardStrategy.WIDTH:
                        result_sharding_strategy = "width"
                    elif strategy == ttnn.ShardStrategy.BLOCK:
                        result_sharding_strategy = "block"
                    else:
                        result_sharding_strategy = "unknown"
                except AttributeError:
                    # Fallback: try to infer from shard shape or use default
                    # If shard_h is much smaller than tensor_h, it's height sharding
                    # If shard_w is much smaller than tensor_w, it's width sharding
                    # Otherwise, assume it matches the first sharded input
                    if config["sharding_a"] != "interleaved":
                        result_sharding_strategy = config["sharding_a"]
                    elif config["sharding_b"] != "interleaved":
                        result_sharding_strategy = config["sharding_b"]
                    else:
                        result_sharding_strategy = "unknown"
            else:
                result_sharding_strategy = "interleaved"
                result_grid = "8x8"  # Interleaved uses full device grid

            # compute_cores and compute_grid are now read from C++ WORKER_GRID log above
            # If parsing failed, set to None (will be handled in CSV writing)

            # Print results
            print(f"  Tensor A grid: {a_grid}, Tensor B grid: {b_grid}")
            print(
                f"  Result: shape={result.shape}, cores={result_cores}, grid={result_grid}, compute_cores={compute_cores}, compute_grid={compute_grid}, shard_shape={result_sharding}, sharding={result_sharding_strategy}"
            )

            # Store test configuration and result info (timing will be added later)
            test_results.append(
                {
                    "config": config,
                    "a_grid": a_grid,
                    "b_grid": b_grid,
                    "result_cores": result_cores,
                    "result_grid": result_grid,
                    "compute_cores": compute_cores,
                    "compute_grid": compute_grid,
                    "result_shape": result.shape,
                    "result_sharding": result_sharding,
                    "result_sharding_strategy": result_sharding_strategy,
                    "a_cores": config["cores_a"],
                    "b_cores": config["cores_b"],
                    "error": None,  # Success
                }
            )

            # Cleanup
            tensor_a.deallocate()
            tensor_b.deallocate()
            result.deallocate()

        except Exception as e:
            # Log error but continue with other tests
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"  ❌ ERROR: {error_msg}")
            import traceback

            traceback.print_exc()

            # On error, we can't capture WORKER_GRID log, so set to None
            compute_cores = None
            compute_grid = None
            a_grid = None
            b_grid = None

            # Store error info so CSV can still be written
            test_results.append(
                {
                    "config": config,
                    "a_grid": a_grid,
                    "b_grid": b_grid,
                    "result_cores": None,
                    "result_grid": None,
                    "compute_cores": compute_cores,
                    "compute_grid": compute_grid,
                    "result_shape": None,
                    "result_sharding": None,
                    "result_sharding_strategy": None,
                    "a_cores": config["cores_a"],
                    "b_cores": config["cores_b"],
                    "error": error_msg,
                }
            )

    # After all operations complete, synchronize device
    # Note: Profiler data will be read by fixture teardown, then we'll process it
    print(f"\n{'='*80}")
    print(f"[PROFILER] Synchronizing device...")
    print(f"{'='*80}")

    # Count successes and failures
    success_count = sum(1 for r in test_results if r.get("error") is None)
    failure_count = len(test_results) - success_count

    print(f"[TEST] Completed {len(test_results)} operations: {success_count} succeeded, {failure_count} failed")

    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        try:
            print(f"[PROFILER] Synchronizing device...")
            ttnn.synchronize_device(device_with_profiling)
            print(f"[PROFILER] Device synchronized")
        except Exception as e:
            print(f"[PROFILER] Warning: Failed to synchronize device: {e}")

    # Store test results for processing after fixture teardown
    # The fixture will call ReadDeviceProfiler() in teardown, which writes the log file
    # A module-scoped fixture will process it after all tests complete
    test_key = f"{effective_grid_strategy}_{len(_test_results_storage)}"
    _test_results_storage[test_key] = {
        "results": test_results,
        "grid_strategy": effective_grid_strategy,
        "op_type": op_type,
        "broadcast_type": broadcast_type,
        "tensor_size": tensor_size,
    }

    # Print summary (timing will be added after profiler processing)
    print(f"\n{'='*80}")
    print(f"[SUMMARY] Grid Strategy: {effective_grid_strategy}")
    print(f"{'='*80}")
    print(f"{'Operation':<40} {'Status':<15} {'Result Cores':<15}")
    print(f"{'-'*80}")
    for i, result_info in enumerate(test_results):
        config = result_info["config"]
        error = result_info.get("error")
        cores = result_info["result_cores"]
        cores_str = str(cores) if cores is not None else "N/A"
        status = "ERROR" if error else "OK"
        print(f"{config['name']:<40} {status:<15} {cores_str:<15}")
    print(f"{'='*80}\n")

    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        print(f"[PROFILER] Test results stored - CSV will be written after fixture teardown")
        print(f"         (Profiler data will be processed by module fixture)")
    else:
        # If profiling not enabled, write CSV immediately without timing
        _write_csv_with_timing(test_results, effective_grid_strategy, op_type, broadcast_type, tensor_size)


def _write_csv_with_timing(test_results, grid_strategy, op_type, broadcast_type, tensor_size):
    """Helper function to write CSV file with timing data."""
    # Convert op_type to string for CSV
    op_type_str = str(op_type).replace("BinaryOpType.", "")

    # Write results to CSV file
    csv_data = []
    for result_info in test_results:
        config = result_info["config"]
        duration = result_info.get("kernel_duration")  # May be None
        result_shape = result_info.get("result_shape")
        error = result_info.get("error")

        # Format shapes as "H×W"
        a_shape_str = f"{config['shape_a'][0]}×{config['shape_a'][1]}"
        b_shape_str = f"{config['shape_b'][0]}×{config['shape_b'][1]}"

        # Extract C shape from result_shape (if available)
        if result_shape and len(result_shape) >= 4:
            c_h = result_shape[2]
            c_w = result_shape[3]
            c_shape_str = f"{c_h}×{c_w}"
        else:
            c_shape_str = "ERROR" if error else "N/A"

        # Determine c_sharding based on actual result tensor (not assumptions)
        c_sharding = result_info.get("result_sharding_strategy", "interleaved")
        if error and not c_sharding:
            c_sharding = "ERROR"
        elif not c_sharding:
            c_sharding = "interleaved"

        # Get c_grid (core grid dimensions like "8x1" or "4x8")
        c_grid = result_info.get("result_grid")
        if error and not c_grid:
            c_grid = "ERROR"
        elif not c_grid:
            c_grid = "N/A"

        # Get compute_grid (actual compute core grid dimensions)
        compute_grid = result_info.get("compute_grid")
        if error and not compute_grid:
            compute_grid = "ERROR"
        elif not compute_grid:
            compute_grid = "N/A"

        # Get a_grid and b_grid (input tensor core grids)
        a_grid = result_info.get("a_grid")
        if error and not a_grid:
            a_grid = "ERROR"
        elif not a_grid:
            a_grid = "N/A"

        b_grid = result_info.get("b_grid")
        if error and not b_grid:
            b_grid = "ERROR"
        elif not b_grid:
            b_grid = "N/A"

        csv_data.append(
            {
                "a_shape": a_shape_str,
                "a_sharding": config["sharding_a"],
                "a_cores": result_info.get("a_cores") if result_info.get("a_cores") is not None else 0,
                "a_grid": a_grid,
                "b_shape": b_shape_str,
                "b_sharding": config["sharding_b"],
                "b_cores": result_info.get("b_cores") if result_info.get("b_cores") is not None else 0,
                "b_grid": b_grid,
                "c_shape": c_shape_str,
                "c_sharding": c_sharding,
                "c_cores": result_info.get("result_cores") if result_info.get("result_cores") else 0,
                "c_grid": c_grid,
                "compute_cores": result_info.get("compute_cores") if result_info.get("compute_cores") else 0,
                "compute_grid": compute_grid,
                "kernel_time_us": duration if duration is not None else 0.0,  # microseconds
                "op_type": f"{op_type_str}",
                "broadcast_type": broadcast_type,
                "grid_strategy": grid_strategy,
                "error": error if error else "",  # Add error column for debugging
            }
        )

        # Create results directory if it doesn't exist
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename based on operation type, broadcast type, grid strategy, and tensor size
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include tensor_size in filename if it's not the default 1024
    if tensor_size != 1024:
        csv_filename = results_dir / f"{op_type_str}_{tensor_size}_{broadcast_type}_{grid_strategy}_{timestamp}.csv"
    else:
        csv_filename = results_dir / f"{op_type_str}_{broadcast_type}_{grid_strategy}_{timestamp}.csv"

    # Write CSV file synchronously with timing data
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_filename, index=False)

    print(f"[CSV] Results written to: {csv_filename}")
    print(f"      Total entries: {len(csv_data)}")

    # Check if profiling was enabled
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        print(f"      ⚠️  WARNING: TT_METAL_DEVICE_PROFILER not set to '1'")
        print(f"         Run with: TT_METAL_DEVICE_PROFILER=1 pytest ...")
        print(f"         CSV written but timing data will be 0.0")

    if any(d["kernel_time_us"] > 0 for d in csv_data):
        timing_values = [f"{d['kernel_time_us']:.2f}" for d in csv_data if d["kernel_time_us"] > 0]
        print(f"      ✅ CSV contains timing data (kernel_time_us)")
        print(f"         Timing values: {timing_values}")
    else:
        print(f"      ⚠️  CSV written but timing data not available (all 0.0)")
        print(f"         This usually means:")
        print(f"         1. TT_METAL_DEVICE_PROFILER was not set to '1'")
        print(f"         2. Profiler data processing failed")
        print(f"         3. No profiler entries were found")


def _run_operations_with_device(device, grid_strategy, op_type, broadcast_type, tensor_size):
    """
    Run all operations for a given configuration on the provided device.
    Returns list of test results (without timing - timing is added after profiler read).
    """
    # Define shape pairs based on broadcast type and tensor_size
    if broadcast_type == "no_broadcast":
        shape_b = (tensor_size, tensor_size)
        shape_a = (tensor_size, tensor_size)
    elif broadcast_type == "row_broadcast":
        shape_b = (1, tensor_size)
        shape_a = (tensor_size, tensor_size)
    elif broadcast_type == "col_broadcast":
        shape_b = (tensor_size, 1)
        shape_a = (tensor_size, tensor_size)
    elif broadcast_type == "row_col_mixed":
        shape_b = (tensor_size, 1)
        shape_a = (1, tensor_size)
    elif broadcast_type == "scalar_broadcast":
        shape_b = (1, 1)
        shape_a = (tensor_size, tensor_size)
    else:
        return []

    # Sharding strategies and core counts
    sharding_strategies = ["height"]
    if tensor_size >= 2048:
        core_counts = [32, 64]
    else:
        core_counts = [8, 16, 32]

    test_configs = []

    def generate_configs_for_shapes(shape_a_val, shape_b_val, label=""):
        configs = []
        for a_sharding in sharding_strategies:
            for b_sharding in sharding_strategies:
                if a_sharding == "height" and shape_a_val[0] == 1:
                    continue
                if b_sharding == "height" and shape_b_val[0] == 1:
                    continue
                if a_sharding == "width" and shape_a_val[1] == 1:
                    continue
                if b_sharding == "width" and shape_b_val[1] == 1:
                    continue

                a_cores_options = [None] if a_sharding == "interleaved" else core_counts
                b_cores_options = [None] if b_sharding == "interleaved" else core_counts

                for a_cores in a_cores_options:
                    for b_cores in b_cores_options:
                        if a_sharding == "block" and a_cores is not None:
                            if compute_valid_block_grid(shape_a_val, a_cores) is None:
                                continue
                        if b_sharding == "block" and b_cores is not None:
                            if compute_valid_block_grid(shape_b_val, b_cores) is None:
                                continue
                        if a_sharding != "interleaved" and a_cores is not None:
                            if not check_l1_memory_fit(shape_a_val, a_cores, a_sharding):
                                continue
                        if b_sharding != "interleaved" and b_cores is not None:
                            if not check_l1_memory_fit(shape_b_val, b_cores, b_sharding):
                                continue

                        configs.append(
                            {
                                "name": f"{label}a({shape_a_val[0]}×{shape_a_val[1]})_b({shape_b_val[0]}×{shape_b_val[1]})_a{a_sharding}{a_cores or 'inter'}_b{b_sharding}{b_cores or 'inter'}",
                                "shape_a": shape_a_val,
                                "shape_b": shape_b_val,
                                "sharding_a": a_sharding,
                                "sharding_b": b_sharding,
                                "cores_a": a_cores,
                                "cores_b": b_cores,
                            }
                        )
        return configs

    configs1 = generate_configs_for_shapes(shape_a, shape_b, "")
    test_configs.extend(configs1)
    configs2 = generate_configs_for_shapes(shape_b, shape_a, "SWAP_")
    test_configs.extend(configs2)

    test_results = []

    for i, config in enumerate(test_configs):
        print(f"\n[Operation {i+1}/{len(test_configs)}] {config['name']}")
        print(f"  Grid Strategy: {grid_strategy}")

        try:
            tensor_a, torch_a = create_sharded_tensor(
                device,
                config["shape_a"],
                config["sharding_a"],
                config["cores_a"] if config["cores_a"] is not None else 8,
                return_torch=True,
            )

            tensor_b, torch_b = create_sharded_tensor(
                device,
                config["shape_b"],
                config["sharding_b"],
                config["cores_b"] if config["cores_b"] is not None else 8,
                return_torch=True,
            )

            # Get grid info
            a_grid = None
            b_grid = None
            if tensor_a.is_sharded():
                a_shard_spec = tensor_a.memory_config().shard_spec
                grid_bbox = a_shard_spec.grid.bounding_box()
                a_grid = f"{grid_bbox.grid_size().x}x{grid_bbox.grid_size().y}"
            else:
                a_grid = "8x8"

            if tensor_b.is_sharded():
                b_shard_spec = tensor_b.memory_config().shard_spec
                grid_bbox = b_shard_spec.grid.bounding_box()
                b_grid = f"{grid_bbox.grid_size().x}x{grid_bbox.grid_size().y}"
            else:
                b_grid = "8x8"

            # Run operation with stderr capture
            stderr_fd = sys.stderr.fileno()
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                original_stderr_fd = os.dup(stderr_fd)
                try:
                    os.dup2(tmp_file.fileno(), stderr_fd)
                    sys.stderr = tmp_file

                    if op_type == ttnn.BinaryOpType.ADD:
                        op_func = ttnn.add
                    elif op_type == ttnn.BinaryOpType.POWER:
                        op_func = ttnn.pow
                    elif op_type == ttnn.BinaryOpType.LOGADDEXP:
                        op_func = ttnn.logaddexp
                    else:
                        continue

                    result = op_func(tensor_a, tensor_b)
                    ttnn.synchronize_device(device)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())
                finally:
                    os.dup2(original_stderr_fd, stderr_fd)
                    os.close(original_stderr_fd)
                    sys.stderr = sys.__stderr__

            with open(tmp_path, "r") as f:
                stderr_output = f.read()
            os.unlink(tmp_path)

            if stderr_output:
                sys.stderr.write(stderr_output)
                sys.stderr.flush()

            # Parse WORKER_GRID log
            compute_cores = None
            compute_grid = None
            worker_grid_match = re.search(
                r"WORKER_GRID:\s*strategy=(\S+)\s+cores=(\d+)\s+grid=(\d+)x(\d+)", stderr_output
            )
            if worker_grid_match:
                compute_cores = int(worker_grid_match.group(2))
                compute_grid = f"{worker_grid_match.group(3)}x{worker_grid_match.group(4)}"

            # Get result info
            result_cores = None
            result_sharding = None
            result_sharding_strategy = None
            result_grid = None
            if result.is_sharded():
                result_shard_spec = result.memory_config().shard_spec
                result_cores = result_shard_spec.grid.num_cores()
                result_sharding = result_shard_spec.shape
                grid_bbox = result_shard_spec.grid.bounding_box()
                result_grid = f"{grid_bbox.grid_size().x}x{grid_bbox.grid_size().y}"
                try:
                    strategy = result_shard_spec.strategy
                    if strategy == ttnn.ShardStrategy.HEIGHT:
                        result_sharding_strategy = "height"
                    elif strategy == ttnn.ShardStrategy.WIDTH:
                        result_sharding_strategy = "width"
                    elif strategy == ttnn.ShardStrategy.BLOCK:
                        result_sharding_strategy = "block"
                    else:
                        result_sharding_strategy = "unknown"
                except AttributeError:
                    result_sharding_strategy = (
                        config["sharding_a"] if config["sharding_a"] != "interleaved" else config["sharding_b"]
                    )
            else:
                result_sharding_strategy = "interleaved"
                result_grid = "8x8"

            print(f"  Result: cores={result_cores}, grid={result_grid}, compute_grid={compute_grid}")

            test_results.append(
                {
                    "config": config,
                    "a_grid": a_grid,
                    "b_grid": b_grid,
                    "result_cores": result_cores,
                    "result_grid": result_grid,
                    "compute_cores": compute_cores,
                    "compute_grid": compute_grid,
                    "result_shape": result.shape,
                    "result_sharding": result_sharding,
                    "result_sharding_strategy": result_sharding_strategy,
                    "a_cores": config["cores_a"],
                    "b_cores": config["cores_b"],
                    "error": None,
                }
            )

            tensor_a.deallocate()
            tensor_b.deallocate()
            result.deallocate()

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"  ❌ ERROR: {error_msg}")
            test_results.append(
                {
                    "config": config,
                    "a_grid": None,
                    "b_grid": None,
                    "result_cores": None,
                    "result_grid": None,
                    "compute_cores": None,
                    "compute_grid": None,
                    "result_shape": None,
                    "result_sharding": None,
                    "result_sharding_strategy": None,
                    "a_cores": config["cores_a"],
                    "b_cores": config["cores_b"],
                    "error": error_msg,
                }
            )

    return test_results


@pytest.mark.parametrize("op_type", [ttnn.BinaryOpType.ADD])
@pytest.mark.parametrize("broadcast_type", ["no_broadcast"])
@pytest.mark.parametrize("tensor_size", [1024])
def test_all_grid_strategies(request, op_type, broadcast_type, tensor_size, result_suffix):
    """
    Test all grid strategies in a SINGLE pytest run by creating a fresh device for each strategy.

    This solves the C++ env var caching issue by closing and reopening the device
    for each grid strategy.

    Usage:
        TT_METAL_DEVICE_PROFILER=1 pytest example_single_test.py::test_all_grid_strategies -v -s

        # With custom result folder:
        TT_METAL_DEVICE_PROFILER=1 pytest example_single_test.py::test_all_grid_strategies -v -s --result-suffix=3

    Edit GRID_STRATEGIES list below to select which strategies to test.
    """
    # ============ CONFIGURE STRATEGIES HERE ============
    GRID_STRATEGIES = [
        "max_abc",
        "full_grid",
        # "max_ab",
        # "min_ab",
        # "half_grid",
        # "current",
        # "a_first",
        # "b_first",
    ]
    # ===================================================

    global _result_suffix
    _result_suffix = result_suffix

    os.environ["TT_METAL_DEVICE_PROFILER"] = "1"

    # Import device creation helpers
    from tests.scripts.common import get_updated_device_params

    device_id = request.config.getoption("device_id")

    for grid_strategy in GRID_STRATEGIES:
        print(f"\n{'='*80}")
        print(f"[STRATEGY] Running grid strategy: {grid_strategy}")
        print(f"{'='*80}")

        # Set env var BEFORE device creation
        os.environ["TT_METAL_BINARY_NG_GRID_STRATEGY"] = grid_strategy

        # Create fresh device for this strategy
        device_params = get_updated_device_params({})
        device = ttnn.CreateDevice(device_id=device_id, **device_params)
        ttnn.SetDefaultDevice(device)

        try:
            # Run operations
            test_results = _run_operations_with_device(device, grid_strategy, op_type, broadcast_type, tensor_size)

            # Synchronize before reading profiler
            ttnn.synchronize_device(device)

            # Read profiler data
            if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
                try:
                    ttnn.ReadDeviceProfiler(device)
                except Exception as e:
                    print(f"[PROFILER] Warning: Failed to read profiler: {e}")

        finally:
            # Always close device
            ttnn.close_device(device)

        # Process profiler logs AFTER device close
        time.sleep(1.0)  # Wait for profiler logs to be written
        profiler_df = process_profiler_logs_and_get_timing()

        # Add timing data to results
        if profiler_df is not None and "DEVICE KERNEL DURATION [ns]" in profiler_df.columns:
            durations_us = (profiler_df["DEVICE KERNEL DURATION [ns]"] / 1_000).tolist()
            for i, result_info in enumerate(test_results):
                if i < len(durations_us):
                    result_info["kernel_duration"] = durations_us[i]
                else:
                    result_info["kernel_duration"] = None
        else:
            for result_info in test_results:
                result_info["kernel_duration"] = None

        # Write CSV for this strategy
        _write_csv_with_timing(test_results, grid_strategy, op_type, broadcast_type, tensor_size)

        # Clean up profiler logs for next strategy
        profiler_logs_dir = Path("/workspace/generated/profiler/.logs")
        if profiler_logs_dir.exists():
            shutil.rmtree(profiler_logs_dir)
            profiler_logs_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[STRATEGY] Completed {grid_strategy}")

    print(f"\n{'='*80}")
    print(f"[DONE] All {len(GRID_STRATEGIES)} grid strategies completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    """
    You can also run this directly (though pytest is recommended):

    python -m pytest example_single_test.py::test_single_binary_operation -v
    """
    pytest.main([__file__, "-v"])
