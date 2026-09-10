# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict
from framework.sweeps_logger import sweeps_logger as logger


def capture_peak_memory(test_module, test_vector: dict, device, use_no_dispatch: bool = True) -> Optional[Dict]:
    """
    Capture peak L1 memory usage per core using graph trace.

    Args:
        test_module: Sweep test module with run() function
        test_vector: Test parameters dict
        device: TTNN device
        use_no_dispatch: If True, use NO_DISPATCH mode (fast, no execution).
                        If False, use NORMAL mode (actual execution, real addresses)

    Returns:
        Dict with per-core and device-level memory metrics:
        {
            'peak_total_per_core': int,       # Total peak per core (CB+L1) in bytes
            'peak_cb_per_core': int,          # Circular buffer peak per core in bytes
            'peak_l1_per_core': int,          # L1 buffers peak per core in bytes
            'num_cores': int,                 # Number of cores used
            'peak_total_aggregate': int,      # Worst-case total (peak_per_core × num_cores) in bytes
            'peak_l1_memory_device': int,     # Actual observed device peak in bytes
        }
        or None if capture fails

        Note: peak_total_aggregate (theoretical) is typically much larger than
        peak_l1_memory_device (actual) for operations with sequential execution.
    """
    metrics = None
    # Set once a NO_DISPATCH capture has actually begun: only from that point can the programs
    # already in the cache have been invalidated. Failures before this (no device, ttnn import,
    # grid query) leave the cache untouched, so there is nothing to clear.
    capture_invalidated_cached_programs = False
    try:
        import ttnn

        grid_size = device.compute_with_storage_grid_size()
        num_cores = grid_size.x * grid_size.y

        mode = ttnn.graph.RunMode.NO_DISPATCH if use_no_dispatch else ttnn.graph.RunMode.NORMAL
        ttnn.graph.begin_graph_capture(mode)
        capture_invalidated_cached_programs = use_no_dispatch

        # Execute test (results not used, just capturing memory profile)
        try:
            test_module.run(**test_vector, device=device)
        except Exception as e:
            # Test may fail but memory capture might still be valid
            logger.debug(f"Test execution failed during memory capture: {e}")

        captured_graph = ttnn.graph.end_graph_capture()

        per_core_usage = ttnn.graph.extract_resource_usage_per_core(captured_graph)

        # Also get device-level peak memory for comparison
        peak_l1_memory_device = ttnn.graph.extract_peak_L1_memory_usage(captured_graph)

        metrics = {
            "peak_total_per_core": per_core_usage.peak_total,
            "peak_cb_per_core": per_core_usage.peak_cb,
            "peak_l1_per_core": per_core_usage.peak_l1,
            "num_cores": num_cores,
            "peak_total_aggregate": per_core_usage.peak_total
            * num_cores,  # Worst-case if all cores peak simultaneously
            "peak_l1_memory_device": peak_l1_memory_device,  # Actual observed peak across device
        }

    except Exception as e:
        # If memory capture fails, return None but don't fail the test
        logger.warning(f"Failed to capture peak memory: {e}")
        metrics = None

    if capture_invalidated_cached_programs:
        # A NO_DISPATCH capture does NOT add program cache entries -- ttnn blocks the insert while
        # the graph hook is blocking (#36772), and the entry count is measurably unchanged across a
        # capture. What it does is leave the ALREADY-cached programs unusable, so a later real run
        # that hits one of them reads back garbage.
        #
        # Measured on wormhole (1x1), traced split vectors, 10 vectors with a capture after each:
        #     no capture ................... 10/10 pass
        #     capture, cache kept ..........  5/10 pass
        #     capture, cache cleared ....... 10/10 pass
        #     capture, program cache OFF ... 10/10 pass   <- corruption needs a cache hit
        #
        # Re-running the SAME config straight after its own capture flips it from pass to fail, so
        # this is not a wrong-program/hash-collision effect: a cache hit only overrides runtime
        # args, so whatever the capture disturbs on the device is never rewritten.
        #
        # Clearing forces the next run to re-create the program, which restores it, at the cost of
        # a recompile -- the same cost the runner already accepts at every module boundary
        # (clear_job_device_program_cache). NORMAL captures dispatch for real and are unaffected.
        try:
            device.clear_program_cache()
        except Exception as exc:
            # Deliberately not swallowed. Returning normally here would hand a still-poisoned cache
            # to the next vector, which is the silent wrong answer this function exists to prevent;
            # failing now pins the problem on the capture that caused it.
            raise RuntimeError(
                "Could not clear the device program cache after a NO_DISPATCH memory capture; its "
                "existing entries can no longer be trusted and would corrupt later vectors"
            ) from exc

    return metrics
