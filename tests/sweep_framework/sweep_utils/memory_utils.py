# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
        Dict with per-core memory metrics:
        {
            'peak_total_per_core': int,  # Total peak per core (bytes)
            'peak_cb_per_core': int,     # CB peak per core (bytes)
            'peak_l1_per_core': int,     # L1 buffers peak per core (bytes)
            'num_cores': int,            # Number of cores
            'peak_total': int,           # Total across all cores (for backwards compat)
        }
        or None if capture fails
    """
    try:
        import ttnn

        # Get core count from device
        grid_size = device.compute_with_storage_grid_size()
        num_cores = grid_size.x * grid_size.y

        mode = ttnn.graph.RunMode.NO_DISPATCH if use_no_dispatch else ttnn.graph.RunMode.NORMAL
        ttnn.graph.begin_graph_capture(mode)

        # Execute test (results not used, just capturing memory profile)
        try:
            test_module.run(**test_vector, device=device)
        except Exception as e:
            # Test may fail but memory capture might still be valid
            logger.debug(f"Test execution failed during memory capture: {e}")

        captured_graph = ttnn.graph.end_graph_capture()

        # Use per-core API (new, recommended)
        usage = ttnn.graph.extract_resource_usage_per_core(captured_graph, num_cores)

        return {
            "peak_total_per_core": usage.peak_total,
            "peak_cb_per_core": usage.peak_cb,
            "peak_l1_per_core": usage.peak_l1,
            "num_cores": num_cores,
            "peak_total": usage.peak_total * num_cores,  # For backwards compatibility
        }

    except Exception as e:
        # If memory capture fails, return None but don't fail the test
        logger.warning(f"Failed to capture peak memory: {e}")
        return None


def capture_peak_memory_with_cache_comparison(test_module, test_vector: dict, device) -> Dict[str, Optional[Dict]]:
    """
    Capture peak L1 memory for both uncached and cached runs.

    Args:
        test_module: Sweep test module
        test_vector: Test parameters
        device: TTNN device

    Returns:
        Dict with 'uncached' and 'cached' peak memory values
    """
    # For memory profiling, NO_DISPATCH mode doesn't distinguish cached vs uncached
    # since it doesn't actually execute. We'll capture once and use for both.
    peak_memory = capture_peak_memory(test_module, test_vector, device, use_no_dispatch=True)

    return {"uncached": peak_memory, "cached": peak_memory}  # Same value since NO_DISPATCH doesn't compile/cache
