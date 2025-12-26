# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict
import ttnn
from framework.sweeps_logger import sweeps_logger as logger


def capture_peak_memory(test_module, test_vector: dict, device, use_no_dispatch: bool = True) -> Optional[int]:
    """
    Capture peak L1 memory usage using graph trace.

    Args:
        test_module: Sweep test module with run() function
        test_vector: Test parameters dict
        device: TTNN device
        use_no_dispatch: If True, use NO_DISPATCH mode (fast, no execution).
                        If False, use NORMAL mode (actual execution, real addresses)

    Returns:
        Peak L1 memory in bytes, or None if capture fails
    """
    try:
        mode = ttnn.graph.RunMode.NO_DISPATCH if use_no_dispatch else ttnn.graph.RunMode.NORMAL
        ttnn.graph.begin_graph_capture(mode)

        # Execute test (results not used, just capturing memory profile)
        try:
            test_module.run(**test_vector, device=device)
        except Exception as e:
            # Test may fail but memory capture might still be valid
            logger.debug(f"Test execution failed during memory capture: {e}")

        captured_graph = ttnn.graph.end_graph_capture()
        peak_l1 = ttnn.graph.extract_peak_L1_memory_usage(captured_graph)

        return peak_l1

    except Exception as e:
        orig_exc = e
        try:
            ttnn.graph.end_graph_capture()
        except Exception as cleanup_exc:
            logger.warning(f"Failed to end graph capture: {cleanup_exc}")
        logger.warning(f"Failed to capture peak memory: {orig_exc}")
        return None


def capture_peak_memory_with_cache_comparison(test_module, test_vector: dict, device) -> Dict[str, Optional[int]]:
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
