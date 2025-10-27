# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Tuple, Dict

from framework.sweeps_logger import sweeps_logger as logger
from tracy.common import PROFILER_LOGS_DIR
from tracy.process_ops_logs import get_device_data_generate_report
from sweep_utils.roofline_utils import get_updated_message


# Device profiler keys to retain in simplified outputs
DEVICE_PERF_KEYS = [
    "DEVICE FW DURATION [ns]",
    "DEVICE KERNEL DURATION [ns]",
    "OP TO OP LATENCY [ns]",
    "DEVICE BRISC FW DURATION [ns]",
    "DEVICE NCRISC FW DURATION [ns]",
]


def gather_single_test_perf(device, test_passed):
    if device is None or device.get_num_devices() > 1:
        logger.error("Multi-device perf is not supported. Failing.")
        return None

    # Read profiler data from device
    logger.info("Reading profiler data from device")
    import ttnn

    ttnn.ReadDeviceProfiler(device)
    logger.info("Reading profiler data from device done")
    try:
        opPerfData = get_device_data_generate_report(
            PROFILER_LOGS_DIR, None, None, None, export_csv=False, cleanup_device_log=True
        )
    except Exception as e:
        logger.warning(f"Failed to get device profiler data: {e}")
        opPerfData = []

    if not test_passed:
        return None
    elif opPerfData == []:
        logger.warning("No profiling data available. Using dummy data for testing purposes.")

        dummy_data = {
            "DEVICE FW DURATION [ns]": 0,
            "DEVICE KERNEL DURATION [ns]": 0,
            "OP TO OP LATENCY [ns]": 0,
            "DEVICE BRISC FW DURATION [ns]": 0,
            "DEVICE NCRISC FW DURATION [ns]": 0,
        }
        return dummy_data
    elif len(opPerfData) > 1:
        logger.info("Composite op detected in device perf measurement. Will aggregate results.")
        try:
            for key in opPerfData[0].keys():
                value = opPerfData[0][key]
                for i in range(1, len(opPerfData)):
                    if key in opPerfData[i]:
                        if type(value) == str:
                            opPerfData[0][key] = str(float(value) + float(opPerfData[i][key]))
                        else:
                            opPerfData[0][key] = value + opPerfData[i][key]
            return opPerfData[0]
        except Exception as e:
            logger.info(e)
            return None
    else:
        return opPerfData[0]


def prepare_program_cache_for_comparison(device) -> None:
    num_entries_before = (
        device.num_program_cache_entries() if hasattr(device, "num_program_cache_entries") else "unknown"
    )
    logger.info(f"Clearing program cache for --perf-with-cache (entries before: {num_entries_before})")
    device.disable_and_clear_program_cache()
    device.enable_program_cache()  # Re-enable for cache comparison
    num_entries_after = (
        device.num_program_cache_entries() if hasattr(device, "num_program_cache_entries") else "unknown"
    )
    logger.info(f"Program cache cleared and re-enabled (entries after: {num_entries_after})")


def execute_test(test_module, test_vector: dict, device) -> Tuple[bool, Any, Optional[float]]:
    results = test_module.run(**test_vector, device=device)
    if isinstance(results, list):
        status, message = results[0]
        e2e_ms = results[1] / 1000000  # Nanoseconds to milliseconds
    else:
        status, message = results
        e2e_ms = None
    return status, message, e2e_ms


def simplify_device_perf(perf: Optional[dict]) -> dict:
    if not perf:
        return {}
    simplified: Dict[str, Any] = {}
    for key in DEVICE_PERF_KEYS:
        if key in perf:
            simplified[key] = perf[key]
    return simplified


def run_with_cache_comparison(
    test_module, test_vector: dict, device, config: Any
) -> Tuple[bool, Any, Dict[str, Optional[float]], Optional[Dict[str, dict]]]:
    # Prepare program cache state
    prepare_program_cache_for_comparison(device)

    # First run (without cache)
    status_uncached, message_uncached, e2e_uncached_ms = execute_test(test_module, test_vector, device)

    device_perf_uncached = None
    if getattr(config, "measure_device_perf", False):
        device_perf_uncached = gather_single_test_perf(device, status_uncached)
        # Clear the profiler log file for the next run to isolate device perf measurements
        from tracy.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG
        import os as _os

        device_log_path = _os.path.join(PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG)
        if _os.path.exists(device_log_path):
            _os.remove(device_log_path)

    # Second run (with cache)
    status_cached, message_cached, e2e_cached_ms = execute_test(test_module, test_vector, device)

    device_perf_cached = None
    if getattr(config, "measure_device_perf", False):
        device_perf_cached = gather_single_test_perf(device, status_cached)

    # Determine combined status and message
    if not status_uncached:
        if status_cached:
            status = False
            message = f"UNCACHED RUN FAILED: {message_uncached} (cached run passed: {message_cached})"
        else:
            status = False
            message = f"BOTH RUNS FAILED - Uncached: {message_uncached}, Cached: {message_cached}"
    elif not status_cached:
        status = False
        message = f"CACHED RUN FAILED: {message_cached} (uncached run passed: {message_uncached})"
    else:
        status = True
        if str(message_uncached) != str(message_cached):
            message = (
                f"BOTH RUNS PASSED BUT MESSAGES DIFFER - " f"Uncached: {message_uncached}, Cached: {message_cached}"
            )
            logger.warning(
                f"Message mismatch between cached and uncached runs: "
                f"uncached={message_uncached}, cached={message_cached}"
            )
        else:
            message = message_uncached

    # e2e perf dict
    e2e_perf = {"uncached": e2e_uncached_ms, "cached": e2e_cached_ms}

    # Device perf dict (simplified) and message augmentation
    if getattr(config, "measure_device_perf", False):
        combined_device_perf = {"uncached": device_perf_uncached, "cached": device_perf_cached}
        if device_perf_uncached or device_perf_cached:
            message = get_updated_message(message, combined_device_perf)

        simplified_perf: Dict[str, dict] = {}
        if device_perf_uncached:
            simplified_perf["uncached"] = simplify_device_perf(device_perf_uncached)
        if device_perf_cached:
            simplified_perf["cached"] = simplify_device_perf(device_perf_cached)
        return status, message, e2e_perf, simplified_perf
    else:
        return status, message, e2e_perf, None


def run_single(
    test_module, test_vector: dict, device, config: Any
) -> Tuple[bool, Any, Optional[float], Optional[dict]]:
    status, message, e2e_ms = execute_test(test_module, test_vector, device)

    if getattr(config, "measure_device_perf", False):
        perf_result = gather_single_test_perf(device, status)
        message = get_updated_message(message, perf_result)
        simplified_perf = simplify_device_perf(perf_result)
        return status, message, e2e_ms, simplified_perf
    else:
        return status, message, e2e_ms, None
