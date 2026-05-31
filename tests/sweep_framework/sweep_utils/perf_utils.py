# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import inspect
import subprocess
import shutil
from pathlib import Path
from typing import Any, List, Optional, Tuple, Dict

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


def clear_disk_kernel_cache() -> None:
    """Clear disk kernel cache for current git hash."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short=10", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        cleared_count = 0
        for kernels_dir in Path.home().glob(f".cache/tt-metal-cache/{git_hash}/*/kernels"):
            if kernels_dir.exists():
                shutil.rmtree(kernels_dir)
                cleared_count += 1

        logger.info(f"Cleared {cleared_count} disk kernel cache directories for git hash {git_hash}")
    except Exception as e:
        logger.warning(f"Failed to clear disk kernel cache: {e}")


def _as_number(value):
    """Return (number, True) if value can be parsed as a float, else (value, False)."""
    try:
        return float(value), True
    except (TypeError, ValueError):
        return value, False


def _reduce_perf_rows(rows: List[dict], reducer) -> dict:
    """Combine perf-row dicts field-by-field, applying `reducer` to numeric fields.

    Non-numeric fields keep the first row's value. String-formatted numbers are kept
    as strings (matching the device report's CSV formatting) so downstream consumers
    (simplify_device_perf / roofline message) behave exactly as before.
    """
    if not rows:
        return {}
    result = dict(rows[0])
    for row in rows[1:]:
        for key, value in row.items():
            num, is_num = _as_number(value)
            if not is_num:
                if key not in result:
                    result[key] = value
                continue
            base, base_is_num = _as_number(result.get(key))
            if not base_is_num:
                result[key] = value
                continue
            combined = reducer(base, num)
            result[key] = str(combined) if isinstance(value, str) else combined
    return result


def aggregate_device_perf(opPerfData: List[dict], num_devices: int = 1) -> Optional[dict]:
    """Collapse raw per-(op, device) profiler rows into a single perf dict.

    ``get_device_data_generate_report`` returns device-only rows that DROP the device
    id / op-code columns, but it emits them grouped by device in contiguous blocks (it
    iterates each device, then that device's ops in chronological order). On a mesh the
    same program runs on every device, so for ``num_devices`` devices the row list is
    ``num_devices`` consecutive blocks of ``ops_per_device`` rows each, with the i-th
    row in every block referring to the same logical op.

    Aggregation is therefore two stages:
      1. Across devices: reduce the i-th op of every device block with ``max`` — a mesh
         op only completes once its slowest device finishes.
      2. Across ops: sum the per-op representatives, matching the original
         single-device "composite op" behaviour (a test may dispatch several ops).

    For a single device (or when the row count is not a clean multiple of
    ``num_devices``) this falls back to the original logic: one op returns its row,
    multiple rows are summed as composite ops.
    """
    if not opPerfData:
        return None

    n = len(opPerfData)
    if num_devices and num_devices > 1 and n % num_devices == 0:
        ops_per_device = n // num_devices
        rows = []
        for i in range(ops_per_device):
            ith_op_across_devices = [opPerfData[d * ops_per_device + i] for d in range(num_devices)]
            rows.append(_reduce_perf_rows(ith_op_across_devices, max))
        logger.info(f"Aggregating device perf: {ops_per_device} op(s) across {num_devices} devices ({n} rows).")
    else:
        if num_devices and num_devices > 1:
            logger.warning(
                f"Device perf rows ({n}) not divisible by device count ({num_devices}); "
                "summing all rows without per-device collapse."
            )
        rows = opPerfData

    if len(rows) == 1:
        return rows[0]
    return _reduce_perf_rows(rows, lambda a, b: a + b)


def gather_single_test_perf(device, test_passed):
    if device is None:
        logger.error("Device is None, cannot gather device perf. Failing.")
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
    else:
        try:
            num_devices = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
            return aggregate_device_perf(opPerfData, num_devices)
        except Exception as e:
            logger.info(e)
            return None


def prepare_program_cache_for_comparison(device) -> None:
    """Clear all cache layers before uncached performance measurement.

    Clears:
    1. Disk kernel cache (persistent)
    2. In-memory HashLookup cache (process-lifetime)
    3. Program cache (keeps it enabled for next run)
    """
    import ttnn

    # Clear disk cache
    clear_disk_kernel_cache()

    # Clear in-memory HashLookup cache
    logger.info("Clearing in-memory HashLookup cache")
    ttnn.device.ClearKernelCache()

    # Clear program cache (but keep it enabled)
    num_entries_before = (
        device.num_program_cache_entries() if hasattr(device, "num_program_cache_entries") else "unknown"
    )
    logger.info(f"Clearing program cache (entries before: {num_entries_before})")
    device.clear_program_cache()
    num_entries_after = (
        device.num_program_cache_entries() if hasattr(device, "num_program_cache_entries") else "unknown"
    )
    logger.info(f"Program cache cleared (entries after: {num_entries_after})")


def execute_test(test_module, test_vector: dict, device) -> Tuple[bool, Any, Optional[float]]:
    # Filter 'device' from test_vector to avoid conflict with explicit device param
    if "device" in test_vector:
        test_vector = {k: v for k, v in test_vector.items() if k != "device"}
    # Convert "__ABSENT__" sentinel values to None (missing columns in multi-config suites)
    # Track which keys were originally absent so sweeps can distinguish "master had key: None"
    # from "master never passed key" — needed to match master trace when an op kwarg was None.
    absent_keys = {k for k, v in test_vector.items() if v == "__ABSENT__"}
    test_vector = {k: (None if v == "__ABSENT__" else v) for k, v in test_vector.items()}

    # Only forward __absent_keys__ when run() can accept it; otherwise the
    # extra kwarg would TypeError any run() without **kwargs.
    try:
        sig = inspect.signature(test_module.run)
        accepts_absent = "__absent_keys__" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_absent = False
    if accepts_absent:
        test_vector["__absent_keys__"] = absent_keys

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
) -> Tuple[bool, Any, Dict[str, Optional[float]], Optional[Dict[str, dict]], Optional[Dict[str, Dict]]]:
    # Capture peak memory (NO_DISPATCH mode) if enabled
    peak_memory = None
    if getattr(config, "measure_memory", False):
        from sweep_utils.memory_utils import capture_peak_memory

        logger.info("Capturing peak memory in NO_DISPATCH mode")
        peak_memory = capture_peak_memory(test_module, test_vector, device)

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
        return status, message, e2e_perf, simplified_perf, peak_memory
    else:
        return status, message, e2e_perf, None, peak_memory


def run_single(
    test_module, test_vector: dict, device, config: Any
) -> Tuple[bool, Any, Optional[float], Optional[dict], Optional[Dict]]:
    status, message, e2e_ms = execute_test(test_module, test_vector, device)

    # Capture peak memory if enabled
    peak_memory = None
    if getattr(config, "measure_memory", False):
        from sweep_utils.memory_utils import capture_peak_memory

        peak_memory = capture_peak_memory(test_module, test_vector, device, use_no_dispatch=True)

    if getattr(config, "measure_device_perf", False):
        perf_result = gather_single_test_perf(device, status)
        message = get_updated_message(message, perf_result)
        simplified_perf = simplify_device_perf(perf_result)
        return status, message, e2e_ms, simplified_perf, peak_memory
    else:
        return status, message, e2e_ms, None, peak_memory
