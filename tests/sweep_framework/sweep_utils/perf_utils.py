# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import inspect
import subprocess
import shutil
from pathlib import Path
from typing import Any, Optional, Tuple, Dict

from framework.sweeps_logger import sweeps_logger as logger
from sweep_utils.roofline_utils import get_updated_message


# Device profiler keys to retain in simplified outputs
DEVICE_PERF_KEYS = [
    "DEVICE FW DURATION [ns]",
    "DEVICE KERNEL DURATION [ns]",
    "DEVICE BRISC KERNEL DURATION [ns]",
    "DEVICE NCRISC KERNEL DURATION [ns]",
    "DEVICE TRISC0 KERNEL DURATION [ns]",
    "DEVICE TRISC1 KERNEL DURATION [ns]",
    "DEVICE TRISC2 KERNEL DURATION [ns]",
    "CORE COUNT",
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


def _resolve_perf_device(device, test_module):
    # Some model_traced ops (add/sdpa/paged_sdpa, conv2d) open their own mesh
    # device inside run() (the fixture yields None) and cache it in a persistent
    # module global that stays open across vectors. The global's name varies by
    # module -- _CUR_DEVICE (add, sdpa, paged_sdpa) or _CONV_DEV (conv2d) -- so
    # fall back through the known names to find the live device for the read.
    if device is not None:
        return device
    for _name in ("_CUR_DEVICE", "_CONV_DEV"):
        d = getattr(test_module, _name, None)
        if d is not None:
            return d
    return None


def gather_single_test_perf(device, test_passed):
    if device is None:
        logger.error("Device perf: no device available. Failing.")
        return None
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        return None

    import ttnn

    # Modern Tracy flow: ReadDeviceProfiler triggers the C++ post-process
    # (TT_METAL_PROFILER_CPP_POST_PROCESS=1), then get_latest_programs_perf_data()
    # returns per-chip analysis results in memory (no CSV). Works on multi-chip
    # meshes (T3K / galaxy); the legacy CSV path only worked single-chip and host-
    # read remote chips mid-run -> inter-chip ethernet timeout.
    logger.info("Reading profiler data from device")
    ttnn.ReadDeviceProfiler(device)
    logger.info("Reading profiler data from device done")

    if not test_passed:
        return None

    try:
        perf_by_chip = ttnn.get_latest_programs_perf_data()
    except Exception as e:
        logger.warning(f"Failed to get device profiler data: {e}")
        return None

    if not perf_by_chip:
        logger.warning("No profiling data available.")
        return None

    # Aggregate per distinct device program, keyed by its execution uid. Each
    # program is replicated across the mesh, so take the max across chips (the
    # bottleneck chip = that program's real latency). A single op may decompose
    # into several device programs (composite op), so sum each analysis across the
    # distinct programs -- matching the legacy CSV path's composite-op summation.
    per_program: Dict[Any, Dict[str, int]] = {}
    core_count = 0
    for _chip, programs in perf_by_chip.items():
        for program in programs:
            core_count = max(core_count, int(getattr(program, "core_count", 0) or 0))
            uid = program.program_execution_uid
            key = (uid.runtime_id, uid.trace_id, uid.trace_id_counter)
            slot = per_program.setdefault(key, {})
            for name, result in program.program_analyses_results.items():
                slot[name] = max(slot.get(name, 0), int(result.duration))

    aggregated: Dict[str, int] = {}
    for slot in per_program.values():
        for name, duration in slot.items():
            aggregated[name] = aggregated.get(name, 0) + duration

    if not aggregated:
        logger.warning("No profiling analyses available.")
        return None

    aggregated["CORE COUNT"] = core_count
    return aggregated


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
        # Each gather's ttnn.ReadDeviceProfiler refreshes the in-memory "latest"
        # program perf data, so the cached run below reads its own data with no
        # legacy CSV-log clearing needed.
        device_perf_uncached = gather_single_test_perf(_resolve_perf_device(device, test_module), status_uncached)

    # Second run (with cache)
    status_cached, message_cached, e2e_cached_ms = execute_test(test_module, test_vector, device)

    device_perf_cached = None
    if getattr(config, "measure_device_perf", False):
        device_perf_cached = gather_single_test_perf(_resolve_perf_device(device, test_module), status_cached)

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
        perf_result = gather_single_test_perf(_resolve_perf_device(device, test_module), status)
        message = get_updated_message(message, perf_result)
        simplified_perf = simplify_device_perf(perf_result)
        return status, message, e2e_ms, simplified_perf, peak_memory
    else:
        return status, message, e2e_ms, None, peak_memory
