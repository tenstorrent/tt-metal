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

# Sentinel returned as the device-perf value when a sweep module opts a vector out of
# profiling by setting _SKIP_DEVICE_PERF (e.g. conv2d's heavy FABRIC_1D path, where the
# profiler read/clock-ARC over the busy fabric hangs). Distinct from None, which means
# "profiler ran but produced nothing" -> FAIL_UNSUPPORTED_DEVICE_PERF. The runner treats
# this sentinel as PASS with device-perf N/A, so an unprofilable-but-correct vector is
# not counted as a failure.
DEVICE_PERF_SKIPPED = "__device_perf_skipped__"

# Distinct from DEVICE_PERF_SKIPPED: the profiler was NOT skipped, its readback THREW
# (e.g. "Invalid packet type" out of DeviceProfiler::readRiscProfilerResults). The runner
# needs to tell the two apart because it treats a readback failure as evidence about the
# DEVICE, combined with the vector's own verdict:
#   readback failed + vector PASSED -> PASS with device-perf N/A, carry on
#   readback failed + vector FAILED -> presume the device is wedged: mark the vector
#                                      NOT_RUN (not a test failure) and end the run
DEVICE_PERF_READBACK_FAILED = "__device_perf_readback_failed__"


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
    # CCL ops (all_gather etc.) don't keep a module-global device; they hold it in
    # ccl_common's persistent _DEVICE_CACHE (kept open across vectors when the
    # profiler is on). Read the live cached device if present. The cache is set to
    # None on teardown/failure, so this self-corrects -- never a stale/closed read.
    # Scan sys.modules instead of importing by name: the sweep module imports
    # ccl_common as "tests.sweep_framework.sweep_utils.ccl_common" while a plain
    # "from sweep_utils import ccl_common" here is a DIFFERENT module object (two
    # PYTHONPATH roots -> two sys.modules entries, two separate _DEVICE_CACHE
    # dicts). Reading the already-imported module that actually owns the device
    # avoids creating a fresh, empty cache.
    import sys

    for _name, _mod in list(sys.modules.items()):
        if _mod is not None and _name.endswith("sweep_utils.ccl_common"):
            cache = getattr(_mod, "_DEVICE_CACHE", None)
            if isinstance(cache, dict) and cache.get("mesh_device") is not None:
                return cache["mesh_device"]
    return None


class ProfilerReadTimeout(RuntimeError):
    """ttnn.ReadDeviceProfiler() did not return within the watchdog budget."""


def _read_device_profiler_with_watchdog(ttnn_mod, device):
    """ttnn.ReadDeviceProfiler(device) with a wall-clock budget.

    Defensive, not a known-defect workaround. A stalled C++ call is not an exception, so the
    try/except around this call cannot see one; the watchdog converts a stall into a
    recoverable "device-perf unavailable" instead of a 300s vector timeout plus reset+retry.

    CORRECTION to an earlier claim in this file's history: this was described as an
    "intermittent 32-chip ReadDeviceProfiler stall". That was wrong. The stall was reproduced
    only with an INCOMPLETE flag set (TT_METAL_DEVICE_PROFILER + CPP_POST_PROCESS, without
    TT_METAL_PROFILER_MID_RUN_DUMP). Without MID_RUN_DUMP, getProgramsPerfDataMidRun() is false,
    get_latest_programs_perf_data() returns nothing, and the profiler data is never consumed --
    the probe logged "perf programs=0" for 42 rounds and then blocked on round 43. With the full
    CI flag set (the set enable_profiler() applies) the same probe ran 80 rounds clean twice, on
    a 32-chip Galaxy. Profiling works on 32 chips; do not use this watchdog as evidence otherwise.

    Budget: TTNN_SWEEP_PROFILER_READ_TIMEOUT_S (default 120s, ~100x the observed 1.2s).
    """
    import threading

    budget = max(1, int(os.environ.get("TTNN_SWEEP_PROFILER_READ_TIMEOUT_S", "120")))
    box = {}

    def _read():
        try:
            ttnn_mod.ReadDeviceProfiler(device)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            box["exc"] = exc

    worker = threading.Thread(target=_read, name="ReadDeviceProfiler-watchdog", daemon=True)
    worker.start()
    worker.join(budget)
    if worker.is_alive():
        raise ProfilerReadTimeout(
            f"ttnn.ReadDeviceProfiler() did not return within {budget}s on a "
            f"{_safe_device_count(ttnn_mod)}-chip mesh; treating device-perf as unavailable "
            "for this vector (thread leaked -- the stuck call cannot be cancelled)."
        )
    if "exc" in box:
        raise box["exc"]


def _safe_device_count(ttnn_mod):
    try:
        return ttnn_mod.get_num_devices()
    except Exception:
        return "?"


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
    try:
        _read_device_profiler_with_watchdog(ttnn, device)
    except Exception as e:
        # A profiler READBACK failure must not OVERWRITE the vector's own verdict. execute_test()
        # has already run the op and its PCC check by the time we get here, so `status` is
        # decided; leaving this call unguarded (while the get_latest_programs_perf_data() call
        # below WAS guarded) let the exception propagate out of the test body and replace that
        # verdict -- and its message -- with a profiler traceback.
        #
        # Note this does NOT decide pass/fail: callers return the original status/message
        # alongside this sentinel, and the runner only maps it to PASS when status is already
        # True. A vector whose PCC failed stays a failure, and now reports the PCC message
        # instead of a misleading profiler error.
        #
        # Seen on Galaxy run 30509849370 job 90770018256, copy_model_traced 75a4...:
        #   04:21:04.466  comp_pcc: One tensor is all zero. PCC undefined; falling back to allclose
        #   04:21:04.468  Reading profiler data from device
        #   04:21:05.699  TT_THROW @ tt_metal/impl/profiler/profiler.cpp:1830: Invalid packet type
        #                 DeviceProfiler::readRiscProfilerResults(...)
        # -> recorded as FAIL_ASSERT_EXCEPTION. The host decoded a marker whose 3-bit
        # packet-type field was 6 or 7 (only 0-5 are valid and all six are handled), i.e. it
        # parsed past the data that iteration wrote -- a stale DEVICE_BUFFER_END_INDEX_*.
        # That vector's PCC result is NOT recoverable from the log, and the comp_pcc branch
        # that fired requires exactly one tensor to be all-zero and the other not, which
        # normally makes the follow-up allclose fail -- so it was probably a real PCC failure
        # being masked by the profiler traceback, not a passing vector being failed.
        #
        # Return the READBACK_FAILED sentinel so the runner can combine it with the
        # vector's own verdict (see the sentinel's definition): perf N/A on a passing
        # vector, wedged-device abort on a failing one.
        logger.warning(f"Device profiler readback failed ({e}); reporting device-perf N/A for this vector.")
        return DEVICE_PERF_READBACK_FAILED
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

    # A sweep module can set _SKIP_DEVICE_PERF (per-vector) to opt this vector out of
    # the profiler read -- e.g. conv2d's heavy FABRIC_1D path, where the profiler's
    # remote-chip AICLK ARC read hangs over the fabric-busy ETH link. Checked AFTER
    # execute_test() since run() sets the flag. dp_skipped -> return the SKIPPED
    # sentinel so the runner marks PASS (perf N/A), not FAIL_UNSUPPORTED_DEVICE_PERF.
    dp_requested = getattr(config, "measure_device_perf", False)
    dp_skipped = dp_requested and getattr(test_module, "_SKIP_DEVICE_PERF", False)
    measure_dp = dp_requested and not dp_skipped

    device_perf_uncached = None
    if measure_dp:
        # Each gather's ttnn.ReadDeviceProfiler refreshes the in-memory "latest"
        # program perf data, so the cached run below reads its own data with no
        # legacy CSV-log clearing needed.
        device_perf_uncached = gather_single_test_perf(_resolve_perf_device(device, test_module), status_uncached)

    # Second run (with cache)
    status_cached, message_cached, e2e_cached_ms = execute_test(test_module, test_vector, device)

    device_perf_cached = None
    if measure_dp:
        device_perf_cached = gather_single_test_perf(_resolve_perf_device(device, test_module), status_cached)

    # A profiler readback failure on either run means no comparable perf pair, so the
    # sentinel replaces the perf value. It must NOT short-circuit the status combination
    # below: returning status_uncached here would report PASS whenever the uncached run
    # passed and the CACHED run failed, masking a cache-only correctness failure under
    # --perf-with-cache --device-perf. Recorded and returned after the combination instead.
    readback_failed = DEVICE_PERF_READBACK_FAILED in (device_perf_uncached, device_perf_cached)

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
    if readback_failed:
        # Combined status/message preserved -- only the perf value is replaced, so a
        # cache-only failure still reports as a failure (and the runner's wedged-device
        # rule sees both the readback failure and that verdict).
        return status, message, e2e_perf, DEVICE_PERF_READBACK_FAILED, peak_memory
    if measure_dp:
        combined_device_perf = {"uncached": device_perf_uncached, "cached": device_perf_cached}
        if device_perf_uncached or device_perf_cached:
            message = get_updated_message(message, combined_device_perf)

        simplified_perf: Dict[str, dict] = {}
        if device_perf_uncached:
            simplified_perf["uncached"] = simplify_device_perf(device_perf_uncached)
        if device_perf_cached:
            simplified_perf["cached"] = simplify_device_perf(device_perf_cached)
        return status, message, e2e_perf, simplified_perf, peak_memory
    elif dp_skipped:
        return status, message, e2e_perf, DEVICE_PERF_SKIPPED, peak_memory
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

    dp_requested = getattr(config, "measure_device_perf", False)
    # Per-vector opt-out: a module sets _SKIP_DEVICE_PERF when the profiler read would
    # hang (e.g. conv2d heavy FABRIC_1D path -> remote-chip AICLK ARC read over fabric).
    # Return the SKIPPED sentinel (not None) so the runner marks PASS, not unsupported.
    if dp_requested and getattr(test_module, "_SKIP_DEVICE_PERF", False):
        return status, message, e2e_ms, DEVICE_PERF_SKIPPED, peak_memory
    if dp_requested:
        perf_result = gather_single_test_perf(_resolve_perf_device(device, test_module), status)
        if perf_result == DEVICE_PERF_READBACK_FAILED:
            # Pass the sentinel through untouched (simplify_device_perf() expects a dict)
            # WITH the original status, which is what the runner keys its decision on.
            return status, message, e2e_ms, DEVICE_PERF_READBACK_FAILED, peak_memory
        message = get_updated_message(message, perf_result)
        simplified_perf = simplify_device_perf(perf_result)
        return status, message, e2e_ms, simplified_perf, peak_memory
    else:
        return status, message, e2e_ms, None, peak_memory
