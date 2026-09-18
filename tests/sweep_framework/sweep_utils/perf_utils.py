# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import inspect
import subprocess
import shutil
from contextlib import contextmanager
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


# ---------------------------------------------------------------------------------------
# Determinism checking (--determinism-runs N)
#
# Sweep modules return only (pass, message) and the results DB keeps no tensors, so
# run-to-run diffing of exported results cannot see bit-level variation -- for eltwise the
# PCC is ~1.0 on every run and would hide it. Instead the same vector is executed N times
# in-process and the raw outputs are compared bit-exactly.
#
# Outputs are captured without touching any sweep module: a ttnn post-operation hook fires
# once per TOP-LEVEL ttnn op (ttnn/decorators.py, POST_OPERATION_HOOKS) with the op and its
# return value, so every ttnn.Tensor produced by the module's run() is recorded in call
# order. Input-creation ops (from_torch etc.) are captured too: if THEY differ between runs
# the module's inputs are not reproducible (unseeded RNG), which is reported as an
# inconclusive check rather than as op non-determinism.
# ---------------------------------------------------------------------------------------

# Top-level ops whose outputs are the module's INPUTS rather than the op under test. A
# difference here means the module regenerated different inputs between runs.
_INPUT_CREATION_OPS = frozenset(
    {
        "ttnn.from_torch",
        "ttnn.to_device",
        "ttnn.from_device",
        "ttnn.copy_host_to_device_tensor",
        "ttnn.allocate_tensor_on_device",
        "ttnn.as_tensor",
        "ttnn.rand",
        "ttnn.uniform",
    }
)

# Module-level opt-out, same convention as _SKIP_DEVICE_PERF: a sweep module sets this
# when the op is non-deterministic by definition (e.g. dropout / random-fill ops).
NON_DETERMINISTIC_BY_DESIGN_ATTR = "_NON_DETERMINISTIC_BY_DESIGN"


def _reseed_host_rngs(seed: int = 0) -> None:
    """Put every host RNG in the same state before each run of a vector.

    ~250 classic eltwise sweeps draw `data_seed = random.randint(...)` inside run() and
    seed torch from it, so without this every run would see different inputs and the
    check could only report "inputs not reproducible". Reseeding here makes those
    modules reproducible run-to-run with no per-module edits. Modules that already seed
    themselves (model-traced: torch.manual_seed(0)) are unaffected.
    """
    import random

    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def _iter_ttnn_tensors(value):
    import ttnn

    if isinstance(value, ttnn.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_ttnn_tensors(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_ttnn_tensors(v)


def _tensor_to_host_shards(tensor):
    """Return the tensor's data as a list of torch tensors, one per device shard.

    Reading shard-by-shard sidesteps mesh composers, so a multi-device tensor is compared
    shard-for-shard without knowing how the module would have composed it.
    """
    import ttnn

    try:
        shards = ttnn.get_device_tensors(tensor)
    except Exception:
        shards = [tensor]
    return [ttnn.to_torch(s) for s in shards]


@contextmanager
def _capture_outputs(reference=None, run_index: int = 1):
    """Yield an _OutputRecorder that sees every top-level ttnn op executed in the block.

    With reference=None (run 1) the recorder snapshots every output. With a reference
    (runs 2..N) it compares each output against the same-position run-1 snapshot as it is
    produced and drops it, so at most ONE run's outputs are ever resident on the host.
    Holding all N runs before comparing OOM-killed the process on large model-traced
    shapes (41 GB RSS for add).

    ttnn's default FastOperation path skips the pre/post-op hooks entirely
    (decorators.py: FastOperation.__call__ only routes to Operation when
    _requires_slow_runtime()). Flipping CONFIG.enable_fast_runtime_mode off for the
    duration is what ttnn.graph's own capture helper does for the same reason; the op
    itself is unchanged, only the Python wrapper around it.
    """
    import ttnn

    recorder = _OutputRecorder(reference=reference, run_index=run_index)
    prev_fast_runtime = ttnn.CONFIG.enable_fast_runtime_mode
    ttnn.CONFIG.enable_fast_runtime_mode = False
    try:
        with ttnn.register_post_operation_hook(recorder):
            yield recorder
    finally:
        ttnn.CONFIG.enable_fast_runtime_mode = prev_fast_runtime


class _OutputRecorder:
    """Post-op hook that snapshots (run 1) or compares-and-discards (runs 2..N) every
    ttnn.Tensor produced by top-level ttnn ops."""

    def __init__(self, reference=None, run_index: int = 1):
        self.reference = reference  # run-1 records, or None when this IS run 1
        self.run_index = run_index
        self.records = []  # run 1: list of (op_name, [torch shard, ...] | None if unreadable)
        self.count = 0  # tensors seen this run (for op-sequence comparison)
        self.mismatch = None  # first difference found (runs 2..N)
        self.capture_errors = 0

    def _snapshot(self, op_name, tensor):
        try:
            return _tensor_to_host_shards(tensor)
        except Exception as exc:  # a snapshot failure must not break the op under test
            self.capture_errors += 1
            logger.debug(f"determinism: could not read output of {op_name}: {exc}")
            return None

    def __call__(self, operation, function_args, function_kwargs, output):
        op_name = getattr(operation, "python_fully_qualified_name", None) or str(operation)
        for tensor in _iter_ttnn_tensors(output):
            idx = self.count
            self.count += 1
            if self.reference is None:
                self.records.append((op_name, self._snapshot(op_name, tensor)))
                continue
            if self.mismatch is not None:
                continue  # already diverged; skip the readback cost for the rest of this run
            if idx >= len(self.reference):
                continue  # extra tensors are reported as an op-sequence mismatch afterwards
            ref_op, ref_shards = self.reference[idx]
            if ref_op != op_name:
                self.mismatch = _sequence_mismatch(
                    op_name,
                    self.run_index,
                    f"tensor #{idx}: run 1 came from {ref_op}, run {self.run_index} from {op_name}",
                )
                continue
            if ref_shards is None:
                continue  # unreadable on run 1; nothing to compare
            shards = self._snapshot(op_name, tensor)
            if shards is None:
                continue
            self.mismatch = _compare_shards(op_name, idx, ref_shards, shards, self.run_index)
            del shards
        return None


def _sequence_mismatch(op, run_index: int, detail: str) -> Dict[str, Any]:
    return {
        "kind": "op_sequence",
        "op": op,
        "divergent_run": run_index,
        "mismatch_elems": None,
        "max_abs_delta": None,
        "detail": detail,
    }


def _compare_shards(op, idx, ref_shards, shards, run_index: int) -> Optional[Dict[str, Any]]:
    """Compare one tensor's shards against its run-1 snapshot; mismatch dict or None."""
    kind = "input" if op in _INPUT_CREATION_OPS else "output"
    if len(ref_shards) != len(shards):
        return {
            "kind": kind,
            "op": op,
            "divergent_run": run_index,
            "mismatch_elems": None,
            "max_abs_delta": None,
            "detail": f"{op} tensor #{idx}: shard count {len(ref_shards)} vs {len(shards)}",
        }
    for shard_i, (ra, rb) in enumerate(zip(ref_shards, shards)):
        if not _bit_identical(ra, rb):
            info = _describe_mismatch(ra, rb)
            shard_txt = f" shard {shard_i}" if len(shards) > 1 else ""
            return {
                "kind": kind,
                "op": op,
                "divergent_run": run_index,
                "mismatch_elems": info["mismatch_elems"],
                "max_abs_delta": info["max_abs_delta"],
                "detail": f"{op} tensor #{idx}{shard_txt} on run {run_index}: {info['detail']}",
            }
    return None


def _bit_identical(a, b) -> bool:
    import torch

    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    # Byte-level compare so NaN payloads and -0.0 vs 0.0 count as differences.
    return torch.equal(
        a.contiguous().flatten().view(torch.uint8),
        b.contiguous().flatten().view(torch.uint8),
    )


def _describe_mismatch(a, b) -> Dict[str, Any]:
    import torch

    if a.shape != b.shape or a.dtype != b.dtype:
        return {
            "mismatch_elems": max(a.numel(), b.numel()),
            "max_abs_delta": None,
            "detail": f"shape/dtype differ: {tuple(a.shape)}/{a.dtype} vs {tuple(b.shape)}/{b.dtype}",
        }
    af, bf = a.to(torch.float32), b.to(torch.float32)
    both_nan = torch.isnan(af) & torch.isnan(bf)
    differs = (af != bf) & ~both_nan
    n_diff = int(differs.sum())
    finite = torch.isfinite(af) & torch.isfinite(bf) & differs
    max_delta = float((af - bf).abs()[finite].max()) if bool(finite.any()) else 0.0
    if n_diff == 0:
        # _bit_identical said the bytes differ but no element compares unequal: the
        # difference is in NaN payloads or the sign of zero.
        detail = f"0 of {a.numel()} elements compare unequal but bytes differ (NaN payload / signed zero)"
    else:
        detail = f"{n_diff} of {a.numel()} elements differ, max|delta|={max_delta:.6g}"
    return {"mismatch_elems": n_diff, "max_abs_delta": max_delta, "detail": detail}


def run_with_determinism_check(
    test_module, test_vector: dict, device, config: Any
) -> Tuple[bool, Any, Optional[float], Optional[dict], Optional[Dict], Dict[str, Any]]:
    """Execute a vector config.determinism_runs times and require bit-identical outputs.

    Returns the usual (status, message, e2e_ms, device_perf, peak_memory) plus a
    `determinism` dict:
        verdict        "deterministic" | "non_deterministic" | "inputs_not_reproducible"
                       | "skipped_by_design" | "not_checked"
        runs           executions performed
        divergent_run  1-based run that first differed (non-deterministic only)
        mismatch_elems / max_abs_delta / detail   description of the first difference

    Run 1 is uncached and runs 2..N hit the program cache, so a cache-path divergence is
    caught as well and reported with divergent_run == 2.

    Only the verdict is folded into status/message: a vector that PASSES its own PCC on
    every run but differs between runs is returned with status=False and a message that
    starts with NON_DETERMINISTIC so the runner can classify it FAIL_NON_DETERMINISTIC.
    A run that fails its own PCC (or raises) is reported as that failure instead.
    """
    runs = max(int(getattr(config, "determinism_runs", 0) or 0), 2)
    base_info: Dict[str, Any] = {"runs": runs, "verdict": "not_checked"}

    if getattr(test_module, NON_DETERMINISTIC_BY_DESIGN_ATTR, False):
        status, message, e2e_ms, device_perf, peak_memory = run_single(test_module, test_vector, device, config)
        base_info["verdict"] = "skipped_by_design"
        if status:
            message = f"{message} | determinism check skipped: op is non-deterministic by design"
        return status, message, e2e_ms, device_perf, peak_memory, base_info

    # Only the program cache is cleared (not the disk kernel cache, which would force a
    # kernel recompile per vector): run 1 then takes the uncached path, runs 2..N the
    # cached one, so a cache-only divergence is covered.
    try:
        device.clear_program_cache()
    except Exception as exc:
        logger.debug(f"determinism: could not clear program cache: {exc}")

    reference = None  # run-1 snapshots; runs 2..N stream-compare against these
    mismatch = None
    first_status = first_message = first_e2e = first_device_perf = first_peak = None
    for i in range(runs):
        _reseed_host_rngs()
        with _capture_outputs(reference=reference, run_index=i + 1) as recorder:
            if i == 0:
                # First run carries the perf / device-perf / memory measurements exactly as
                # run_single would, so --determinism-runs composes with those flags.
                status, message, e2e_ms, device_perf, peak_memory = run_single(test_module, test_vector, device, config)
                first_status, first_message, first_e2e, first_device_perf, first_peak = (
                    status,
                    message,
                    e2e_ms,
                    device_perf,
                    peak_memory,
                )
            else:
                status, message, _ = execute_test(test_module, test_vector, device)
        if not status:
            # The vector itself failed on this run: that is the finding, not determinism.
            fail_msg = message if i == 0 else f"RUN {i + 1}/{runs} FAILED: {message} (run 1: {first_message})"
            base_info["verdict"] = "not_checked"
            base_info["detail"] = f"run {i + 1} failed its own check"
            return False, fail_msg, first_e2e, first_device_perf, first_peak, base_info
        if i == 0:
            reference = recorder.records
            continue
        mismatch = recorder.mismatch
        if mismatch is None and recorder.count != len(reference):
            mismatch = _sequence_mismatch(
                None, i + 1, f"run 1 produced {len(reference)} tensors, run {i + 1} produced {recorder.count}"
            )
        if mismatch is not None:
            break  # first divergence is the finding; later runs add nothing

    # Every run passed its own check; the streamed comparison decides the verdict.
    n_tensors = len(reference)
    if mismatch is None:
        base_info["verdict"] = "deterministic"
        base_info["mismatch_elems"] = 0
        base_info["compared_tensors"] = n_tensors
        if n_tensors == 0:
            base_info["verdict"] = "not_checked"
            base_info["detail"] = "no ttnn tensor outputs were captured"
            message = f"{first_message} | determinism NOT CHECKED: no ttnn outputs captured"
        else:
            message = f"{first_message} | deterministic: {n_tensors} tensor(s) bit-identical over {runs} runs"
        return True, message, first_e2e, first_device_perf, first_peak, base_info

    base_info.update(
        {
            "divergent_run": mismatch["divergent_run"],
            "mismatch_elems": mismatch["mismatch_elems"],
            "max_abs_delta": mismatch["max_abs_delta"],
            "op": mismatch["op"],
            "detail": mismatch["detail"],
        }
    )
    if mismatch["kind"] == "input":
        # The module fed different inputs to the op on this run, so the outputs cannot be
        # compared. Not a failure of the op; flag the module for seeding.
        base_info["verdict"] = "inputs_not_reproducible"
        logger.warning(f"determinism: inputs differ between runs, module needs a fixed seed: {mismatch['detail']}")
        message = f"{first_message} | determinism INCONCLUSIVE (inputs not reproducible): {mismatch['detail']}"
        return True, message, first_e2e, first_device_perf, first_peak, base_info

    base_info["verdict"] = "non_deterministic"
    logger.error(f"determinism: {mismatch['detail']}")
    message = f"NON_DETERMINISTIC: {mismatch['detail']} (every run passed PCC: {first_message})"
    return False, message, first_e2e, first_device_perf, first_peak, base_info


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
