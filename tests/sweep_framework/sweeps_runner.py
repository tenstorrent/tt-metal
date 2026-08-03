# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# standard
import argparse
import builtins
import collections
import datetime as dt
import importlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from queue import Empty

# third party
import enlighten
import framework.tt_smi_util as tt_smi_util

try:
    from faster_fifo import Queue  # faster IPC; not available on aarch64 Linux
except ImportError:
    from multiprocessing import Queue

# tt
from framework.device_fixtures import default_device
from framework.result_destination import ResultDestinationFactory
from framework.serialize import deserialize, deserialize_vector_structured
from framework.constants import parse_mesh_suffix
from framework.statuses import TestStatus, VectorValidity
from framework.sweeps_logger import sweeps_logger as logger
from framework.vector_source import VectorSourceFactory
from sweep_utils.perf_utils import (
    run_single,
    run_with_cache_comparison,
    DEVICE_PERF_SKIPPED,
    DEVICE_PERF_READBACK_FAILED,
)


@dataclass
class SweepsConfig:
    """Configuration object for sweeps runner"""

    module_name: str | None = None
    suite_name: str | None = None
    vector_source: str = "vectors_export"
    file_path: str | None = None
    vector_id: str | None = None
    result_destination: str = "results_export"
    watcher: bool = False
    measure_perf: bool = False
    measure_perf_with_cache: bool = False
    measure_device_perf: bool = False
    measure_memory: bool = False
    dry_run: bool = False
    sweeps_tag: str | None = None
    skip_modules: str | None = None
    skip_on_timeout: bool = False
    keep_invalid: bool = False
    summary: bool = False
    run_contents: str | None = None
    arch_name: str | None = None
    main_proc_verbose: bool = False
    trace_params: bool = False
    # Restrict the run to vectors whose mesh is 1D ("1d": rows==1 or cols==1) or
    # 2D ("2d": both >1). Used to split CCL ops (e.g. all_gather) into separate
    # jobs per fabric family (1D -> FABRIC_1D/RING, 2D -> FABRIC_2D) so a single
    # process never does a live FABRIC_1D->FABRIC_2D control-plane transition,
    # whose first post-transition op hangs on T3K CI. None = run all meshes.
    mesh_dims: str | None = None
    fail_on_test_failure: bool = False


def create_config_from_args(args) -> SweepsConfig:
    """Create configuration object from parsed arguments"""

    config = SweepsConfig(
        module_name=args.module_name,
        suite_name=args.suite_name,
        vector_source=args.vector_source,
        file_path=args.file_path,
        vector_id=args.vector_id,
        result_destination=args.result_dest,
        watcher=args.watcher,
        measure_perf=args.perf,
        measure_perf_with_cache=args.perf_with_cache,
        measure_device_perf=args.device_perf,
        measure_memory=args.measure_memory,
        dry_run=args.dry_run,
        sweeps_tag=args.tag,
        skip_modules=args.skip_modules,
        skip_on_timeout=args.skip_on_timeout,
        keep_invalid=args.keep_invalid,
        summary=args.summary,
        main_proc_verbose=args.main_proc_verbose,
        trace_params=args.trace_params,
        mesh_dims=args.mesh_dims,
        fail_on_test_failure=args.fail_on_test_failure,
    )

    # Validate and set ARCH_NAME
    allowed_arch = {"blackhole", "wormhole_b0"}
    arch_env = os.getenv("ARCH_NAME") or os.getenv("IRD_ARCH_NAME")
    if not arch_env:
        logger.error("ARCH_NAME must be set in environment and be one of ['blackhole', 'wormhole_b0']")
        exit(1)
    arch_env = arch_env.strip()
    if arch_env not in allowed_arch:
        logger.error(f"Invalid ARCH_NAME '{arch_env}'. Must be one of ['blackhole', 'wormhole_b0']")
        exit(1)
    config.arch_name = arch_env

    return config


def validate_arguments(args, parser):
    # Define validation rules as tuples of (condition, error_message)
    validation_rules = [
        # Module name dependencies
        (args.vector_id and not args.module_name, "Module name is required if vector id is specified."),
        (args.file_path and not args.module_name, "Module name is required if file path is specified."),
        (
            args.vector_source == "file" and not args.module_name,
            "Module name is required when test vector source is 'file'.",
        ),
        # File path constraints
        (
            args.file_path and args.vector_source == "vectors_export",
            "File path should not be specified when test vector source is 'vectors_export'.",
        ),
    ]

    # Check each validation rule
    for condition, error_message in validation_rules:
        if condition:
            parser.print_help()
            logger.error(error_message)
            exit(1)

    # Validate that skip modules is only used when running all modules
    if args.skip_modules and args.module_name:
        logger.error("Skip modules is only supported when running all modules.")
        exit(1)

    # Validate performance measurement flags
    # Disabled while e2e perf measurement is disabled
    if getattr(args, "perf_with_cache", False) and args.perf:
        logger.error(
            "Cannot use both --perf and --perf-with-cache flags simultaneously. Use --perf-with-cache to get both cached and uncached performance measurements."
        )
        exit(1)

    logger.info("All argument validations passed successfully.")


def get_all_modules():
    sweeps_path = Path(__file__).parent / "sweeps"
    for file in sorted(sweeps_path.glob("**/*.py")):
        sweep_name = str(Path(file).relative_to(sweeps_path))[:-3].replace("/", ".")
        yield sweep_name


DEFAULT_TIMEOUT = 30
TIMEOUT_KEY = "TIMEOUT"
SWEEPS_SUBDIR_NAME = "sweeps"
PY_SUFFIX = ".py"


def get_timeout(test_module_name):
    """We need to grab the test's timeout without loading the test module"""

    sweep_root_path = Path(__file__).resolve().parent
    test_source_name = test_module_name.replace(".", "/") + PY_SUFFIX
    test_path = sweep_root_path / SWEEPS_SUBDIR_NAME / test_source_name

    if not (test_path.exists() and test_path.is_file()):
        return DEFAULT_TIMEOUT

    timeout = DEFAULT_TIMEOUT
    with test_path.open("rt") as fh:
        for line in fh:
            if TIMEOUT_KEY in line:
                try:
                    timeout = int(line.split("=")[-1].strip())
                except (ValueError, IndexError):
                    # Malformed/unparseable TIMEOUT line — keep the default
                    # timeout already assigned and stop scanning.
                    break
    return timeout


def sanitize_inputs(test_vectors):
    info_field_names = ["sweep_name", "suite_name", "input_hash", "traced_source", "traced_machine_info"]
    header_info = []
    for vector in test_vectors:
        header = dict()
        for field in info_field_names:
            if field in vector:
                header[field] = vector.pop(field)
        if "timestamp" in vector:
            vector.pop("timestamp")
        if "tag" in vector:
            vector.pop("tag")
        header_info.append(header)
    return header_info, test_vectors


def get_devices(test_module):
    try:
        return test_module.mesh_device_fixture()
    except:
        return default_device()


# Single-host clusters top out at 8 devices (N150=1, N300=2, T3K=8); more than
# this means a multi-host Galaxy box, where the persistent job device is enabled.
_SINGLE_HOST_MAX_DEVICES = 8


def _is_galaxy_job() -> bool:
    """Whether this job runs on Galaxy — the only place the per-module device
    reopen force-reinitializes dispatch and wedges a core, so the only place the
    persistent worker + job-device reuse is enabled. In CI RUNNER_LABEL identifies
    the box (topology-6u / g0*glx* = Galaxy) without a device query; locally fall
    back to the device count (>8). TTNN_SWEEP_JOB_DEVICE_FORCE=1 forces it on
    (validation on smaller clusters)."""
    if os.environ.get("TTNN_SWEEP_JOB_DEVICE_FORCE") == "1":
        return True
    rl = os.environ.get("RUNNER_LABEL", "").lower()
    if rl:
        return "6u" in rl or "galaxy" in rl or "glx" in rl
    try:
        import ttnn

        return ttnn.get_num_devices() > _SINGLE_HOST_MAX_DEVICES
    except Exception:
        return False


def get_hostname():
    return subprocess.check_output(["uname", "-n"]).decode("ascii").strip()


def get_username():
    """Get the username - GitHub Actions actor for CI, local USER for development"""
    # In GitHub Actions, use the actor who triggered the workflow
    if os.getenv("GITHUB_ACTOR"):
        return os.environ["GITHUB_ACTOR"]
    # Fall back to local USER environment variable for development
    return os.environ.get("USER", "unknown")


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception as e:
        return "Couldn't get git hash!"


def get_git_author():
    """Get the git author name from the latest commit"""
    try:
        # Get the author of the latest commit on the current branch
        return (
            subprocess.check_output(["git", "log", "-1", "--pretty=format:%an"], stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
    except Exception as e:
        return "Unknown"


def get_git_branch():
    """Get the current git branch name"""
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("ascii").strip()
    except Exception as e:
        return "Unknown"


def get_initiated_by():
    """Get the user who initiated the run - username for dev, CI pipeline name for CI/CD"""
    # Check if we're in a CI environment
    ci_pipeline = os.getenv("GITHUB_WORKFLOW") or os.getenv("CI_PIPELINE_NAME")
    if ci_pipeline:
        return ci_pipeline
    else:
        return get_username()


def get_github_pipeline_id() -> int | None:
    """Get a CI pipeline identifier suitable for joining CICD metadata tables.

    Prefer GitHub Actions run id if present; otherwise fall back to generic CI_PIPELINE_ID.
    Returns an int when available, otherwise None.
    """
    run_id = os.getenv("GITHUB_RUN_NUMBER") or os.getenv("GITHUB_RUN_ID")
    if not run_id:
        return None
    try:
        return int(run_id)
    except ValueError:
        # Unexpected non-integer; keep it unset for type consistency
        return None


# Sentinel the parent sends to end the persistent worker (a module task is a
# (module_name, serialized_vector) tuple, so this string never collides).
_WORKER_CLOSE = "__worker_close__"
# Grace period (seconds) to let the persistent job worker drain and close its
# device after _WORKER_CLOSE before it is force-killed.
_WORKER_JOIN_TIMEOUT_S = 30
_WORKER_KILL_GRACE_S = 5


def run(input_queue, output_queue, config: SweepsConfig):
    """Persistent, module-agnostic worker: one process per job that runs every
    module's vectors, so the job-level device cache in mesh_tensor_utils
    (TTNN_SWEEP_JOB_DEVICE) reuses ONE open device across all modules that share a
    device config, reopening only when the config actually changes. Each queue
    item is (module_name, serialized_vector); _WORKER_CLOSE ends the worker.

    Each module is entered through its own mesh_device_fixture (get_devices) as
    before, but because create_mesh_device caches and ttnn.close_mesh_device is
    deferred for the cached device, consecutive modules with the same config reuse
    the device instead of the per-module reopen that force-reinitializes dispatch
    on Galaxy. The program cache is cleared at each module boundary so a new module
    doesn't collide with an earlier one's kernels on the reused device.
    """
    if config.trace_params:
        try:
            from tests.sweep_framework.framework.preop_arg_capture import enable_preop_capture

            enable_preop_capture()
        except Exception as e:
            logger.warning(f"Could not enable operation tracing: {e}")

    from tests.sweep_framework.sweep_utils.mesh_tensor_utils import clear_job_device_program_cache, close_job_device

    module_cache = {}
    cur_module = None
    cur_gen = None  # current module's device fixture generator
    cur_device = None

    def _exhaust_fixture():
        nonlocal cur_gen
        if cur_gen is not None:
            try:
                for _ in cur_gen:  # run the fixture past its yield -> ttnn.close_mesh_device (deferred for cached)
                    pass
            except Exception as e:
                logger.warning(f"Worker fixture teardown failed (continuing): {e}")
            cur_gen = None

    try:
        while True:
            try:
                item = input_queue.get(block=True, timeout=5)
            except Empty:
                # Between modules / waiting for the next vector — keep the device
                # open and keep waiting; the worker only exits on the sentinel.
                continue
            if item == _WORKER_CLOSE:
                return

            module_name, test_vector = item
            test_module = module_cache.get(module_name)
            if test_module is None:
                test_module = importlib.import_module("sweeps." + module_name)
                module_cache[module_name] = test_module

            if module_name != cur_module:
                _exhaust_fixture()
                # Clear the reused device's program cache so this module starts
                # clean (no cross-module kernel-binary collision, kernel.cpp:443).
                clear_job_device_program_cache()
                # Advance cur_module BEFORE opening so a failed open isn't retried
                # for every subsequent vector of the same module.
                cur_module = module_name
                try:
                    cur_gen = get_devices(test_module)
                    cur_device, _device_name = next(cur_gen)
                except AssertionError as e:
                    # Device failed to open for this module: emit exactly ONE
                    # result for THIS request and move on. Do NOT fall through to
                    # also run the vector — a second output_queue.put() desyncs the
                    # persistent worker (the extra result is consumed as the next
                    # vector's result, misattributing every later result in the job).
                    cur_device = None
                    output_queue.put([False, "DEVICE EXCEPTION: " + str(e), None, None, None])
                    continue

            test_vector = deserialize_vector_structured(test_vector)
            try:
                if config.measure_perf_with_cache:
                    status, message, e2e_perf, device_perf, peak_memory = run_with_cache_comparison(
                        test_module, test_vector, cur_device, config
                    )
                else:
                    status, message, e2e_perf, device_perf, peak_memory = run_single(
                        test_module, test_vector, cur_device, config
                    )
                output_queue.put(
                    [
                        status,
                        message,
                        e2e_perf,
                        device_perf if config.measure_device_perf else None,
                        peak_memory if config.measure_memory else None,
                    ]
                )
            except Exception as e:
                if config.main_proc_verbose:
                    logger.exception(e)
                output_queue.put([False, str(e), None, None, None])
    finally:
        _exhaust_fixture()
        close_job_device()


MAX_RETRIES = 1


def _create_main_proc_runner(module_name, input_queue, output_queue, config):
    """Create a persistent runner for main process mode that keeps device open.

    Returns (runner_function, cleanup_context) tuple.
    The runner_function executes a single test vector.
    The cleanup_context must be exited to close the device.
    """
    # Enable operation tracing if --trace-params is set (pre-op argument capture;
    # see framework/preop_arg_capture.py and the note in run()).
    if config.trace_params:
        try:
            from tests.sweep_framework.framework.preop_arg_capture import enable_preop_capture

            enable_preop_capture()
            logger.info("Operation tracing enabled in main process mode")
        except Exception as e:
            logger.warning(f"Could not enable operation tracing: {e}")

    test_module = importlib.import_module("sweeps." + module_name)

    # Open device once and keep it open
    device_gen = get_devices(test_module)
    device, device_name = next(device_gen)
    logger.info(f"Device opened: {device_name}")

    def runner(test_vector):
        """Execute a single test vector using the persistent device."""
        try:
            # Deserialize the test vector (same as subprocess mode)
            test_vector = deserialize_vector_structured(test_vector)

            if config.measure_perf_with_cache:
                status, message, e2e_perf, device_perf, peak_memory = run_with_cache_comparison(
                    test_module, test_vector, device, config
                )
            else:
                status, message, e2e_perf, device_perf, peak_memory = run_single(
                    test_module, test_vector, device, config
                )
            output_queue.put(
                [
                    status,
                    message,
                    e2e_perf,
                    device_perf if config.measure_device_perf else None,
                    peak_memory if config.measure_memory else None,
                ]
            )
        except Exception as e:
            if config.main_proc_verbose:
                logger.exception(e)
            status, message = False, str(e)
            output_queue.put([status, message, None, None, None])

    # Return runner function and device generator for cleanup
    return runner, device_gen


def _kill_child(p, timeout_before_rejoin):
    """Terminate/kill a child process gracefully then forcefully."""
    if p is None:
        return
    logger.warning(f"Killing child process {p.pid}...")
    p.terminate()
    p.join(timeout_before_rejoin)
    if p.is_alive():
        logger.error(f"Child process {p.pid} did not terminate, killing it.")
        p.kill()
        p.join()


def _attempt_vector(
    test_vector, module_name, input_queue, output_queue, config, timeout, child_mode, p, main_proc_runner
):
    """Send a single vector to the child process and collect the result.

    Returns (response_tuple, p) on success.
    Raises Empty on timeout.
    """
    if child_mode and (p is None or not p.is_alive()):
        p = Process(target=run, args=(input_queue, output_queue, config))
        p.start()

    if p is None and main_proc_runner is not None:
        main_proc_runner(test_vector)
    else:
        # persistent worker is module-agnostic: tag each vector with its module
        input_queue.put((module_name, test_vector))

    response = output_queue.get(block=True, timeout=timeout)
    return response, p


def _populate_result_from_response(result, response, config, suite_name, input_hash):
    """Parse a child-process response tuple into the result dict."""
    status, message, e2e_perf, device_perf, peak_memory = (
        response[0],
        response[1],
        response[2],
        response[3],
        response[4],
    )
    result["message"] = message

    logger.info(f"Test status: {status}")
    logger.info(f"Test message: {message}")
    logger.info(f"Test e2e perf: {e2e_perf}")
    logger.info(f"Test device perf: {device_perf}")

    if status:
        if config.measure_device_perf:
            if device_perf == DEVICE_PERF_READBACK_FAILED:
                # The profiler readback threw, but this vector's own PCC PASSED. A wrong
                # profiler buffer says nothing about a correct op result, so keep the PASS
                # and carry on to the next vector with device-perf N/A.
                logger.warning(
                    "Device profiler readback failed but the vector PASSED; recording PASS with "
                    "device-perf N/A and continuing."
                )
                result["status"] = TestStatus.PASS
                result["device_perf"] = None
            elif device_perf == DEVICE_PERF_SKIPPED:
                # Module opted this vector out of profiling (unsupported config, e.g.
                # conv2d heavy FABRIC_1D -> profiler ARC read hangs). PCC passed, so
                # PASS with device-perf N/A -- not a failure.
                result["status"] = TestStatus.PASS
                result["device_perf"] = None
            elif device_perf is None and _should_skip_device_profiler(config):
                # The profiler was intentionally gated off for this run (e.g. conv2d on
                # a multi-chip mesh, or CCL on FABRIC_2D), so there is no device-perf by
                # design. PCC passed -> PASS with device-perf N/A, not a failure.
                result["status"] = TestStatus.PASS
                result["device_perf"] = None
            elif device_perf is None:
                result["status"] = TestStatus.FAIL_UNSUPPORTED_DEVICE_PERF
            else:
                result["status"] = TestStatus.PASS
                if config.measure_perf_with_cache and isinstance(device_perf, dict):
                    result["device_perf_uncached"] = device_perf.get("uncached")
                    result["device_perf_cached"] = device_perf.get("cached")
                else:
                    result["device_perf"] = device_perf
        else:
            result["status"] = TestStatus.PASS
    else:
        result["exception"] = message
        if config.measure_device_perf and device_perf == DEVICE_PERF_READBACK_FAILED:
            # The vector FAILED and the profiler readback ALSO threw. Two independent
            # readers of the device disagreeing with expectations at once is treated as
            # evidence the device itself is bad, not that this vector is a bad test: the
            # host decoded a corrupt profiler marker (only 0-5 are valid packet types) on
            # the same vector whose result did not match. So do NOT count it as a test
            # failure -- mark it NOT_RUN and end the run, rather than feeding more vectors
            # to a device we no longer trust.
            #
            # This check deliberately comes FIRST, ahead of the OOM/Watcher/infra
            # classification below, so the rule is unambiguous: readback failure + failed
            # vector always means "stop", whatever the vector's own message said.
            #
            # Tradeoff, measured on run 30509849370 job 90770018256: the device does not
            # always stay bad. There, copy 75a4... was the FIRST vector and hit exactly
            # this combination, yet the following 6 vectors (cos x4, div x2) passed with
            # PCC ~1.0 on the same device before div 46d243e2 genuinely hung. Ending the
            # run at the first occurrence forfeits those 6 passes. That is the intended
            # behaviour here -- prefer stopping early on a suspect device over continuing.
            logger.error(
                f"Device profiler readback failed AND the vector failed for input_hash='{input_hash}'. "
                "Treating the device as wedged: marking this vector NOT_RUN (not a test failure) "
                "and ending the run."
            )
            result["status"] = TestStatus.NOT_RUN
            result["device_perf"] = None
            result["_infra_abort"] = True
            result["_abort_suite"] = True
        else:
            # NOTE: keep this classification inside the else -- the OOM/Watcher chain below
            # is a separate statement from the DEVICE EXCEPTION log, so leaving it
            # unguarded would overwrite the NOT_RUN set above with FAIL_ASSERT_EXCEPTION.
            if "DEVICE EXCEPTION" in str(message):
                logger.error(
                    f"DEVICE EXCEPTION: Device could not be initialized. "
                    f"The following assertion was thrown: {message}"
                )
                logger.info("Device error detected. The suite will be aborted after this test.")
            if "Out of Memory: Not enough space to allocate" in str(message):
                result["status"] = TestStatus.FAIL_L1_OUT_OF_MEM
            elif "Watcher" in str(message):
                result["status"] = TestStatus.FAIL_WATCHER
            elif _is_infra_failure_message(message):
                # Infrastructure-class failure: either a fabric / control-plane
                # bring-up failure (mesh never initialized, so this vector's op kernel
                # never ran) or a device-fatal wedge (a bad core run state surfaced as
                # "Read unexpected run_mailbox value"). Both are environment faults,
                # not test-vector faults -- mark NOT_RUN rather than
                # FAIL_ASSERT_EXCEPTION. _execute_vector_with_retry detects the same
                # signatures and exits the run early so the remaining vectors are not
                # each re-reported as false failures on a device that won't recover.
                result["status"] = TestStatus.NOT_RUN
            else:
                result["status"] = TestStatus.FAIL_ASSERT_EXCEPTION

    if suite_name.lower().startswith("xfail"):
        if result["status"] == TestStatus.PASS:
            result["status"] = TestStatus.XPASS
            logger.warning(f"UNEXPECTED PASS: Test in XFail suite '{suite_name}' passed unexpectedly: {input_hash}")
        elif result["status"] in [
            TestStatus.FAIL_ASSERT_EXCEPTION,
            TestStatus.FAIL_L1_OUT_OF_MEM,
            TestStatus.FAIL_WATCHER,
            TestStatus.FAIL_UNSUPPORTED_DEVICE_PERF,
        ]:
            result["status"] = TestStatus.XFAIL
            logger.info(f"EXPECTED FAILURE: Test in XFail suite '{suite_name}' failed as expected: {input_hash}")

    if config.measure_perf_with_cache and e2e_perf:
        result["e2e_perf"] = e2e_perf
        result["e2e_perf_uncached"] = e2e_perf.get("uncached") if isinstance(e2e_perf, dict) else None
        result["e2e_perf_cached"] = e2e_perf.get("cached") if isinstance(e2e_perf, dict) else None
    elif config.measure_perf and e2e_perf:
        result["e2e_perf"] = e2e_perf
    else:
        result["e2e_perf"] = None

    if config.measure_memory and peak_memory:
        if isinstance(peak_memory, dict):
            result["peak_l1_memory_per_core"] = peak_memory.get("peak_total_per_core")
            result["peak_cb_per_core"] = peak_memory.get("peak_cb_per_core")
            result["peak_l1_buffers_per_core"] = peak_memory.get("peak_l1_per_core")
            result["num_cores"] = peak_memory.get("num_cores")
            result["peak_l1_memory_aggregate"] = peak_memory.get("peak_total_aggregate")
            result["peak_l1_memory_device"] = peak_memory.get("peak_l1_memory_device")
    else:
        result["peak_l1_memory_per_core"] = None
        result["peak_cb_per_core"] = None
        result["peak_l1_buffers_per_core"] = None
        result["num_cores"] = None
        result["peak_l1_memory_aggregate"] = None
        result["peak_l1_memory_device"] = None


# Signatures of a device-level hang that the child process catches and *returns*
# as a normal exception (status=False) rather than triggering the Python-side
# watchdog timeout. Once the mesh hangs in fetch-queue dispatch, every subsequent
# vector throws the same error, so we must reset the device (and, under
# skip-on-timeout, abort the rest of the suite) instead of spinning for the
# entire job and getting the runner cancelled on the wall-clock cap.
_DEVICE_HANG_SIGNATURES = (
    "device timeout in fetch queue wait",
    "potential hang detected",
    "completion reader queue is not empty",
    "device hang or timeout occurred",
)


def _is_device_hang_message(message) -> bool:
    """Return True if a returned exception message indicates a device hang."""
    if not message:
        return False
    msg = str(message).lower()
    return any(sig in msg for sig in _DEVICE_HANG_SIGNATURES)


# Signatures of a device-level *fatal* wedge (distinct from a hang): one vector
# leaves a core in a bad run state, so the NEXT program launch on that device
# aborts reading the stale run mailbox ("Read unexpected run_mailbox value:
# 0x40"). This is treated as an infrastructure failure (see
# _is_infra_failure_message): the device does not recover within the job — on a
# Galaxy the wedge cascades into dispatch hangs, slow resets, and all-zero
# outputs that mis-report as PCC failures on every remaining vector — so rather
# than reset+continue we exit the whole run early and mark the rest NOT_RUN.
_DEVICE_FATAL_SIGNATURES = (
    "unexpected run_mailbox value",
    "read unexpected run_mailbox",
)


def _is_device_fatal_message(message) -> bool:
    """Return True if a returned exception indicates a device-fatal wedge (a core
    left in a bad run state, surfaced on the next launch as an unexpected
    run_mailbox value)."""
    if not message:
        return False
    msg = str(message).lower()
    return any(sig in msg for sig in _DEVICE_FATAL_SIGNATURES)


# Signatures of a transient kernel-ELF build/load failure. When the persisted
# cache is cold (CI clears it before the job) and FABRIC_2D opens all chips at
# once, the fabric_erisc_router ELF is built+loaded concurrently across devices;
# a device can load a partially-written ELF and throw "tt_elffile.cpp:405". The
# build itself completes (the .elf is written), so simply resetting and retrying
# the SAME vector in a fresh child — whose cache is now warm — loads it cleanly.
# (Observed on T3K [2D] all_gather: first FABRIC_2D config 0dd01b5f after a cache
# clear.) Genuinely-corrupt ELFs just exhaust the retries and fail as before.
_ELF_LOAD_RETRY_SIGNATURES = (
    "tt_elffile.cpp",
    "failed to generate binaries",
)


def _is_elf_load_error(message) -> bool:
    """Return True if a returned exception looks like a transient kernel-ELF
    build/load failure that a warm-cache retry should clear."""
    if not message:
        return False
    msg = str(message).lower()
    return any(sig in msg for sig in _ELF_LOAD_RETRY_SIGNATURES)


# Signatures of a fabric / control-plane bring-up failure — the mesh could not
# be initialized at all, so NO op kernel ever ran. The canonical case is the
# fabric topology mapper failing to fit the mesh-graph descriptor (MGD) onto the
# discovered physical topology, e.g. on a Galaxy where an ethernet edge has
# degraded below the required channel count and auto-discovery yields a
# non-uniform degree histogram:
#     TT_FATAL @ .../topology_mapper.cpp:546: mapping_result.success
#     Graph specified in MGD could not fit in the discovered physical topology
# This is an ENVIRONMENT fault, not a test-vector fault, and it is sticky: the
# same host state makes every subsequent vector throw the identical error. Left
# unhandled it falls through to FAIL_ASSERT_EXCEPTION and mis-reports the whole
# suite as a wall of test failures. We instead classify it as NOT_RUN and abort
# the suite early (see _execute_vector_with_retry / execute_suite).
# NOTE: match on the specific mapping-failure text, NOT on the bare filename
# "topology_mapper.cpp". That file emits TT_FATALs for several unrelated
# conditions; keying off the filename alone would reclassify any of them as a
# sticky infra abort and kill the whole sweep on a false positive.
_FABRIC_INFRA_SIGNATURES = (
    "could not fit in the discovered physical topology",
    "mapping_result.success",
    "inter-mesh mapping failed",
    "intra-mesh mapping failed",
)


def _is_fabric_infra_message(message) -> bool:
    """Return True if a returned exception indicates a fabric / control-plane
    bring-up failure (the mesh never initialized, so no vector actually ran).
    These are environment faults, not test-vector faults."""
    if not message:
        return False
    msg = str(message).lower()
    return any(sig in msg for sig in _FABRIC_INFRA_SIGNATURES)


def _is_infra_failure_message(message) -> bool:
    """Return True for any infrastructure-class failure that has degraded the
    host: either a fabric / control-plane bring-up failure (the mesh never came
    up) or a device-fatal wedge (a core left in a bad run state, surfaced as
    "Read unexpected run_mailbox value: 0x40" on the next launch). Both are
    environment faults, not test-vector faults, and both are STICKY — the device
    does not recover within the job (a wedge cascades into dispatch hangs, slow
    Galaxy resets, and all-zero/garbage outputs that mis-report as PCC failures).
    So rather than reset+continue on a machine that "once degraded is always
    degraded", we classify the vector NOT_RUN and exit the whole run early."""
    return _is_fabric_infra_message(message) or _is_device_fatal_message(message)


def _set_crash_hang_defaults(result):
    """Populate result fields for a FAIL_CRASH_HANG outcome."""
    result["status"] = TestStatus.FAIL_CRASH_HANG
    result["exception"] = "TEST TIMED OUT (CRASH / HANG)"
    result["end_time_ts"] = dt.datetime.now(dt.timezone.utc)
    result["e2e_perf"] = None
    result["peak_l1_memory_per_core"] = None
    result["peak_cb_per_core"] = None
    result["peak_l1_buffers_per_core"] = None
    result["num_cores"] = None
    result["peak_l1_memory_aggregate"] = None
    result["peak_l1_memory_device"] = None


def _mark_infra_abort(result, reason: str):
    """Classify a result as an infrastructure abort (NOT_RUN) and stop the run.

    Mirrors the _is_infra_failure_message path: the vector is NOT a test failure,
    the suite aborts unconditionally and run_sweeps exits the run early, so the job
    surfaces one infrastructure error instead of a wall of false results.
    """
    result["status"] = TestStatus.NOT_RUN
    result["exception"] = "INFRASTRUCTURE ERROR (degraded host): " + reason
    result["end_time_ts"] = dt.datetime.now(dt.timezone.utc)
    result["_child_process"] = None
    result["_abort_suite"] = True
    result["_infra_abort"] = True


def _reset_or_infra_abort(reset_util, result, input_hash) -> bool:
    """Reset the devices. Returns True on success.

    When every configured reset mechanism is exhausted (ResetFailed) the host is
    unrecoverable for this job, so mark the result as an infra abort and return
    False; the caller must return `result` immediately rather than respawn a child
    against a wedged device. Previously ResetFailed propagated as an uncaught
    exception (all five reset call sites were bare), which is now reachable on
    Galaxy since the known-bad `tt-smi -r all` fallback was removed.
    """
    try:
        reset_util.reset()
        return True
    except tt_smi_util.ResetFailed as e:
        logger.error(
            f"DEVICE RESET FAILED for input_hash='{input_hash}': {e}. All reset mechanisms are "
            f"exhausted — the host is unrecoverable for this job; exiting the run early instead "
            f"of launching further vectors against a wedged device."
        )
        _mark_infra_abort(result, f"device reset failed ({e})")
        return False


def _execute_vector_with_retry(
    test_vector,
    module_name,
    input_queue,
    output_queue,
    config,
    suite_name,
    input_hash,
    timeout,
    timeout_before_rejoin,
    reset_util,
    child_mode,
    p,
    result,
    main_proc_runner=None,
):
    """Execute a single test vector with up to MAX_RETRIES retries on timeout.

    On timeout: kill child -> tt-smi reset -> spawn new child -> retry.
    If the retry also times out: mark as FAIL_CRASH_HANG -> tt-smi reset.

    Returns the result dict with two internal keys:
      _child_process  – the (possibly new) child Process
      _abort_suite    – True if skip_on_timeout should be honoured
    """
    for attempt in range(1 + MAX_RETRIES):
        try:
            response, p = _attempt_vector(
                test_vector,
                module_name,
                input_queue,
                output_queue,
                config,
                timeout,
                child_mode,
                p,
                main_proc_runner,
            )
            _populate_result_from_response(result, response, config, suite_name, input_hash)

            # The child returned a result, but it may carry a device-hang
            # exception (e.g. "device timeout in fetch queue wait, potential
            # hang detected"). The Python watchdog never fired because the
            # child responded within the timeout, yet the mesh is now wedged
            # and every later vector will throw the same error. Treat this like
            # a hang: kill/reset the device and (under skip-on-timeout) abort
            # the suite so we recover instead of spinning for the whole job.
            # An intermittent dispatch hang (system_memory_manager.cpp:757 "device
            # timeout") -- the device accumulates dispatch state over the long
            # sequential suite and a vector's dispatch occasionally stalls past the
            # hang-detector, even though the SAME config passes on a clean device
            # (verified: conv2d 1df14794 etc. pass 4/4 in isolation; profiler-off CI
            # runs still hit it, so it is NOT profiler-related). A device reset clears
            # that state, so reset + RETRY the vector -- it runs clean on the next
            # attempt. Falls through to the abort path below only if it hangs AGAIN on
            # the last attempt (a genuine, non-transient hang).
            # NOT on Galaxy: the reset-then-retry recovery re-opens the mesh device, and a
            # SECOND device open inside a Galaxy job re-enters the force-reinit race this
            # framework's one-device-per-job design exists to avoid. Observed in run
            # 30324574397: after a dispatch-hang reset the reopen succeeded and the very
            # next operation blocked forever (49 min mid-vector / 24 min in teardown),
            # invisible to the per-vector watchdog because the block is below Python.
            # On Galaxy fall through to the abort path instead of retrying.
            if _is_device_hang_message(result.get("message")) and attempt < MAX_RETRIES and not _is_galaxy_job():
                logger.warning(
                    f"DEVICE HANG (likely intermittent dispatch-state stall) for "
                    f"input_hash='{input_hash}': {result.get('message')}. Resetting + retrying on a "
                    f"clean device (attempt {attempt + 1}/{1 + MAX_RETRIES})."
                )
                _kill_child(p, timeout_before_rejoin)
                p = None
                if not _reset_or_infra_abort(reset_util, result, input_hash):
                    return result
                if child_mode:
                    p = Process(target=run, args=(input_queue, output_queue, config))
                    p.start()
                continue

            if _is_device_hang_message(result.get("message")):
                logger.error(
                    f"DEVICE HANG detected for input_hash='{input_hash}': {result.get('message')}. "
                    f"Resetting devices and aborting suite."
                )
                _kill_child(p, timeout_before_rejoin)
                p = None
                result["status"] = TestStatus.FAIL_CRASH_HANG
                result["exception"] = str(result.get("message", "DEVICE HANG"))
                result["end_time_ts"] = dt.datetime.now(dt.timezone.utc)
                if not _reset_or_infra_abort(reset_util, result, input_hash):
                    return result
                # On Galaxy, stop the whole run rather than continue: every later module
                # re-opens the mesh device, and a reopen after a hang re-enters the
                # force-reinit race (see the retry branch above). The hanging vector keeps
                # its FAIL_CRASH_HANG status -- a genuine op hang is still reported as a
                # test failure, not hidden -- but the remaining vectors are marked NOT_RUN
                # instead of being run against a device we cannot safely reopen.
                if _is_galaxy_job():
                    logger.error(
                        "DEVICE HANG on Galaxy: a device reopen after a hang re-enters the "
                        "force-reinit race, so exiting the run early instead of continuing."
                    )
                    result["_child_process"] = None
                    result["_abort_suite"] = True
                    result["_infra_abort"] = True
                    return result
                if child_mode:
                    p = Process(target=run, args=(input_queue, output_queue, config))
                    p.start()
                result["_child_process"] = p
                # OR, not assignment: never downgrade an abort another rule already
                # requested for this vector (e.g. the wedged-device rule in
                # _populate_result_from_response) just because skip_on_timeout is off.
                result["_abort_suite"] = result.get("_abort_suite", False) or config.skip_on_timeout
                return result

            # A transient kernel-ELF build/load failure (tt_elffile.cpp:405),
            # typically a cold-cache concurrent fabric_erisc_router build race on
            # the first FABRIC_2D open. The .elf is written by the time we get
            # here, so reset + respawn (warm cache) and retry the SAME vector;
            # it loads cleanly on the next attempt. Fall through to fail only
            # after exhausting retries.
            if _is_elf_load_error(result.get("message")) and attempt < MAX_RETRIES:
                logger.warning(
                    f"KERNEL ELF load failure (likely cold-cache concurrent build) for "
                    f"input_hash='{input_hash}': {result.get('message')}. "
                    f"Resetting + retrying on warm cache (attempt {attempt + 1}/{1 + MAX_RETRIES})."
                )
                _kill_child(p, timeout_before_rejoin)
                p = None
                if not _reset_or_infra_abort(reset_util, result, input_hash):
                    return result
                if child_mode:
                    p = Process(target=run, args=(input_queue, output_queue, config))
                    p.start()
                continue

            # An infrastructure-class failure that has degraded the host:
            #   * fabric / control-plane bring-up (topology_mapper.cpp: the MGD
            #     mesh graph could not be fit onto the discovered physical
            #     topology — e.g. a Galaxy ethernet link degraded below its
            #     required channel count). The mesh never came up, so no op kernel
            #     ran, and every subsequent vector throws the identical error.
            #   * device-fatal wedge ("Read unexpected run_mailbox value: 0x40"):
            #     a core left in a bad run state. The device does NOT recover for
            #     the rest of the job — it cascades into dispatch hangs, slow
            #     Galaxy resets, and all-zero/garbage outputs that mis-report as
            #     PCC failures on every remaining vector (observed: run 29887189384
            #     — one wedged host turned 6 suites into a wall of false PCC/assert
            #     results and burned the 60-min wall-clock).
            # Retrying/resetting will not heal a degraded machine, so abort the
            # suite immediately and mark it NOT_RUN. execute_suite honours
            # _infra_abort unconditionally (regardless of skip_on_timeout) and
            # marks the remaining vectors NOT_RUN; run_sweeps then exits the whole
            # run early, so the job surfaces one infrastructure error instead of a
            # wall of false FAIL_ASSERT_EXCEPTION / PCC results.
            if _is_infra_failure_message(result.get("message")):
                logger.error(
                    f"INFRASTRUCTURE ERROR (degraded host) for input_hash='{input_hash}': "
                    f"{result.get('message')}. The device is degraded and will not recover within "
                    f"this job — exiting the run early instead of reporting false failures."
                )
                _kill_child(p, timeout_before_rejoin)
                p = None
                result["status"] = TestStatus.NOT_RUN
                result["exception"] = "INFRASTRUCTURE ERROR (degraded host): " + str(result.get("message", ""))
                result["end_time_ts"] = dt.datetime.now(dt.timezone.utc)
                result["_child_process"] = p
                result["_abort_suite"] = True
                result["_infra_abort"] = True
                return result

            result["_child_process"] = p
            # setdefault, NOT assignment. This is the NORMAL in-loop return -- the path a
            # vector takes whenever the child returned a response -- so it is the path the
            # wedged-device rule (profiler readback failed + vector failed, set in
            # _populate_result_from_response) actually reaches. Assigning False here threw
            # the abort away, leaving execute_suite with _infra_abort but no _abort_suite:
            # it skipped the mark-remaining-NOT_RUN-and-break branch and kept feeding the
            # rest of the suite to the device just declared wedged.
            result.setdefault("_abort_suite", False)
            return result

        except Empty:
            is_last_attempt = attempt == MAX_RETRIES
            _kill_child(p, timeout_before_rejoin)
            p = None

            if not is_last_attempt:
                logger.warning(
                    f"TEST TIMED OUT (attempt {attempt + 1}/{1 + MAX_RETRIES}) for "
                    f"input_hash='{input_hash}'. Resetting devices and retrying..."
                )
                if not _reset_or_infra_abort(reset_util, result, input_hash):
                    return result
                if child_mode:
                    p = Process(target=run, args=(input_queue, output_queue, config))
                    p.start()
                continue

            logger.warning(
                f"TEST TIMED OUT after {1 + MAX_RETRIES} attempt(s) for "
                f"input_hash='{input_hash}'. Marking as FAIL_CRASH_HANG."
            )
            _set_crash_hang_defaults(result)
            result["original_vector_data"] = result.get("original_vector_data", test_vector)
            result["end_time_ts"] = dt.datetime.now(dt.timezone.utc)
            result["timestamp"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            result["host"] = get_hostname()
            result["user"] = get_username()
            if not _reset_or_infra_abort(reset_util, result, input_hash):
                return result

            if child_mode:
                p = Process(target=run, args=(input_queue, output_queue, config))
                p.start()

            result["_child_process"] = p
            # Preserve an abort already requested by a previous attempt's
            # _populate_result_from_response rather than downgrading it.
            result["_abort_suite"] = result.get("_abort_suite", False) or config.skip_on_timeout
            return result

    result["_child_process"] = p
    # setdefault, NOT assignment: _populate_result_from_response may already have set
    # _abort_suite=True (wedged-device rule: profiler readback failed + vector failed).
    # Overwriting it with False left execute_suite with infra_aborted set but no
    # abort_suite, so it skipped the mark-remaining-NOT_RUN-and-break branch and kept
    # feeding every remaining vector of the suite to the device just declared wedged --
    # the run only ended at the next module boundary, defeating the fail-fast entirely.
    result.setdefault("_abort_suite", False)
    return result


def execute_suite(test_vectors, pbar_manager, suite_name, module_name, header_info, config: SweepsConfig, worker=None):
    # runs a single suite in a test vector
    results = []
    invalid_vectors_count = 0
    # child_mode is False if any of dry_run, vector_id, or main_proc_verbose are truthy
    child_mode = not (config.dry_run or config.vector_id or config.main_proc_verbose)
    # Set True when an infrastructure-class failure (fabric/control-plane bring-up
    # or a device-fatal run_mailbox wedge) aborts this suite. A degraded mesh stays
    # degraded for the rest of the job, so run_sweeps uses this to stop the whole
    # run instead of re-hitting the dead device per suite.
    infra_aborted = False
    # A ``worker`` dict (from run_sweeps) means one persistent worker process + its
    # queues span ALL modules in the job, so the job-level device is opened once
    # and reused (TTNN_SWEEP_JOB_DEVICE). We borrow its queues/process here and DON'T
    # spawn or close it — just hand the (possibly respawned-on-reset) process back.
    # Without one (debug/standalone runs), keep the old per-suite queues + worker.
    owns_worker = worker is None
    if owns_worker:
        input_queue = Queue()
        output_queue = Queue()
        p = None
    else:
        input_queue = worker["input_queue"]
        output_queue = worker["output_queue"]
        p = worker["p"]
    timeout = get_timeout(module_name)
    suite_pbar = pbar_manager.counter(total=len(test_vectors), desc=f"Suite: {suite_name}", leave=False)
    reset_util = tt_smi_util.ResetUtil(config.arch_name)
    timeout_before_rejoin = 5

    # For main process mode, create a persistent runner that keeps device open
    main_proc_runner = None
    main_proc_context = None
    if not child_mode and not config.dry_run:
        logger.info("Running in main process mode - device will remain open for all vectors in suite")
        main_proc_runner, main_proc_context = _create_main_proc_runner(module_name, input_queue, output_queue, config)

    if child_mode and owns_worker:
        p = Process(target=run, args=(input_queue, output_queue, config))
        p.start()

    for i, test_vector in enumerate(test_vectors):
        input_hash = header_info[i].get("input_hash", "N/A")
        logger.info(f"Executing test: Module='{module_name}', Suite='{suite_name}', Input Hash='{input_hash}'")
        if config.dry_run:
            logger.info(f"Would have executed test for vector {test_vector}")
            suite_pbar.update()
            continue
        result = dict()

        # Capture the original test vector data BEFORE any modifications
        original_vector_data = test_vector.copy()
        result["start_time_ts"] = dt.datetime.now(dt.timezone.utc)
        result["input_hash"] = input_hash
        validity = deserialize(test_vector["validity"]).split(".")[-1]
        if validity == VectorValidity.INVALID:
            invalid_vectors_count += 1
            if not config.keep_invalid:
                # Skip this vector entirely - don't add to results
                suite_pbar.update()
                continue
            else:
                # Include invalid vector in results with NOT_RUN status
                result["status"] = TestStatus.NOT_RUN
                result["exception"] = "INVALID VECTOR: " + test_vector["invalid_reason"]
                result["e2e_perf"] = None
        else:
            test_vector.pop("invalid_reason")
            test_vector.pop("status")
            test_vector.pop("validity")

            import ttnn.operation_tracer

            try:
                ttnn.operation_tracer.set_sweep_source_hash(input_hash)
                result = _execute_vector_with_retry(
                    test_vector,
                    module_name,
                    input_queue,
                    output_queue,
                    config,
                    suite_name,
                    input_hash,
                    timeout,
                    timeout_before_rejoin,
                    reset_util,
                    child_mode,
                    p,
                    result,
                    main_proc_runner,
                )
                p = result.pop("_child_process", p)
                abort_suite = result.pop("_abort_suite", False)
                # A fabric / control-plane bring-up failure aborts the suite
                # unconditionally (not gated on skip_on_timeout): the mesh is
                # down, so no remaining vector can be meaningfully tested.
                infra_abort = result.pop("_infra_abort", False)
                if infra_abort:
                    infra_aborted = True

                if abort_suite:
                    if infra_abort or config.skip_on_timeout:
                        results.append(result)
                        suite_pbar.update()
                        skip_reason = (
                            "SKIPPED — INFRASTRUCTURE ERROR ABORTED SUITE"
                            if infra_abort
                            else "SKIPPED DUE TO PREVIOUS TIMEOUT"
                        )
                        logger.info(
                            "Skipping remaining tests in suite due to "
                            + ("infrastructure error." if infra_abort else "timeout.")
                        )
                        for j in range(i + 1, len(test_vectors)):
                            remaining_vector = test_vectors[j]
                            skipped_result = dict()
                            skipped_result["input_hash"] = header_info[j].get("input_hash", "N/A")
                            skipped_result["start_time_ts"] = dt.datetime.now(dt.timezone.utc)
                            skipped_result["original_vector_data"] = remaining_vector.copy()
                            skipped_result["status"] = TestStatus.NOT_RUN
                            skipped_result["exception"] = skip_reason
                            skipped_result["e2e_perf"] = None
                            skipped_result["end_time_ts"] = dt.datetime.now(dt.timezone.utc)
                            skipped_result["timestamp"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
                            skipped_result["host"] = get_hostname()
                            skipped_result["user"] = get_username()
                            results.append(skipped_result)
                            suite_pbar.update()
                        break
            except tt_smi_util.ResetFailed as e:
                # Every reset mechanism failed: the device is wedged and cannot be
                # recovered on this host. Continuing would re-hang + re-reset every
                # remaining vector and burn the whole job timeout, so abort the
                # suite now (regardless of skip-on-timeout) and mark the rest NOT_RUN.
                logger.error(f"Device reset failed unrecoverably: {e}. Aborting remaining tests in suite.")
                result["status"] = TestStatus.FAIL_CRASH_HANG
                result["exception"] = str(e)
                # This path breaks before the common footer that stamps this; set it here
                # so the abort record carries original_vector_data like every other result.
                result["original_vector_data"] = original_vector_data
                result["e2e_perf"] = None
                result["end_time_ts"] = dt.datetime.now(dt.timezone.utc)
                result["timestamp"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
                result["host"] = get_hostname()
                result["user"] = get_username()
                results.append(result)
                suite_pbar.update()
                for j in range(i + 1, len(test_vectors)):
                    remaining_vector = test_vectors[j]
                    skipped_result = dict()
                    skipped_result["input_hash"] = header_info[j].get("input_hash", "N/A")
                    skipped_result["start_time_ts"] = dt.datetime.now(dt.timezone.utc)
                    skipped_result["original_vector_data"] = remaining_vector.copy()
                    skipped_result["status"] = TestStatus.NOT_RUN
                    skipped_result["exception"] = "SKIPPED DUE TO UNRECOVERABLE DEVICE RESET"
                    skipped_result["e2e_perf"] = None
                    skipped_result["end_time_ts"] = dt.datetime.now(dt.timezone.utc)
                    skipped_result["timestamp"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
                    skipped_result["host"] = get_hostname()
                    skipped_result["user"] = get_username()
                    results.append(skipped_result)
                    suite_pbar.update()
                p = None
                break
            except Exception as e:
                logger.exception(f"Unexpected error executing vector: {e}")
                result["status"] = TestStatus.FAIL_ASSERT_EXCEPTION
                result["exception"] = str(e)
                result["e2e_perf"] = None
            finally:
                ttnn.operation_tracer.set_sweep_source_hash(None)

        # Add the original test vector data to the result
        result["original_vector_data"] = original_vector_data
        result["end_time_ts"] = dt.datetime.now(dt.timezone.utc)
        result["timestamp"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        result["host"] = get_hostname()
        result["user"] = get_username()

        suite_pbar.update()
        results.append(result)

        # Abort the suite if a fatal device error was encountered
        if "DEVICE EXCEPTION" in str(result.get("exception", "")):
            logger.error("Aborting test suite due to fatal device error.")
            if p and p.is_alive():
                p.terminate()
                p.join()
            p = None  # dead; run_sweeps respawns a fresh worker for the next module
            break

    if owns_worker:
        if p is not None:
            # The worker (run) is persistent and only exits on _WORKER_CLOSE — send it
            # so p.join() doesn't block forever (it no longer self-exits on an idle queue).
            try:
                if p.is_alive():
                    input_queue.put(_WORKER_CLOSE)
                    p.join(timeout=_WORKER_JOIN_TIMEOUT_S)
            except Exception:
                pass
            if p.is_alive():
                _kill_child(p, timeout_before_rejoin)
        # Cleanup main process context (close device)
        if main_proc_context is not None:
            try:
                next(main_proc_context)
            except StopIteration:
                # generator already exhausted (device already closed) — nothing left to clean up
                pass
            logger.info("Device closed in main process mode")
    else:
        # Persistent worker: hand the (possibly respawned/killed) process back to
        # run_sweeps so it and its one open job device carry over to the next module.
        worker["p"] = p

    suite_pbar.close()
    return results, invalid_vectors_count, infra_aborted


def _vector_mesh_pair(vector, extended_sources=True):
    """The (rows, cols) mesh a raw (pre-sanitize) vector was traced on, or None.

    Mirrors how the sweep bodies themselves derive it (tensor_placement first). Several
    modules -- add_model_traced.py and linear_model_traced.py among them -- pin
    os.environ["MESH_DEVICE_SHAPE"] to THIS value per vector before opening their device,
    so the mesh shape is a real per-vector component of _job_device_key, not a per-job
    constant.
    """

    def _parse_two_ints(value):
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return int(value[0]), int(value[1])
            except (TypeError, ValueError):
                return None
        if isinstance(value, str):
            nums = re.findall(r"\d+", value)
            if len(nums) >= 2:
                return int(nums[0]), int(nums[1])
        return None

    # 1) Explicit tensor placement mesh_device_shape (model_traced vectors). Same key
    #    order the modules use before pinning MESH_DEVICE_SHAPE.
    keys = ("input_a_tensor_placement", "input_tensor_tensor_placement")
    if extended_sources:
        # Extra sources are used for device-key GROUPING only, never for the --mesh-dims
        # filter: they make the shape determinable for more vectors, and a vector the
        # filter previously could not classify was always KEPT. Widening it there could
        # start dropping vectors, which is a behaviour change well beyond this fix.
        keys = (
            "input_a_tensor_placement",
            "input_b_tensor_placement",
            "input_tensor_b_tensor_placement",
            "input_tensor_tensor_placement",
        )
    for key in keys:
        tp = vector.get(key)
        if isinstance(tp, dict):
            pair = _parse_two_ints(tp.get("mesh_device_shape"))
            if pair:
                return pair
    # 2) Explicit mesh_shape field (generality / lead_model vectors).
    pair = _parse_two_ints(vector.get("mesh_shape"))
    if pair:
        return pair
    # 3) mesh_device descriptor.
    md = vector.get("mesh_device")
    if isinstance(md, dict):
        pair = _parse_two_ints(md.get("shape") or md.get("repr", ""))
        if pair:
            return pair
    # 4) Per-vector traced_machine_info -- add_model_traced's own last resort, and every
    #    vector records it. Grouping only (see above).
    if extended_sources:
        ti = vector.get("traced_machine_info")
        for entry in ti if isinstance(ti, list) else [ti]:
            if isinstance(entry, dict):
                pair = _parse_two_ints(entry.get("mesh_device_shape"))
                if pair:
                    return pair
    # 5) .mesh_RxC suffix on the stored sweep/suite name.
    for key in ("sweep_name", "suite_name"):
        name = vector.get(key)
        if isinstance(name, str):
            ms = parse_mesh_suffix(name)
            if ms:
                return (ms[0], ms[1])
    return None


def _vector_mesh_dims(vector) -> str | None:
    """Classify a vector's mesh as '1d' (a unit axis -> FABRIC_1D/RING), '2d' (both axes
    > 1 -> FABRIC_2D), or None when the shape can't be determined (never filtered out)."""
    pair = _vector_mesh_pair(vector, extended_sources=False)
    if pair is None:
        return None
    return "1d" if (pair[0] == 1 or pair[1] == 1) else "2d"


def _filter_vectors_by_mesh_dims(vectors, mesh_dims):
    """Keep only vectors matching the requested mesh dimensionality.

    Vectors whose mesh can't be determined are kept (never silently dropped);
    they are rare for CCL ops (which always carry a tensor placement).
    """
    if not mesh_dims:
        return vectors
    kept, dropped, undetermined = [], 0, 0
    for v in vectors:
        dims = _vector_mesh_dims(v)
        if dims is None:
            undetermined += 1
            kept.append(v)
        elif dims == mesh_dims:
            kept.append(v)
        else:
            dropped += 1
    logger.info(
        f"mesh-dims filter '{mesh_dims}': kept {len(kept)} vector(s), dropped {dropped} "
        f"(mesh mismatch){f', {undetermined} undetermined kept' if undetermined else ''}."
    )
    return kept


def _vector_device_group(vector, env_axis):
    """Device-key group for a vector: (mesh shape, dispatch axis).

    Both components matter because both are part of _job_device_key:

    - mesh shape: several modules (add_model_traced, linear_model_traced, ...) pin
      os.environ["MESH_DEVICE_SHAPE"] to THIS vector's traced shape before opening, so a
      job's vectors legitimately span [4,8]/[8,4]/[4,4]/[1,32]/[1,1]. Grouping on the axis
      alone would interleave those and could ADD reopens rather than remove them, which is
      the opposite of the point. Mesh shape leads the key since it is the coarser split.
    - axis: 'row'/'col' when the vector's grids force one, else the pass's
      TTNN_DISPATCH_AXIS (what a module passing None inherits), else 'auto'.

    The axis is a HEURISTIC, not a reproduction of each module's logic -- e.g. linear's
    gather_in0 path deliberately ignores the nominal compute width and keys off output/hop
    grids, while the shared scanner classifies the nominal width. A mis-predicted axis only
    costs an extra reopen (it can never change a result), and the log line reports the
    transitions actually achieved, so a wrong hint shows up as a smaller-than-expected win.
    """
    mesh = _vector_mesh_pair(vector)
    try:
        from split_vectors_by_axis import vector_dispatch_axis_hint

        hint = vector_dispatch_axis_hint(vector)
    except Exception:
        hint = None
    if hint is None:
        hint = env_axis if env_axis in ("col", "row") else "auto"
    return (mesh, hint)


def _order_vectors_by_device_key(vectors, module_name, suite_name):
    """Stable-sort vectors so same-device vectors run back to back.

    The 8 model_traced modules that open their own device per vector derive the
    dispatch axis from each vector's shard/compute grid, so in file order the job
    device key flips ROW<->COL repeatedly and each flip is a close+reopen. On Galaxy a
    reopen is the event that wedges a dispatch core (run_mailbox=0x40), so grouping
    vectors by the device they need cuts reopens from O(transitions) to O(distinct
    keys): measured 44 -> 9 over the 293 Galaxy vectors in those modules.

    Vectors matching the current pass's axis go first so the device the worker already
    opened is reused for the longest initial stretch.

    This only ever REORDERS: the result is verified to be a permutation of the input by
    object identity, so no vector can be duplicated (run twice) or dropped. If that
    check or anything else fails, the original order is returned unchanged -- a few
    extra reopens are always preferable to running a vector twice or losing one.
    Set TTNN_SWEEP_NO_VECTOR_REORDER=1 to disable.
    """
    if os.environ.get("TTNN_SWEEP_NO_VECTOR_REORDER") == "1" or len(vectors) < 2:
        return vectors
    try:
        env_axis = os.environ.get("TTNN_DISPATCH_AXIS", "").strip().lower()
        groups = [_vector_device_group(v, env_axis) for v in vectors]
        first_seen = {}
        for g in groups:
            first_seen.setdefault(g, len(first_seen))
        if len(first_seen) < 2:
            return vectors  # nothing to gain -- leave the order byte-identical

        # Order groups by FIRST APPEARANCE, and keep original order within a group. This
        # keeps the first vector's device first (so the device already open is not
        # immediately swapped) and yields exactly len(groups)-1 transitions, without
        # needing to know which device is currently open.
        order = sorted(range(len(vectors)), key=lambda i: (first_seen[groups[i]], i))
        reordered = [vectors[i] for i in order]

        # Permutation check by identity: same objects, same multiplicity, none lost.
        if len(reordered) != len(vectors) or collections.Counter(map(id, reordered)) != collections.Counter(
            map(id, vectors)
        ):
            logger.warning(
                f"vector reorder for {module_name}/{suite_name} did not produce a permutation "
                f"({len(vectors)} in, {len(reordered)} out) -- keeping the original order."
            )
            return vectors
        moved = builtins.sum(1 for i, j in enumerate(order) if i != j)
        counts = ", ".join(
            f"mesh{g[0] if g[0] else '?'}/{g[1]}={groups.count(g)}"
            for g in sorted(first_seen, key=lambda g: first_seen[g])
        )
        logger.info(
            f"vector order for {module_name}/{suite_name}: grouped {len(vectors)} vector(s) by "
            f"(mesh shape, dispatch axis) ({counts}); {moved} moved, {len(first_seen) - 1} device "
            f"reopen(s) instead of {builtins.sum(1 for a, b in zip(groups, groups[1:]) if a != b)}."
        )
        return reordered
    except Exception as e:
        logger.warning(f"vector reorder for {module_name}/{suite_name} failed ({e}) -- keeping the original order.")
        return vectors


def run_sweeps(
    module_names,
    config: SweepsConfig,
):
    pbar_manager = enlighten.get_manager()

    # Set up vector source based on config
    source_kwargs = {}
    if config.vector_source == "file":
        source_kwargs = {
            "file_path": config.file_path,
        }
    # vectors_export uses default kwargs
    vector_source = VectorSourceFactory.create_source(config.vector_source, **source_kwargs)

    # Set up result destination based on config
    result_kwargs = {}
    # results_export and superset use default kwargs
    result_dest = ResultDestinationFactory.create_destination(config.result_destination, **result_kwargs)

    # Initialize run metadata and run record
    run_id = None
    final_status = "success"

    if not config.dry_run:
        run_metadata = {
            "initiated_by": get_initiated_by(),
            "host": get_hostname(),
            "card_type": config.arch_name,
            "runner_label": os.getenv("RUNNER_LABEL"),  # CI runner label (e.g., N150, N300, BH-LoudBox)
            "run_type": "sweeps",
            "run_contents": config.run_contents,
            "git_author": get_git_author(),
            "git_branch_name": get_git_branch(),
            "git_commit_sha": git_hash(),
            "github_pipeline_id": get_github_pipeline_id(),
            "run_start_ts": dt.datetime.now(dt.timezone.utc),
            "status": "success",
        }
        run_id = result_dest.initialize_run(run_metadata)
        if run_id:
            logger.info(f"Initialized run with id: {run_id}")

    # Unified processing regardless of source
    # Summary counters
    total_vectors_run = 0  # total number of test cases (vectors)
    total_tests_run = 0  # total number of suites executed
    total_invalid_vectors = 0  # total number of invalid vectors (skipped)
    module_suite_test_count = {}  # module_name -> {suite_name: count}
    max_test_cases_module = None  # find the module with the most test cases
    max_test_cases_per_module = 0
    # Track test status counts across the entire run (only meaningful for non-dry runs)
    status_counts = {}
    # Set True when a suite aborts on a fabric/control-plane infra failure; used
    # to stop the whole run early (a degraded device stays degraded for the job).
    infra_aborted = False

    module_pbar = pbar_manager.counter(total=len(module_names), desc="Modules", leave=False)

    # One persistent worker for the whole job spans every module so the job-level
    # device cache reuses ONE open device across modules that share a config — the
    # fix for the per-module device reopen that force-reinitializes dispatch on
    # Galaxy. Gated to Galaxy: single-host (N150/N300/T3K) has no reopen-wedge, and
    # has modules that take a single-device path (ttnn.open_device) that would
    # collide with a held mesh device — so single-host keeps the ORIGINAL
    # per-module-child model (job_worker=None -> execute_suite owns_worker path).
    # Debug modes (dry_run/vector_id/main_proc_verbose) also keep per-suite workers.
    job_child_mode = not (config.dry_run or config.vector_id or config.main_proc_verbose)
    job_worker = None
    if job_child_mode and _is_galaxy_job():
        # Prime the device-count cache in THIS (main) process before the worker
        # opens the job device — result export's card-type fallback queries the
        # count (constructs a cluster), which would collide with the worker's held
        # device (CHIP_IN_USE) if queried live. No-op when RUNNER_LABEL is set (CI).
        if not os.environ.get("RUNNER_LABEL"):
            try:
                from framework.result_destination import prime_device_count

                prime_device_count()
            except Exception:
                pass
        # Enable job-level device reuse in create_mesh_device (inherited by the
        # forked worker). Only vectors sharing a device config reach a given
        # process (two-pass splits by dispatch axis), so the cached device is
        # reused, not reconfigured, within a job.
        os.environ["TTNN_SWEEP_JOB_DEVICE"] = "1"
        job_worker = {"input_queue": Queue(), "output_queue": Queue(), "p": None}
        job_worker["p"] = Process(target=run, args=(job_worker["input_queue"], job_worker["output_queue"], config))
        job_worker["p"].start()

    try:
        for module_name in module_names:
            if config.suite_name:
                # Filter to only the specified suite
                all_suites = vector_source.get_available_suites(module_name)
                if config.suite_name not in all_suites:
                    logger.warning(
                        f"Suite '{config.suite_name}' not found in module '{module_name}'. Available suites: {all_suites}"
                    )
                    continue  # or exit with error
                suites = [config.suite_name]
            else:
                suites = vector_source.get_available_suites(module_name)

            for suite in suites:
                suite_start_time = dt.datetime.now(dt.timezone.utc)

                vectors = vector_source.load_vectors(module_name, suite, config.vector_id)
                vectors = _filter_vectors_by_mesh_dims(vectors, config.mesh_dims)
                # Group same-device vectors together BEFORE sanitize_inputs, which builds
                # header_info as a list positionally parallel to test_vectors -- reordering
                # after it would misattribute every result to the wrong vector.
                vectors = _order_vectors_by_device_key(vectors, module_name, suite)
                # Update summary counters
                total_vectors_run += len(vectors)
                total_tests_run += 1
                module_suite_test_count.setdefault(module_name, {})
                module_suite_test_count[module_name][suite] = module_suite_test_count[module_name].get(suite, 0) + len(
                    vectors
                )
                # Track max per module (for dry run summary)
                module_total = builtins.sum(module_suite_test_count[module_name].values())
                if module_total > max_test_cases_per_module:
                    max_test_cases_per_module = module_total
                    max_test_cases_module = module_name
                if not vectors:
                    logger.warning(f"No vectors found for module {module_name}, suite {suite}")
                    continue
                header_info, test_vectors = sanitize_inputs(vectors)
                results, invalid_vectors_count, infra_aborted = execute_suite(
                    test_vectors, pbar_manager, suite, module_name, header_info, config, worker=job_worker
                )
                total_invalid_vectors += invalid_vectors_count

                suite_end_time = dt.datetime.now(dt.timezone.utc)
                logger.info(f"Completed tests for module {module_name}, suite {suite}.")

                # Export results
                if not config.dry_run and results:
                    if config.summary:
                        # Aggregate status counts for summary
                        for res in results:
                            st = res.get("status")
                            if st is not None:
                                key = getattr(st, "name", None)
                                if key is None:
                                    val = getattr(st, "value", None)
                                    key = str(val) if val is not None else str(st)
                                status_counts[key] = status_counts.get(key, 0) + 1

                    run_context = {
                        "run_id": run_id,
                        "test_start_time": suite_start_time,
                        "test_end_time": suite_end_time,
                        "git_hash": git_hash(),
                    }
                    try:
                        test_status = result_dest.export_results(header_info, results, run_context)
                        if test_status == "failure":
                            final_status = "failure"
                    except Exception as e:
                        logger.exception(f"Failed to export results for {module_name}, suite {suite}: {e}")
                        final_status = "failure"
                        # continue with other suites

                # A degraded mesh (fabric / control-plane bring-up failure) stays
                # degraded for the rest of the job — every remaining suite and
                # module would re-hit the same dead device and re-report the same
                # infra error. Stop the whole run now, after exporting this
                # suite's NOT_RUN results, and finalize as a failure so the job is
                # visibly red for the infrastructure fault (not a silent skip).
                if infra_aborted:
                    logger.error(
                        "Infrastructure error (degraded host: fabric topology mapping or a "
                        "device-fatal run_mailbox wedge) detected; the device is degraded for the "
                        "remainder of this job. Aborting the entire run early."
                    )
                    final_status = "failure"
                    break

            if infra_aborted:
                break
            module_pbar.update()
    except Exception as e:
        logger.error(f"Error during sweep execution: {e}")
        final_status = "failure"
        raise
    finally:
        # Shut down the persistent job worker (its finally closes the job device).
        if job_worker is not None:
            wp = job_worker.get("p")
            try:
                if wp is not None and wp.is_alive():
                    job_worker["input_queue"].put(_WORKER_CLOSE)
                    wp.join(timeout=_WORKER_JOIN_TIMEOUT_S)
            except Exception:
                pass
            if wp is not None and wp.is_alive():
                _kill_child(wp, _WORKER_KILL_GRACE_S)
        if not config.dry_run:
            result_dest.finalize_run(run_id, final_status)
            logger.info(f"Finalized run with status: {final_status}")
        module_pbar.close()

        # Emit summary if requested
        if config.summary:
            if config.dry_run:
                logger.info("--- DRY RUN SUMMARY ---")
                logger.info(f"Total tests (modules) that would have been run: {len(module_names)}")
                logger.info(f"Total test cases (vectors) that would have been run: {total_vectors_run}")
            else:
                logger.info("=== EXECUTION SUMMARY ===")
                logger.info(f"Total tests (module-suite combinations) executed: {total_tests_run}")
                logger.info(f"Total test cases (vectors) executed: {total_vectors_run}")
                if config.keep_invalid:
                    logger.info(f"Total invalid vectors (included in results as NOT_RUN): {total_invalid_vectors}")
                else:
                    logger.info(f"Total invalid vectors (excluded from results): {total_invalid_vectors}")
                # Status breakdown across all executed tests
                if status_counts:
                    logger.info("\n=== TEST STATUS COUNTS ===")
                    for status_name in sorted(status_counts.keys()):
                        logger.info(f"{status_name}: {status_counts[status_name]}")

            # Detailed breakdown by module and suite
            if module_suite_test_count:
                logger.info("\n=== DETAILED BREAKDOWN BY MODULE AND SUITE ===")
                for mod in sorted(module_suite_test_count.keys()):
                    module_total = builtins.sum(module_suite_test_count[mod].values())
                    logger.info(f"Module: {mod} (Total: {module_total} test cases)")
                    for suite_name in sorted(module_suite_test_count[mod].keys()):
                        test_count = module_suite_test_count[mod][suite_name]
                        logger.info(f"  └─ Suite: {suite_name} ({test_count} test cases)")

            # Extra dry-run insight: max test cases per module
            if config.dry_run and max_test_cases_module:
                logger.info(
                    f"\nMaximum test cases per module: {max_test_cases_per_module} (in {max_test_cases_module})"
                )

    # Derive failure from actual per-test statuses, not from export_results() return value
    # (export_results unconditionally returns "success" for file-based destinations).
    if config.fail_on_test_failure and status_counts:
        from tests.sweep_framework.framework.statuses import TestStatus

        fail_status_names = {
            TestStatus.FAIL_ASSERT_EXCEPTION.name,
            TestStatus.FAIL_CRASH_HANG.name,
            TestStatus.FAIL_L1_OUT_OF_MEM.name,
            TestStatus.FAIL_WATCHER.name,
            TestStatus.FAIL_UNSUPPORTED_DEVICE_PERF.name,
        }
        failed_count = sum(count for name, count in status_counts.items() if name in fail_status_names)
        if failed_count > 0:
            final_status = "failure"
            logger.error(f"{failed_count} test case(s) failed/crashed/hung")

    return final_status, infra_aborted


def get_module_names(config: SweepsConfig):
    """Extract module names based on configuration"""
    if not config.module_name:
        module_names = list(get_all_modules())
        logger.info(f"Running all modules.")
        if config.skip_modules:
            skip_modules_set = {name.strip() for name in config.skip_modules.split(",")}
            module_names = [name for name in module_names if name not in skip_modules_set]
            logger.info(f"But skipping: {', '.join(skip_modules_set)}")
        return module_names

    # Parse selectors and expand directory-like prefixes to all contained modules
    selectors = [name.strip() for name in config.module_name.split(",") if name.strip()]
    all_modules = list(get_all_modules())

    expanded: list[str] = []
    seen: set[str] = set()
    for sel in selectors:
        # Exact matches first
        matches = [m for m in all_modules if m == sel or m.startswith(sel + ".")]
        if not matches:
            logger.warning(f"No modules matched selector '{sel}'.")
            continue
        for m in matches:
            if m not in seen:
                expanded.append(m)
                seen.add(m)

    if not expanded:
        logger.error("No modules matched any provided selectors.")
        exit(1)

    logger.info(f"Expanded module selectors {selectors} to {len(expanded)} modules to run.")
    return expanded


def get_run_contents(config: SweepsConfig):
    """Generate run contents description based on configuration"""
    if config.module_name or config.suite_name:
        run_contents_details = []
        if config.module_name:
            run_contents_details.append(f"{config.module_name}")
        if config.suite_name:
            run_contents_details.append(f"{config.suite_name}")
        return ", ".join(run_contents_details)
    else:
        return "all_sweeps"


def enable_watcher():
    logger.info("Enabling Watcher")
    os.environ["TT_METAL_WATCHER"] = "120"
    os.environ["TT_METAL_WATCHER_APPEND"] = "1"


def disable_watcher():
    logger.info("Disabling Watcher")
    os.environ.pop("TT_METAL_WATCHER", None)
    os.environ.pop("TT_METAL_WATCHER_APPEND", None)


def _is_multidevice_ccl_module(module_name):
    """True if module_name is a CCL / multi-device op (all_gather, all_reduce,
    reduce_scatter, all_to_all, all_broadcast). Used to decide whether the device
    profiler is safe to enable for the run -- see _should_skip_device_profiler."""
    if not module_name:
        return False
    _ccl = ("all_gather", "all_reduce", "reduce_scatter", "all_to_all", "all_broadcast")
    return any(any(c in m for c in _ccl) for m in str(module_name).split(",") if m)


def _should_skip_device_profiler(config):
    """Whether to skip enabling the device profiler for this run. The profiler is a
    process-global toggle (TT_METAL_DEVICE_PROFILER, read once at startup; kernels are
    JIT-compiled with instrumentation), so it can't be disabled per-vector -- if it's
    unsafe for any vector the run will hit, skip it for the whole run.

    CCL on a 2D mesh: with the profiler on, the cq_prefetch dispatch kernel that
    FABRIC_2D pushes onto idle-erisc overflows the idle-erisc code region
    (idle_erisc.elf 0x5544 > 0x5390) and wedges the erisc cores at mesh open
    (run_mailbox 0x40). FABRIC_1D CCL (mesh_dims=="1d") profiles fine -> keep it.

    (conv2d is NOT skipped: a controlled A/B on relay T3K -- looping the 5 known-hanging
    SDXL configs 787ff3a2/8c7a2cf1/1df14794/46992c20/f350bce3 with
    TT_METAL_OPERATION_TIMEOUT_SECONDS=30 -- shows the large 3x3 height-sharded
    auto-sliced/haloed conv2d dispatch deadlock (system_memory_manager.cpp:702/757)
    reproduces with the profiler OFF at the same onset (~10-15 cumulative executions,
    round 3) as with it ON, so skipping the profiler does NOT avoid it. It is a genuine
    intermittent conv2d/dispatch op-level deadlock, handled by reset+retry in
    _execute_vector_with_retry; the heavy-conv profiler GATHER hang is handled per-vector
    by conv2d's _SKIP_DEVICE_PERF.)

    Vectors of a skipped run report device-perf N/A and PASS (not
    FAIL_UNSUPPORTED_DEVICE_PERF) -- see _populate_result_from_response."""
    if _is_multidevice_ccl_module(config.module_name) and config.mesh_dims != "1d":
        return True
    return False


def enable_profiler():
    logger.info("Enabling Device Profiler")
    os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
    # NOTE: ENABLE_TRACY is deliberately NOT set here. It is a CMake option
    # (-DENABLE_TRACY=ON in setup.py; the profiler test scripts only ever print "Make sure
    # this test runs in a build with cmake option ENABLE_TRACY=ON"). Setting it as an env var
    # was a no-op -- this was the ONLY runtime reference to it in the repo -- and it gave the
    # false impression that Tracy was being turned on when Tracy support is fixed at build
    # time. Check the built binary instead and say so plainly.
    os.environ["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"
    # C++ post-process exposes per-chip perf in memory via
    # ttnn._ttnn.profiler.get_latest_programs_perf_data(); required for the
    # modern (multi-chip-safe) device-perf read in perf_utils.gather_single_test_perf.
    os.environ["TT_METAL_PROFILER_CPP_POST_PROCESS"] = "1"
    # Surface a non-Tracy build instead of silently collecting nothing: the two APIs the
    # device-perf path depends on only exist in a Tracy-enabled build.
    try:
        import ttnn

        missing = [n for n in ("ReadDeviceProfiler", "get_latest_programs_perf_data") if not hasattr(ttnn, n)]
        if missing:
            logger.warning(
                f"Device profiler requested but this ttnn build is missing {missing} -- it was very "
                "likely built without -DENABLE_TRACY=ON. Device-perf will be reported N/A."
            )
    except Exception as e:
        logger.warning(f"Could not verify Tracy support in this ttnn build ({e}).")


def disable_profiler():
    logger.info("Disabling Device Profiler")
    os.environ.pop("TT_METAL_DEVICE_PROFILER", None)
    os.environ.pop("ENABLE_TRACY", None)
    os.environ.pop("TT_METAL_PROFILER_MID_RUN_DUMP", None)
    os.environ.pop("TT_METAL_PROFILER_CPP_POST_PROCESS", None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sweep Test Runner",
        description="Run test vector suites from generated vector database.",
    )
    parser.add_argument(
        "--module-name",
        required=False,
        help=(
            "Module selector(s). Comma-separated. Accepts full module names (e.g. 'eltwise.unary.relu.relu') "
            "or directory-like prefixes to run all contained modules (e.g. 'eltwise', 'eltwise.unary', 'matmul')."
        ),
    )
    parser.add_argument("--suite-name", required=False, help="Suite of Test Vectors to run, or all tests if omitted.")

    parser.add_argument(
        "--vector-source",
        required=True,
        choices=["file", "vectors_export"],
        help="Test vector source. Available presets are ['file', 'vectors_export']",
    )

    parser.add_argument("--file-path", required=False, help="Read and execute test vectors from a specified file path.")

    parser.add_argument(
        "--mesh-dims",
        required=False,
        choices=["1d", "2d"],
        default=None,
        help=(
            "Restrict the run to vectors whose mesh is 1D (rows==1 or cols==1) or 2D (both >1). "
            "Splits CCL ops into separate jobs per fabric family so one process never does a live "
            "FABRIC_1D->FABRIC_2D transition. Omit to run all meshes."
        ),
    )

    parser.add_argument(
        "--vector-id", required=False, help="Specify vector id with a module name to run an individual test vector."
    )

    parser.add_argument(
        "--result-dest",
        required=True,
        choices=["results_export", "superset"],
        help="Specify test result destination. Available presets are ['results_export', 'superset']",
    )

    parser.add_argument(
        "--watcher", action="store_true", required=False, help="Add this flag to run sweeps with watcher enabled."
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        required=False,
        help="Add this flag to measure e2e perf, for op tests with performance markers.",
    )

    parser.add_argument(
        "--perf-with-cache",
        action="store_true",
        required=False,
        help="Add this flag to measure e2e perf with and without program cache. Runs each test twice to capture both cached and uncached performance.",
    )

    parser.add_argument(
        "--device-perf",
        required=False,
        action="store_true",
        help="Measure device perf using device profiler. REQUIRES PROFILER BUILD!",
    )

    parser.add_argument(
        "--measure-memory",
        required=False,
        action="store_true",
        help="Capture L1 memory usage per core using graph trace (NO_DISPATCH mode). Memory data will be included in test results.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=False,
        help="Add this flag to perform a dry run.",
    )

    parser.add_argument(
        "--tag",
        required=False,
        default=os.getenv("USER"),
        help="Custom tag for the vectors you are running. This is to keep copies separate from other people's test vectors. By default, this will be your username. You are able to specify a tag when generating tests using the generator.",
    )

    parser.add_argument(
        "--skip-modules",
        required=False,
        help="Comma-separated list of modules to skip when running all modules.",
    )

    parser.add_argument(
        "--skip-on-timeout",
        action="store_true",
        required=False,
        help="Skip remaining tests in suite when a test times out. Default behavior is to not skip.",
    )

    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        required=False,
        help="Include invalid vectors in results with NOT_RUN status. Default behavior is to exclude invalid vectors from results entirely.",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        required=False,
        help="Log a detailed execution or dry-run summary at the end of the run.",
    )

    parser.add_argument(
        "--main-proc-verbose",
        action="store_true",
        required=False,
        help="Run tests in parent process (disables hang detection). Required for Tracy profiling and debugging. Prints test exceptions to stdout.",
    )

    parser.add_argument(
        "--trace-params",
        action="store_true",
        required=False,
        help="Enable tracing of operation parameters (serializes all ttnn operation inputs to files). Outputs to generated/ttnn/reports/operation_parameters/",
    )

    parser.add_argument(
        "--fail-on-test-failure",
        action="store_true",
        required=False,
        help="Exit with non-zero status if any test case fails, crashes, or hangs. Use in CI to mark the job as failed.",
    )

    args = parser.parse_args(sys.argv[1:])

    # Argument validation
    validate_arguments(args, parser)

    # Create sweeps config object
    config = create_config_from_args(args)

    if config.watcher:
        enable_watcher()

    if config.measure_device_perf and not _should_skip_device_profiler(config):
        enable_profiler()
    elif config.measure_device_perf:
        logger.info(
            f"Skipping device profiler for {config.module_name!r} "
            f"(mesh_dims={config.mesh_dims!r}, MESH_DEVICE_SHAPE={os.environ.get('MESH_DEVICE_SHAPE')!r}): "
            "the profiler is process-global and unsafe for this run -- CCL on FABRIC_2D overflows the "
            "idle-erisc code region at mesh open (idle_erisc.elf 0x5544 > 0x5390; run_mailbox 0x40). "
            "Such vectors report device-perf N/A and PASS. 1D-only CCL keeps device-perf."
        )

    # Generate run contents description
    config.run_contents = get_run_contents(config)

    logger.info(
        f"Running current sweeps with tag: {config.sweeps_tag} using {config.vector_source} test vector source, outputting to {config.result_destination}."
    )

    # Log performance measurement configuration
    if config.measure_perf_with_cache:
        logger.info(
            "Performance measurement: Enabled with cache measurement (runs each test twice to capture both cached and uncached performance)"
        )
    elif config.measure_perf:
        logger.info("Performance measurement: Enabled (single run, uncached performance only)")
    else:
        logger.info("Performance measurement: Disabled")

    # Log memory measurement configuration
    if config.measure_memory:
        logger.info("Memory measurement: Enabled (using graph trace NO_DISPATCH mode)")
    else:
        logger.info("Memory measurement: Disabled")

    if config.skip_on_timeout:
        logger.info("Timeout behavior: Skip remaining tests in suite when a test times out.")
    else:
        logger.info("Timeout behavior: Continue running remaining tests in suite when a test times out.")

    # Parse modules for running specific tests
    module_names = get_module_names(config)

    final_status, infra_aborted = run_sweeps(
        module_names,
        config=config,
    )

    if config.watcher:
        disable_watcher()

    if config.measure_device_perf:
        disable_profiler()

    # An infrastructure abort (degraded host: fabric topology mapping failure or a
    # device-fatal run_mailbox wedge) always forces a nonzero exit, independent of
    # --fail-on-test-failure. The sweep workflow only sets that input for the Lead
    # Models and Model Traced suites, so without this the vast majority of suites
    # would exit 0 (green) on a degraded machine and hide the real infra fault.
    if infra_aborted:
        logger.error("Exiting with failure: infrastructure error degraded the host mid-run (see log above)")
        sys.exit(1)

    if config.fail_on_test_failure and final_status == "failure":
        logger.error("Exiting with failure: one or more test cases did not pass (--fail-on-test-failure)")
        sys.exit(1)
