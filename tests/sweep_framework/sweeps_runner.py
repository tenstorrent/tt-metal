# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import os
import pathlib
import enlighten
import importlib
import datetime as dt
from multiprocessing import Process
from faster_fifo import Queue
from queue import Empty
import builtins
from framework.statuses import VectorValidity, TestStatus
import framework.tt_smi_util as tt_smi_util
from framework.sweeps_logger import sweeps_logger as logger
from framework.vector_source import VectorSourceFactory
from framework.serialize import deserialize
import subprocess
from dataclasses import dataclass
from typing import Optional
from framework.result_destination import ResultDestinationFactory
from tt_metal.tools.profiler.process_ops_logs import get_device_data_generate_report
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR
from sweep_utils.roofline_utils import get_updated_message

PROCESS_TERMINATION_TIMEOUT_SECONDS = 5


@dataclass
class SweepsConfig:
    """Configuration object for sweeps runner"""

    module_name: Optional[str] = None
    suite_name: Optional[str] = None
    vector_source: str = "elastic"
    file_path: Optional[str] = None
    vector_id: Optional[str] = None
    result_destination: str = "elastic"
    watcher: bool = False
    measure_perf: bool = False
    measure_device_perf: bool = False
    dry_run: bool = False
    sweeps_tag: Optional[str] = None
    skip_modules: Optional[str] = None
    skip_on_timeout: bool = False
    elastic_connection_string: Optional[str] = None
    elastic_username: Optional[str] = None
    elastic_password: Optional[str] = None
    summary: bool = False
    run_contents: str = None


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
        measure_device_perf=args.device_perf,
        dry_run=args.dry_run,
        sweeps_tag=args.tag,
        skip_modules=args.skip_modules,
        skip_on_timeout=args.skip_on_timeout,
        summary=args.summary,
    )

    if config.vector_source == "elastic" or config.result_destination == "elastic":
        from framework.elastic_config import get_elastic_url

        config.elastic_connection_string = get_elastic_url("corp")

        # Acquire once
        config.elastic_username = os.getenv("ELASTIC_USERNAME")
        config.elastic_password = os.getenv("ELASTIC_PASSWORD")
        if not config.elastic_username or not config.elastic_password:
            raise ValueError("ELASTIC_USERNAME and ELASTIC_PASSWORD must be set in environment variables")

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
            args.file_path and args.vector_source in ["elastic", "vectors_export"],
            "File path should not be specified when test vector source is 'elastic' or 'vectors_export'.",
        ),
    ]

    # Check each validation rule
    for condition, error_message in validation_rules:
        if condition:
            parser.print_help()
            logger.error(error_message)
            exit(1)

    # Environment variable validation for elastic database
    if args.vector_source == "elastic" or args.result_dest == "elastic":
        elastic_username = os.getenv("ELASTIC_USERNAME")
        elastic_password = os.getenv("ELASTIC_PASSWORD")

        if not elastic_username or not elastic_password:
            logger.error("ELASTIC_USERNAME and ELASTIC_PASSWORD must be set in the environment variables.")
            exit(1)

    # Validate that skip modules is only used when running all modules
    if args.skip_modules and args.module_name:
        logger.error("Skip modules is only supported when running all modules.")
        exit(1)

    logger.info("All argument validations passed successfully.")


def get_all_modules():
    sweeps_path = pathlib.Path(__file__).parent / "sweeps"
    for file in sorted(sweeps_path.glob("**/*.py")):
        sweep_name = str(pathlib.Path(file).relative_to(sweeps_path))[:-3].replace("/", ".")
        yield sweep_name


def get_timeout(test_module):
    try:
        timeout = test_module.TIMEOUT
    except:
        timeout = 30
    return timeout


def sanitize_inputs(test_vectors):
    info_field_names = ["sweep_name", "suite_name", "vector_id", "input_hash"]
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


def gather_single_test_perf(device, test_passed):
    if device.get_num_devices() > 1:
        logger.error("Multi-device perf is not supported. Failing.")
        return None
    # Read profiler data from device
    ttnn.ReadDeviceProfiler(device)
    opPerfData = get_device_data_generate_report(
        PROFILER_LOGS_DIR, None, None, None, export_csv=False, cleanup_device_log=True
    )
    if not test_passed:
        return None
    elif opPerfData == []:
        logger.error("No profiling data available. Ensure you are running with the profiler build.")
        return None
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


def get_hostname():
    return subprocess.check_output(["uname", "-n"]).decode("ascii").strip()


def get_username():
    return os.environ["USER"]


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception as e:
        return "Couldn't get git hash!"


def get_git_author():
    """Get the git author name"""
    try:
        return subprocess.check_output(["git", "config", "user.name"]).decode("ascii").strip()
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


def get_github_pipeline_id() -> Optional[int]:
    """Get a CI pipeline identifier suitable for joining CICD metadata tables.

    Prefer GitHub Actions run id if present; otherwise fall back to generic CI_PIPELINE_ID.
    Returns an int when available, otherwise None.
    """
    run_id = os.getenv("GITHUB_RUN_ID") or os.getenv("CI_PIPELINE_ID")
    if not run_id:
        return None
    try:
        return int(run_id)
    except ValueError:
        # Unexpected non-integer; keep it unset for type consistency
        return None


def run(test_module, input_queue, output_queue, config: SweepsConfig):
    device_generator = get_devices(test_module)
    try:
        device, device_name = next(device_generator)
        logger.info(f"Opened device configuration, {device_name}.")
    except AssertionError as e:
        output_queue.put([False, "DEVICE EXCEPTION: " + str(e), None, None])
        return
    try:
        while True:
            test_vector = input_queue.get(block=True, timeout=1)
            test_vector = deserialize_vector_structured(test_vector)
            try:
                results = test_module.run(**test_vector, device=device)
                if type(results) == list:
                    status, message = results[0]
                    e2e_perf = results[1] / 1000000  # Nanoseconds to milliseconds
                else:
                    status, message = results
                    e2e_perf = None
            except Exception as e:
                status, message = False, str(e)
                e2e_perf = None
            if config.measure_device_perf:
                perf_result = gather_single_test_perf(device, status)
                message = get_updated_message(message, perf_result)
                output_queue.put([status, message, e2e_perf, perf_result])
            else:
                output_queue.put([status, message, e2e_perf, None])
    except Empty as e:
        try:
            # Run teardown in mesh_device_fixture
            next(device_generator)
        except StopIteration:
            logger.info(f"Closed device configuration, {device_name}.")


def execute_suite(test_module, test_vectors, pbar_manager, suite_name, module_name, header_info, config: SweepsConfig):
    results = []
    input_queue = Queue()
    output_queue = Queue()
    p = None
    timeout = get_timeout(test_module)
    suite_pbar = pbar_manager.counter(total=len(test_vectors), desc=f"Suite: {suite_name}", leave=False)
    arch = ttnn.get_arch_name()
    reset_util = tt_smi_util.ResetUtil(arch)

    if len(test_vectors) > 1 and not config.dry_run:
        p = Process(target=run, args=(test_module, input_queue, output_queue, config))
        p.start()

    for i, test_vector in enumerate(test_vectors):
        vector_id = header_info[i].get("vector_id", "N/A")
        logger.info(f"Executing test: Module='{module_name}', Suite='{suite_name}', Vector ID='{vector_id}'")
        if config.dry_run:
            logger.info(f"Would have executed test for vector {test_vector}")
            suite_pbar.update()
            continue
        result = dict()

        result["start_time_ts"] = dt.datetime.now()

        # Capture the original test vector data BEFORE any modifications
        original_vector_data = test_vector.copy()

        validity = deserialize(test_vector["validity"])

        if validity == VectorValidity.INVALID:
            result["status"] = TestStatus.NOT_RUN
            result["exception"] = "INVALID VECTOR: " + test_vector["invalid_reason"]
            result["e2e_perf"] = None
        else:
            test_vector.pop("invalid_reason")
            test_vector.pop("status")
            test_vector.pop("validity")

            try:
                if config.measure_perf:
                    # Run one time before capturing result to deal with compile-time slowdown of perf measurement
                    input_queue.put(test_vector)
                    if p is None:
                        logger.info(
                            "Executing test (first run, e2e perf is enabled) on parent process (to allow debugger support) because there is only one test vector. Hang detection is disabled."
                        )
                        run(test_module, input_queue, output_queue, config)
                    output_queue.get(block=True, timeout=timeout)
                input_queue.put(test_vector)
                if p is None:
                    logger.info(
                        "Executing test on parent process (to allow debugger support) because there is only one test vector. Hang detection is disabled."
                    )
                    run(test_module, input_queue, output_queue, config)
                response = output_queue.get(block=True, timeout=timeout)
                status, message, e2e_perf, device_perf = (
                    response[0],
                    response[1],
                    response[2],
                    response[3],
                )
                if status and config.measure_device_perf and device_perf is None:
                    result["status"] = TestStatus.FAIL_UNSUPPORTED_DEVICE_PERF
                    result["message"] = message
                elif status and config.measure_device_perf:
                    result["status"] = TestStatus.PASS
                    result["message"] = message
                    result["device_perf"] = device_perf
                elif status:
                    result["status"] = TestStatus.PASS
                    result["message"] = message
                else:
                    if "DEVICE EXCEPTION" in message:
                        logger.error(
                            "DEVICE EXCEPTION: Device could not be initialized. The following assertion was thrown: "
                            + message,
                        )
                        logger.info("Device error detected. The suite will be aborted after this test.")
                    if "Out of Memory: Not enough space to allocate" in message:
                        result["status"] = TestStatus.FAIL_L1_OUT_OF_MEM
                    elif "Watcher" in message:
                        result["status"] = TestStatus.FAIL_WATCHER
                    else:
                        result["status"] = TestStatus.FAIL_ASSERT_EXCEPTION
                    result["exception"] = message
                if e2e_perf and config.measure_perf:
                    result["e2e_perf"] = e2e_perf
                else:
                    result["e2e_perf"] = None
            except Empty as e:
                if p:
                    logger.warning(f"TEST TIMED OUT, Killing child process {p.pid} and running tt-smi...")
                    p.terminate()
                    p.join(PROCESS_TERMINATION_TIMEOUT_SECONDS)  # Wait for graceful process termination
                    if p.is_alive():
                        logger.error(f"Child process {p.pid} did not terminate, killing it.")
                        p.kill()
                        p.join()
                    p = None
                    reset_util.reset()

                result["status"], result["exception"] = TestStatus.FAIL_CRASH_HANG, "TEST TIMED OUT (CRASH / HANG)"
                result["e2e_perf"] = None
                result["original_vector_data"] = original_vector_data
                result["end_time_ts"] = dt.datetime.now()
                result["timestamp"] = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                result["host"] = get_hostname()
                result["user"] = get_username()
                suite_pbar.update()
                results.append(result)

                # Check if we should skip remaining tests in the suite
                if config.skip_on_timeout:
                    # Skip all remaining tests in the suite
                    logger.info("Skipping remaining tests in suite due to timeout.")
                    for j in range(i + 1, len(test_vectors)):
                        remaining_vector = test_vectors[j]
                        skipped_result = dict()
                        skipped_result["start_time_ts"] = dt.datetime.now()
                        skipped_result["original_vector_data"] = remaining_vector.copy()
                        skipped_result["status"] = TestStatus.NOT_RUN
                        skipped_result["exception"] = "SKIPPED DUE TO PREVIOUS TIMEOUT"
                        skipped_result["e2e_perf"] = None
                        skipped_result["end_time_ts"] = dt.datetime.now()
                        skipped_result["timestamp"] = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        skipped_result["host"] = get_hostname()
                        skipped_result["user"] = get_username()
                        results.append(skipped_result)
                        suite_pbar.update()

                    # Abort the suite
                    break
                else:
                    logger.info("Continuing with remaining tests in suite despite timeout.")
                    # Continue to the next test vector without breaking

        # Add the original test vector data to the result
        result["original_vector_data"] = original_vector_data

        result["end_time_ts"] = dt.datetime.now()
        result["timestamp"] = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result["host"] = get_hostname()
        result["user"] = get_username()

        suite_pbar.update()
        results.append(result)

        # Abort the suite if a fatal device error was encountered
        if "DEVICE EXCEPTION" in result.get("exception", ""):
            logger.error("Aborting test suite due to fatal device error.")
            if p and p.is_alive():
                p.terminate()
                p.join()
            break

    if p is not None:
        p.join()

    suite_pbar.close()
    return results


def run_sweeps(
    module_names,
    config: SweepsConfig,
):
    pbar_manager = enlighten.get_manager()

    # Set up vector source based on config
    source_kwargs = {}
    if config.vector_source == "elastic":
        source_kwargs = {
            "connection_string": config.elastic_connection_string,
            "username": config.elastic_username,
            "password": config.elastic_password,
            "tag": config.sweeps_tag,
        }
    elif config.vector_source == "file":
        source_kwargs = {
            "file_path": config.file_path,
        }
    vector_source = VectorSourceFactory.create_source(config.vector_source, **source_kwargs)

    # Set up result destination based on config
    result_kwargs = {}
    if config.result_destination == "elastic":
        result_kwargs = {
            "connection_string": config.elastic_connection_string,
            "username": config.elastic_username,
            "password": config.elastic_password,
        }
    elif config.result_destination == "results_export":
        # Optionally: pass a custom export_dir via config if you add it later
        # result_kwargs = {"export_dir": pathlib.Path("...")}
        pass

    result_dest = ResultDestinationFactory.create_destination(config.result_destination, **result_kwargs)

    # Initialize run metadata and run record
    run_id = None
    final_status = "success"

    if not config.dry_run:
        run_metadata = {
            "initiated_by": get_initiated_by(),
            "host": get_hostname(),
            "card_type": ttnn.get_arch_name(),
            "run_type": "sweeps",
            "run_contents": config.run_contents,
            "git_author": get_git_author(),
            "git_branch_name": get_git_branch(),
            "git_commit_sha": git_hash(),
            "github_pipeline_id": get_github_pipeline_id(),
            "run_start_ts": dt.datetime.now(),
            "status": "success",
        }
        run_id = result_dest.initialize_run(run_metadata)
        if run_id:
            logger.info(f"Initialized run with id: {run_id}")

    # Unified processing regardless of source
    # Summary counters
    total_vectors_run = 0  # total number of test cases (vectors)
    total_tests_run = 0  # total number of suites executed
    module_suite_test_count = {}  # module_name -> {suite_name: count}
    max_test_cases_module = None  # find the module with the most test cases
    max_test_cases_per_module = 0

    module_pbar = pbar_manager.counter(total=len(module_names), desc="Modules", leave=False)
    try:
        for module_name in module_names:
            test_module = importlib.import_module("sweeps." + module_name)
            suites = vector_source.get_available_suites(module_name)

            for suite in suites:
                suite_start_time = dt.datetime.now()

                vectors = vector_source.load_vectors(module_name, suite)
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
                header_info, test_vectors = sanitize_inputs(vectors)
                results = execute_suite(
                    test_module, test_vectors, pbar_manager, suite, module_name, header_info, config
                )

                suite_end_time = dt.datetime.now()
                logger.info(f"Completed tests for module {module_name}, suite {suite}.")

                # Export results
                if not config.dry_run and results:
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
                        logger.error(f"Failed to export results for {module_name}, suite {suite}: {e}")
                        final_status = "failure"
                        # continue with other suites

            module_pbar.update()
    except Exception as e:
        logger.error(f"Error during sweep execution: {e}")
        final_status = "failure"
        raise
    finally:
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
    os.environ.pop("TT_METAL_WATCHER")
    os.environ.pop("TT_METAL_WATCHER_APPEND")


def enable_profiler():
    logger.info("Enabling Device Profiler")
    os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
    os.environ["ENABLE_TRACY"] = "1"


def disable_profiler():
    logger.info("Disabling Device Profiler")
    os.environ.pop("TT_METAL_DEVICE_PROFILER")
    os.environ.pop("ENABLE_TRACY")


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
        choices=["elastic", "file", "vectors_export"],
        help="Test vector source. Available presets are ['elastic', 'file', 'vectors_export']",
    )

    parser.add_argument("--file-path", required=False, help="Read and execute test vectors from a specified file path.")

    parser.add_argument(
        "--vector-id", required=False, help="Specify vector id with a module name to run an individual test vector."
    )

    parser.add_argument(
        "--result-dest",
        required=True,
        choices=["elastic", "postgres", "results_export"],
        help="Specify test result destination. Available presets are ['elastic', 'postgres', 'results_export']",
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
        "--device-perf",
        required=False,
        action="store_true",
        help="Measure device perf using device profiler. REQUIRES PROFILER BUILD!",
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
        "--summary",
        action="store_true",
        required=False,
        help="Log a detailed execution or dry-run summary at the end of the run.",
    )

    args = parser.parse_args(sys.argv[1:])

    # Argument validation
    validate_arguments(args, parser)

    # Create sweeps config object
    config = create_config_from_args(args)

    # Import Elasticsearch if using elastic database
    if config.vector_source == "elastic" or config.result_destination == "elastic":
        from elasticsearch import Elasticsearch, NotFoundError
        from framework.elastic_config import *

    if config.watcher:
        enable_watcher()

    if config.measure_device_perf:
        enable_profiler()

    # Generate run contents description
    config.run_contents = get_run_contents(config)

    logger.info(
        f"Running current sweeps with tag: {config.sweeps_tag} using {config.vector_source} test vector source, outputting to {config.result_destination}."
    )

    if config.skip_on_timeout:
        logger.info("Timeout behavior: Skip remaining tests in suite when a test times out.")
    else:
        logger.info("Timeout behavior: Continue running remaining tests in suite when a test times out.")

    # Parse modules for running specific tests
    module_names = get_module_names(config)

    from ttnn import *
    from framework.serialize import *
    from framework.device_fixtures import default_device

    run_sweeps(
        module_names,
        config=config,
    )

    if config.watcher:
        disable_watcher()

    if config.measure_device_perf:
        disable_profiler()
