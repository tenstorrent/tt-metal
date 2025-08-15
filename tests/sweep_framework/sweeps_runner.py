# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import pathlib
import importlib
import datetime as dt
import os
import json
import enlighten
from tt_metal.tools.profiler.process_ops_logs import get_device_data_generate_report
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR
from multiprocessing import Process
from faster_fifo import Queue
from queue import Empty
import subprocess
from framework.statuses import TestStatus, VectorValidity, VectorStatus
import framework.tt_smi_util as tt_smi_util
from elasticsearch import Elasticsearch, NotFoundError
from framework.elastic_config import *
from framework.sweeps_logger import sweeps_logger as logger
from framework.database import (
    postgres_connection,
    initialize_postgres_database,
    push_run,
    update_run,
    push_test,
    generate_error_signature,
    map_test_status_to_run_status,
    get_postgres_config,
)
from sweep_utils.roofline_utils import get_updated_message

# psycopg2 import handling is now centralized in framework/database.py
# Constants
PROCESS_TERMINATION_TIMEOUT_SECONDS = 5  # Time to wait for graceful process termination


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception as e:
        return "Couldn't get git hash!"


def get_hostname():
    return subprocess.check_output(["uname", "-n"]).decode("ascii").strip()


def get_username():
    return os.environ["USER"]


def get_devices(test_module):
    try:
        return test_module.mesh_device_fixture()
    except:
        return default_device()


def gather_single_test_perf(device, test_passed):
    if device.get_num_devices() > 1:
        logger.error("Multi-device perf is not supported. Failing.")
        return None
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


def run(test_module, input_queue, output_queue):
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
            # Use appropriate deserialization based on database backend
            if DATABASE_BACKEND == "postgres":
                test_vector = deserialize_vector_for_postgres(test_vector)
            else:  # DATABASE_BACKEND == "elastic"
                test_vector = deserialize_vector(test_vector)
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
            if MEASURE_DEVICE_PERF:
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


def execute_suite(test_module, test_vectors, pbar_manager, suite_name, module_name, header_info):
    results = []
    input_queue = Queue()
    output_queue = Queue()
    p = None
    timeout = get_timeout(test_module)
    suite_pbar = pbar_manager.counter(total=len(test_vectors), desc=f"Suite: {suite_name}", leave=False)
    arch = ttnn.get_arch_name()
    reset_util = tt_smi_util.ResetUtil(arch)

    if len(test_vectors) > 1 and not DRY_RUN:
        p = Process(target=run, args=(test_module, input_queue, output_queue))
        p.start()

    for i, test_vector in enumerate(test_vectors):
        vector_id = header_info[i].get("vector_id", "N/A")
        logger.info(f"Executing test: Module='{module_name}', Suite='{suite_name}', Vector ID='{vector_id}'")
        if DRY_RUN:
            print(f"Would have executed test for vector {test_vector}")
            suite_pbar.update()
            continue
        result = dict()

        result["start_time_ts"] = dt.datetime.now()

        # Capture the original test vector data BEFORE any modifications
        original_vector_data = test_vector.copy()

        # Use appropriate deserialization based on database backend
        if DATABASE_BACKEND == "postgres":
            validity = deserialize_for_postgres(test_vector["validity"])
        else:  # DATABASE_BACKEND == "elastic"
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
                if MEASURE_PERF:
                    # Run one time before capturing result to deal with compile-time slowdown of perf measurement
                    input_queue.put(test_vector)
                    if p is None:
                        logger.info(
                            "Executing test (first run, e2e perf is enabled) on parent process (to allow debugger support) because there is only one test vector. Hang detection is disabled."
                        )
                        run(test_module, input_queue, output_queue)
                    output_queue.get(block=True, timeout=timeout)
                input_queue.put(test_vector)
                if p is None:
                    logger.info(
                        "Executing test on parent process (to allow debugger support) because there is only one test vector. Hang detection is disabled."
                    )
                    run(test_module, input_queue, output_queue)
                response = output_queue.get(block=True, timeout=timeout)
                status, message, e2e_perf, device_perf = (
                    response[0],
                    response[1],
                    response[2],
                    response[3],
                )
                if status and MEASURE_DEVICE_PERF and device_perf is None:
                    result["status"] = TestStatus.FAIL_UNSUPPORTED_DEVICE_PERF
                    result["message"] = message
                elif status and MEASURE_DEVICE_PERF:
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
                if e2e_perf and MEASURE_PERF:
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
                if SKIP_REMAINING_ON_TIMEOUT:
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


def get_suite_vectors(client, vector_index, suite):
    response = client.search(
        index=vector_index,
        query={
            "bool": {
                "must": [
                    {"match": {"status": str(VectorStatus.CURRENT)}},
                    {"match": {"suite_name.keyword": suite}},
                    {"match": {"tag.keyword": SWEEPS_TAG}},
                ]
            }
        },
        size=10000,
    )
    test_ids = [hit["_id"] for hit in response["hits"]["hits"]]
    test_vectors = [hit["_source"] for hit in response["hits"]["hits"]]
    for i in range(len(test_ids)):
        test_vectors[i]["vector_id"] = test_ids[i]
    header_info, test_vectors = sanitize_inputs(test_vectors)
    return header_info, test_vectors


def export_test_results_json(header_info, results):
    if len(results) == 0:
        return
    module_name = header_info[0]["sweep_name"]
    EXPORT_DIR_PATH = pathlib.Path(__file__).parent / "results_export"
    EXPORT_PATH = EXPORT_DIR_PATH / str(module_name + ".json")

    if not EXPORT_DIR_PATH.exists():
        EXPORT_DIR_PATH.mkdir()

    curr_git_hash = git_hash()
    for result in results:
        result["git_hash"] = curr_git_hash

    new_data = []

    for i in range(len(results)):
        result = header_info[i]
        for elem in results[i].keys():
            if elem == "device_perf":
                result[elem] = results[i][elem]
                continue
            result[elem] = serialize_for_postgres(results[i][elem])
        new_data.append(result)

    if EXPORT_PATH.exists():
        with open(EXPORT_PATH, "r") as file:
            old_data = json.load(file)
        new_data = old_data + new_data
        with open(EXPORT_PATH, "w") as file:
            json.dump(new_data, file, indent=2)
    else:
        with open(EXPORT_PATH, "w") as file:
            json.dump(new_data, file, indent=2)


def find_vector_files_for_modules(module_names):
    """Find vector files for specified modules in the vectors_export directory"""
    vectors_export_dir = pathlib.Path(__file__).parent / "vectors_export"

    if not vectors_export_dir.exists():
        logger.error(f"Vectors export directory not found: {vectors_export_dir}")
        return {}

    module_files = {}
    for module_name in module_names:
        # Look for JSON files that match the module name pattern
        # Module name format: "category.subcategory.test_name"
        # File name format: "category.subcategory.test_name.json"
        potential_files = list(vectors_export_dir.glob(f"{module_name}.json"))
        if potential_files:
            module_files[module_name] = potential_files[0]
            logger.info(f"Found vector file for module '{module_name}': {potential_files[0]}")
        else:
            logger.warning(f"No vector file found for module '{module_name}' in {vectors_export_dir}")
            # Try to find similar files for debugging
            similar_files = list(vectors_export_dir.glob(f"*{module_name.split('.')[-1]}*.json"))
            if similar_files:
                logger.info(f"Similar files found: {[f.name for f in similar_files[:5]]}")

    return module_files


# Database wrapper functions that use the common database module with sweep-specific parameters
def create_sweep_run(
    initiated_by,
    host,
    git_author,
    git_branch_name,
    git_commit_hash,
    start_time_ts,
    status,
    run_contents=None,
    device=None,
):
    """Create a new sweep run record in the database."""
    return push_run(
        initiated_by=initiated_by,
        host=host,
        git_author=git_author,
        git_branch_name=git_branch_name,
        git_commit_hash=git_commit_hash,
        start_time_ts=start_time_ts,
        status=status,
        run_contents=run_contents,
        device=device,
        run_type="sweep",
        env=POSTGRES_ENV,
    )


def push_test(run_id, header_info, test_results, test_start_time, test_end_time):
    """Push test result to PostgreSQL database"""
    if not test_results:
        logger.info("No test results to push to PostgreSQL database.")
        return "success"

    try:
        with postgres_connection(POSTGRES_ENV) as (conn, cursor):
            sweep_name = header_info[0]["sweep_name"]

            test_insert_query = """
            INSERT INTO tests (run_id, name, start_time_ts, end_time_ts, status)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """
            cursor.execute(test_insert_query, (run_id, sweep_name, test_start_time, test_end_time, "success"))
            test_id = cursor.fetchone()[0]

            # Insert test cases in batch
            test_statuses = []
            testcase_insert_query = """
            INSERT INTO sweep_testcases (
                test_id, name, start_time_ts, end_time_ts,
                status, suite_name, test_vector, message, exception,
                e2e_perf, device_perf, error_signature
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """

            # Prepare all testcase values for batch insert
            batch_values = []
            for i, result in enumerate(test_results):
                # Map test status to db status
                db_status = map_test_status_to_db_status(result.get("status", None))
                test_statuses.append(db_status)
                testcase_name = f"{sweep_name}_{header_info[i].get('vector_id', 'unknown')}"
                exception_text = result.get("exception", None)
                error_sig = generate_error_signature(exception_text)
                # Create testcase record
                testcase_values = (
                    test_id,
                    testcase_name,
                    result.get("start_time_ts", None),
                    result.get("end_time_ts", None),
                    db_status,
                    header_info[i].get("suite_name", None),
                    json.dumps(result.get("original_vector_data", None)),
                    result.get("message", None),
                    exception_text,
                    result.get("e2e_perf", None),
                    json.dumps(result.get("device_perf")) if result.get("device_perf") else None,
                    error_sig,
                )
                batch_values.append(testcase_values)

            # Execute batch insert
            if batch_values:
                cursor.executemany(testcase_insert_query, batch_values)
                logger.info(
                    f"Successfully pushed {len(batch_values)} testcase results to PostgreSQL database for test {test_id}"
                )

            # Update test status based on testcase results
            test_status = map_test_status_to_run_status(test_statuses)
            test_update_query = """
            UPDATE tests
            SET status = %s
            WHERE id = %s
            """
            cursor.execute(test_update_query, (test_status, test_id))

            return test_status
    except Exception as e:
        logger.error(f"Failed to push test result to PostgreSQL database: {e}")
        raise


def push_failed_test(run_id, name, test_start_time, test_end_time, status):
    """Push failed test result to PostgreSQL database"""
    try:
        with postgres_connection(POSTGRES_ENV) as (conn, cursor):
            test_insert_query = """
            INSERT INTO tests (run_id, name, start_time_ts, end_time_ts, status)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """
            cursor.execute(test_insert_query, (run_id, name, test_start_time, test_end_time, status))
            test_id = cursor.fetchone()[0]
            return test_id
    except Exception as e:
        logger.error(f"Failed to push failed test result to PostgreSQL database: {e}")
        raise


def run_multiple_modules_json(module_names, suite_name, run_contents=None, vector_id=None):
    """Run multiple modules using JSON files from vectors_export directory"""
    pbar_manager = enlighten.get_manager()

    # Find vector files for each module
    module_files = find_vector_files_for_modules(module_names)

    if not module_files:
        logger.error("No vector files found for any of the specified modules")
        return

    total_vectors_run = 0
    total_tests_run = 0  # Track total tests (module-suite combinations)

    # Track detailed breakdown
    module_suite_breakdown = {}  # module_name -> {suite_name: test_count}

    # Initialize database if needed
    if not DRY_RUN:
        initialize_postgres_database(POSTGRES_ENV)

        initiated_by = get_initiated_by()
        host = get_hostname()
        git_author = get_git_author()
        git_branch_name = get_git_branch()
        git_commit_hash = git_hash()
        status = "success"
        run_start_time = dt.datetime.now()
        device = ttnn.get_arch_name()
        run_id = create_sweep_run(
            initiated_by,
            host,
            git_author,
            git_branch_name,
            git_commit_hash,
            run_start_time,
            status,
            run_contents,
            device,
        )
    else:
        run_id = None
        status = "success"

    should_continue = True

    # Process each module
    for module_name, vector_file in module_files.items():
        if not should_continue:
            break

        logger.info(f"Processing module: {module_name} from file: {vector_file}")

        try:
            with open(vector_file, "r") as file:
                data = json.load(file)

            suites_to_process = []
            if vector_id:
                found_vector = False
                for suite_key, suite_content in data.items():
                    if vector_id in suite_content:
                        vector = suite_content[vector_id]
                        vector["vector_id"] = vector_id
                        suites_to_process.append((suite_key, [vector]))
                        found_vector = True
                        break  # Found the vector, no need to check other suites
                if not found_vector:
                    logger.warning(f"Vector ID '{vector_id}' not found in module '{module_name}'. Skipping.")
                    continue
            else:
                for suite_key, suite_content in data.items():
                    if suite_name and suite_name != suite_key:
                        continue  # user only wants to run a specific suite

                    for input_hash in suite_content:
                        suite_content[input_hash]["vector_id"] = input_hash
                    vectors = list(suite_content.values())
                    suites_to_process.append((suite_key, vectors))

            for suite, vectors in suites_to_process:
                if not should_continue:
                    break

                # Track totals for both dry run and actual run
                total_vectors_run += len(vectors)
                total_tests_run += 1
                module_suite_breakdown[module_name] = module_suite_breakdown.get(module_name, {})
                module_suite_breakdown[module_name][suite] = module_suite_breakdown[module_name].get(suite, 0) + len(
                    vectors
                )

                if DRY_RUN:
                    continue  # Skip actual execution in dry run

                # Verify the module name matches
                if vectors and vectors[0]["sweep_name"] != module_name:
                    logger.warning(f"Module name mismatch: expected {module_name}, got {vectors[0]['sweep_name']}")
                    continue
                test_start_time = dt.datetime.now()
                # Import and run the test module
                try:
                    test_module = importlib.import_module("sweeps." + module_name)
                    header_info, test_vectors = sanitize_inputs(vectors)
                    logger.info(f"Executing tests for module {module_name}, suite {suite}")
                    results = execute_suite(test_module, test_vectors, pbar_manager, suite, module_name, header_info)
                    test_end_time = dt.datetime.now()
                    logger.info(f"Completed tests for module {module_name}, suite {suite}.")
                    logger.info(f"Tests Executed - {len(results)}")

                    try:
                        test_status = push_test(run_id, header_info, results, test_start_time, test_end_time)
                        if test_status == "failure":
                            status = "failure"
                    except Exception as e:
                        logger.error("Stopping execution due to database error")
                        should_continue = False
                        break

                except ImportError as e:
                    logger.error(f"Failed to import module {module_name}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Failed to execute module {module_name}: {e}")
                    if not DRY_RUN:
                        test_end_time = dt.datetime.now()
                        push_failed_test(run_id, module_name, test_start_time, test_end_time, "failure")
                    continue
        except FileNotFoundError:
            logger.error(f"Vector file not found: {vector_file}")
            continue
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {vector_file}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error processing file {vector_file}: {e}")
            continue

    if not DRY_RUN:
        run_end_time = dt.datetime.now()
        update_run(run_id, run_end_time, status, POSTGRES_ENV)
        logger.info(f"Run status: {status}")

        # Display execution summary
        logger.info("=== EXECUTION SUMMARY ===")
        logger.info(f"Total tests (module-suite combinations) executed: {total_tests_run}")
        logger.info(f"Total test cases (vectors) executed: {total_vectors_run}")

        # Show detailed breakdown by module and suite
        if module_suite_breakdown:
            logger.info("\n=== DETAILED BREAKDOWN BY MODULE AND SUITE ===")
            for module_name in sorted(module_suite_breakdown.keys()):
                module_total = 0
                for count in module_suite_breakdown[module_name].values():
                    module_total += count
                logger.info(f"Module: {module_name} (Total: {module_total} test cases)")
                for suite_name in sorted(module_suite_breakdown[module_name].keys()):
                    test_count = module_suite_breakdown[module_name][suite_name]
                    logger.info(f"  └─ Suite: {suite_name} ({test_count} test cases)")
    else:
        logger.info("--- DRY RUN SUMMARY ---")
        logger.info(f"Total tests (modules) that would have been run: {len(module_files)}")
        logger.info(f"Total test cases (vectors) that would have been run: {total_vectors_run}")

        # Show detailed breakdown by module and suite
        logger.info("\n=== DETAILED BREAKDOWN BY MODULE AND SUITE ===")
        for module_name in sorted(module_suite_breakdown.keys()):
            module_total = 0
            for count in module_suite_breakdown[module_name].values():
                module_total += count
            logger.info(f"Module: {module_name} (Total: {module_total} test cases)")
            for suite_name in sorted(module_suite_breakdown[module_name].keys()):
                test_count = module_suite_breakdown[module_name][suite_name]
                logger.info(f"  └─ Suite: {suite_name} ({test_count} test cases)")

        # Find the module with the maximum number of test cases
        max_test_cases_per_module = 0
        max_test_cases_module = None

        for module_name, vector_file in module_files.items():
            try:
                with open(vector_file, "r") as file:
                    data = json.load(file)

                module_test_cases = 0
                for suite_key, suite_content in data.items():
                    module_test_cases += len(suite_content)

                if module_test_cases > max_test_cases_per_module:
                    max_test_cases_per_module = module_test_cases
                    max_test_cases_module = module_name

            except Exception as e:
                logger.warning(f"Could not analyze module {module_name} for max test cases: {e}")

        if max_test_cases_module:
            logger.info(f"\nMaximum test cases per module: {max_test_cases_per_module} (in {max_test_cases_module})")


def run_sweeps_json(module_names, suite_name, run_contents=None, vector_id=None):
    """Run sweeps from JSON files - supports single or multiple modules"""
    if isinstance(module_names, str):
        # this path is when we are exporting a single module
        pbar_manager = enlighten.get_manager()
        with open(READ_FILE, "r") as file:
            print(READ_FILE)
            data = json.load(file)

            suites_to_process = []
            if vector_id:
                found_vector = False
                for suite_key, suite_content in data.items():
                    if vector_id in suite_content:
                        vector = suite_content[vector_id]
                        vector["vector_id"] = vector_id
                        suites_to_process.append((suite_key, [vector]))
                        found_vector = True
                        break  # Found the vector, no need to check other suites
                if not found_vector:
                    logger.error(f"Vector ID '{vector_id}' not found in '{READ_FILE}'.")
                    return
            else:
                for suite_key, suite_content in data.items():
                    if suite_name and suite_name != suite_key:
                        continue  # user only wants to run a specific suite

                    for input_hash in suite_content:
                        suite_content[input_hash]["vector_id"] = input_hash
                    vectors = list(suite_content.values())
                    suites_to_process.append((suite_key, vectors))

            total_vectors_run = 0
            total_tests_run = 0  # Track total tests (suites)
            suite_breakdown = {}  # suite_name -> test_count
            for suite, vectors in suites_to_process:
                # Track totals for both dry run and actual run
                total_vectors_run += len(vectors)
                total_tests_run += 1
                suite_breakdown[suite] = len(vectors)

                if DRY_RUN:
                    continue  # Skip actual execution in dry run

                module_name = vectors[0]["sweep_name"]
                test_module = importlib.import_module("sweeps." + module_name)
                header_info, test_vectors = sanitize_inputs(vectors)
                logger.info(f"Executing tests for module {module_name}, suite {suite}")
                results = execute_suite(test_module, test_vectors, pbar_manager, suite, module_name, header_info)
                logger.info(f"Completed tests for module {module_name}, suite {suite}.")
                logger.info(f"Tests Executed - {len(results)}")
                if DATABASE_BACKEND == "postgres":
                    logger.info("Dumping results to PostgreSQL database.")
                    export_test_results_postgres(
                        header_info, results, dt.datetime.now(), dt.datetime.now(), run_contents
                    )
                else:
                    logger.info("Dumping results to JSON file.")
                    export_test_results_json(header_info, results)
            # Display summary
            if DRY_RUN:
                logger.info("--- DRY RUN SUMMARY ---")
                logger.info(f"Total tests (suites) that would have been run: {total_tests_run}")
                logger.info(f"Total test cases (vectors) that would have been run: {total_vectors_run}")

                # Show detailed breakdown by suite
                logger.info(f"\n=== DETAILED BREAKDOWN BY MODULE AND SUITE ===")
                logger.info(f"Module: {module_names} (Total: {total_vectors_run} test cases)")
                for suite_name in sorted(suite_breakdown.keys()):
                    test_count = suite_breakdown[suite_name]
                    logger.info(f"  └─ Suite: {suite_name} ({test_count} test cases)")

                logger.info(f"\nMaximum test cases per module: {total_vectors_run} (in {module_names})")
            else:
                logger.info("=== EXECUTION SUMMARY ===")
                logger.info(f"Total tests (suites) executed: {total_tests_run}")
                logger.info(f"Total test cases (vectors) executed: {total_vectors_run}")

                # Show detailed breakdown by suite
                if suite_breakdown:
                    logger.info(f"\n=== DETAILED BREAKDOWN BY MODULE AND SUITE ===")
                    logger.info(f"Module: {module_names} (Total: {total_vectors_run} test cases)")
                    for suite_name in sorted(suite_breakdown.keys()):
                        test_count = suite_breakdown[suite_name]
                        logger.info(f"  └─ Suite: {suite_name} ({test_count} test cases)")
    else:
        # Multiple modules
        run_multiple_modules_json(module_names, suite_name, run_contents, vector_id=vector_id)


def run_sweeps(module_name, suite_name, vector_id, skip_modules=None, run_contents=None):
    try:
        client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))
    except Exception as e:
        logger.error(f"Error connecting to Elasticsearch: {e}")
        exit(1)
    pbar_manager = enlighten.get_manager()

    sweeps_path = pathlib.Path(__file__).parent / "sweeps"
    total_modules_run = 0
    total_vectors_run = 0

    # Track maximum test cases per module for dry run
    max_test_cases_per_module = 0
    max_test_cases_module = None

    # Track detailed breakdown for dry run
    module_suite_breakdown = {}  # module_name -> {suite_name: test_count}

    if not module_name:
        all_modules = sorted(sweeps_path.glob("**/*.py"))
        if DRY_RUN:
            logger.info(f"DRY RUN: Found {len(all_modules)} modules to process.")
        for file in all_modules:
            sweep_name = str(pathlib.Path(file).relative_to(sweeps_path))[:-3].replace("/", ".")
            if skip_modules and sweep_name in skip_modules:
                logger.info(f"Skipping module {sweep_name} due to --skip-modules flag.")
                continue
            total_modules_run += 1
            test_module = importlib.import_module("sweeps." + sweep_name)
            vector_index = VECTOR_INDEX_PREFIX + sweep_name
            logger.info(f"Executing tests for module {sweep_name}...")
            try:
                if not suite_name:
                    response = client.search(
                        index=vector_index,
                        query={"match": {"tag.keyword": SWEEPS_TAG}},
                        aggregations={"suites": {"terms": {"field": "suite_name.keyword", "size": 10000}}},
                    )
                    suites = [suite["key"] for suite in response["aggregations"]["suites"]["buckets"]]
                else:
                    response = client.search(
                        index=vector_index,
                        query={
                            "bool": {
                                "must": [
                                    {"match": {"tag.keyword": SWEEPS_TAG}},
                                    {"match": {"suite_name.keyword": suite_name}},
                                ]
                            }
                        },
                        aggregations={"suites": {"terms": {"field": "suite_name.keyword", "size": 10000}}},
                    )
                    suites = [suite["key"] for suite in response["aggregations"]["suites"]["buckets"]]
                if len(suites) == 0:
                    if not suite_name:
                        logger.info(
                            f"No suites found for module {sweep_name}, with tag {SWEEPS_TAG}. If you meant to run the CI suites of tests, use '--tag ci-main' in your test command, otherwise, run the parameter generator with your own tag and try again. Continuing..."
                        )
                    else:
                        logger.info(
                            f"No suite named {suite_name} found for module {sweep_name}, with tag {SWEEPS_TAG}. If you meant to run the CI suite of tests, use '--tag ci-main' in your test command, otherwise, run the parameter generator with your own tag and try again. Continuing..."
                        )
                    continue

                module_pbar = pbar_manager.counter(total=len(suites), desc=f"Module: {sweep_name}", leave=False)
                module_test_cases = 0  # Track test cases for this module
                module_suite_breakdown[sweep_name] = {}  # Initialize suite breakdown for this module
                for suite in suites:
                    logger.info(f"Executing tests for module {sweep_name}, suite {suite}.")
                    header_info, test_vectors = get_suite_vectors(client, vector_index, suite)
                    total_vectors_run += len(test_vectors)
                    module_test_cases += len(test_vectors)  # Add to module count
                    module_suite_breakdown[sweep_name][suite] = len(test_vectors)  # Track suite breakdown
                    results = execute_suite(test_module, test_vectors, pbar_manager, suite, sweep_name, header_info)
                    logger.info(f"Completed tests for module {sweep_name}, suite {suite}.")
                    if not DRY_RUN:
                        logger.info(f"Tests Executed - {len(results)}")
                        export_test_results(header_info, results)
                    module_pbar.update()
                module_pbar.close()

                # Update maximum test cases tracking
                if module_test_cases > max_test_cases_per_module:
                    max_test_cases_per_module = module_test_cases
                    max_test_cases_module = sweep_name
            except NotFoundError as e:
                logger.info(f"No test vectors found for module {sweep_name}. Skipping...")
                continue
            except Exception as e:
                logger.error(e)
                continue

    else:
        total_modules_run = 1
        try:
            test_module = importlib.import_module("sweeps." + module_name)
        except ModuleNotFoundError as e:
            logger.error(f"No module found with name {module_name}")
            exit(1)
        vector_index = VECTOR_INDEX_PREFIX + module_name

        if vector_id:
            test_vector = client.get(index=vector_index, id=vector_id)["_source"]
            test_vector["vector_id"] = vector_id
            header_info, test_vectors = sanitize_inputs([test_vector])
            total_vectors_run += len(test_vectors)
            # Update maximum test cases tracking for single vector
            if len(test_vectors) > max_test_cases_per_module:
                max_test_cases_per_module = len(test_vectors)
                max_test_cases_module = module_name
            # Track suite breakdown for single vector
            module_suite_breakdown[module_name] = {"Single Vector": len(test_vectors)}
            results = execute_suite(test_module, test_vectors, pbar_manager, "Single Vector", module_name, header_info)
            if not DRY_RUN:
                export_test_results(header_info, results)
        else:
            try:
                if not suite_name:
                    response = client.search(
                        index=vector_index,
                        query={"match": {"tag.keyword": SWEEPS_TAG}},
                        aggregations={"suites": {"terms": {"field": "suite_name.keyword", "size": 10000}}},
                        size=10000,
                    )
                    suites = [suite["key"] for suite in response["aggregations"]["suites"]["buckets"]]
                    if len(suites) == 0:
                        logger.info(
                            f"No suites found for module {module_name}, with tag {SWEEPS_TAG}. If you meant to run the CI suites of tests, use '--tag ci-main' in your test command, otherwise, run the parameter generator with your own tag and try again."
                        )
                        return

                    module_test_cases = 0  # Track test cases for this module
                    module_suite_breakdown[module_name] = {}  # Initialize suite breakdown for this module
                    for suite in suites:
                        logger.info(f"Executing tests for module {module_name}, suite {suite}.")
                        header_info, test_vectors = get_suite_vectors(client, vector_index, suite)
                        total_vectors_run += len(test_vectors)
                        module_test_cases += len(test_vectors)  # Add to module count
                        module_suite_breakdown[module_name][suite] = len(test_vectors)  # Track suite breakdown
                        results = execute_suite(
                            test_module, test_vectors, pbar_manager, suite, module_name, header_info
                        )
                        logger.info(f"Completed tests for module {module_name}, suite {suite}.")
                        if not DRY_RUN:
                            logger.info(f"Tests Executed - {len(results)}")
                            export_test_results(header_info, results)
                    # Update maximum test cases tracking
                    if module_test_cases > max_test_cases_per_module:
                        max_test_cases_per_module = module_test_cases
                        max_test_cases_module = module_name
                else:
                    logger.info(f"Executing tests for module {module_name}, suite {suite_name}.")
                    header_info, test_vectors = get_suite_vectors(client, vector_index, suite_name)
                    total_vectors_run += len(test_vectors)
                    # Update maximum test cases tracking for single suite
                    if len(test_vectors) > max_test_cases_per_module:
                        max_test_cases_per_module = len(test_vectors)
                        max_test_cases_module = module_name
                    # Track suite breakdown for single suite
                    module_suite_breakdown[module_name] = {suite_name: len(test_vectors)}
                    results = execute_suite(
                        test_module, test_vectors, pbar_manager, suite_name, module_name, header_info
                    )
                    logger.info(f"Completed tests for module {module_name}, suite {suite_name}.")
                    if not DRY_RUN:
                        logger.info(f"Tests Executed - {len(results)}")
                        export_test_results(header_info, results)
            except Exception as e:
                logger.info(e)
    # Display summary
    if DRY_RUN:
        logger.info("--- DRY RUN SUMMARY ---")
        logger.info(f"Total tests (modules) that would have been run: {total_modules_run}")
        logger.info(f"Total test cases (vectors) that would have been run: {total_vectors_run}")

        # Show detailed breakdown by module and suite
        if module_suite_breakdown:
            logger.info("\n=== DETAILED BREAKDOWN BY MODULE AND SUITE ===")
            for module_name in sorted(module_suite_breakdown.keys()):
                module_total = 0
                for count in module_suite_breakdown[module_name].values():
                    module_total += count
                logger.info(f"Module: {module_name} (Total: {module_total} test cases)")
                for suite_name in sorted(module_suite_breakdown[module_name].keys()):
                    test_count = module_suite_breakdown[module_name][suite_name]
                    logger.info(f"  └─ Suite: {suite_name} ({test_count} test cases)")

        if max_test_cases_module:
            logger.info(f"\nMaximum test cases per module: {max_test_cases_per_module} (in {max_test_cases_module})")
    else:
        logger.info("=== EXECUTION SUMMARY ===")
        logger.info(f"Total tests (modules) executed: {total_modules_run}")
        logger.info(f"Total test cases (vectors) executed: {total_vectors_run}")

        # Show detailed breakdown by module and suite
        if module_suite_breakdown:
            logger.info("\n=== DETAILED BREAKDOWN BY MODULE AND SUITE ===")
            for module_name in sorted(module_suite_breakdown.keys()):
                module_total = 0
                for count in module_suite_breakdown[module_name].values():
                    module_total += count
                logger.info(f"Module: {module_name} (Total: {module_total} test cases)")
                for suite_name in sorted(module_suite_breakdown[module_name].keys()):
                    test_count = module_suite_breakdown[module_name][suite_name]
                    logger.info(f"  └─ Suite: {suite_name} ({test_count} test cases)")
    client.close()


# Export test output (msg), status, exception (if applicable), git hash, timestamp, test vector, test UUID?,
def export_test_results(header_info, results):
    if len(results) == 0:
        return
    client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))
    sweep_name = header_info[0]["sweep_name"]
    results_index = RESULT_INDEX_PREFIX + sweep_name

    curr_git_hash = git_hash()
    for result in results:
        result["git_hash"] = curr_git_hash

    for i in range(len(results)):
        result = header_info[i]
        for elem in results[i].keys():
            if elem == "device_perf":
                result[elem] = results[i][elem]
                continue
            # Skip problematic fields that were added for PostgreSQL functionality
            if elem in ["start_time_ts", "end_time_ts", "original_vector_data"]:
                continue
            result[elem] = serialize(results[i][elem])
        client.index(index=results_index, body=result)
    logger.info(f"Successfully exported {len(results)} results to Elasticsearch")
    client.close()


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


def map_test_status_to_db_status(test_status):
    """Map TestStatus enum to database status string"""
    status_mapping = {
        TestStatus.PASS: "pass",
        TestStatus.FAIL_ASSERT_EXCEPTION: "fail_assert_exception",
        TestStatus.FAIL_L1_OUT_OF_MEM: "fail_l1_out_of_mem",
        TestStatus.FAIL_WATCHER: "fail_watcher",
        TestStatus.FAIL_CRASH_HANG: "fail_crash_hang",
        TestStatus.FAIL_UNSUPPORTED_DEVICE_PERF: "fail_unsupported_device_perf",
        TestStatus.NOT_RUN: "skipped",
    }
    return status_mapping.get(test_status, "error")


def export_test_results_postgres(header_info, results, run_start_time, run_end_time, run_contents=None):
    """Export test results to PostgreSQL database"""
    if len(results) == 0:
        return

    # Initialize database if needed
    initialize_postgres_database()

    try:
        with postgres_connection(POSTGRES_ENV) as (conn, cursor):
            # Get git information
            curr_git_hash = git_hash()
            git_author = get_git_author()
            git_branch = get_git_branch()
            initiated_by = get_initiated_by()
            host = get_hostname()
            device = ttnn.get_arch_name()

            # Create a new run record
            run_insert_query = """
            INSERT INTO runs (
                initiated_by, host, git_author, git_branch_name, git_commit_hash,
                start_time_ts, end_time_ts, status, run_contents, device, type
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """

            run_values = (
                initiated_by,
                host,
                git_author,
                git_branch,
                curr_git_hash,
                run_start_time,
                run_end_time,
                "success",  # Will be updated after processing all results
                run_contents,
                device,
                "sweep",
            )

            cursor.execute(run_insert_query, run_values)
            run_id = cursor.fetchone()[0]

            # Group results by test module (sweep_name)
            test_groups = {}
            for i, result in enumerate(results):
                sweep_name = header_info[i]["sweep_name"]
                if sweep_name not in test_groups:
                    test_groups[sweep_name] = []
                test_groups[sweep_name].append((i, result))

            all_test_statuses = []
            test_index = 0  # Track which test we're processing

            # Process each test module
            for sweep_name, test_results_group in test_groups.items():
                # Use the corresponding test start/end time from the lists
                test_start_time = (
                    min(r.get("start_time_ts") for _, r in test_results_group)
                    if test_results_group
                    else dt.datetime.now()
                )
                test_end_time = (
                    max(r.get("end_time_ts") for _, r in test_results_group)
                    if test_results_group
                    else dt.datetime.now()
                )

                test_insert_query = """
                INSERT INTO tests (
                    run_id, name, start_time_ts, end_time_ts, status
                ) VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """

                test_values = (
                    run_id,
                    sweep_name,
                    test_start_time,
                    test_end_time,
                    "success",  # Will be updated after processing all testcases
                )

                cursor.execute(test_insert_query, test_values)
                test_id = cursor.fetchone()[0]

                test_statuses = []
                testcase_insert_query = """
                INSERT INTO sweep_testcases (
                    test_id, name, start_time_ts, end_time_ts,
                    status, suite_name, test_vector, message, exception,
                    e2e_perf, device_perf, error_signature
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """

                # Prepare all testcase values for batch insert
                batch_values = []
                for idx, result in test_results_group:
                    header = header_info[idx]

                    # Map test status to database status
                    db_status = map_test_status_to_db_status(result.get("status"))
                    test_statuses.append(db_status)

                    exception_text = result.get("exception", None)
                    error_sig = generate_error_signature(exception_text)

                    testcase_values = (
                        test_id,
                        f"{sweep_name}_{header.get('vector_id', 'unknown')}",
                        result.get("start_time_ts"),
                        result.get("end_time_ts"),
                        db_status,
                        header.get("suite_name"),
                        json.dumps(result.get("original_vector_data")),
                        result.get("message"),
                        exception_text,
                        result.get("e2e_perf"),
                        json.dumps(result.get("device_perf")) if result.get("device_perf") else None,
                        error_sig,
                    )
                    batch_values.append(testcase_values)

                # Execute batch insert
                if batch_values:
                    cursor.executemany(testcase_insert_query, batch_values)

                # Update test status based on testcase results
                test_status = map_test_status_to_run_status(test_statuses)

                test_update_query = """
                UPDATE tests SET status = %s WHERE id = %s
                """
                cursor.execute(test_update_query, (test_status, test_id))

                all_test_statuses.append(test_status)
                test_index += 1  # Move to next test

            # Update run status based on all test results
            run_status = map_test_status_to_run_status(all_test_statuses)

            run_update_query = """
            UPDATE runs SET status = %s WHERE id = %s
            """
            cursor.execute(run_update_query, (run_status, run_id))

            logger.info(f"Successfully exported {len(results)} results to PostgreSQL with run_id: {run_id}")

    except Exception as e:
        logger.error(f"Failed to export results to PostgreSQL: {e}")
        raise


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
        "--elastic",
        required=False,
        default="corp",
        help="Elastic Connection String for the vector and results database. Available presets are ['corp', 'cloud']",
    )
    parser.add_argument(
        "--module-name",
        required=False,
        help="Test Module Name(s). For PostgreSQL with local files, can be comma-separated list (e.g., 'eltwise.unary.relu.relu,matmul.short.matmul'). For Elasticsearch, single module name only.",
    )
    parser.add_argument("--suite-name", required=False, help="Suite of Test Vectors to run, or all tests if omitted.")
    parser.add_argument(
        "--vector-id", required=False, help="Specify vector id with a module name to run an individual test vector."
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
        "--read-file", required=False, help="Read and execute test vectors from a specified file path instead of ES."
    )
    parser.add_argument(
        "--database",
        required=False,
        default="elastic",
        choices=["elastic", "postgres"],
        help="Database backend for storing results. Available options: ['elastic', 'postgres']",
    )
    parser.add_argument(
        "--postgres-env",
        required=False,
        default="dev",
        choices=["dev", "prod"],
        help="PostgreSQL environment configuration. Available options: ['dev', 'prod']",
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

    args = parser.parse_args(sys.argv[1:])
    if args.module_name or args.suite_name:
        run_contents_details = []
        if args.module_name:
            run_contents_details.append(f"{args.module_name}")
        if args.suite_name:
            run_contents_details.append(f"{args.suite_name}")
        run_contents = ", ".join(run_contents_details)
    else:
        run_contents = "all_sweeps"

    if not args.module_name and args.vector_id:
        parser.print_help()
        logger.error("Module name is required if vector id is specified.")
        exit(1)

    global READ_FILE
    # Only import Elasticsearch if using elastic database and not reading from file
    if not args.read_file and args.database == "elastic":
        from elasticsearch import Elasticsearch, NotFoundError
        from framework.elastic_config import *

        global ELASTIC_CONNECTION_STRING
        ELASTIC_CONNECTION_STRING = get_elastic_url(args.elastic)
        READ_FILE = None
    elif args.read_file:
        if not args.module_name:
            logger.error("You must specify a module with a local file.")
            exit(1)
        READ_FILE = args.read_file
    else:
        # Using PostgreSQL database or other non-elastic backend
        READ_FILE = None

    global MEASURE_PERF
    MEASURE_PERF = args.perf

    global MEASURE_DEVICE_PERF
    MEASURE_DEVICE_PERF = args.device_perf

    global DRY_RUN
    DRY_RUN = args.dry_run

    global SWEEPS_TAG
    SWEEPS_TAG = args.tag

    global DATABASE_BACKEND
    DATABASE_BACKEND = args.database

    global POSTGRES_ENV
    POSTGRES_ENV = args.postgres_env

    global SKIP_REMAINING_ON_TIMEOUT
    SKIP_REMAINING_ON_TIMEOUT = args.skip_on_timeout

    logger.info(f"Running current sweeps with tag: {SWEEPS_TAG} using {DATABASE_BACKEND} backend.")

    if SKIP_REMAINING_ON_TIMEOUT:
        logger.info("Timeout behavior: Skip remaining tests in suite when a test times out.")
    else:
        logger.info("Timeout behavior: Continue running remaining tests in suite when a test times out.")

    if args.watcher:
        enable_watcher()

    if MEASURE_DEVICE_PERF:
        enable_profiler()

    from ttnn import *
    from framework.serialize import *
    from framework.device_fixtures import default_device
    from framework.sweeps_logger import sweeps_logger as logger

    # Parse module names
    if args.module_name:
        if DATABASE_BACKEND == "postgres" and not args.read_file:
            # For PostgreSQL without read-file, support comma-separated module names
            module_names = [name.strip() for name in args.module_name.split(",")]
            logger.info(f"Running multiple modules: {module_names}")
        else:
            # Use Elasticsearch or with read-file, use single module name
            module_names = args.module_name
    else:
        module_names = None

    skip_modules_set = set()
    if args.skip_modules:
        if args.module_name:
            logger.warning("--skip-modules is only supported when running all modules. Ignoring this flag.")
        else:
            skip_modules_set = {name.strip() for name in args.skip_modules.split(",")}
            logger.info(f"Skipping modules: {', '.join(skip_modules_set)}")

    # Determine which execution path to take
    use_json_runner = READ_FILE or DATABASE_BACKEND == "postgres"

    if use_json_runner:
        effective_module_names = module_names
        # For postgres, if no modules are specified, find all available modules
        if DATABASE_BACKEND == "postgres" and not module_names:
            all_module_names = list(get_all_modules())
            if skip_modules_set:
                effective_module_names = [name for name in all_module_names if name not in skip_modules_set]
            else:
                effective_module_names = all_module_names

            logger.info("Running modules:")
            for module_name in effective_module_names:
                logger.info(f"  {module_name}")
        run_sweeps_json(effective_module_names, args.suite_name, run_contents=run_contents, vector_id=args.vector_id)
    else:
        # Exporting results to Elasticsearch
        # check that ELASTIC_USERNAME and ELASTIC_PASSWORD are set
        if ELASTIC_USERNAME is None or ELASTIC_PASSWORD is None:
            logger.error("ELASTIC_USERNAME and ELASTIC_PASSWORD must be set in the environment variables.")
            exit(1)
        else:
            logger.info(f"Exporting results to Elasticsearch")
        run_sweeps(module_names, args.suite_name, args.vector_id, skip_modules_set, run_contents=run_contents)

    if args.watcher:
        disable_watcher()

    if MEASURE_DEVICE_PERF:
        disable_profiler()
