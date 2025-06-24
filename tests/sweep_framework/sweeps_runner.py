# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import pathlib
import importlib
import datetime as dt
import os
import json
import uuid
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
from sweep_utils.roofline_utils import get_updated_message
import psycopg2
from psycopg2.extras import RealDictCursor

ARCH = os.getenv("ARCH_NAME")


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
    ttnn.DumpDeviceProfiler(device)
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
        output_queue.put([False, "DEVICE EXCEPTION: " + str(e), None, None, None])
        return
    try:
        while True:
            test_vector = input_queue.get(block=True, timeout=1)
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
                output_queue.put([status, message, e2e_perf, perf_result, device_name])
            else:
                output_queue.put([status, message, e2e_perf, None, device_name])
    except Empty as e:
        try:
            # Run teardown in mesh_device_fixture
            next(device_generator)
        except StopIteration:
            logger.info(f"Closed device configuration, {device_name}.")


def get_timeout(test_module):
    try:
        timeout = test_module.TIMEOUT
    except:
        timeout = 30
    return timeout


def execute_suite(test_module, test_vectors, pbar_manager, suite_name):
    results = []
    input_queue = Queue()
    output_queue = Queue()
    p = None
    timeout = get_timeout(test_module)
    suite_pbar = pbar_manager.counter(total=len(test_vectors), desc=f"Suite: {suite_name}", leave=False)
    reset_util = tt_smi_util.ResetUtil(ARCH)
    for test_vector in test_vectors:
        if DRY_RUN:
            print(f"Would have executed test for vector {test_vector}")
            continue
        result = dict()

        # Capture the original test vector data BEFORE any modifications
        original_vector_data = test_vector.copy()

        if deserialize(test_vector["validity"]) == VectorValidity.INVALID:
            result["status"] = TestStatus.NOT_RUN
            result["exception"] = "INVALID VECTOR: " + test_vector["invalid_reason"]
            result["e2e_perf"] = None
        else:
            test_vector.pop("invalid_reason")
            test_vector.pop("status")
            test_vector.pop("validity")
            if p is None and len(test_vectors) > 1:
                p = Process(target=run, args=(test_module, input_queue, output_queue))
                p.start()
            try:
                if MEASURE_PERF:
                    # Run one time before capturing result to deal with compile-time slowdown of perf measurement
                    input_queue.put(test_vector)
                    if len(test_vectors) == 1:
                        logger.info(
                            "Executing test (first run, e2e perf is enabled) on parent process (to allow debugger support) because there is only one test vector. Hang detection is disabled."
                        )
                        run(test_module, input_queue, output_queue)
                    output_queue.get(block=True, timeout=timeout)
                input_queue.put(test_vector)
                if len(test_vectors) == 1:
                    logger.info(
                        "Executing test on parent process (to allow debugger support) because there is only one test vector. Hang detection is disabled."
                    )
                    run(test_module, input_queue, output_queue)
                response = output_queue.get(block=True, timeout=timeout)
                status, message, e2e_perf, device_perf, device_name = (
                    response[0],
                    response[1],
                    response[2],
                    response[3],
                    response[4],
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
                        logger.info("Skipping test suite because of device error, proceeding...")
                        return []
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
                result["device"] = device_name
            except Empty as e:
                logger.warning(f"TEST TIMED OUT, Killing child process {p.pid} and running tt-smi...")
                p.terminate()
                p = None
                reset_util.reset()
                result["status"], result["exception"] = TestStatus.FAIL_CRASH_HANG, "TEST TIMED OUT (CRASH / HANG)"
                result["e2e_perf"] = None

        # Add the original test vector data to the result
        result["original_vector_data"] = original_vector_data

        result["timestamp"] = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result["host"] = get_hostname()
        result["user"] = get_username()

        suite_pbar.update()
        results.append(result)

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
            header[field] = vector.pop(field)
        vector.pop("timestamp")
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
            result[elem] = serialize(results[i][elem])
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


def run_multiple_modules_json(module_names, suite_name):
    """Run multiple modules using JSON files from vectors_export directory"""
    pbar_manager = enlighten.get_manager()

    # Find vector files for each module
    module_files = find_vector_files_for_modules(module_names)

    if not module_files:
        logger.error("No vector files found for any of the specified modules")
        return

    all_results = []
    all_header_info = []

    # Process each module
    for module_name, vector_file in module_files.items():
        logger.info(f"Processing module: {module_name} from file: {vector_file}")

        try:
            with open(vector_file, "r") as file:
                data = json.load(file)

                for suite in data:
                    if suite_name and suite_name != suite:
                        continue  # user only wants to run a specific suite

                    # Prepare vectors for this suite
                    for input_hash in data[suite]:
                        data[suite][input_hash]["vector_id"] = input_hash
                    vectors = [data[suite][input_hash] for input_hash in data[suite]]

                    # Verify the module name matches
                    if vectors and vectors[0]["sweep_name"] != module_name:
                        logger.warning(f"Module name mismatch: expected {module_name}, got {vectors[0]['sweep_name']}")
                        continue

                    # Import and run the test module
                    try:
                        test_module = importlib.import_module("sweeps." + module_name)
                        header_info, test_vectors = sanitize_inputs(vectors)
                        logger.info(f"Executing tests for module {module_name}, suite {suite}")
                        results = execute_suite(test_module, test_vectors, pbar_manager, suite)
                        logger.info(f"Completed tests for module {module_name}, suite {suite}.")
                        logger.info(f"Tests Executed - {len(results)}")

                        # Collect results for batch export
                        all_results.extend(results)
                        all_header_info.extend(header_info)

                    except ImportError as e:
                        logger.error(f"Failed to import module {module_name}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Failed to execute module {module_name}: {e}")
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

    # Export all results to PostgreSQL in a single run
    if all_results and DATABASE_BACKEND == "postgres":
        logger.info("Dumping all results to PostgreSQL database.")
        export_test_results_postgres(all_header_info, all_results)
    elif all_results:
        logger.info("Dumping results to JSON file.")
        export_test_results_json(all_header_info, all_results)
    else:
        logger.warning("No results to export")


def run_sweeps_json(module_names, suite_name):
    """Run sweeps from JSON files - supports single or multiple modules"""
    if isinstance(module_names, str):
        # Single module
        pbar_manager = enlighten.get_manager()
        with open(READ_FILE, "r") as file:
            print(READ_FILE)
            data = json.load(file)
            for suite in data:
                if suite_name and suite_name != suite:
                    continue  # user only wants to run a specific suite

                for input_hash in data[suite]:
                    data[suite][input_hash]["vector_id"] = input_hash
                vectors = [data[suite][input_hash] for input_hash in data[suite]]
                module_name = vectors[0]["sweep_name"]
                test_module = importlib.import_module("sweeps." + module_name)
                header_info, test_vectors = sanitize_inputs(vectors)
                logger.info(f"Executing tests for module {module_name}, suite {suite}")
                results = execute_suite(test_module, test_vectors, pbar_manager, suite)
                logger.info(f"Completed tests for module {module_name}, suite {suite}.")
                logger.info(f"Tests Executed - {len(results)}")
                if DATABASE_BACKEND == "postgres":
                    logger.info("Dumping results to PostgreSQL database.")
                    export_test_results_postgres(header_info, results)
                else:
                    logger.info("Dumping results to JSON file.")
                    export_test_results_json(header_info, results)
    else:
        # Multiple modules
        run_multiple_modules_json(module_names, suite_name)


def run_sweeps(module_name, suite_name, vector_id):
    client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))
    pbar_manager = enlighten.get_manager()

    sweeps_path = pathlib.Path(__file__).parent / "sweeps"

    if not module_name:
        for file in sorted(sweeps_path.glob("**/*.py")):
            sweep_name = str(pathlib.Path(file).relative_to(sweeps_path))[:-3].replace("/", ".")
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
                for suite in suites:
                    logger.info(f"Executing tests for module {sweep_name}, suite {suite}.")
                    header_info, test_vectors = get_suite_vectors(client, vector_index, suite)
                    results = execute_suite(test_module, test_vectors, pbar_manager, suite)
                    logger.info(f"Completed tests for module {sweep_name}, suite {suite}.")
                    logger.info(f"Tests Executed - {len(results)}")
                    if DATABASE_BACKEND == "postgres":
                        export_test_results_postgres(header_info, results)
                    else:
                        export_test_results(header_info, results)
                    module_pbar.update()
                module_pbar.close()
            except NotFoundError as e:
                logger.info(f"No test vectors found for module {sweep_name}. Skipping...")
                continue
            except Exception as e:
                logger.error(e)
                continue

    else:
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
            results = execute_suite(test_module, test_vectors, pbar_manager, "Single Vector")
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

                    for suite in suites:
                        logger.info(f"Executing tests for module {module_name}, suite {suite}.")
                        header_info, test_vectors = get_suite_vectors(client, vector_index, suite)
                        results = execute_suite(test_module, test_vectors, pbar_manager, suite)
                        logger.info(f"Completed tests for module {module_name}, suite {suite}.")
                        logger.info(f"Tests Executed - {len(results)}")
                        if DATABASE_BACKEND == "postgres":
                            export_test_results_postgres(header_info, results)
                        else:
                            export_test_results(header_info, results)
                else:
                    logger.info(f"Executing tests for module {module_name}, suite {suite_name}.")
                    header_info, test_vectors = get_suite_vectors(client, vector_index, suite_name)
                    results = execute_suite(test_module, test_vectors, pbar_manager, suite_name)
                    logger.info(f"Completed tests for module {module_name}, suite {suite_name}.")
                    logger.info(f"Tests Executed - {len(results)}")
                    if DATABASE_BACKEND == "postgres":
                        export_test_results_postgres(header_info, results)
                    else:
                        export_test_results(header_info, results)
            except Exception as e:
                logger.info(e)

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
            result[elem] = serialize(results[i][elem])
        client.index(index=results_index, body=result)

    client.close()


def initialize_postgres_database():
    """Initialize PostgreSQL database with required tables"""
    pg_config = get_postgres_config(POSTGRES_ENV)

    try:
        conn = psycopg2.connect(**pg_config)
        cursor = conn.cursor()

        # Create the Run table
        create_run_table_query = """
        CREATE TABLE IF NOT EXISTS runs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            initiated_by VARCHAR(255) NOT NULL,
            git_author VARCHAR(255),
            git_branch_name VARCHAR(255),
            git_commit_hash VARCHAR(50),
            start_time_ts TIMESTAMP NOT NULL,
            end_time_ts TIMESTAMP,
            status VARCHAR(20) NOT NULL CHECK (status IN ('success', 'failure', 'error', 'cancelled')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        # Create the Test table
        create_test_table_query = """
        CREATE TABLE IF NOT EXISTS tests (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            start_time_ts TIMESTAMP NOT NULL,
            end_time_ts TIMESTAMP,
            status VARCHAR(20) NOT NULL CHECK (status IN ('success', 'failure', 'error', 'cancelled', 'skipped')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        # Create the Sweep Testcase table
        create_testcase_table_query = """
        CREATE TABLE IF NOT EXISTS sweep_testcases (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            test_id UUID NOT NULL REFERENCES tests(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            device VARCHAR(255),
            host VARCHAR(255) NOT NULL,
            start_time_ts TIMESTAMP NOT NULL,
            end_time_ts TIMESTAMP,
            status VARCHAR(100) NOT NULL,
            suite_name VARCHAR(255) NOT NULL,
            test_vector JSONB,
            message TEXT,
            exception TEXT,
            e2e_perf FLOAT,
            device_perf JSONB
        );
        """

        # Execute table creation queries
        cursor.execute(create_run_table_query)
        cursor.execute(create_test_table_query)
        cursor.execute(create_testcase_table_query)

        # Create indexes for better query performance
        create_indexes_query = """
        CREATE INDEX IF NOT EXISTS idx_runs_initiated_by ON runs(initiated_by);
        CREATE INDEX IF NOT EXISTS idx_runs_git_commit_hash ON runs(git_commit_hash);
        CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
        CREATE INDEX IF NOT EXISTS idx_runs_start_time_ts ON runs(start_time_ts);

        CREATE INDEX IF NOT EXISTS idx_tests_run_id ON tests(run_id);
        CREATE INDEX IF NOT EXISTS idx_tests_name ON tests(name);
        CREATE INDEX IF NOT EXISTS idx_tests_status ON tests(status);
        CREATE INDEX IF NOT EXISTS idx_tests_start_time_ts ON tests(start_time_ts);

        CREATE INDEX IF NOT EXISTS idx_sweep_testcases_test_id ON sweep_testcases(test_id);
        CREATE INDEX IF NOT EXISTS idx_sweep_testcases_name ON sweep_testcases(name);
        CREATE INDEX IF NOT EXISTS idx_sweep_testcases_suite_name ON sweep_testcases(suite_name);
        CREATE INDEX IF NOT EXISTS idx_sweep_testcases_status ON sweep_testcases(status);
        CREATE INDEX IF NOT EXISTS idx_sweep_testcases_host ON sweep_testcases(host);
        CREATE INDEX IF NOT EXISTS idx_sweep_testcases_start_time_ts ON sweep_testcases(start_time_ts);
        """

        cursor.execute(create_indexes_query)

        conn.commit()
        logger.info("Successfully initialized PostgreSQL database with runs, tests, and sweep_testcases tables")

    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL database: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


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


def map_test_status_to_run_status(test_statuses):
    """Map test statuses to overall run status"""
    if not test_statuses:
        return "error"

    # If any test failed, the run failed
    if any(status in ["failure", "error"] for status in test_statuses):
        return "failure"
    # If any test was cancelled, the run was cancelled
    elif any(status == "cancelled" for status in test_statuses):
        return "cancelled"
    # If all tests passed or were skipped, the run succeeded
    elif all(status in ["success", "skipped"] for status in test_statuses):
        return "success"
    else:
        return "error"


def map_test_status_to_db_status(test_status):
    """Map TestStatus enum to database status string"""
    status_mapping = {
        TestStatus.PASS: "success",
        TestStatus.FAIL_ASSERT_EXCEPTION: "failure",
        TestStatus.FAIL_L1_OUT_OF_MEM: "failure",
        TestStatus.FAIL_WATCHER: "failure",
        TestStatus.FAIL_CRASH_HANG: "failure",
        TestStatus.FAIL_UNSUPPORTED_DEVICE_PERF: "failure",
        TestStatus.NOT_RUN: "skipped",
    }
    return status_mapping.get(test_status, "error")


def export_test_results_postgres(header_info, results):
    """Export test results to PostgreSQL database"""
    if len(results) == 0:
        return

    # Initialize database if needed
    initialize_postgres_database()

    # Get PostgreSQL connection
    pg_config = get_postgres_config(POSTGRES_ENV)

    try:
        conn = psycopg2.connect(**pg_config)
        cursor = conn.cursor()

        # Get git information
        curr_git_hash = git_hash()
        git_author = get_git_author()
        git_branch = get_git_branch()
        initiated_by = get_initiated_by()

        # Get current timestamp for run start
        run_start_time = dt.datetime.now()

        # Create a new run record
        run_insert_query = """
        INSERT INTO runs (
            initiated_by, git_author, git_branch_name, git_commit_hash,
            start_time_ts, status
        ) VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """

        run_values = (
            initiated_by,
            git_author,
            git_branch,
            curr_git_hash,
            run_start_time,
            "success",  # Will be updated after processing all results
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

        # Process each test module
        for sweep_name, test_results in test_groups.items():
            # Create a test record for this module
            test_start_time = dt.datetime.now()

            test_insert_query = """
            INSERT INTO tests (
                run_id, name, start_time_ts, status
            ) VALUES (%s, %s, %s, %s)
            RETURNING id
            """

            test_values = (
                run_id,
                sweep_name,
                test_start_time,
                "success",  # Will be updated after processing all testcases
            )

            cursor.execute(test_insert_query, test_values)
            test_id = cursor.fetchone()[0]

            test_statuses = []

            # Process each test case within this test module
            for idx, result in test_results:
                header = header_info[idx]

                # Map test status to database status
                db_status = map_test_status_to_db_status(result.get("status"))
                test_statuses.append(db_status)

                # Create testcase record
                testcase_insert_query = """
                INSERT INTO sweep_testcases (
                    test_id, name, device, host, start_time_ts, end_time_ts,
                    status, suite_name, test_vector, message, exception,
                    e2e_perf, device_perf
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """

                # Parse timestamp
                timestamp_str = result.get("timestamp")
                start_time = (
                    dt.datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S") if timestamp_str else dt.datetime.now()
                )

                testcase_values = (
                    test_id,
                    f"{sweep_name}_{header.get('vector_id', 'unknown')}",
                    result.get("device"),
                    result.get("host"),
                    start_time,
                    None,  # end_time_ts - could be calculated if we track duration
                    db_status,
                    header.get("suite_name"),
                    json.dumps(result.get("original_vector_data")),  # test_vector now stores original vector data
                    result.get("message"),
                    result.get("exception"),
                    result.get("e2e_perf"),
                    json.dumps(result.get("device_perf")) if result.get("device_perf") else None,
                )

                cursor.execute(testcase_insert_query, testcase_values)

            # Update test status based on testcase results
            test_status = map_test_status_to_run_status(test_statuses)
            test_end_time = dt.datetime.now()

            test_update_query = """
            UPDATE tests SET status = %s, end_time_ts = %s WHERE id = %s
            """
            cursor.execute(test_update_query, (test_status, test_end_time, test_id))

            all_test_statuses.append(test_status)

        # Update run status based on all test results
        run_status = map_test_status_to_run_status(all_test_statuses)
        run_end_time = dt.datetime.now()

        run_update_query = """
        UPDATE runs SET status = %s, end_time_ts = %s WHERE id = %s
        """
        cursor.execute(run_update_query, (run_status, run_end_time, run_id))

        conn.commit()
        logger.info(f"Successfully exported {len(results)} results to PostgreSQL with run_id: {run_id}")

    except Exception as e:
        logger.error(f"Failed to export results to PostgreSQL: {e}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


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


def get_postgres_config(env="prod"):
    if env == "prod":
        return {
            "host": "corp_postgres_host",
            "port": 5432,
            "database": "sweeps_results",
            "user": "username",
            "password": "password",
        }
    elif env == "dev":
        return {
            "host": "ep-misty-surf-a5lm1q6p-pooler.us-east-2.aws.neon.tech",
            "port": 5432,
            "database": "sweeps",
            "user": "sweeps_owner",
            "password": "npg_TEBDYL0pUXs4",
        }
    else:
        raise ValueError(f"Unknown PostgreSQL environment: {env}")


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
        help="Custom tag for the vectors you are running. This is to keep copies seperate from other people's test vectors. By default, this will be your username. You are able to specify a tag when generating tests using the generator.",
    )
    parser.add_argument(
        "--read-file", required=False, help="Read and execute test vectors from a specified file path instead of ES."
    )
    parser.add_argument(
        "--database",
        required=False,
        default="postgres",
        choices=["elasticsearch", "postgres"],
        help="Database backend for storing results. Available options: ['elasticsearch', 'postgres']",
    )
    parser.add_argument(
        "--postgres-env",
        required=False,
        default="dev",
        choices=["dev", "prod"],
        help="PostgreSQL environment configuration. Available options: ['dev', 'prod']",
    )

    args = parser.parse_args(sys.argv[1:])
    if not args.module_name and args.vector_id:
        parser.print_help()
        logger.error("Module name is required if vector id is specified.")
        exit(1)

    global READ_FILE
    if not args.read_file:
        from elasticsearch import Elasticsearch, NotFoundError
        from framework.elastic_config import *

        global ELASTIC_CONNECTION_STRING
        ELASTIC_CONNECTION_STRING = get_elastic_url(args.elastic)
        READ_FILE = None
    else:
        if not args.module_name:
            logger.error("You must specify a module with a local file.")
            exit(1)
        READ_FILE = args.read_file

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

    logger.info(f"Running current sweeps with tag: {SWEEPS_TAG} using {DATABASE_BACKEND} backend.")

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
            # For Elasticsearch or with read-file, use single module name
            module_names = args.module_name
    else:
        module_names = None

    # Determine which execution path to take
    if READ_FILE:
        # Using explicit read-file argument
        run_sweeps_json(module_names, args.suite_name)
    elif DATABASE_BACKEND == "postgres" and args.module_name and not args.read_file:
        # Using PostgreSQL with module names but no read-file - use automatic file discovery
        run_sweeps_json(module_names, args.suite_name)
    else:
        # Using Elasticsearch or no module names specified
        run_sweeps(module_names, args.suite_name, args.vector_id)

    if args.watcher:
        disable_watcher()

    if MEASURE_DEVICE_PERF:
        disable_profiler()
