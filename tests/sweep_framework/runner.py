# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import pathlib
import importlib
import datetime
import os
import enlighten
from multiprocessing import Process, Queue
from queue import Empty
import subprocess
from statuses import TestStatus, VectorValidity, VectorStatus
import tt_smi_util
from device_fixtures import default_device
from elasticsearch import Elasticsearch, NotFoundError
from elastic_config import *

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


def run(test_module, input_queue, output_queue):
    device_generator = get_devices(test_module)
    try:
        device, device_name = next(device_generator)
        print(f"SWEEPS: Opened device configuration, {device_name}.")
    except AssertionError as e:
        output_queue.put([False, "DEVICE EXCEPTION: " + str(e), None])
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
            output_queue.put([status, message, e2e_perf])
    except Empty as e:
        try:
            # Run teardown in mesh_device_fixture
            next(device_generator)
        except StopIteration:
            print(f"SWEEPS: Closed device configuration, {device_name}.")


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
    for test_vector in test_vectors:
        if DRY_RUN:
            print(f"Would have executed test for vector {test_vector}")
            continue
        result = dict()
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
                        print(
                            "SWEEPS: Executing test (first run, e2e perf is enabled) on parent process (to allow debugger support) because there is only one test vector. Hang detection is disabled."
                        )
                        run(test_module, input_queue, output_queue)
                    output_queue.get(block=True, timeout=timeout)
                input_queue.put(test_vector)
                if len(test_vectors) == 1:
                    print(
                        "SWEEPS: Executing test on parent process (to allow debugger support) because there is only one test vector. Hang detection is disabled."
                    )
                    run(test_module, input_queue, output_queue)
                response = output_queue.get(block=True, timeout=timeout)
                status, message, e2e_perf = response[0], response[1], response[2]
                if status:
                    result["status"] = TestStatus.PASS
                    result["message"] = message
                else:
                    if "DEVICE EXCEPTION" in message:
                        print(
                            "SWEEPS: DEVICE EXCEPTION: Device could not be initialized. The following assertion was thrown: ",
                            message,
                        )
                        print("SWEEPS: Skipping test suite because of device error, proceeding...")
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
            except Empty as e:
                print(f"SWEEPS: TEST TIMED OUT, Killing child process {p.pid} and running tt-smi...")
                p.terminate()
                p = None
                tt_smi_util.run_tt_smi(ARCH)
                result["status"], result["exception"] = TestStatus.FAIL_CRASH_HANG, "TEST TIMED OUT (CRASH / HANG)"
                result["e2e_perf"] = None
        result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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


def run_sweeps(module_name, suite_name, vector_id):
    client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))
    pbar_manager = enlighten.get_manager()

    sweeps_path = pathlib.Path(__file__).parent / "sweeps"

    if not module_name:
        for file in sorted(sweeps_path.glob("**/*.py")):
            sweep_name = str(pathlib.Path(file).relative_to(sweeps_path))[:-3].replace("/", ".")
            test_module = importlib.import_module("sweeps." + sweep_name)
            vector_index = VECTOR_INDEX_PREFIX + sweep_name
            print(f"SWEEPS: Executing tests for module {sweep_name}...")
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
                        print(
                            f"SWEEPS: No suites found for module {sweep_name}, with tag {SWEEPS_TAG}. If you meant to run the CI suites of tests, use '--tag ci-main' in your test command, otherwise, run the parameter generator with your own tag and try again. Continuing..."
                        )
                    else:
                        print(
                            f"SWEEPS: No suite named {suite_name} found for module {sweep_name}, with tag {SWEEPS_TAG}. If you meant to run the CI suite of tests, use '--tag ci-main' in your test command, otherwise, run the parameter generator with your own tag and try again. Continuing..."
                        )
                    continue

                module_pbar = pbar_manager.counter(total=len(suites), desc=f"Module: {sweep_name}", leave=False)
                for suite in suites:
                    print(f"SWEEPS: Executing tests for module {sweep_name}, suite {suite}.")
                    header_info, test_vectors = get_suite_vectors(client, vector_index, suite)
                    results = execute_suite(test_module, test_vectors, pbar_manager, suite)
                    print(f"SWEEPS: Completed tests for module {sweep_name}, suite {suite}.")
                    print(f"SWEEPS: Tests Executed - {len(results)}")
                    export_test_results(header_info, results)
                    module_pbar.update()
                module_pbar.close()
            except NotFoundError as e:
                print(f"SWEEPS: No test vectors found for module {sweep_name}. Skipping...")
                continue
            except Exception as e:
                print(e)
                continue

    else:
        try:
            test_module = importlib.import_module("sweeps." + module_name)
        except ModuleNotFoundError as e:
            print(f"SWEEPS: No module found with name {module_name}")
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
                        print(
                            f"SWEEPS: No suites found for module {module_name}, with tag {SWEEPS_TAG}. If you meant to run the CI suites of tests, use '--tag ci-main' in your test command, otherwise, run the parameter generator with your own tag and try again."
                        )
                        return

                    for suite in suites:
                        print(f"SWEEPS: Executing tests for module {module_name}, suite {suite}.")
                        header_info, test_vectors = get_suite_vectors(client, vector_index, suite)
                        results = execute_suite(test_module, test_vectors, pbar_manager, suite)
                        print(f"SWEEPS: Completed tests for module {module_name}, suite {suite}.")
                        print(f"SWEEPS: Tests Executed - {len(results)}")
                        export_test_results(header_info, results)
                else:
                    print(f"SWEEPS: Executing tests for module {module_name}, suite {suite_name}.")
                    header_info, test_vectors = get_suite_vectors(client, vector_index, suite_name)
                    results = execute_suite(test_module, test_vectors, pbar_manager, suite_name)
                    print(f"SWEEPS: Completed tests for module {module_name}, suite {suite_name}.")
                    print(f"SWEEPS: Tests Executed - {len(results)}")
                    export_test_results(header_info, results)
            except Exception as e:
                print(e)

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
            result[elem] = serialize(results[i][elem])
        client.index(index=results_index, body=result)

    client.close()


def enable_watcher():
    print("SWEEPS: Enabling Watcher")
    os.environ["TT_METAL_WATCHER"] = "120"
    os.environ["TT_METAL_WATCHER_APPEND"] = "1"


def disable_watcher():
    print("SWEEPS: Disabling Watcher")
    os.environ.pop("TT_METAL_WATCHER")
    os.environ.pop("TT_METAL_WATCHER_APPEND")


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
    parser.add_argument("--module-name", required=False, help="Test Module Name, or all tests if omitted.")
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

    args = parser.parse_args(sys.argv[1:])
    if not args.module_name and args.vector_id:
        parser.print_help()
        print("ERROR: Module name is required if vector id is specified.")
        exit(1)

    global ELASTIC_CONNECTION_STRING
    ELASTIC_CONNECTION_STRING = get_elastic_url(args.elastic)

    global MEASURE_PERF
    MEASURE_PERF = args.perf

    global DRY_RUN
    DRY_RUN = args.dry_run

    global SWEEPS_TAG
    SWEEPS_TAG = args.tag

    print(f"SWEEPS: Running current sweeps with tag: {SWEEPS_TAG}.")

    if args.watcher:
        enable_watcher()

    from ttnn import *
    from serialize import *

    run_sweeps(args.module_name, args.suite_name, args.vector_id)

    if args.watcher:
        disable_watcher()
