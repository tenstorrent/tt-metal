# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import pathlib
import importlib
import datetime
import os
from multiprocessing import Process, Queue
from queue import Empty
import subprocess
from statuses import TestStatus, VectorStatus
import architecture
from elasticsearch import Elasticsearch, NotFoundError

ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception as e:
        return "Couldn't get git hash!"


def run(test_module, input_queue, output_queue):
    device = ttnn.open_device(0)
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
        ttnn.close_device(device)
        exit(0)


def execute_batch(test_module, test_vectors):
    results = []
    input_queue = Queue()
    output_queue = Queue()
    p = None
    for test_vector in test_vectors:
        result = dict()
        if deserialize(test_vector["status"]) == VectorStatus.INVALID:
            result["status"] = TestStatus.NOT_RUN
            result["exception"] = "INVALID VECTOR: " + test_vector["invalid_reason"]
            result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            results.append(result)
            continue
        else:
            test_vector.pop("status")
        if p is None:
            p = Process(target=run, args=(test_module, input_queue, output_queue))
            p.start()
        try:
            if MEASURE_PERF:
                # Run one time before capturing result to deal with compile-time slowdown of perf measurement
                input_queue.put(test_vector)
                output_queue.get(block=True, timeout=30)
            input_queue.put(test_vector)
            response = output_queue.get(block=True, timeout=30)
            status, message, e2e_perf = response[0], response[1], response[2]
            if status:
                result["status"] = TestStatus.PASS
                result["message"] = message
            else:
                if "Out of Memory: Not enough space to allocate" in message:
                    result["status"] = TestStatus.FAIL_L1_OUT_OF_MEM
                elif "Watcher" in message:
                    result["status"] = TestStatus.FAIL_WATCHER
                else:
                    result["status"] = TestStatus.FAIL_ASSERT_EXCEPTION
                result["exception"] = message
            if e2e_perf and MEASURE_PERF:
                result["e2e_perf"] = e2e_perf
        except Empty as e:
            print(f"SWEEPS: TEST TIMED OUT, Killing child process {p.pid} and running tt-smi...")
            p.terminate()
            p = None
            smi_dir = architecture.tt_smi_path(ARCH)
            smi_process = subprocess.run([smi_dir, "-tr", "0"])
            if smi_process.returncode == 0:
                print("SWEEPS: TT-SMI Reset Complete Successfully")
            result["status"], result["exception"] = TestStatus.FAIL_CRASH_HANG, "TEST TIMED OUT (CRASH / HANG)"
        result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        results.append(result)
    if p is not None:
        p.join()

    return results


def sanitize_inputs(test_vectors):
    info_field_names = ["sweep_name", "batch_name", "vector_id"]
    header_info = []
    for vector in test_vectors:
        header = dict()
        for field in info_field_names:
            header[field] = vector.pop(field)
        vector.pop("timestamp")
        header_info.append(header)
    return header_info, test_vectors


def get_batch_vectors(client, vector_index, batch):
    response = client.search(index=vector_index, query={"match": {"batch_name": batch}}, size=10000)
    test_ids = [hit["_id"] for hit in response["hits"]["hits"]]
    test_vectors = [hit["_source"] for hit in response["hits"]["hits"]]
    for i in range(len(test_ids)):
        test_vectors[i]["vector_id"] = test_ids[i]
    header_info, test_vectors = sanitize_inputs(test_vectors)
    return header_info, test_vectors


def run_sweeps(module_name, batch_name, vector_id):
    client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=("elastic", ELASTIC_PASSWORD))

    sweeps_path = pathlib.Path(__file__).parent / "sweeps"

    if not module_name:
        for file in sorted(sweeps_path.glob("*.py")):
            sweep_name = str(pathlib.Path(file).relative_to(sweeps_path))[:-3]
            test_module = importlib.import_module("sweeps." + sweep_name)
            vector_index = sweep_name + "_test_vectors"
            print(f"SWEEPS: Executing tests for module {sweep_name}...")
            try:
                response = client.search(
                    index=vector_index, aggregations={"batches": {"terms": {"field": "batch_name.keyword"}}}
                )
                batches = [batch["key"] for batch in response["aggregations"]["batches"]["buckets"]]
                if len(batches) == 0:
                    continue

                for batch in batches:
                    print(f"SWEEPS: Executing tests for module {sweep_name}, batch {batch}.")
                    header_info, test_vectors = get_batch_vectors(client, vector_index, batch)
                    results = execute_batch(test_module, test_vectors)
                    print(f"SWEEPS: Completed tests for module {sweep_name}, batch {batch}.")
                    print(f"SWEEPS: Tests Executed - {len(results)}")
                    export_test_results(header_info, results)
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
        vector_index = module_name + "_test_vectors"

        if vector_id:
            test_vector = client.get(index=vector_index, id=vector_id)["_source"]
            test_vector["vector_id"] = vector_id
            header_info, test_vectors = sanitize_inputs([test_vector])
            results = execute_batch(test_module, test_vectors)
            export_test_results(header_info, results)
        else:
            try:
                if not batch_name:
                    response = client.search(
                        index=vector_index, aggregations={"batches": {"terms": {"field": "batch_name.keyword"}}}
                    )
                    batches = [batch["key"] for batch in response["aggregations"]["batches"]["buckets"]]
                    if len(batches) == 0:
                        return

                    for batch in batches:
                        print(f"SWEEPS: Executing tests for module {module_name}, batch {batch}.")
                        header_info, test_vectors = get_batch_vectors(client, vector_index, batch)
                        results = execute_batch(test_module, test_vectors)
                        print(f"SWEEPS: Completed tests for module {module_name}, batch {batch}.")
                        print(f"SWEEPS: Tests Executed - {len(results)}")
                        export_test_results(header_info, results)
                else:
                    print(f"SWEEPS: Executing tests for module {module_name}, batch {batch_name}.")
                    header_info, test_vectors = get_batch_vectors(client, vector_index, batch_name)
                    results = execute_batch(test_module, test_vectors)
                    print(f"SWEEPS: Completed tests for module {module_name}, batch {batch_name}.")
                    print(f"SWEEPS: Tests Executed - {len(results)}")
                    export_test_results(header_info, results)
            except Exception as e:
                print(e)

    client.close()


# Export test output (msg), status, exception (if applicable), git hash, timestamp, test vector, test UUID?,
def export_test_results(header_info, results):
    if len(results) == 0:
        return
    client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=("elastic", ELASTIC_PASSWORD))
    sweep_name = header_info[0]["sweep_name"]
    results_index = sweep_name + "_test_results"

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
        "--elastic", required=False, help="Elastic Connection String for the vector and results database."
    )
    parser.add_argument("--module-name", required=False, help="Test Module Name, or all tests if omitted.")
    parser.add_argument("--batch-name", required=False, help="Batch of Test Vectors to run, or all tests if omitted.")
    parser.add_argument(
        "--vector-id", required=False, help="Specify vector id with a module name to run an individual test vector."
    )
    parser.add_argument(
        "--arch",
        required=True,
        choices=["grayskull", "wormhole", "wormhole_b0", "blackhole"],
        help="Device architecture",
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

    args = parser.parse_args(sys.argv[1:])

    if not args.module_name and args.batch_name:
        parser.print_help()
        print("ERROR: Module name is required if batch name is specified.")
        exit(1)

    if not args.module_name and args.vector_id:
        parser.print_help()
        print("ERROR: Module name is required if vector id is specified.")

    global ELASTIC_CONNECTION_STRING
    ELASTIC_CONNECTION_STRING = args.elastic if args.elastic else "http://localhost:9200"

    global ARCH
    ARCH = architecture.str_to_arch(args.arch)

    global MEASURE_PERF
    MEASURE_PERF = args.perf

    if args.watcher:
        enable_watcher()

    from ttnn import *
    from serialize import *

    run_sweeps(args.module_name, args.batch_name, args.vector_id)

    if args.watcher:
        disable_watcher()
