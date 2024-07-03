# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import pathlib
import importlib
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from multiprocessing import Process, Queue
from queue import Empty
import subprocess
from ttnn import *
from pymongo import MongoClient
from serialize import *
from test_status import TestStatus
import architecture


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception as e:
        raise RuntimeError("Couldn't get git hash!") from e


def run(test_module, test_vector, queue):
    device = ttnn.open_device(0)
    test_vector = deserialize_vector(test_vector)
    try:
        status, message = test_module.run(**test_vector, device=device)
    except Exception as e:
        status, message = False, str(e)
    ttnn.close_device(device)
    queue.put([status, message])


def execute_test(test_module, test_vector):
    result = dict()

    q = Queue()
    p = Process(target=run, args=(test_module, test_vector, q))
    try:
        p.start()
        response = q.get(block=True, timeout=5)
        status, message = response[0], response[1]
        if not status:
            result["status"] = TestStatus.FAIL_ASSERT_EXCEPTION
        else:
            result["status"] = TestStatus.PASS
        result["test_output"] = message
    except Empty as e:
        print(f"TEST TIMED OUT, Killing child process {p.pid} and running tt-smi...")
        p.terminate()
        smi_dir = architecture.tt_smi_path(ARCH)
        smi_process = subprocess.run([smi_dir, "-tr", "0"])
        if smi_process.returncode == 0:
            print("TT-SMI Reset Complete Successfully")
        result["status"], result["exception"] = TestStatus.FAIL_CRASH_HANG, "TEST TIMED OUT (CRASH / HANG)"
    except Exception as e:
        result["status"] = TestStatus.FAIL_ASSERT_EXCEPTION
        result["exception"] = str(e)

    result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return result


def execute_tests(test_module, test_vectors):
    tests_run = 0
    results = []
    for test_vector in test_vectors:
        tests_run += 1
        result = execute_test(test_module, test_vector)
        results.append(result)

    print("TESTS RUN: ", tests_run)
    return results


def sanitize_inputs(test_vectors):
    info_column_names = ["sweep_name", "batch_name"]
    header_info = []
    for vector in test_vectors:
        header = dict()
        for col in info_column_names:
            header[col] = vector.pop(col)
        vector.pop("timestamp")
        header["vector_id"] = vector.pop("_id")
        header_info.append(header)
    return header_info, test_vectors


def run_sweeps(module_name, batch_name):
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client.test_vectors

    sweeps_path = pathlib.Path(__file__).parent / "sweeps"

    if not module_name:
        for file in sorted(sweeps_path.glob("*.py")):
            sweep_name = str(pathlib.Path(file).relative_to(sweeps_path))[:-3]
            test_module = importlib.import_module("sweeps." + sweep_name)
            collection_name = sweep_name + "_test_vectors"
            collection = db[collection_name]

            try:
                test_vectors = list(collection.find())
                if len(test_vectors) == 0:
                    continue

                header_info, test_vectors = sanitize_inputs(test_vectors)
                results = execute_tests(test_module, test_vectors)
                export_test_results(header_info, results)
            except Exception as e:
                print(e)
                continue

    else:
        test_module = importlib.import_module("sweeps." + module_name)
        collection_name = module_name + "_test_vectors"
        collection = db[collection_name]

        try:
            if not batch_name:
                test_vectors = list(collection.find())
            else:
                test_vectors = list(collection.find({"batch_name": batch_name}))

            if len(test_vectors) == 0:
                return

            header_info, test_vectors = sanitize_inputs(test_vectors)
            # param_names = test_vectors[0].keys()
            # test_vectors = [[deserialize(vector[elem]) for elem in vector] for vector in test_vectors]
            # test_vectors = [dict(zip(param_names, vector)) for vector in test_vectors]
            results = execute_tests(test_module, test_vectors)
            export_test_results(header_info, results)
        except Exception as e:
            print(e)

    client.close()


# Export test output (msg), status, exception (if applicable), git hash, timestamp, test vector, test UUID?,
def export_test_results(header_info, results):
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client.test_results
    sweep_name = header_info[0]["sweep_name"]
    collection = db[sweep_name + "_test_results"]

    try:
        git = git_hash()
        for result in results:
            result["git_hash"] = git
    except:
        pass

    serialized_results = []
    for i in range(len(results)):
        serialized_results.append(header_info[i])
        for elem in results[i].keys():
            serialized_results[i][elem] = serialize(results[i][elem])
    collection.insert_many(serialized_results)

    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sweep Test Runner",
        description="Run test vector suites from generated vector database.",
    )

    parser.add_argument("--mongo", required=False, help="Mongo Connection String for the vector and results database.")
    parser.add_argument("--module-name", required=False, help="Test Module Name, or all tests if omitted.")
    parser.add_argument("--batch-name", required=False, help="Batch of Test Vectors to run, or all tests if omitted.")
    parser.add_argument(
        "--arch",
        required=True,
        choices=["grayskull", "wormhole", "wormhole_b0", "blackhole"],
        help="Device architecture",
    )

    args = parser.parse_args(sys.argv[1:])

    if not args.module_name and args.batch_name:
        parser.print_help()
        print("ERROR: Module name is required if batch id is specified.")
        exit(1)

    global MONGO_CONNECTION_STRING
    MONGO_CONNECTION_STRING = args.mongo if args.mongo else "mongodb://localhost:27017"

    global ARCH
    ARCH = architecture.str_to_arch(args.arch)

    run_sweeps(args.module_name, args.batch_name)
