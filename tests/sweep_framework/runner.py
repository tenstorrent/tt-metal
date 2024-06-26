# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import pathlib
import importlib
import datetime
from ttnn import *
from pymongo import MongoClient
from serialize import *


def git_hash():
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception as e:
        raise RuntimeError("Couldn't get git hash!") from e


def execute_tests(sweep_name, test_module, test_vectors):
    device = ttnn.open_device(device_id=0)

    for test_vector in test_vectors:
        try:
            status, pcc = test_module.run(**test_vector, device=device)
            test_vector["status"] = status
            test_vector["pcc"] = pcc
        except Exception as e:
            test_vector["status"] = False
            test_vector["exception"] = str(e)
        test_vector["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        test_vector["sweep_name"] = sweep_name

    ttnn.close_device(device)
    return test_vectors


def sanitize_inputs(test_vectors):
    info_column_names = ["_id", "sweep_name", "timestamp", "batch_id"]
    for vector in test_vectors:
        for col in info_column_names:
            vector.pop(col)
    return test_vectors


def run_sweeps(module_name, batch_id):
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

                test_vectors = sanitize_inputs(test_vectors)
                param_names = test_vectors[0].keys()
                test_vectors = [[deserialize(vector[elem]) for elem in vector] for vector in test_vectors]
                test_vectors = [dict(zip(param_names, vector)) for vector in test_vectors]
                results = execute_tests(sweep_name, test_module, test_vectors)
                export_test_results(results)
            except Exception as e:
                print(e)
                continue

    else:
        test_module = importlib.import_module("sweeps." + module_name)
        collection_name = module_name + "_test_vectors"
        collection = db[collection_name]

        try:
            if not batch_id:
                test_vectors = list(collection.find())
            else:
                test_vectors = list(collection.find({"batch_id": batch_id}))

            if len(test_vectors) == 0:
                return

            test_vectors = sanitize_inputs(test_vectors)
            param_names = test_vectors[0].keys()
            test_vectors = [[deserialize(vector[elem]) for elem in vector] for vector in test_vectors]
            test_vectors = [dict(zip(param_names, vector)) for vector in test_vectors]
            results = execute_tests(module_name, test_module, test_vectors)
            export_test_results(results)
        except Exception as e:
            print(e)

    client.close()


# Export test output (msg), status, exception (if applicable), git hash, timestamp, test vector, test UUID?,
def export_test_results(results):
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client.test_results
    sweep_name = results[0]["sweep_name"]
    collection = db[sweep_name + "_results"]

    try:
        git = git_hash()
        for result in results:
            result["git_hash"] = git
    except:
        pass

    serialized_results = []
    for i in range(len(results)):
        serialized_results.append(dict())
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
    parser.add_argument("--batch-id", required=False, help="Batch of Test Vectors to run, or all tests if omitted.")
    parser.add_argument(
        "--arch",
        required=True,
        choices=["grayskull", "wormhole", "wormhole_b0", "blackhole"],
        help="Device architecture",
    )

    args = parser.parse_args(sys.argv[1:])

    if not args.module_name and args.batch_id:
        parser.print_help()
        print("ERROR: Module name is required if batch id is specified.")
        exit(1)

    global MONGO_CONNECTION_STRING
    MONGO_CONNECTION_STRING = args.mongo if args.mongo else "mongodb://localhost:27017"

    run_sweeps(args.module_name, args.batch_id)
