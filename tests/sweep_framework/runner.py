# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import sqlite3
import pathlib
import importlib
import ttnn

from ttnn.types import *

RESULTS_DB = pathlib.Path(__file__).parent / "results" / "results.sqlite"


def execute_tests(test_module, test_vectors):
    device = ttnn.open_device(device_id=0)
    status_results = []
    msg_results = []

    for test_vector in test_vectors:
        status, message = test_module.run(**test_vector, device=device)
        status_results.append(status)
        msg_results.append(message)

    ttnn.close_device(device)
    return zip(status_results, msg_results, test_vectors)


def flatten_results(results):
    pass


def sanitize_inputs(test_vectors, cursor_description):
    test_vectors = [list(vector) for vector in test_vectors]
    column_names = [description[0] for description in cursor_description]
    test_vectors = [dict(zip(column_names, vector)) for vector in test_vectors]
    info_column_names = ["sweep_name", "timestamp", "batch_id"]
    for vector in test_vectors:
        for col in info_column_names:
            vector.pop(col)
        for elem in vector:
            vector[elem] = eval(vector[elem])
    return test_vectors


def run_sweeps(module_name, batch_id):
    connection = sqlite3.connect(VECTOR_DB)
    cursor = connection.cursor()

    sweeps_path = pathlib.Path(__file__).parent / "sweeps"

    if not module_name:
        for file in sorted(sweeps_path.glob("*.py")):
            sweep_name = str(pathlib.Path(file).relative_to(sweeps_path))[:-3]
            test_module = importlib.import_module("sweeps." + sweep_name)
            table_name = sweep_name + "_test_vectors"

            try:
                test_vectors_query = f"SELECT * FROM {table_name}"
                cursor.execute(test_vectors_query)
                test_vectors = cursor.fetchall()
                if len(test_vectors) == 0:
                    continue

                test_vectors = sanitize_inputs(test_vectors, cursor.description)
                results = execute_tests(test_module, test_vectors)
                results = flatten_results(results)
                export_test_results(results)
            except:
                continue

    elif module_name and not batch_id:
        test_module = importlib.import_module("sweeps." + module_name)
        table_name = module_name + "_test_vectors"

        try:
            test_vectors_query = f"SELECT * FROM {table_name}"
            cursor.execute(test_vectors_query)
            test_vectors = cursor.fetchall()
            if len(test_vectors) == 0:
                return

            test_vectors = sanitize_inputs(test_vectors, cursor.description)
            results = execute_tests(test_module, test_vectors)
            results = flatten_results(results)
            export_test_results(results)
        except:
            return

    elif module_name and batch_id:
        test_module = importlib.import_module("sweeps." + module_name)
        table_name = module_name + "_test_vectors"

        try:
            test_vectors_query = f"SELECT * FROM {table_name} WHERE batch_id={batch_id}"
            cursor.execute(test_vectors_query)
            test_vectors = cursor.fetchall()
            if len(test_vectors) == 0:
                return

            test_vectors = sanitize_inputs(test_vectors, cursor.description)
            results = execute_tests(test_module, test_vectors)
            results = flatten_results(results)
            export_test_results(results)
        except:
            return

    connection.commit()
    connection.close()


# Export test output (msg), status, exception (if applicable), git hash, timestamp, test vector, test UUID?,
def export_test_results(results):
    connection = sqlite3.connect(RESULTS_DB)
    cursor = connection.cursor()
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sweep Test Runner",
        description="Run test vector suites from generated vector database.",
    )

    parser.add_argument("--vector-db", required=True, help="Path to the vector database.")
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

    global VECTOR_DB
    VECTOR_DB = pathlib.Path(__file__).parent / args.vector_db

    run_sweeps(args.module_name, args.batch_id)
