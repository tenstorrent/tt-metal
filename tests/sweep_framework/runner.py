# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import sqlite3
import pathlib
import importlib


def import_test_vectors(module_name, batch_id):
    connection = sqlite3.connect(VECTOR_DB)
    cursor = connection.cursor()

    table_name = module_name + "_test_vectors"
    query = f"SELECT * FROM ? WHERE batch_id=?"
    cursor.execute(query, table_name, batch_id)

    vectors = cursor.fetchall()
    connection.close()

    return vectors


def _run_single_test(test_vector, run_function):
    pass


def run_single_module(test_vector, run_function):
    pass


def run_sweeps(module_name, batch_id):
    connection = sqlite3.connect(VECTOR_DB)
    cursor = connection.cursor()

    if not module_name:
        tables_query = "SELECT name FROM sqlite_schema WHERE type='table' AND name not like 'sqlite_%'"
        cursor.execute(tables_query)
        tables = cursor.fetchall()

        for table in tables:
            module_name = table.replace("_test_vectors", "")
            test_module = importlib.import_module("sweeps." + module_name)

    elif module_name and not batch_id:
        pass
    elif module_name and batch_id:
        pass


def export_test_results():
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
        "--arch", required=True, choices=["grayskull", "wormhole", "wormhole_b0", "blackhole"], help="Output Directory"
    )

    args = parser.parse_args(sys.argv[1:])

    if not args.module_name and args.batch_id:
        parser.print_help()
        print("ERROR: Module name is required if batch id is specified.")
        exit(1)

    global VECTOR_DB
    VECTOR_DB = pathlib.Path(__file__).parent / args.vector_db

    run_sweeps(args.module_name, args.batch_id)
