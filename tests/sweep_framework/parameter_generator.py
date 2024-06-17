# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import z3
import importlib
import pathlib
import sqlite3
import datetime
import shortuuid

from architecture import str_to_arch
from permutations import *
from sql_utils import *

SWEEPS_DIR = pathlib.Path(__file__).parent
SWEEP_SOURCES_DIR = SWEEPS_DIR / "sweeps"


# Generate vectors from module parameters
def generate_vectors(test_module, arch):
    parameters = test_module.parameters

    vectors = permutations(parameters)
    return vectors


# Perform any post-gen validation to the resulting vectors.
def validate_vectors(vectors) -> None:
    pass


# Output the individual test vectors from solver.model()
def export_test_vectors(vectors):
    vectors = list(vectors)
    # Perhaps we export with some sort of readable id, which can be passed to a runner to run specific sets of input vectors. (export seed as well for reproducability)
    connection = sqlite3.connect(str(OUTPUT_DIR) + "/vectors.sqlite")
    cursor = connection.cursor()

    parameter_names = get_parameter_names(vectors[0])
    batch_id = str(shortuuid.uuid())
    # TODO: Duplicate batch check?
    column_names = ["sweep_name", "timestamp", "batch_id"] + parameter_names

    table_name = MODULE_NAME + "_test_vectors"
    table = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_names_to_sql_string(column_names)})"
    cursor.execute(table)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for vector in vectors:
        row = [MODULE_NAME, current_time, batch_id] + list(get_parameter_values(parameter_names, vector))
        row = [str(value) for value in row]
        row_placeholders = ", ".join(["?"] * len(row))
        command = f"INSERT INTO {table_name} VALUES ({row_placeholders})"
        cursor.execute(command, row)

    connection.commit()
    connection.close()


# Generate one or more sets of test vectors depending on module_name
def generate_tests(module_name, arch):
    global MODULE_NAME
    MODULE_NAME = module_name

    if not module_name:
        for file_name in sorted(SWEEP_SOURCES_DIR.glob("**/*.py")):
            vectors = generate_vectors(pathlib.Path(file_name).relative_to(SWEEP_SOURCES_DIR), arch)
            validate_vectors(vectors)
            export_test_vectors(vectors)
    else:
        vectors = generate_vectors(importlib.import_module("sweeps." + module_name[:3]), arch)  # Macro this
        validate_vectors(vectors)
        export_test_vectors(vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sweep Test Vector Generator",
        description="Generate test vector suites for the specified module.",
    )

    parser.add_argument("--module-name", required=False, help="Test Module Name, or all tests if omitted")
    parser.add_argument("--output-dir", required=True, help="Output Directory")
    parser.add_argument("--seed", required=False, default=0, help="Seed for random value generation")
    parser.add_argument(
        "--arch", required=True, choices=["grayskull", "wormhole", "wormhole_b0", "blackhole"], help="Output Directory"
    )

    args = parser.parse_args(sys.argv[1:])

    global OUTPUT_DIR
    OUTPUT_DIR = pathlib.Path(__file__).parent / args.output_dir

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_tests(args.module_name, str_to_arch(args.arch))
