# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import importlib
import pathlib
import datetime
import shortuuid

from architecture import str_to_arch
from permutations import *
from sql_utils import *
from serialize import serialize, deserialize
from pymongo import MongoClient

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


# Output the individual test vectors.
def export_test_vectors(vectors):
    vectors = list(vectors)
    # Perhaps we export with some sort of readable id, which can be passed to a runner to run specific sets of input vectors. (export seed as well for reproducability)
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client.test_vectors

    parameter_names = get_parameter_names(vectors[0])
    batch_id = str(shortuuid.uuid())
    # TODO: Duplicate batch check?

    table_name = MODULE_NAME + "_test_vectors"
    collection = db[table_name]

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    serialized_vectors = []

    for i in range(len(vectors)):
        serialized_vectors.append(dict())
        serialized_vectors[i]["sweep_name"] = MODULE_NAME
        serialized_vectors[i]["timestamp"] = current_time
        serialized_vectors[i]["batch_id"] = batch_id
        for elem in vectors[i].keys():
            serialized_vectors[i][elem] = serialize(vectors[i][elem])

    collection.insert_many(serialized_vectors)


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
    parser.add_argument("--mongo", required=False, help="Mongo Connection String for vector database.")
    parser.add_argument("--seed", required=False, default=0, help="Seed for random value generation")
    parser.add_argument(
        "--arch",
        required=True,
        choices=["grayskull", "wormhole", "wormhole_b0", "blackhole"],
        help="Device Architecture",
    )

    args = parser.parse_args(sys.argv[1:])

    global MONGO_CONNECTION_STRING
    MONGO_CONNECTION_STRING = args.mongo if args.mongo else "mongodb://localhost:27017"

    generate_tests(args.module_name, str_to_arch(args.arch))
