# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import importlib
import pathlib
import datetime
import os

from permutations import *
from serialize import serialize
from elasticsearch import Elasticsearch
from statuses import VectorStatus

ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")

SWEEPS_DIR = pathlib.Path(__file__).parent
SWEEP_SOURCES_DIR = SWEEPS_DIR / "sweeps"


# Generate vectors from module parameters
def generate_vectors(test_module):
    parameters = test_module.parameters

    vectors = []
    for batch in parameters:
        print(f"SWEEPS: Generating test vectors for batch {batch}.")
        batch_vectors = list(permutations(parameters[batch]))
        for v in batch_vectors:
            v["batch_name"] = batch
            v["status"] = VectorStatus.VALID
        vectors += batch_vectors
        print(f"SWEEPS: Generated {len(batch_vectors)} test vectors for batch {batch}.")
    return vectors


# Perform any post-gen validation to the resulting vectors.
def invalidate_vectors(test_module, vectors) -> None:
    if "invalidate_vector" not in dir(test_module):
        return
    for vector in vectors:
        invalid, reason = test_module.invalidate_vector(vector)
        if invalid:
            vector["status"] = VectorStatus.INVALID
            vector["invalid_reason"] = reason


# Output the individual test vectors.
def export_test_vectors(module_name, vectors):
    # Perhaps we export with some sort of readable id, which can be passed to a runner to run specific sets of input vectors. (export seed as well for reproducability)
    client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=("elastic", ELASTIC_PASSWORD))

    # TODO: Duplicate batch check?

    index_name = module_name + "_test_vectors"

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for i in range(len(vectors)):
        vector = dict()
        vector["sweep_name"] = module_name
        vector["timestamp"] = current_time
        for elem in vectors[i].keys():
            vector[elem] = serialize(vectors[i][elem])
        client.index(index=index_name, body=vector)


# Generate one or more sets of test vectors depending on module_name
def generate_tests(module_name):
    if not module_name:
        for file_name in sorted(SWEEP_SOURCES_DIR.glob("**/*.py")):
            module_name = str(pathlib.Path(file_name).relative_to(SWEEP_SOURCES_DIR))[:-3]
            print(f"SWEEPS: Generating test vectors for module {module_name}.")
            test_module = importlib.import_module("sweeps." + module_name)
            vectors = generate_vectors(test_module)
            invalidate_vectors(test_module, vectors)
            export_test_vectors(module_name, vectors)
            print(f"SWEEPS: Finished generating test vectors for module {module_name}.\n\n")
    else:
        test_module = importlib.import_module("sweeps." + module_name)
        print(f"SWEEPS: Generating test vectors for module {module_name}.")
        vectors = generate_vectors(test_module)
        invalidate_vectors(test_module, vectors)
        export_test_vectors(module_name, vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sweep Test Vector Generator",
        description="Generate test vector suites for the specified module.",
    )

    parser.add_argument("--module-name", required=False, help="Test Module Name, or all tests if omitted")
    parser.add_argument("--elastic", required=False, help="Elastic Connection String for vector database.")

    args = parser.parse_args(sys.argv[1:])

    global ELASTIC_CONNECTION_STRING
    ELASTIC_CONNECTION_STRING = args.elastic if args.elastic else "http://localhost:9200"

    generate_tests(args.module_name)
