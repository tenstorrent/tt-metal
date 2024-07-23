# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import importlib
import pathlib
import datetime
import os
import hashlib

from permutations import *
from serialize import serialize
from elasticsearch import Elasticsearch, NotFoundError
from statuses import VectorValidity, VectorStatus

ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")

SWEEPS_DIR = pathlib.Path(__file__).parent
SWEEP_SOURCES_DIR = SWEEPS_DIR / "sweeps"


# Generate vectors from module parameters
def generate_vectors(module_name):
    test_module = importlib.import_module("sweeps." + module_name)
    parameters = test_module.parameters

    for suite in parameters:
        print(f"SWEEPS: Generating test vectors for suite {suite}.")
        suite_vectors = list(permutations(parameters[suite]))
        for v in suite_vectors:
            v["suite_name"] = suite
            v["validity"] = VectorValidity.VALID
            v["status"] = VectorStatus.CURRENT
            v["sweep_name"] = module_name

        invalidate_vectors(test_module, suite_vectors)
        export_suite_vectors(module_name, suite, suite_vectors)


# Perform any post-gen validation to the resulting vectors.
def invalidate_vectors(test_module, vectors) -> None:
    if "invalidate_vector" not in dir(test_module):
        return
    for vector in vectors:
        invalid, reason = test_module.invalidate_vector(vector)
        if invalid:
            vector["validity"] = VectorValidity.INVALID
            vector["invalid_reason"] = reason


# Output the individual test vectors.
def export_suite_vectors(module_name, suite_name, vectors):
    # Perhaps we export with some sort of readable id, which can be passed to a runner to run specific sets of input vectors. (export seed as well for reproducability)
    client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=("elastic", ELASTIC_PASSWORD))

    index_name = module_name + "_test_vectors"

    try:
        response = client.search(
            index=index_name,
            query={
                "bool": {
                    "must": [{"match": {"status": str(VectorStatus.CURRENT)}}, {"match": {"suite_name": suite_name}}]
                }
            },
            size=10000,
        )["hits"]["hits"]
        old_vector_ids = set(vector["_id"] for vector in response)
    except NotFoundError as e:
        old_vector_ids = set()
        pass

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    new_vector_ids = set()
    serialized_vectors = dict()
    for i in range(len(vectors)):
        vector = dict()
        for elem in vectors[i].keys():
            vector[elem] = serialize(vectors[i][elem])
        id = hashlib.sha224(str(vectors[i]).encode("utf-8")).hexdigest()
        new_vector_ids.add(id)
        vector["timestamp"] = current_time
        serialized_vectors[id] = vector

    if old_vector_ids == new_vector_ids:
        print(
            f"SWEEPS: Vectors generated for module {module_name}, suite {suite_name} already exist, and have not changed. Skipping..."
        )
        return
    else:
        print(
            f"SWEEPS: New vectors found for module {module_name}, suite {suite_name}. Archiving old vectors and saving new suite."
        )
        for old_vector_id in old_vector_ids:
            client.update(index=index_name, id=old_vector_id, doc={"status": str(VectorStatus.ARCHIVED)})
        for new_vector_id in serialized_vectors.keys():
            try:
                existing = client.get(index=index_name, id=new_vector_id)
                print(existing)
                print(new_vector_id)
                print(serialized_vectors[new_vector_id])
                exit(0)
            except:
                pass
            client.index(index=index_name, id=new_vector_id, body=serialized_vectors[new_vector_id])
        print(f"SWEEPS: Generated {len(serialized_vectors)} test vectors for suite {suite_name}.")


# Generate one or more sets of test vectors depending on module_name
def generate_tests(module_name):
    if not module_name:
        for file_name in sorted(SWEEP_SOURCES_DIR.glob("**/*.py")):
            module_name = str(pathlib.Path(file_name).relative_to(SWEEP_SOURCES_DIR))[:-3]
            print(f"SWEEPS: Generating test vectors for module {module_name}.")
            generate_vectors(module_name)
            print(f"SWEEPS: Finished generating test vectors for module {module_name}.\n\n")
    else:
        print(f"SWEEPS: Generating test vectors for module {module_name}.")
        generate_vectors(module_name)


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
