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
from elastic_config import *

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
            v["invalid_reason"] = ""
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
    client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))

    index_name = VECTOR_INDEX_PREFIX + module_name
    warnings = []

    try:
        response = client.search(
            index=index_name,
            query={
                "bool": {
                    "must": [
                        {"match": {"status": str(VectorStatus.CURRENT)}},
                        {"match": {"suite_name.keyword": suite_name}},
                    ]
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
            vector[elem] = serialize(vectors[i][elem], warnings)
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
            f"SWEEPS: New vectors found for module {module_name}, suite {suite_name}. Archiving old vectors and saving new suite. This step may take several minutes."
        )
        for old_vector_id in old_vector_ids:
            client.update(index=index_name, id=old_vector_id, doc={"status.keyword": str(VectorStatus.ARCHIVED)})
        for new_vector_id in serialized_vectors.keys():
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


def clean_module(module_name):
    client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))
    vector_index = VECTOR_INDEX_PREFIX + module_name

    if not client.indices.exists(index=vector_index):
        print(f"SWEEPS: Could not clean vectors for module {module_name} as there is no corresponding index.")
        exit(1)

    update_script = {"source": f"ctx._source.status = '{str(VectorStatus.ARCHIVED)}'", "lang": "painless"}
    client.update_by_query(index=vector_index, query={"match_all": {}}, script=update_script, refresh=True)
    print(f"SWEEPS: Marked all vectors in index {vector_index} as archived. Proceeding with generation...")

    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sweep Test Vector Generator",
        description="Generate test vector suites for the specified module.",
    )

    parser.add_argument("--module-name", required=False, help="Test Module Name, or all tests if omitted")
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        required=False,
        help="Must be set with module_name. Setting this flag will mark ALL old vectors for an sweep as Archived, and generate the new set. Use this if you make a mistake when generating your vectors and want to refresh to the current state of your op test file.",
    )
    parser.add_argument(
        "--elastic",
        required=False,
        default=ELASTIC_DEFAULT_URL,
        help="Elastic Connection String for vector database.",
    )

    args = parser.parse_args(sys.argv[1:])

    global ELASTIC_CONNECTION_STRING
    ELASTIC_CONNECTION_STRING = args.elastic

    if args.clean and not args.module_name:
        print("SWEEPS: The clean flag must be set in conjunction with a module name.")
        exit(1)
    elif args.clean:
        clean_module(args.module_name)

    generate_tests(args.module_name)
