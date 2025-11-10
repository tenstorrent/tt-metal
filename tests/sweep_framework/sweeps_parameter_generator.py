# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import importlib
import pathlib
import datetime
import os
import hashlib
import json
import random

from framework.permutations import *
from framework.serialize import serialize, serialize_structured
from framework.statuses import VectorValidity, VectorStatus
from framework.sweeps_logger import sweeps_logger as logger

SWEEPS_DIR = pathlib.Path(__file__).parent
SWEEP_SOURCES_DIR = SWEEPS_DIR / "sweeps"


# Shuffle control (set in __main__ when --randomize is provided)
SHUFFLE_SEED = None
DO_RANDOMIZE = False


# Generate vectors from module parameters
def generate_vectors(module_name):
    test_module = importlib.import_module("sweeps." + module_name)
    parameters = test_module.parameters

    for suite in parameters:
        logger.info(f"Generating test vectors for suite {suite}.")
        suite_vectors = list(permutations(parameters[suite]))
        for v in suite_vectors:
            v["suite_name"] = suite
            v["validity"] = VectorValidity.VALID
            v["invalid_reason"] = ""
            v["status"] = VectorStatus.CURRENT
            v["sweep_name"] = module_name

        invalidate_vectors(test_module, suite_vectors)
        export_suite_vectors_json(module_name, suite, suite_vectors)


# Perform any post-gen validation to the resulting vectors.
def invalidate_vectors(test_module, vectors) -> None:
    if "invalidate_vector" not in dir(test_module):
        return
    for vector in vectors:
        invalid, reason = test_module.invalidate_vector(vector)
        if invalid:
            vector["validity"] = VectorValidity.INVALID
            vector["invalid_reason"] = reason


def export_suite_vectors_json(module_name, suite_name, vectors):
    EXPORT_DIR_PATH = SWEEPS_DIR / "vectors_export"
    EXPORT_PATH = EXPORT_DIR_PATH / str(module_name + ".json")
    if not EXPORT_DIR_PATH.exists():
        EXPORT_DIR_PATH.mkdir()

    # Randomize order only when explicitly requested via --randomize
    if DO_RANDOMIZE:
        rng = random.Random(SHUFFLE_SEED)
        rng.shuffle(vectors)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    serialized_vectors = dict()
    warnings = []
    for i in range(len(vectors)):
        vector = dict()
        for elem in vectors[i].keys():
            vector[elem] = serialize_structured(vectors[i][elem], warnings)
        input_hash = hashlib.sha224(str(vector).encode("utf-8")).hexdigest()
        vector["timestamp"] = current_time
        vector["input_hash"] = input_hash
        vector["tag"] = SWEEPS_TAG
        serialized_vectors[input_hash] = vector

    if EXPORT_PATH.exists():
        with open(EXPORT_PATH, "r") as file:
            data = json.load(file)
        with open(EXPORT_PATH, "w") as file:
            data[suite_name] = serialized_vectors
            json.dump(data, file, indent=2)
    else:
        with open(EXPORT_PATH, "w") as file:
            json.dump({suite_name: serialized_vectors}, file, indent=2)
    logger.info(f"SWEEPS: Generated {len(vectors)} test vectors for suite {suite_name}.")


# Generate one or more sets of test vectors depending on module_name
def generate_tests(module_name, skip_modules=None):
    skip_modules_set = set()
    if skip_modules:
        skip_modules_set = {name.strip() for name in skip_modules.split(",")}
        logger.info(f"Skipping modules: {', '.join(skip_modules_set)}")

    if not module_name:
        for file_name in sorted(SWEEP_SOURCES_DIR.glob("**/*.py")):
            module_name = str(pathlib.Path(file_name).relative_to(SWEEP_SOURCES_DIR))[:-3].replace("/", ".")
            if module_name in skip_modules_set:
                logger.info(f"Skipping module {module_name} (in skip list).")
                continue
            logger.info(f"Generating test vectors for module {module_name}.")
            try:
                generate_vectors(module_name)
                logger.info(f"Finished generating test vectors for module {module_name}.\n\n")
            except Exception as e:
                logger.error(f"Failed to generate vectors for module {module_name}: {e}")
                logger.info(f"Skipping module {module_name} due to import/generation error.\n\n")
    else:
        if module_name in skip_modules_set:
            logger.info(f"Skipping module {module_name} (in skip list).")
            return
        logger.info(f"Generating test vectors for module {module_name}.")
        try:
            generate_vectors(module_name)
        except Exception as e:
            logger.error(f"Failed to generate vectors for module {module_name}: {e}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sweep Test Vector Generator",
        description="Generate test vector suites for the specified module.",
    )

    parser.add_argument("--module-name", required=False, help="Test Module Name, or all tests if omitted")
    parser.add_argument(
        "--tag",
        required=False,
        default=os.getenv("USER"),
        help="Custom tag for the vectors you are generating. This is to keep copies separate from other people's test vectors. By default, this will be your username. You are able to specify a tag when running tests using the runner.",
    )
    parser.add_argument("--explicit", required=False, action="store_true")
    parser.add_argument(
        "--dump-file",
        required=False,
        action="store_true",
        help="[DEPRECATED] This flag is now the default behavior. Elasticsearch support has been removed. Vectors are always dumped to disk in JSON format.",
    )
    parser.add_argument(
        "--randomize",
        required=False,
        type=int,
        help="Randomize the order of vectors to allow reproducible order.",
    )
    parser.add_argument(
        "--skip-modules",
        required=False,
        help="Comma-separated list of module names to skip during generation",
    )

    args = parser.parse_args(sys.argv[1:])

    # Elasticsearch support has been removed. Vectors are always dumped to disk.
    if args.dump_file:
        logger.warning(
            "The --dump-file flag is deprecated. Elasticsearch support has been removed. "
            "Vectors are now always dumped to disk in JSON format by default."
        )

    global SWEEPS_TAG
    SWEEPS_TAG = args.tag

    if args.tag == "ci-main" and not args.explicit:
        logger.error("The ci-main tag is reserved for CI only.")
        exit(1)

    logger.info(f"Running current generation with tag: {SWEEPS_TAG}.")
    logger.info("Vectors will be exported to: tests/sweep_framework/vectors_export/")

    # Enable reproducible shuffling only when --randomize is provided
    if args.randomize is not None:
        SHUFFLE_SEED = int(args.randomize)
        DO_RANDOMIZE = True
        logger.info(f"Randomize seed: {SHUFFLE_SEED}")
    else:
        DO_RANDOMIZE = False
        SHUFFLE_SEED = None

    generate_tests(args.module_name, args.skip_modules)
