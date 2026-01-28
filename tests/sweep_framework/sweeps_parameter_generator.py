# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import hashlib
import importlib
import json
import os
import pathlib
import random
import sys
import ast
from collections import defaultdict

from framework.constants import format_mesh_suffix
from framework.permutations import permutations
from framework.serialize import serialize_structured
from framework.statuses import VectorStatus, VectorValidity
from framework.sweeps_logger import sweeps_logger as logger
from tests.sweep_framework.master_config_loader import MasterConfigLoader

SWEEPS_DIR = pathlib.Path(__file__).parent
SWEEP_SOURCES_DIR = SWEEPS_DIR / "sweeps"


# Shuffle control (set in __main__ when --randomize is provided)
SHUFFLE_SEED = None
DO_RANDOMIZE = False


def get_mesh_shape_from_vector(vector):
    """Extract mesh_device_shape from traced_machine_info, default to [1, 1] for single-chip.

    Args:
        vector: Dictionary containing vector parameters including traced_machine_info

    Returns:
        tuple: (rows, cols) representing mesh shape, e.g., (2, 4) or (1, 1) for single-chip
    """
    machine_info = vector.get("traced_machine_info")
    if machine_info and isinstance(machine_info, list) and len(machine_info) > 0:
        # Check if mesh_device_shape is directly in machine_info (old format)
        mesh_shape = machine_info[0].get("mesh_device_shape")
        if mesh_shape and isinstance(mesh_shape, list) and len(mesh_shape) == 2:
            return tuple(mesh_shape)

        # Check if mesh_device_shape is inside tensor_placements (new format)
        tensor_placements = machine_info[0].get("tensor_placements")
        if tensor_placements and isinstance(tensor_placements, list) and len(tensor_placements) > 0:
            # Parse mesh_device_shape from string format "[2, 4]" to list
            mesh_shape_str = tensor_placements[0].get("mesh_device_shape", "")
            if isinstance(mesh_shape_str, str):
                # Parse "[2, 4]" format
                try:
                    mesh_shape = ast.literal_eval(mesh_shape_str)
                    if isinstance(mesh_shape, list) and len(mesh_shape) == 2:
                        return tuple(mesh_shape)
                except (ValueError, SyntaxError) as e:
                    # Invalid or malformed mesh_device_shape string; fall back to default (1, 1)
                    logger.debug(f"Failed to parse mesh_device_shape '{mesh_shape_str}': {e}")
            elif isinstance(mesh_shape_str, list) and len(mesh_shape_str) == 2:
                return tuple(mesh_shape_str)

    return (1, 1)  # Default: single-chip


def group_vectors_by_mesh_shape(vectors):
    """Group vectors by their mesh_device_shape.

    Args:
        vectors: List of vector dictionaries

    Returns:
        dict: Mapping of mesh_shape tuple to list of vectors with that mesh shape
    """

    grouped = defaultdict(list)
    for vector in vectors:
        mesh_shape = get_mesh_shape_from_vector(vector)
        grouped[mesh_shape].append(vector)

    return grouped


# Generate vectors from module parameters
def generate_vectors(module_name, model_traced, suite_name=None):
    # Configure MasterConfigLoader filter BEFORE importing sweep modules
    # This replaces the previous environment variable approach for cleaner control
    MasterConfigLoader.set_lead_models_filter(model_traced == "lead")

    # Import or reload the module to pick up the filter setting
    # Note: Reload is still needed because sweep modules define parameters at import time
    module_path = "sweeps." + module_name
    if module_path in sys.modules:
        # Force reload if module was already imported with different filter setting
        test_module = importlib.reload(sys.modules[module_path])
    else:
        test_module = importlib.import_module(module_path)

    parameters = test_module.parameters

    for suite in parameters:
        # Skip suite if suite_name filter is specified and doesn't match
        if suite_name and suite != suite_name:
            logger.info(f"Skipping suite {suite} (filtering to {suite_name}).")
            continue

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


def _serialize_vectors(vectors):
    """Serialize vectors and compute their hashes.

    Args:
        vectors: List of vector dictionaries to serialize

    Returns:
        dict: Dictionary mapping input_hash to serialized vector
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    serialized_vectors = dict()
    warnings = []

    for i in range(len(vectors)):
        vector = dict()
        for elem in vectors[i].keys():
            vector[elem] = serialize_structured(vectors[i][elem], warnings)
        input_hash = _compute_vector_hash(vector)
        vector["timestamp"] = current_time
        vector["input_hash"] = input_hash
        vector["tag"] = SWEEPS_TAG
        serialized_vectors[input_hash] = vector

    return serialized_vectors


def _compute_vector_hash(vector):
    """Compute SHA224 hash of a serialized vector.

    Args:
        vector: Dictionary representing a vector

    Returns:
        str: Hexadecimal hash string
    """
    try:
        # Deterministic JSON serialization for stable hashing
        hash_input = json.dumps(vector, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        # Fallback if non-serializable objects are present
        hash_input = str(vector)
    return hashlib.sha224(hash_input.encode("utf-8")).hexdigest()


def _backup_corrupted_json_file(path: pathlib.Path) -> None:
    """Rename a corrupted JSON file to a timestamped backup for inspection."""
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_suffix(path.suffix + f".corrupted.{ts}")
        path.rename(backup_path)
        logger.warning(f"Backed up corrupted JSON file from {path} to {backup_path}")
    except OSError as e:
        logger.warning(f"Failed to back up corrupted JSON file {path}: {e}")


def validate_exported_vectors(export_path, module_name, suite_name):
    """Validate that exported JSON file can be read back correctly.

    Args:
        export_path: Path to the exported JSON file
        module_name: Name of the module
        suite_name: Name of the suite

    Returns:
        bool: True if validation succeeds, False otherwise
    """
    try:
        if not export_path.exists():
            logger.warning(f"Validation failed: export file does not exist at {export_path}")
            return False

        with open(export_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            logger.warning(f"Validation failed: exported file is not a dictionary")
            return False

        if suite_name not in data:
            logger.warning(f"Validation failed: suite '{suite_name}' not found in exported file")
            return False

        suite_data = data[suite_name]
        if not isinstance(suite_data, dict):
            logger.warning(f"Validation failed: suite data is not a dictionary")
            return False

        # Check that vectors have required fields
        for vector_id, vector_data in suite_data.items():
            if not isinstance(vector_data, dict):
                logger.warning(f"Validation failed: vector {vector_id} is not a dictionary")
                return False
            required_fields = ["input_hash", "timestamp", "tag"]
            for field in required_fields:
                if field not in vector_data:
                    logger.warning(f"Validation failed: vector {vector_id} missing required field '{field}'")
                    return False

        return True
    except json.JSONDecodeError as e:
        logger.warning(f"Validation failed: JSON decode error - {e}")
        return False
    except IOError as e:
        logger.warning(f"Validation failed: IO error - {e}")
        return False
    except Exception as e:
        logger.warning(f"Validation failed: unexpected error - {e}")
        return False


def export_suite_vectors_json(module_name, suite_name, vectors):
    """Export test vectors to JSON files grouped by mesh shape with atomic writes and deduplication.

    Vectors are grouped by mesh_device_shape and written to separate files:
    - model_traced.op__mesh_2x4.json (for [2, 4] mesh)
    - model_traced.op__mesh_1x1.json (for single-chip)

    IMPORTANT: The mesh suffix is used ONLY for filename routing, NOT for modifying
    the sweep_name field. This ensures stable full_test_name and input_hash values
    for historical comparison in Superset dashboards. The mesh configuration is
    already captured in traced_machine_info within the vector data.

    Args:
        module_name: Name of the test module
        suite_name: Name of the test suite
        vectors: List of vector dictionaries to export
    """
    # Group vectors by mesh shape
    grouped_vectors = group_vectors_by_mesh_shape(vectors)

    # Export each mesh group to a separate file
    for mesh_shape, mesh_vectors in grouped_vectors.items():
        # Skip empty groups
        if not mesh_vectors:
            continue

        # Generate mesh-specific filename (NOT sweep_name)
        # The mesh suffix is for file routing only, not for modifying vector metadata
        mesh_suffix = format_mesh_suffix(mesh_shape)
        mesh_module_name = f"{module_name}{mesh_suffix}"

        # Export vectors WITHOUT modifying sweep_name
        # The mesh info is already in traced_machine_info; sweep_name stays stable
        # for historical comparison (full_test_name in Superset)
        _export_mesh_vectors_to_file(mesh_module_name, suite_name, mesh_vectors)


def _export_mesh_vectors_to_file(module_name, suite_name, vectors):
    """Internal function to export vectors for a specific mesh shape to JSON file.

    Args:
        module_name: Name including mesh suffix (e.g., 'model_traced.gelu__mesh_2x4')
        suite_name: Name of the test suite
        vectors: List of vector dictionaries for this mesh shape
    """
    EXPORT_DIR_PATH = SWEEPS_DIR / "vectors_export"
    EXPORT_PATH = EXPORT_DIR_PATH / f"{module_name}.json"

    # Create export directory with proper error handling
    try:
        EXPORT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        logger.error(f"Permission denied creating export directory {EXPORT_DIR_PATH}: {e}")
        raise
    except OSError as e:
        logger.error(f"Failed to create export directory {EXPORT_DIR_PATH}: {e}")
        raise

    # Randomize order only when explicitly requested via --randomize
    if DO_RANDOMIZE:
        rng = random.Random(SHUFFLE_SEED)
        rng.shuffle(vectors)

    # Serialize vectors
    serialized_vectors = _serialize_vectors(vectors)

    # Load existing data and check for deduplication
    existing_data = {}
    existing_hashes = set()

    if EXPORT_PATH.exists():
        try:
            with open(EXPORT_PATH, "r", encoding="utf-8") as file:
                existing_data = json.load(file)

            # Check if suite already exists and compare hashes
            if suite_name in existing_data:
                existing_suite_data = existing_data[suite_name]
                if isinstance(existing_suite_data, dict):
                    # Only consider hashes for vectors with the same tag to avoid cross-tag dedup collisions
                    existing_hashes = {
                        input_hash
                        for input_hash, vec in existing_suite_data.items()
                        if isinstance(vec, dict) and vec.get("tag") == SWEEPS_TAG
                    }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse existing JSON file {EXPORT_PATH}: {e}. Will overwrite after backup.")
            _backup_corrupted_json_file(EXPORT_PATH)
            existing_data = {}
        except IOError as e:
            logger.warning(f"Failed to read existing file {EXPORT_PATH}: {e}. Will overwrite after backup if possible.")
            try:
                _backup_corrupted_json_file(EXPORT_PATH)
            except Exception as e:
                logger.warning(f"Failed to backup corrupted JSON file {EXPORT_PATH}: {e}. Will overwrite after backup.")
            existing_data = {}
        except Exception as e:
            logger.warning(f"Unexpected error reading existing file {EXPORT_PATH}: {e}. Will overwrite after backup.")
            _backup_corrupted_json_file(EXPORT_PATH)
            existing_data = {}

    # Check for deduplication: skip write if vectors haven't changed
    new_hashes = set(serialized_vectors.keys())
    if existing_hashes == new_hashes:
        logger.info(
            f"Vectors generated for module {module_name}, suite {suite_name} already exist with tag {SWEEPS_TAG}, "
            f"and have not changed. ({len(existing_hashes)} existing tests). Skipping..."
        )
        return

    # Prepare data for atomic write
    data_to_write = existing_data.copy()
    data_to_write[suite_name] = serialized_vectors

    # Atomic write using temporary file
    tmp_path = EXPORT_PATH.with_suffix(EXPORT_PATH.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(data_to_write, file, indent=2)
            file.flush()
            try:
                os.fsync(file.fileno())
            except OSError:
                # fsync may fail on some systems, but file is still written
                pass

        # Atomic replace
        os.replace(tmp_path, EXPORT_PATH)

        # Validate the exported file
        if not validate_exported_vectors(EXPORT_PATH, module_name, suite_name):
            logger.warning(
                f"Validation failed after export. File was written to {EXPORT_PATH}. "
                f"If issues persist, delete the file and regenerate."
            )

        # Extract mesh shape from module name for logging
        if "__mesh_" in module_name:
            mesh_info = module_name.split("__mesh_")[1]
            logger.info(f"SWEEPS: Generated {len(vectors)} test vectors for suite {suite_name} (mesh {mesh_info}).")
        else:
            logger.info(f"SWEEPS: Generated {len(vectors)} test vectors for suite {suite_name}.")
    except (IOError, OSError) as e:
        logger.error(f"Failed to write vectors to {EXPORT_PATH}: {e}")
        raise
    finally:
        # Ensure temporary file is removed on success or failure
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                # It's safe to ignore errors when deleting the temporary file (file may not exist)
                pass


# Generate one or more sets of test vectors depending on module_name
def generate_tests(module_name, skip_modules=None, model_traced=None, suite_name=None):
    skip_modules_set = set()
    if skip_modules:
        skip_modules_set = {name.strip() for name in skip_modules.split(",")}
        logger.info(f"Skipping modules: {', '.join(skip_modules_set)}")

    if suite_name:
        logger.info(f"Filtering to suite: {suite_name}")

    if not module_name:
        # Determine which directory to search based on model_traced_only flag
        if model_traced is not None:
            search_dir = SWEEP_SOURCES_DIR / "model_traced"
            logger.info("Generating test vectors for model_traced operations only.")
            # Only search directly in model_traced directory, not subdirectories
            glob_pattern = "*.py"
        else:
            search_dir = SWEEP_SOURCES_DIR
            glob_pattern = "**/*.py"

        for file_name in sorted(search_dir.glob(glob_pattern)):
            module_name = str(pathlib.Path(file_name).relative_to(SWEEP_SOURCES_DIR))[:-3].replace("/", ".")
            if module_name in skip_modules_set:
                logger.info(f"Skipping module {module_name} (in skip list).")
                continue
            logger.info(f"Generating test vectors for module {module_name}.")
            try:
                generate_vectors(module_name, model_traced, suite_name)
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
            generate_vectors(module_name, model_traced, suite_name)
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
    parser.add_argument(
        "--model-traced",
        required=False,
        type=str,
        nargs="?",
        const="all",
        default=None,
        choices=["all", "lead"],
        help="Generate test vectors for model traced operations. Options: 'all' (default if flag provided) or 'lead' (only lead models like DeepSeek). Omit flag to generate all sweeps.",
    )
    parser.add_argument(
        "--suite-name",
        required=False,
        type=str,
        help="Generate vectors for a specific suite only (e.g., 'nightly', 'model_traced'). Omit to generate all suites.",
    )

    args = parser.parse_args(sys.argv[1:])

    global SWEEPS_TAG
    SWEEPS_TAG = args.tag

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

    generate_tests(args.module_name, args.skip_modules, args.model_traced, args.suite_name)
