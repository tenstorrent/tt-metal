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
from collections import defaultdict

from framework.constants import format_hardware_suffix, format_mesh_suffix, normalize_hardware_group
from framework.permutations import permutations
from framework.serialize import serialize_structured
from framework.statuses import VectorStatus, VectorValidity
from framework.sweeps_logger import sweeps_logger as logger
from framework.vector_metadata import (
    get_hardware_from_vector,
    get_mesh_shape_from_vector,
    normalize_mesh_shape_for_manifest,
)

SWEEPS_DIR = pathlib.Path(__file__).parent
SWEEP_SOURCES_DIR = SWEEPS_DIR / "sweeps"
REPO_ROOT = SWEEPS_DIR.parent.parent
EXPORT_DIR_PATH = SWEEPS_DIR / "vectors_export"
EXPORT_MANIFEST_PATH = EXPORT_DIR_PATH / "export_manifest.json"


# Shuffle control (set in __main__ when --randomize is provided)
SHUFFLE_SEED = None
DO_RANDOMIZE = False
VECTOR_GROUPING_MODE = "mesh"
SWEEPS_TAG = os.getenv("USER")
EXPORT_MANIFEST_STATE = None


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


def group_vectors_by_hardware(vectors):
    """Group vectors by their traced hardware tuple."""
    grouped = defaultdict(list)
    for vector in vectors:
        hardware = get_hardware_from_vector(vector)
        grouped[hardware].append(vector)

    return grouped


def _collect_mesh_shapes_from_vectors(vectors):
    """Unique mesh shapes inferred from traced_machine_info (same logic as grouping)."""
    shapes = set()
    for vector in vectors:
        mesh_shape = get_mesh_shape_from_vector(vector)
        if mesh_shape is None:
            continue
        normalized = normalize_mesh_shape_for_manifest(mesh_shape)
        if normalized is not None:
            shapes.add(tuple(normalized))
    return [[rows, cols] for rows, cols in sorted(shapes)]


def _normalize_hardware_for_manifest(hardware_group):
    """Normalize hardware routing metadata to a stable manifest shape."""
    if hardware_group is None:
        return None

    board_type, device_series, card_count = hardware_group
    board_type, device_series, card_count = normalize_hardware_group(board_type, device_series, card_count)
    return {
        "board_type": board_type,
        "device_series": device_series,
        "card_count": card_count,
    }


def _collect_trace_ids(vectors):
    """Collect any trace IDs present in generated vectors."""
    trace_ids = set()
    for vector in vectors:
        value = vector.get("trace_ids", vector.get("trace_id"))
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple, set)) else [value]
        for item in values:
            try:
                trace_ids.add(int(item))
            except (TypeError, ValueError):
                continue
    return sorted(trace_ids)


def initialize_export_manifest(
    *,
    tag,
    group_by,
    module_name_filter,
    suite_name_filter,
    skip_modules,
    model_traced,
    randomize_enabled,
    randomize_seed,
    use_db,
    mesh_shape_filter,
    master_trace,
):
    """Initialize the per-run export manifest state."""
    global EXPORT_MANIFEST_STATE
    EXPORT_MANIFEST_STATE = {
        "generation": {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "tag": tag,
            "group_by": group_by,
            "module_name_filter": module_name_filter,
            "suite_name_filter": suite_name_filter,
            "skip_modules": sorted(skip_modules),
            "model_traced": model_traced,
            "randomize": {"enabled": randomize_enabled, "seed": randomize_seed},
            "use_db": use_db,
            "mesh_shape_filter": mesh_shape_filter,
            "master_trace": master_trace,
        },
        "files": {},
    }


def _require_export_manifest_state():
    """Return the initialized export manifest state."""
    if EXPORT_MANIFEST_STATE is None:
        raise RuntimeError("Export manifest state was not initialized before vector export.")
    return EXPORT_MANIFEST_STATE


def _record_export_manifest_entry(
    *,
    module_name,
    base_module_name,
    suite_name,
    vectors,
    serialized_vectors,
    export_path,
    grouping_kind,
    group_key,
):
    """Record manifest metadata for a grouped vector export."""
    state = _require_export_manifest_state()
    entries = state["files"]

    entry = entries.setdefault(
        module_name,
        {
            "module_name": module_name,
            "base_module_name": base_module_name,
            "file_path": str(export_path.relative_to(REPO_ROOT)),
            "grouping_kind": grouping_kind,
            "hardware_group": None,
            "mesh_shapes": [],
            "suite_names": [],
            "vector_count": 0,
            "input_hash_count": 0,
            "trace_ids": [],
        },
    )

    if grouping_kind == "hw":
        entry["hardware_group"] = _normalize_hardware_for_manifest(group_key)
    elif grouping_kind == "mesh":
        normalized = normalize_mesh_shape_for_manifest(group_key)
        if normalized is not None:
            entry["mesh_shapes"] = [normalized]

    batch_mesh_shapes = _collect_mesh_shapes_from_vectors(vectors)
    if batch_mesh_shapes:
        existing = {tuple(pair) for pair in entry["mesh_shapes"]}
        existing.update(tuple(pair) for pair in batch_mesh_shapes)
        entry["mesh_shapes"] = [[r, c] for r, c in sorted(existing)]

    if suite_name not in entry["suite_names"]:
        entry["suite_names"].append(suite_name)
        entry["suite_names"].sort()

    entry["vector_count"] += len(vectors)
    entry["input_hash_count"] += len(serialized_vectors)
    entry["trace_ids"] = sorted(set(entry["trace_ids"]).union(_collect_trace_ids(vectors)))


def write_export_manifest():
    """Persist the export manifest. This is required for successful generation."""
    state = _require_export_manifest_state()
    EXPORT_DIR_PATH.mkdir(parents=True, exist_ok=True)

    files = sorted(state["files"].values(), key=lambda entry: entry["module_name"])
    manifest = {
        "generation": state["generation"],
        "file_count": len(files),
        "files": files,
    }

    tmp_path = EXPORT_MANIFEST_PATH.with_suffix(EXPORT_MANIFEST_PATH.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
            file.flush()
            try:
                os.fsync(file.fileno())
            except OSError as e:
                logger.warning("Failed to fsync export manifest temp file %s: %s", tmp_path, e)
        os.replace(tmp_path, EXPORT_MANIFEST_PATH)
    except (IOError, OSError) as e:
        raise RuntimeError(f"Failed to write export manifest to {EXPORT_MANIFEST_PATH}: {e}") from e
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# Generate vectors for each suite in test module parameters
def generate_vectors(module_name, model_traced, suite_name=None):
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

    If a vector has a config_hash from the database, use that as the input_hash
    for direct correlation with ttnn_ops.ttnn_configuration. Otherwise, compute
    a hash from the serialized vector content.

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

        # Use config_hash from database if available, otherwise compute from content
        # This enables direct correlation with ttnn_ops.ttnn_configuration.config_hash
        config_hash = vectors[i].get("config_hash")
        if config_hash:
            input_hash = config_hash
        else:
            input_hash = _compute_vector_hash(vector)

        vector["timestamp"] = current_time
        vector["input_hash"] = input_hash
        vector["tag"] = SWEEPS_TAG
        vector = _normalize_serialized_vector_metadata(vector)
        if input_hash in serialized_vectors:
            serialized_vectors[input_hash] = _merge_duplicate_serialized_vectors(serialized_vectors[input_hash], vector)
        else:
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


def _normalize_metadata_list(value):
    """Normalize metadata fields to a list while preserving order."""
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _dedupe_metadata_items(items):
    """Deduplicate metadata items deterministically."""
    deduped = []
    seen = set()

    for item in items:
        try:
            key = json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            key = repr(item)

        if key in seen:
            continue

        seen.add(key)
        deduped.append(item)

    return deduped


def _merge_duplicate_serialized_vectors(existing_vector, new_vector):
    """Merge metadata for vectors that share the same config_hash/input_hash."""
    merged = dict(existing_vector)

    merged_sources = _dedupe_metadata_items(
        _normalize_metadata_list(existing_vector.get("traced_source"))
        + _normalize_metadata_list(new_vector.get("traced_source"))
    )
    if merged_sources:
        merged["traced_source"] = merged_sources[0] if len(merged_sources) == 1 else merged_sources

    merged_machine_info = _dedupe_metadata_items(
        _normalize_metadata_list(existing_vector.get("traced_machine_info"))
        + _normalize_metadata_list(new_vector.get("traced_machine_info"))
    )
    if merged_machine_info:
        merged["traced_machine_info"] = merged_machine_info[0] if len(merged_machine_info) == 1 else merged_machine_info

    return merged


def _normalize_serialized_vector_metadata(vector):
    """Normalize exported metadata fields to stable shapes."""
    normalized = dict(vector)
    normalized["traced_source"] = _normalize_metadata_list(vector.get("traced_source"))
    return normalized


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
    """Group suite test vectors by routing target and export each group to its JSON file.

    Supported grouping modes:
    - mesh: vectors are grouped by mesh_device_shape and written to files like
      model_traced.op.mesh_2x4.json
    - hw: vectors are grouped by traced hardware and written to files like
      model_traced.op.hw_wormhole_n300_1c.json

    IMPORTANT: The grouping suffix is used ONLY for filename routing, NOT for
    modifying the sweep_name field. This ensures stable full_test_name and
    input_hash values for historical comparison in Superset dashboards. The
    routing metadata is already captured in traced_machine_info within the vector data.

    Args:
        module_name: Name of the test module
        suite_name: Name of the test suite
        vectors: List of vector dictionaries to export
    """
    if VECTOR_GROUPING_MODE == "hw":
        grouped_vectors = group_vectors_by_hardware(vectors)
        format_group_suffix = lambda group_key: format_hardware_suffix(*group_key)
    else:
        grouped_vectors = group_vectors_by_mesh_shape(vectors)
        format_group_suffix = format_mesh_suffix

    # Export each group to a separate file
    for group_key, grouped_subset in grouped_vectors.items():
        if not grouped_subset:
            continue

        # A None group means the vector has no routing restriction, so keep the
        # base module name and let any compatible runner pick up the file.
        grouped_module_name = module_name if group_key is None else f"{module_name}{format_group_suffix(group_key)}"
        grouping_kind = "ungrouped" if group_key is None else VECTOR_GROUPING_MODE

        # Export vectors WITHOUT modifying sweep_name.
        # Routing info is already present in traced_machine_info; sweep_name stays
        # stable for historical comparison (full_test_name in Superset).
        _export_grouped_vectors_to_file(
            grouped_module_name,
            suite_name,
            grouped_subset,
            base_module_name=module_name,
            grouping_kind=grouping_kind,
            group_key=group_key,
        )


def _export_grouped_vectors_to_file(module_name, suite_name, vectors, *, base_module_name, grouping_kind, group_key):
    """Export one grouped vector file with deduplication and atomic writes.

    Args:
        module_name: Name including any grouping suffix
        suite_name: Name of the test suite
        vectors: List of vector dictionaries for this group
    """
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
        _record_export_manifest_entry(
            module_name=module_name,
            base_module_name=base_module_name,
            suite_name=suite_name,
            vectors=vectors,
            serialized_vectors=serialized_vectors,
            export_path=EXPORT_PATH,
            grouping_kind=grouping_kind,
            group_key=group_key,
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

        _record_export_manifest_entry(
            module_name=module_name,
            base_module_name=base_module_name,
            suite_name=suite_name,
            vectors=vectors,
            serialized_vectors=serialized_vectors,
            export_path=EXPORT_PATH,
            grouping_kind=grouping_kind,
            group_key=group_key,
        )

        # Extract grouping suffix from module name for logging.
        if ".mesh_" in module_name:
            mesh_info = module_name.split(".mesh_")[1]
            logger.info(f"SWEEPS: Generated {len(vectors)} test vectors for suite {suite_name} (mesh {mesh_info}).")
        elif ".hw_" in module_name:
            hardware_info = module_name.split(".hw_")[1]
            logger.info(
                f"SWEEPS: Generated {len(vectors)} test vectors for suite {suite_name} (hardware {hardware_info})."
            )
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


# Generate vectors for selected modules
def generate_tests(module_name, skip_modules=None, model_traced=None, suite_name=None):
    skip_modules_set = set()
    if skip_modules:
        skip_modules_set = {name.strip() for name in skip_modules.split(",")}
        logger.info(f"Skipping modules: {', '.join(skip_modules_set)}")

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
            write_export_manifest()
            logger.info(f"Wrote export manifest to {EXPORT_MANIFEST_PATH}")
            return
        logger.info(f"Generating test vectors for module {module_name}.")
        try:
            generate_vectors(module_name, model_traced, suite_name)
        except Exception as e:
            logger.error(f"Failed to generate vectors for module {module_name}: {e}")
            raise

    write_export_manifest()
    logger.info(f"Wrote export manifest to {EXPORT_MANIFEST_PATH}")


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
    parser.add_argument(
        "--use-db",
        action="store_true",
        help="Load configurations from PostgreSQL database instead of JSON file. Requires TTNN_OPS_DATABASE_URL or POSTGRES_* environment variables.",
    )
    parser.add_argument(
        "--mesh-shape",
        required=False,
        type=str,
        help="Filter configurations to specific mesh shape (e.g., '2x4', '1x1').",
    )
    parser.add_argument(
        "--master-trace",
        required=False,
        type=str,
        help="Path to a master trace JSON file (e.g., ttnn_operations_master_*.json). "
        "Overrides the default file resolution in MasterConfigLoader.",
    )
    parser.add_argument(
        "--group-by",
        required=False,
        type=str,
        default="mesh",
        choices=["mesh", "hw"],
        help="Group exported vector files by 'mesh' (default) or 'hw'.",
    )

    args = parser.parse_args(sys.argv[1:])

    # Set environment variable before importing MasterConfigLoader or any sweep module
    if args.model_traced == "lead":
        os.environ["TTNN_LEAD_MODELS_ONLY"] = "1"
        logger.info("=" * 80)
        logger.info("Lead models filter enabled.")
        logger.info(
            "Set TTNN_LEAD_MODELS_ONLY={} for tests.sweep_framework.master_config_loader_v2 "
            "to filter traced configs to lead-model sources from model_tracer/sweep_manifest.yaml",
            os.environ.get("TTNN_LEAD_MODELS_ONLY"),
        )
        logger.info("=" * 80)
    else:
        os.environ["TTNN_LEAD_MODELS_ONLY"] = "0"

    SWEEPS_TAG = args.tag
    VECTOR_GROUPING_MODE = args.group_by

    logger.info(f"Running current generation with tag: {SWEEPS_TAG}.")
    logger.info("Vectors will be exported to: tests/sweep_framework/vectors_export/")
    logger.info(f"Vector export grouping mode: {VECTOR_GROUPING_MODE}")

    # Enable reproducible shuffling only when --randomize is provided
    if args.randomize is not None:
        SHUFFLE_SEED = int(args.randomize)
        DO_RANDOMIZE = True
        logger.info(f"Randomize seed: {SHUFFLE_SEED}")
    else:
        DO_RANDOMIZE = False
        SHUFFLE_SEED = None

    # Import MasterConfigLoader only after TTNN_LEAD_MODELS_ONLY is set.
    # This module snapshots the env var into shared filter state at import time,
    # and sweep modules use that state immediately when they build parameters on import.
    from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader

    # Configure database mode if --use-db flag is provided
    # DEPRECATE TO REMOVE COMPLEXITY. ALWAYS RELY ON PRE-GENERATION OF CONFIGS.
    if args.use_db:
        MasterConfigLoader.set_database_mode(True)
        logger.info("Database mode enabled: Loading configurations from PostgreSQL")

    # Configure mesh filter if --mesh-shape is provided
    if args.mesh_shape:
        try:
            rows, cols = map(int, args.mesh_shape.lower().split("x"))
            MasterConfigLoader.set_mesh_filter((rows, cols))
            logger.info(f"Mesh filter enabled: {rows}x{cols}")
        except ValueError:
            logger.error(f"Invalid mesh shape format: {args.mesh_shape}. Use format like '2x4' or '1x1'.")
            sys.exit(1)

    if args.master_trace:
        resolved = os.path.abspath(args.master_trace)
        MasterConfigLoader.set_master_file_path(resolved)
        logger.info(f"Master trace override: {resolved}")
    else:
        resolved = None

    skip_modules_set = set()
    if args.skip_modules:
        skip_modules_set = {name.strip() for name in args.skip_modules.split(",")}

    initialize_export_manifest(
        tag=SWEEPS_TAG,
        group_by=VECTOR_GROUPING_MODE,
        module_name_filter=args.module_name,
        suite_name_filter=args.suite_name,
        skip_modules=skip_modules_set,
        model_traced=args.model_traced,
        randomize_enabled=DO_RANDOMIZE,
        randomize_seed=SHUFFLE_SEED,
        use_db=args.use_db,
        mesh_shape_filter=args.mesh_shape,
        master_trace=resolved,
    )

    generate_tests(args.module_name, args.skip_modules, args.model_traced, args.suite_name)
