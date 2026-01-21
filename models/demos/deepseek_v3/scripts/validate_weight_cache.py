# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from models.demos.deepseek_v3.utils.config_dataclass import SavedWeight
from models.demos.deepseek_v3.utils.config_helpers import TENSOR_CACHE_EXTENSION
from models.demos.deepseek_v3.utils.weight_config import (
    WeightConfigEncoder,
    locked_file,
    try_decode_saved_weight,
    validate_weight_config_paths,
)

# Regex patterns for cache directory structure
CACHE_DIR_PATTERN = re.compile(r"(\d+)_layers/mesh_(\d+)x(\d+)$")


def discover_cache_directories(root_path: Path) -> list[Path]:
    """
    Find all cache directories by their structure pattern.

    Looks for directories matching the pattern: */{num_layers}_layers/mesh_{rows}x{cols}

    Args:
        root_path: Root directory to search recursively

    Returns:
        List of cache directory paths matching the mesh pattern
    """
    cache_dirs = []

    for mesh_dir in root_path.rglob("mesh_*"):
        # Check if it's a directory and matches expected pattern
        if mesh_dir.is_dir():
            path_str = str(mesh_dir)
            if CACHE_DIR_PATTERN.search(path_str):
                cache_dirs.append(mesh_dir)

    return sorted(cache_dirs)


def extract_cache_info(cache_dir: Path) -> dict[str, Any] | None:
    """
    Extract cache information from directory path.

    Args:
        cache_dir: Cache directory path

    Returns:
        Dictionary with num_layers, mesh_rows, mesh_cols, or None if pattern doesn't match
    """
    path_str = str(cache_dir)
    match = CACHE_DIR_PATTERN.search(path_str)
    if match:
        return {
            "num_layers": int(match.group(1)),
            "mesh_rows": int(match.group(2)),
            "mesh_cols": int(match.group(3)),
        }
    return None


def collect_saved_weights(weight_config: Any, path_prefix: str = "") -> list[tuple[str, SavedWeight]]:
    """
    Recursively collect all SavedWeight objects from a weight config.

    Args:
        weight_config: Weight configuration (dict, list, tuple, or nested structures)
        path_prefix: Prefix for path identification

    Returns:
        List of (path_key, SavedWeight) tuples
    """
    saved_weights = []

    if isinstance(weight_config, dict):
        entries = weight_config.items()
    elif isinstance(weight_config, (list, tuple)):
        entries = enumerate(weight_config)
    else:
        return saved_weights

    for key, entry in entries:
        if entry is None:
            continue
        current_prefix = f"{path_prefix}.{key}" if path_prefix else str(key)

        if isinstance(entry, SavedWeight):
            saved_weights.append((current_prefix, entry))
        else:
            # Recursively collect from nested structures
            saved_weights.extend(collect_saved_weights(entry, current_prefix))

    return saved_weights


def repair_absolute_paths(cache_dir: Path, weight_config: Any) -> tuple[Any, int]:
    """
    Repair absolute paths in weight config by converting them to relative paths.

    Args:
        cache_dir: Cache directory path (base for relative paths)
        weight_config: Weight configuration (dict, list, tuple, or nested structures)

    Returns:
        Tuple of (repaired_weight_config, num_repaired) where num_repaired is the count of fixed paths
    """
    if isinstance(weight_config, dict):
        repaired = {}
        total_repaired = 0
        for key, value in weight_config.items():
            if value is None:
                repaired[key] = None
            else:
                repaired_value, count = repair_absolute_paths(cache_dir, value)
                repaired[key] = repaired_value
                total_repaired += count
        return repaired, total_repaired
    elif isinstance(weight_config, (list, tuple)):
        repaired = []
        total_repaired = 0
        for item in weight_config:
            if item is None:
                repaired.append(None)
            else:
                repaired_item, count = repair_absolute_paths(cache_dir, item)
                repaired.append(repaired_item)
                total_repaired += count
        # Preserve tuple type if input was a tuple
        return (tuple(repaired) if isinstance(weight_config, tuple) else repaired), total_repaired
    elif isinstance(weight_config, SavedWeight):
        if weight_config.path.is_absolute():
            # Validate that the absolute path is within the cache directory
            cache_dir_resolved = cache_dir.resolve()
            abs_path_resolved = weight_config.path.resolve()
            try:
                # Try to get relative path
                rel_path = abs_path_resolved.relative_to(cache_dir_resolved)
            except ValueError:
                # Path is not within cache_dir
                raise ValueError(f"Absolute path '{weight_config.path}' is not within cache directory '{cache_dir}'")
            # Verify the file exists
            if not abs_path_resolved.exists():
                raise ValueError(f"Absolute path points to non-existent file: {abs_path_resolved}")
            # Create new SavedWeight with relative path
            return SavedWeight(path=rel_path, memory_config=weight_config.memory_config), 1
        # Already relative, no repair needed
        return weight_config, 0
    else:
        # For other types (None, primitives, etc.), return as-is
        return weight_config, 0


def find_orphaned_files(cache_dir: Path, referenced_paths: set[Path]) -> list[Path]:
    """
    Find tensorbin files in cache directory that are not referenced in config.

    Args:
        cache_dir: Cache directory to search
        referenced_paths: Set of absolute paths that are referenced in config

    Returns:
        List of orphaned file paths
    """
    orphaned = []
    weights_dir = cache_dir / "weights"

    if weights_dir.exists():
        for file_path in weights_dir.rglob(f"*{TENSOR_CACHE_EXTENSION}"):
            resolved_path = file_path.resolve()
            if resolved_path not in referenced_paths:
                orphaned.append(file_path)

    return sorted(orphaned)


def validate_cache_directory(cache_dir: Path, verbose: bool = False) -> dict[str, Any]:
    """
    Validate a single cache directory and return statistics.

    Args:
        cache_dir: Cache directory path
        verbose: Whether to include detailed error information

    Returns:
        Dictionary with validation results and statistics
    """
    config_path = cache_dir / "config.json"
    cache_info = extract_cache_info(cache_dir)

    result = {
        "cache_dir": cache_dir,
        "cache_info": cache_info,
        "config_exists": config_path.exists(),
        "valid": False,
        "errors": [],
        "warnings": [],
        "orphaned_files": [],
        "stats": {
            "total_files": 0,
            "total_size": 0,
            "absolute_paths": 0,
            "missing_files": 0,
            "invalid_extensions": 0,
            "orphaned_files": 0,
        },
    }

    if not result["config_exists"]:
        result["errors"].append("config.json not found")
        return result

    # Load and decode config (with shared lock to prevent reading during writes)
    try:
        with locked_file(config_path, "r", exclusive=False) as f:
            weight_config = json.load(f, object_hook=try_decode_saved_weight)
    except Exception as e:
        result["errors"].append(f"Failed to load config.json: {e}")
        return result

    # Collect all SavedWeight objects
    saved_weights = collect_saved_weights(weight_config)
    result["stats"]["total_files"] = len(saved_weights)

    # Validate paths
    validation_errors = []
    referenced_paths = set()

    for path_key, saved_weight in saved_weights:
        # Check for absolute paths
        if saved_weight.path.is_absolute():
            result["stats"]["absolute_paths"] += 1
            error_msg = f"Absolute path at '{path_key}': {saved_weight.path}"
            validation_errors.append(error_msg)
            if verbose:
                result["errors"].append(error_msg)

        # Check extension
        if saved_weight.path.suffix != TENSOR_CACHE_EXTENSION:
            result["stats"]["invalid_extensions"] += 1
            error_msg = (
                f"Invalid extension at '{path_key}': {saved_weight.path.suffix} (expected {TENSOR_CACHE_EXTENSION})"
            )
            validation_errors.append(error_msg)
            if verbose:
                result["errors"].append(error_msg)

        # Resolve and check file existence
        effective_path = cache_dir / saved_weight.path
        referenced_paths.add(effective_path.resolve())

        if not effective_path.exists():
            result["stats"]["missing_files"] += 1
            error_msg = f"Missing file at '{path_key}': {effective_path} (original: {saved_weight.path})"
            validation_errors.append(error_msg)
            if verbose:
                result["errors"].append(error_msg)
        else:
            # Calculate file size
            try:
                result["stats"]["total_size"] += effective_path.stat().st_size
            except OSError as e:
                if verbose:
                    result["warnings"].append(f"Could not stat '{path_key}' at {effective_path}: {e}")

    # Use existing validation function to catch any additional issues
    try:
        validate_weight_config_paths(cache_dir, weight_config)
    except ValueError as e:
        # Only add if not already captured above
        error_str = str(e)
        if error_str not in validation_errors:
            validation_errors.append(error_str)
            if verbose:
                result["errors"].append(error_str)

    # Find orphaned files
    orphaned = find_orphaned_files(cache_dir, referenced_paths)
    result["stats"]["orphaned_files"] = len(orphaned)
    result["orphaned_files"] = orphaned  # Store for later display
    if orphaned and verbose:
        result["warnings"].extend([f"Orphaned file: {f}" for f in orphaned])

    # Determine if cache is valid
    result["valid"] = (
        len(validation_errors) == 0
        and result["stats"]["missing_files"] == 0
        and result["stats"]["absolute_paths"] == 0
        and result["stats"]["invalid_extensions"] == 0
    )

    if not result["valid"] and not verbose:
        # Collect error summaries even if not verbose
        if result["stats"]["absolute_paths"] > 0:
            result["errors"].append(f"{result['stats']['absolute_paths']} absolute path(s) found")
        if result["stats"]["missing_files"] > 0:
            result["errors"].append(f"{result['stats']['missing_files']} missing file(s)")
        if result["stats"]["invalid_extensions"] > 0:
            result["errors"].append(f"{result['stats']['invalid_extensions']} invalid extension(s)")

    return result


def repair_cache_directory(cache_dir: Path, verbose: bool = False) -> dict[str, Any]:
    """
    Repair absolute paths in a cache directory's config.json.

    Args:
        cache_dir: Cache directory path
        verbose: Whether to print detailed information

    Returns:
        Dictionary with repair results
    """
    config_path = cache_dir / "config.json"
    result = {
        "cache_dir": cache_dir,
        "repaired": False,
        "num_repaired": 0,
        "errors": [],
    }

    if not config_path.exists():
        result["errors"].append("config.json not found")
        return result

    # Load and decode config (with exclusive lock for write)
    try:
        with locked_file(config_path, "r", exclusive=False) as f:
            weight_config = json.load(f, object_hook=try_decode_saved_weight)
    except Exception as e:
        result["errors"].append(f"Failed to load config.json: {e}")
        return result

    # Repair absolute paths
    try:
        repaired_config, num_repaired = repair_absolute_paths(cache_dir, weight_config)
        result["num_repaired"] = num_repaired

        if num_repaired > 0:
            # Write repaired config back to disk (with exclusive lock)
            with locked_file(config_path, "w", exclusive=True) as f:
                json.dump(repaired_config, f, cls=WeightConfigEncoder, indent=2)
            result["repaired"] = True
            if verbose:
                print(f"Repaired {num_repaired} absolute path(s) in {cache_dir}")
    except Exception as e:
        result["errors"].append(f"Failed to repair paths: {e}")
        return result

    return result


def collect_cache_statistics(cache_dirs: list[Path], verbose: bool = False) -> dict[str, Any]:
    """
    Collect comprehensive statistics across all caches.

    Args:
        cache_dirs: List of cache directory paths
        verbose: Whether to include detailed error information

    Returns:
        Dictionary with aggregated statistics
    """
    results = []
    total_stats = {
        "total_caches": len(cache_dirs),
        "valid_caches": 0,
        "invalid_caches": 0,
        "total_files": 0,
        "total_size": 0,
        "total_absolute_paths": 0,
        "total_missing_files": 0,
        "total_invalid_extensions": 0,
        "total_orphaned_files": 0,
    }

    for cache_dir in cache_dirs:
        result = validate_cache_directory(cache_dir, verbose)
        results.append(result)

        # Aggregate statistics
        stats = result["stats"]
        total_stats["total_files"] += stats["total_files"]
        total_stats["total_size"] += stats["total_size"]
        total_stats["total_absolute_paths"] += stats["absolute_paths"]
        total_stats["total_missing_files"] += stats["missing_files"]
        total_stats["total_invalid_extensions"] += stats["invalid_extensions"]
        total_stats["total_orphaned_files"] += stats["orphaned_files"]

        if result["valid"]:
            total_stats["valid_caches"] += 1
        else:
            total_stats["invalid_caches"] += 1

    return {
        "summary": total_stats,
        "results": results,
    }


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def print_summary(stats: dict[str, Any]) -> None:
    """
    Print detailed text summary.

    Args:
        stats: Statistics dictionary from collect_cache_statistics
    """
    summary = stats["summary"]
    results = stats["results"]

    print("=" * 80)
    print("WEIGHT CACHE VALIDATION SUMMARY")
    print("=" * 80)
    print()

    # Summary section
    print("OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total cache directories:     {summary['total_caches']}")
    print(f"  Valid:                    {summary['valid_caches']}")
    print(f"  Invalid:                  {summary['invalid_caches']}")
    print(f"Total weight files:          {summary['total_files']}")
    print(f"Total cache size:            {format_size(summary['total_size'])}")
    print()

    # Error summary
    if (
        summary["total_absolute_paths"] > 0
        or summary["total_missing_files"] > 0
        or summary["total_invalid_extensions"] > 0
    ):
        print("ERROR SUMMARY")
        print("-" * 80)
        if summary["total_absolute_paths"] > 0:
            print(f"  Absolute paths found:      {summary['total_absolute_paths']}")
        if summary["total_missing_files"] > 0:
            print(f"  Missing files:             {summary['total_missing_files']}")
        if summary["total_invalid_extensions"] > 0:
            print(f"  Invalid extensions:        {summary['total_invalid_extensions']}")
        print()

    # Warnings
    if summary["total_orphaned_files"] > 0:
        print("WARNINGS")
        print("-" * 80)
        print(f"  Orphaned files:            {summary['total_orphaned_files']}")
        print()

    # Per-cache breakdown
    print("PER-CACHE BREAKDOWN")
    print("-" * 80)
    for result in results:
        cache_dir = result["cache_dir"]
        cache_info = result["cache_info"]
        stats = result["stats"]
        status = "✓ VALID" if result["valid"] else "✗ INVALID"

        if cache_info:
            info_str = f"{cache_info['num_layers']} layers, mesh {cache_info['mesh_rows']}x{cache_info['mesh_cols']}"
        else:
            info_str = "unknown structure"

        print(f"{status}  {cache_dir}")
        print(f"      {info_str}")
        print(f"      Files: {stats['total_files']}, Size: {format_size(stats['total_size'])}")

        if not result["valid"]:
            if result["errors"]:
                print(f"      Errors:")
                for error in result["errors"][:5]:  # Limit to 5 errors per cache
                    print(f"        - {error}")
                if len(result["errors"]) > 5:
                    print(f"        ... and {len(result['errors']) - 5} more error(s)")

        if result["warnings"]:
            print(f"      Warnings:")
            for warning in result["warnings"][:3]:  # Limit to 3 warnings per cache
                print(f"        - {warning}")
            if len(result["warnings"]) > 3:
                print(f"        ... and {len(result['warnings']) - 3} more warning(s)")
        print()

    # Detailed error section
    invalid_results = [r for r in results if not r["valid"]]
    if invalid_results:
        print("DETAILED ERROR INFORMATION")
        print("-" * 80)
        for result in invalid_results:
            if result["errors"]:
                print(f"\n{result['cache_dir']}:")
                for error in result["errors"]:
                    print(f"  ERROR: {error}")
        print()

    # Orphaned files section
    results_with_orphans = [r for r in results if r["stats"]["orphaned_files"] > 0]
    if results_with_orphans:
        print("ORPHANED FILES")
        print("-" * 80)
        for result in results_with_orphans:
            cache_dir = result["cache_dir"]
            orphaned = result.get("orphaned_files", [])
            if orphaned:
                print(f"\n{cache_dir}:")
                for orphan in orphaned:
                    print(f"  {orphan}")
        print()

    print("=" * 80)


def main():
    """Main entry point with argparse."""
    parser = argparse.ArgumentParser(description="Validate weight cache directories and print a comprehensive summary.")

    # Create mutually exclusive group for cache discovery methods
    discovery_group = parser.add_mutually_exclusive_group()
    discovery_group.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory to search for cache directories (default: current directory or DEEPSEEK_V3_CACHE env var)",
    )
    discovery_group.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Validate a single cache directory (directory containing config.json). Mutually exclusive with --root.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed error information for each cache",
    )
    parser.add_argument(
        "--repair-absolute-paths",
        action="store_true",
        help="Repair absolute paths by converting them to relative paths in config.json files",
    )

    args = parser.parse_args()

    # Handle single cache directory mode
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)

        if not cache_dir.exists():
            print(f"Error: Cache directory does not exist: {cache_dir}")
            return 1

        if not cache_dir.is_dir():
            print(f"Error: Path is not a directory: {cache_dir}")
            return 1

        config_path = cache_dir / "config.json"
        if not config_path.exists():
            print(f"Error: config.json not found in directory: {cache_dir}")
            return 1

        # Use the single directory
        cache_dirs = [cache_dir]
        print(f"Validating cache directory: {cache_dir}")
        print()
    else:
        # Recursive search mode
        # Determine root path
        if args.root:
            root_path = Path(args.root)
        else:
            cache_env = os.getenv("DEEPSEEK_V3_CACHE")
            if cache_env:
                root_path = Path(cache_env)
            else:
                root_path = Path.cwd()

        if not root_path.exists():
            print(f"Error: Root path does not exist: {root_path}")
            return 1

        print(f"Searching for cache directories in: {root_path}")
        print()

        # Discover cache directories
        cache_dirs = discover_cache_directories(root_path)

        if not cache_dirs:
            print("No cache directories found.")
            return 0

        print(f"Found {len(cache_dirs)} cache directory(ies)")
        print()

    # Repair absolute paths if requested
    if args.repair_absolute_paths:
        print("Repairing absolute paths...")
        print()
        total_repaired = 0
        repair_errors = []
        for cache_dir in cache_dirs:
            repair_result = repair_cache_directory(cache_dir, verbose=args.verbose)
            total_repaired += repair_result["num_repaired"]
            if repair_result["errors"]:
                repair_errors.extend([(cache_dir, error) for error in repair_result["errors"]])

        if repair_errors:
            print("REPAIR ERRORS")
            print("-" * 80)
            for cache_dir, error in repair_errors:
                print(f"  {cache_dir}: {error}")
            print()

        if total_repaired > 0:
            print(f"Repaired {total_repaired} absolute path(s) across {len(cache_dirs)} cache directory(ies)")
            print()
        else:
            print("No absolute paths found to repair.")
            print()

    # Collect statistics
    stats = collect_cache_statistics(cache_dirs, verbose=args.verbose)

    # Print summary
    print_summary(stats)

    # Return exit code based on validation results
    if stats["summary"]["invalid_caches"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
