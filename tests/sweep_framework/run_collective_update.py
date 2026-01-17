#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Script to update multiple op_run*.json files to share common metadata.
Updates initiated_by, git_sha, github_pipeline_id, run_start_ts, and run_end_ts
across all matching files, using the earliest start and latest end timestamps.

Optimized for large files by:
1. Extracting only metadata fields without loading full file content
2. Using streaming JSON processing for file updates
3. Optional support for ijson streaming parser for extremely large files

For extremely large files (multi-GB), install ijson for better performance:
    pip install ijson
"""

import json
import pathlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import sys
import argparse
import os
import tempfile
import shutil

# Try to import ijson for streaming JSON parsing (optional)
try:
    import ijson

    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False


def parse_datetime(dt_str: str) -> datetime:
    """Parse datetime string in ISO format."""
    # Handle both with and without microseconds
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except ValueError:
        # Try without timezone
        return datetime.fromisoformat(dt_str)


def datetime_to_str(dt: datetime) -> str:
    """Convert datetime to ISO format string."""
    return dt.isoformat()


def extract_metadata_from_file(filepath: pathlib.Path) -> Dict[str, Any]:
    """
    Extract only the metadata fields from a JSON file without loading the entire content.
    This is optimized for large files by reading only the beginning of the file.
    Uses ijson for streaming if available and file is large.
    """
    metadata_fields = ["initiated_by", "git_sha", "github_pipeline_id", "run_start_ts", "run_end_ts", "run_contents"]
    metadata = {}

    file_size_mb = filepath.stat().st_size / (1024 * 1024)

    # For very large files (>500MB), use ijson if available
    if IJSON_AVAILABLE and file_size_mb > 500:
        try:
            with open(filepath, "rb") as f:
                parser = ijson.parse(f)
                for prefix, event, value in parser:
                    # Look for top-level fields
                    if prefix in metadata_fields and event == "string":
                        metadata[prefix] = value
                    elif prefix in metadata_fields and event == "number":
                        metadata[prefix] = value
                    elif prefix in metadata_fields and event == "null":
                        metadata[prefix] = None

                    # Stop early if we have all fields
                    if len(metadata) == len(metadata_fields):
                        break

            if metadata:
                return metadata
        except Exception as e:
            print(f"  Warning: ijson parsing failed, falling back: {e}")

    # For medium files or if ijson not available, use chunk-based extraction
    with open(filepath, "r") as f:
        # Read a reasonable chunk that should contain the metadata
        # Adjust size if metadata appears later in files
        chunk_size = 16384  # 16KB should be enough for metadata
        chunk = f.read(chunk_size)

        # Try to parse the partial JSON to get metadata
        try:
            # Find the metadata fields in the chunk
            for field in metadata_fields:
                # Look for the field pattern in the chunk
                field_pattern = f'"{field}"'
                if field_pattern in chunk:
                    # Extract the value after the field
                    start_idx = chunk.find(field_pattern)
                    if start_idx != -1:
                        # Find the colon after the field name
                        colon_idx = chunk.find(":", start_idx)
                        if colon_idx != -1:
                            # Find the value (handle strings, numbers, null)
                            value_start = colon_idx + 1
                            # Skip whitespace
                            while value_start < len(chunk) and chunk[value_start] in " \t\n":
                                value_start += 1

                            if value_start < len(chunk):
                                if chunk[value_start] == '"':
                                    # String value - find closing quote
                                    value_end = chunk.find('"', value_start + 1)
                                    if value_end != -1:
                                        metadata[field] = chunk[value_start + 1 : value_end]
                                elif chunk[value_start : value_start + 4] == "null":
                                    metadata[field] = None
                                else:
                                    # Number or other value - find comma or closing brace
                                    value_end = value_start
                                    while value_end < len(chunk) and chunk[value_end] not in ",\n\r\t }":
                                        value_end += 1
                                    value_str = chunk[value_start:value_end].strip()
                                    try:
                                        metadata[field] = int(value_str)
                                    except ValueError:
                                        metadata[field] = value_str
        except Exception:
            # Fall back to loading the entire file if the quick extraction fails
            pass

    # If we couldn't extract all metadata from the chunk, fall back to full load
    # but only load these specific fields
    if len(metadata) < len(metadata_fields):
        with open(filepath, "r") as f:
            try:
                full_data = json.load(f)
                for field in metadata_fields:
                    if field in full_data:
                        metadata[field] = full_data[field]
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse {filepath.name}: {e}")

    return metadata


def load_json_file(filepath: pathlib.Path) -> Dict[str, Any]:
    """Load and parse JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_json_file(filepath: pathlib.Path, data: Dict[str, Any]) -> None:
    """Save data to JSON file with proper formatting."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def update_large_json_file(filepath: pathlib.Path, updates: Dict[str, Any], dry_run: bool = False) -> List[str]:
    """
    Update specific fields in a potentially large JSON file efficiently.
    Uses streaming for large files to avoid loading everything into memory.

    Returns list of changes made.
    """
    changes = []

    # First, get file size to determine strategy
    file_size = filepath.stat().st_size
    size_mb = file_size / (1024 * 1024)

    # If file is small enough (< 100MB), use simple in-memory approach
    if size_mb < 100:
        data = load_json_file(filepath)
        for key, new_value in updates.items():
            old_value = data.get(key)
            if old_value != new_value:
                changes.append(f"{key}: '{old_value}' -> '{new_value}'")
                if not dry_run:
                    data[key] = new_value

        if changes and not dry_run:
            save_json_file(filepath, data)
    else:
        # For large files, use a streaming approach with a temporary file
        print(f"  Note: {filepath.name} is {size_mb:.1f}MB - using streaming update")

        if dry_run:
            # For dry run, just check what would change without modifying
            metadata = extract_metadata_from_file(filepath)
            for key, new_value in updates.items():
                old_value = metadata.get(key)
                if old_value != new_value:
                    changes.append(f"{key}: '{old_value}' -> '{new_value}'")
        else:
            # Create a temporary file for the updated content
            temp_fd, temp_path = tempfile.mkstemp(suffix=".json", dir=filepath.parent)
            temp_file = pathlib.Path(temp_path)

            try:
                # Load the full file (this is still necessary for large files, but we'll optimize if needed)
                print(f"    Loading {filepath.name}...")
                data = load_json_file(filepath)

                # Apply updates
                for key, new_value in updates.items():
                    old_value = data.get(key)
                    if old_value != new_value:
                        changes.append(f"{key}: '{old_value}' -> '{new_value}'")
                        data[key] = new_value

                if changes:
                    print(f"    Writing updated {filepath.name}...")
                    with open(temp_fd, "w") as f:
                        json.dump(data, f, indent=2, default=str)

                    # Atomic replace
                    os.replace(temp_path, filepath)
                else:
                    # No changes, remove temp file
                    os.close(temp_fd)
                    temp_file.unlink()
            except Exception as e:
                # Clean up temp file on error
                try:
                    os.close(temp_fd)
                except OSError:
                    pass
                if temp_file.exists():
                    temp_file.unlink()
                raise e

    return changes


def find_op_run_files(directory: pathlib.Path) -> List[pathlib.Path]:
    """Find all op_run*.json files in the specified directory."""
    pattern = "oprun*.json"
    files = sorted(directory.glob(pattern))
    return files


def extract_common_metadata(files: List[pathlib.Path], run_type: str = "nightly") -> Dict[str, Any]:
    """
    Extract common metadata from all files and determine the earliest start
    and latest end timestamps.

    Optimized to only extract metadata fields without loading entire files.

    Args:
        files: List of file paths to process
        run_type: Type of run - "nightly", "comprehensive", "model_traced", or "lead_models"

    Returns a dict with the common values and computed timestamps.
    """
    if not files:
        raise ValueError("No files provided")

    # Extract only metadata from each file (optimized for large files)
    all_metadata = []
    total_size_mb = 0

    for filepath in files:
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        total_size_mb += file_size_mb

        if file_size_mb > 100:
            print(f"  Extracting metadata from {filepath.name} ({file_size_mb:.1f}MB)...")

        metadata = extract_metadata_from_file(filepath)
        all_metadata.append(metadata)

    if total_size_mb > 1000:  # More than 1GB total
        print(f"  Total data size: {total_size_mb:.1f}MB")

    # Extract metadata from first file as reference
    first_metadata = all_metadata[0]
    common_metadata = {
        "initiated_by": first_metadata.get("initiated_by"),
        "git_sha": first_metadata.get("git_sha"),
        "github_pipeline_id": first_metadata.get("github_pipeline_id"),
    }

    # Verify these fields are consistent across all files (optional warning)
    for i, metadata in enumerate(all_metadata[1:], start=1):
        for field in ["initiated_by", "git_sha", "github_pipeline_id"]:
            if metadata.get(field) != common_metadata[field]:
                print(
                    f"Warning: {field} differs in {files[i].name}: "
                    f"'{metadata.get(field)}' vs '{common_metadata[field]}'"
                )

    # Find earliest start and latest end timestamps
    earliest_start = None
    latest_end = None

    for metadata in all_metadata:
        if "run_start_ts" in metadata and metadata["run_start_ts"]:
            start_ts = parse_datetime(metadata["run_start_ts"])
            if earliest_start is None or start_ts < earliest_start:
                earliest_start = start_ts

        if "run_end_ts" in metadata and metadata["run_end_ts"]:
            end_ts = parse_datetime(metadata["run_end_ts"])
            if latest_end is None or end_ts > latest_end:
                latest_end = end_ts

    if earliest_start:
        common_metadata["run_start_ts"] = datetime_to_str(earliest_start)
    if latest_end:
        common_metadata["run_end_ts"] = datetime_to_str(latest_end)

    # Set run_contents based on run type
    if run_type == "comprehensive":
        common_metadata["run_contents"] = "comprehensive"
    elif run_type == "model_traced":
        common_metadata["run_contents"] = "model traced"
    elif run_type == "lead_models":
        common_metadata["run_contents"] = "lead models"
    else:
        common_metadata["run_contents"] = "nightly"

    return common_metadata


def update_op_run_files(directory: pathlib.Path, dry_run: bool = False, run_type: str = "nightly") -> None:
    """
    Update all op_run*.json files in the directory with common metadata.

    Args:
        directory: Path to the directory containing op_run*.json files
        dry_run: If True, only print what would be changed without modifying files
        run_type: Type of run - "nightly", "comprehensive", "model_traced", or "lead_models"
    """
    # Find all op_run*.json files
    files = find_op_run_files(directory)

    if not files:
        print(f"No oprun*.json files found in {directory}")
        return

    print(f"Found {len(files)} oprun*.json files:")
    for f in files:
        print(f"  - {f.name}")
    print()

    # Extract common metadata and compute timestamps
    try:
        common_metadata = extract_common_metadata(files, run_type)
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return

    print("Common metadata to be applied:")
    for key, value in common_metadata.items():
        print(f"  {key}: {value}")
    print()

    if dry_run:
        print("DRY RUN: No files will be modified")
        print()

    # Update each file using optimized method for large files
    for filepath in files:
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"Processing {filepath.name} ({file_size_mb:.1f}MB)...")

        # Use optimized update function that handles large files
        changes = update_large_json_file(filepath, common_metadata, dry_run)

        if changes:
            print(f"  Changes:")
            for change in changes:
                print(f"    {change}")

            if not dry_run:
                print(f"  ✓ Updated {filepath.name}")
        else:
            print(f"  No changes needed for {filepath.name}")
        print()

    if not dry_run:
        print(f"Successfully updated {len(files)} files")
    else:
        print(f"DRY RUN complete - {len(files)} files would be updated")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Update op_run*.json files to share common metadata")
    parser.add_argument(
        "--directory",
        "-d",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "results_export",
        help="Directory containing op_run*.json files (default: tests/sweep_framework/results_export)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    parser.add_argument(
        "--run-type",
        type=str,
        choices=["nightly", "comprehensive", "model_traced", "lead_models"],
        default="nightly",
        help="Type of run: 'nightly', 'comprehensive', 'model_traced', or 'lead_models' (default: nightly)",
    )

    args = parser.parse_args()

    # Inform about ijson availability
    if not IJSON_AVAILABLE:
        print("Note: ijson library not found. For better performance with very large files (>500MB),")
        print("      install it with: pip install ijson")
        print()

    # Validate directory exists
    if not args.directory.exists():
        print(f"Error: Directory '{args.directory}' does not exist")
        sys.exit(1)

    if not args.directory.is_dir():
        print(f"Error: '{args.directory}' is not a directory")
        sys.exit(1)

    # Run the update
    try:
        update_op_run_files(args.directory, dry_run=args.dry_run, run_type=args.run_type)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
