# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
JSON serialization support for tt-triage results.

This module provides structured JSON output for triage results, preserving
data types and structure for post-processing and analysis.
"""

from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass, asdict
from datetime import datetime, timedelta
from typing import Any
import json


@dataclass
class CheckResult:
    """Result of a check operation."""

    success: bool
    message: str


@dataclass
class ScriptResult:
    """Container for single script execution result."""

    name: str
    execution_time_seconds: float
    status: str  # "success", "failed", "skipped"
    failure_message: str | None
    dependency_failed: bool
    is_data_provider: bool
    checks: list[dict[str, Any]]  # List of {success: bool, message: str}
    result: dict[str, Any] | None  # {type: str, data: Any}


@dataclass
class TriageMetadata:
    """Metadata about the triage run."""

    timestamp: str  # ISO 8601 format
    command: str
    execution_duration_seconds: float
    tt_triage_version: str


@dataclass
class TriageSummary:
    """Summary statistics for the triage run."""

    total_scripts: int
    successful: int
    failed: int
    skipped: int
    total_failures: int


@dataclass
class TriageRunResult:
    """Container for complete triage run data."""

    metadata: dict[str, Any]
    scripts: list[dict[str, Any]]
    summary: dict[str, Any]


def json_serializer_default(value: Any) -> Any:
    """
    Convert value to JSON-serializable format (preserves types, not converted to string).

    This is different from the string serializers used for console output.
    Numbers stay as numbers, nested structures are preserved.

    Args:
        value: The value to serialize

    Returns:
        JSON-serializable representation of the value
    """
    # Import here to avoid circular dependencies
    try:
        from ttexalens.device import Device
        from ttexalens.coordinate import OnChipCoordinate
        from ttexalens.elf import ElfVariable
    except ImportError:
        # If ttexalens is not available, we can still handle basic types
        Device = None
        OnChipCoordinate = None
        ElfVariable = None

    if value is None:
        return None

    # Handle ttexalens types
    if Device is not None and isinstance(value, Device):
        return {"_type": "Device", "id": value._id}

    if OnChipCoordinate is not None and isinstance(value, OnChipCoordinate):
        return {
            "_type": "OnChipCoordinate",
            "x": value.x,
            "y": value.y,
            "device_id": value.device._id if hasattr(value, "device") else None,
        }

    if ElfVariable is not None and isinstance(value, ElfVariable):
        try:
            return {
                "_type": "ElfVariable",
                "name": value.name if hasattr(value, "name") else None,
                "address": value.address if hasattr(value, "address") else None,
                "size": value.size if hasattr(value, "size") else None,
                "as_list": value.as_list() if hasattr(value, "as_list") else None,
            }
        except:
            return {"_type": "ElfVariable", "str": str(value)}

    # Handle datetime types
    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, timedelta):
        return {
            "_type": "timedelta",
            "total_seconds": value.total_seconds(),
            "days": value.days,
            "seconds": value.seconds,
            "microseconds": value.microseconds,
        }

    # Handle iterables (but not strings)
    if isinstance(value, (list, tuple)):
        return [json_serializer_default(item) for item in value]

    if isinstance(value, dict):
        return {k: json_serializer_default(v) for k, v in value.items()}

    # Handle dataclasses recursively
    if is_dataclass(value) and not isinstance(value, type):
        return dataclass_to_json_dict(value)

    # Handle basic types (int, float, str, bool)
    if isinstance(value, (int, float, str, bool)):
        return value

    # For everything else, convert to string
    return str(value)


def dataclass_to_json_dict(obj: Any, verbose_level: int = 0) -> dict[str, Any]:
    """
    Convert a dataclass instance to a JSON-serializable dictionary.

    Handles field metadata including:
    - recurse: Flattens nested dataclasses
    - serialized_name: Uses custom field names
    - verbose: Respects verbosity levels
    - dont_serialize: Skips internal fields

    Args:
        obj: The dataclass instance to convert
        verbose_level: Current verbosity level (0, 1, 2)

    Returns:
        Dictionary with JSON-serializable values
    """
    if not is_dataclass(obj):
        return json_serializer_default(obj)

    result: dict[str, Any] = {}

    for field in fields(obj):
        metadata = field.metadata

        # Skip field if it requires higher verbosity level
        if metadata.get("verbose", 0) > verbose_level:
            continue

        # Skip fields marked as dont_serialize
        if metadata.get("dont_serialize", False):
            continue

        value = getattr(obj, field.name)

        # Handle recurse fields (flatten nested dataclasses)
        if metadata.get("recurse", False):
            if is_dataclass(value):
                # Merge nested fields into parent
                nested_dict = dataclass_to_json_dict(value, verbose_level)
                result.update(nested_dict)
            else:
                # If not a dataclass, just include it normally
                field_name = metadata.get("serialized_name", field.name)
                result[field_name] = json_serializer_default(value)
        else:
            # Use serialized_name if provided, otherwise use field name
            field_name = metadata.get("serialized_name", field.name)
            result[field_name] = json_serializer_default(value)

    return result


def serialize_result_to_json(result: Any, verbose_level: int = 0) -> dict[str, Any] | None:
    """
    Serialize a script result to JSON format.

    Args:
        result: The result object from script.run()
        verbose_level: Current verbosity level

    Returns:
        Dictionary containing serialized result or None
    """
    if result is None:
        return None

    # Handle single dataclass
    if is_dataclass(result):
        return {"type": "single", "data": dataclass_to_json_dict(result, verbose_level)}

    # Handle list of dataclasses
    if isinstance(result, list):
        if all(is_dataclass(item) for item in result):
            return {
                "type": "list",
                "count": len(result),
                "data": [dataclass_to_json_dict(item, verbose_level) for item in result],
            }
        else:
            # Mixed list, serialize each item
            return {"type": "list", "count": len(result), "data": [json_serializer_default(item) for item in result]}

    # For other types, just serialize directly
    return {"type": "other", "data": json_serializer_default(result)}


def write_json_output(output_data: TriageRunResult, filepath: str, pretty: bool = True) -> None:
    """
    Write triage results to a JSON file.

    Args:
        output_data: The complete triage run results
        filepath: Path to output JSON file
        pretty: If True, format with indentation (default: True)
    """
    indent = 2 if pretty else None

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(asdict(output_data), f, indent=indent, ensure_ascii=False)


def create_triage_metadata(
    timestamp: datetime, command: str, execution_duration_seconds: float, version: str = "1.0"
) -> dict[str, Any]:
    """
    Create metadata dictionary for triage run.

    Args:
        timestamp: Start time of the run
        command: Command line that was executed
        execution_duration_seconds: Total execution time
        version: Version identifier

    Returns:
        Metadata dictionary
    """
    return {
        "timestamp": timestamp.isoformat(),
        "tt_triage_version": version,
        "command": command,
        "execution_duration_seconds": execution_duration_seconds,
    }


def create_summary(scripts: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Create summary statistics from script results.

    Args:
        scripts: List of script result dictionaries

    Returns:
        Summary dictionary with statistics
    """
    total = len(scripts)
    successful = sum(1 for s in scripts if s["status"] == "success")
    failed = sum(1 for s in scripts if s["status"] == "failed")
    skipped = sum(1 for s in scripts if s["status"] == "skipped")
    total_failures = sum(
        len(s.get("checks", [])) for s in scripts if any(not c.get("success", True) for c in s.get("checks", []))
    )

    return {
        "total_scripts": total,
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "total_failures": total_failures,
    }
