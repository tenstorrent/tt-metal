#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test script for JSON serialization functionality.

This script tests the JSON serialization of triage results without
needing actual hardware or ttexalens context.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import sys
import os

# Add triage directory to path
sys.path.insert(0, os.path.dirname(__file__))

from json_serializer import (
    json_serializer_default,
    dataclass_to_json_dict,
    serialize_result_to_json,
    write_json_output,
    create_triage_metadata,
    create_summary,
    TriageRunResult,
)
from triage import triage_field, hex_serializer


@dataclass
class MockCoordinate:
    """Mock coordinate for testing."""

    x: int
    y: int


@dataclass
class TestData:
    """Test dataclass similar to ArcCheckData."""

    name: str = triage_field("Name")
    value: int = triage_field("Value", hex_serializer)
    duration: timedelta = triage_field("Duration")
    coord: MockCoordinate = triage_field("Coordinate")


def test_basic_serialization():
    """Test basic JSON serialization of simple types."""
    print("Testing basic serialization...")

    # Test various types
    assert json_serializer_default(None) is None
    assert json_serializer_default(42) == 42
    assert json_serializer_default(3.14) == 3.14
    assert json_serializer_default("hello") == "hello"
    assert json_serializer_default(True) is True

    # Test list
    result = json_serializer_default([1, 2, 3])
    assert result == [1, 2, 3]

    # Test dict
    result = json_serializer_default({"a": 1, "b": 2})
    assert result == {"a": 1, "b": 2}

    # Test timedelta
    td = timedelta(seconds=123.45)
    result = json_serializer_default(td)
    assert result["_type"] == "timedelta"
    assert result["total_seconds"] == 123.45

    print("✓ Basic serialization tests passed")


def test_dataclass_serialization():
    """Test dataclass serialization."""
    print("Testing dataclass serialization...")

    coord = MockCoordinate(x=1, y=2)
    data = TestData(name="test", value=0xDEADBEEF, duration=timedelta(seconds=100), coord=coord)

    result = dataclass_to_json_dict(data)

    # Check fields are present
    assert "Name" in result  # Uses serialized_name
    assert result["Name"] == "test"

    # Check value (should be int, not hex string)
    assert "Value" in result
    assert result["Value"] == 0xDEADBEEF

    # Check duration (should be structured)
    assert "Duration" in result
    assert result["Duration"]["_type"] == "timedelta"
    assert result["Duration"]["total_seconds"] == 100

    # Check coordinate (nested dataclass should be serialized)
    assert "Coordinate" in result

    print("✓ Dataclass serialization tests passed")


def test_result_serialization():
    """Test full result serialization."""
    print("Testing result serialization...")

    coord = MockCoordinate(x=1, y=2)
    data1 = TestData(name="test1", value=0x100, duration=timedelta(seconds=10), coord=coord)
    data2 = TestData(name="test2", value=0x200, duration=timedelta(seconds=20), coord=coord)

    # Test single item
    result = serialize_result_to_json(data1)
    assert result["type"] == "single"
    assert "data" in result
    assert result["data"]["Name"] == "test1"

    # Test list
    result = serialize_result_to_json([data1, data2])
    assert result["type"] == "list"
    assert result["count"] == 2
    assert len(result["data"]) == 2
    assert result["data"][0]["Name"] == "test1"
    assert result["data"][1]["Name"] == "test2"

    # Test None
    result = serialize_result_to_json(None)
    assert result is None

    print("✓ Result serialization tests passed")


def test_json_output():
    """Test complete JSON output generation."""
    print("Testing complete JSON output...")

    # Create test data
    timestamp = datetime.now()
    command = "test command"

    metadata = create_triage_metadata(timestamp=timestamp, command=command, execution_duration_seconds=45.2)

    # Create mock scripts
    scripts = [
        {
            "name": "test_script_1.py",
            "execution_time_seconds": 2.3,
            "status": "success",
            "failure_message": None,
            "dependency_failed": False,
            "is_data_provider": False,
            "checks": [{"success": True, "message": "All checks passed"}],
            "result": {"type": "single", "data": {"Name": "test", "Value": 42}},
        },
        {
            "name": "test_script_2.py",
            "execution_time_seconds": 1.5,
            "status": "failed",
            "failure_message": "Test failure",
            "dependency_failed": False,
            "is_data_provider": False,
            "checks": [{"success": False, "message": "Check failed"}],
            "result": None,
        },
    ]

    summary = create_summary(scripts)

    assert summary["total_scripts"] == 2
    assert summary["successful"] == 1
    assert summary["failed"] == 1
    assert summary["skipped"] == 0

    # Create complete structure
    output_data = TriageRunResult(metadata=metadata, scripts=scripts, summary=summary)

    # Write to file
    test_file = "/tmp/test_triage_output.json"
    write_json_output(output_data, test_file, pretty=True)

    # Read back and verify
    with open(test_file, "r") as f:
        loaded = json.load(f)

    assert "metadata" in loaded
    assert "scripts" in loaded
    assert "summary" in loaded
    assert loaded["metadata"]["command"] == command
    assert len(loaded["scripts"]) == 2
    assert loaded["summary"]["total_scripts"] == 2

    # Clean up
    os.remove(test_file)

    print("✓ JSON output tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("JSON Serialization Test Suite")
    print("=" * 60)
    print()

    try:
        test_basic_serialization()
        test_dataclass_serialization()
        test_result_serialization()
        test_json_output()

        print()
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
