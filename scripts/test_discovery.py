#!/usr/bin/env python3
"""
Test script for program factory discovery

This script tests the discovery approach on a few sample files to verify
it correctly identifies program factories.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import the discovery module
sys.path.append(str(Path(__file__).parent))

from discover_program_factories import ProgramFactoryDiscoverer


def test_discovery():
    """Test the discovery functionality."""
    print("Testing Program Factory Discovery...")

    discoverer = ProgramFactoryDiscoverer()

    # Test file scanning
    print("\n1. Testing file scanning...")
    operations_dir = Path("ttnn/cpp/ttnn/operations")

    if not operations_dir.exists():
        print(f"Warning: Operations directory {operations_dir} does not exist")
        return

    # Find device directories
    device_dirs = []
    for op_dir in operations_dir.iterdir():
        if op_dir.is_dir():
            device_dir = op_dir / "device"
            if device_dir.exists():
                device_dirs.append(device_dir)

    print(f"Found {len(device_dirs)} device directories:")
    for device_dir in device_dirs:
        print(f"  - {device_dir}")

    # Test scanning a few specific files
    print("\n2. Testing specific file scanning...")

    # Look for some known program factory files
    test_files = [
        "ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_program_factory.hpp",
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_program_factory.hpp",
        "ttnn/cpp/ttnn/operations/full/device/full_device_operation.hpp",
    ]

    for test_file in test_files:
        file_path = Path(test_file)
        if file_path.exists():
            print(f"\nScanning {test_file}...")
            factories = discoverer.scan_file(file_path)
            print(f"  Found {len(factories)} factories:")
            for factory in factories:
                print(f"    - {factory.name} (line {factory.line_number})")
                print(f"      Function: {factory.function_name}")
                print(f"      Program Creation: {factory.has_program_creation}")
                print(f"      CreateKernel: {factory.has_create_kernel}")
                print(f"      CreateCircularBuffer: {factory.has_create_circular_buffer}")
        else:
            print(f"  File {test_file} not found")

    # Test full discovery
    print("\n3. Testing full discovery...")
    factories = discoverer.discover_factories()
    print(f"Total factories found: {len(factories)}")

    # Group by file to show multiple factories per file
    factories_by_file = {}
    for factory in factories:
        if factory.file_path not in factories_by_file:
            factories_by_file[factory.file_path] = []
        factories_by_file[factory.file_path].append(factory)

    files_with_multiple = {file: factories for file, factories in factories_by_file.items() if len(factories) > 1}

    if files_with_multiple:
        print(f"\nFiles with multiple factories: {len(files_with_multiple)}")
        for file, factories_list in files_with_multiple.items():
            print(f"\n  {file} ({len(factories_list)} factories):")
            for factory in factories_list:
                print(f"    - {factory.name} (line {factory.line_number})")

    if factories:
        print("\nSample factories:")
        for i, factory in enumerate(factories[:5]):  # Show first 5
            print(f"  {i+1}. {factory.name}")
            print(f"     File: {factory.file_path}")
            print(f"     Function: {factory.function_name}")
            print(f"     Program Creation: {factory.has_program_creation}")
            print(f"     CreateKernel: {factory.has_create_kernel}")
            print(f"     CreateCircularBuffer: {factory.has_create_circular_buffer}")


def test_patterns():
    """Test the regex patterns used for discovery."""
    print("\n4. Testing regex patterns...")

    # Test patterns on sample code
    sample_code = """
    static cached_program_t TypecastProgramFactory::create(
        const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {

        Program program{};

        CreateCircularBuffer(program, all_cores, cb_config);
        CreateKernel(program, kernel_path, cores, config);

        return {std::move(program), shared_vars};
    }
    """

    discoverer = ProgramFactoryDiscoverer()

    # Test function pattern matching
    for pattern in discoverer.function_patterns:
        matches = re.findall(pattern, sample_code)
        if matches:
            print(f"  Pattern matched: {pattern[:50]}...")
            print(f"    Matches: {matches}")

    # Test program creation detection
    has_program = bool(re.search(r"Program\s+program\s*\{|Program\s+program\s*;|CreateProgram", sample_code))
    has_kernel = bool(re.search(r"CreateKernel", sample_code))
    has_cb = bool(re.search(r"CreateCircularBuffer", sample_code))

    print(f"  Program creation detected: {has_program}")
    print(f"  CreateKernel detected: {has_kernel}")
    print(f"  CreateCircularBuffer detected: {has_cb}")


if __name__ == "__main__":
    import re

    test_discovery()
    test_patterns()

    print("\nTest completed!")
