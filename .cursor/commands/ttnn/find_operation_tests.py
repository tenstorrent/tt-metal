#!/usr/bin/env python3
"""
Find unit tests for a given operation and identify which CI/CD pipelines run those tests.

Usage:
    python find_operation_tests.py <operation_name> [--operation-path <path>]

Example:
    python find_operation_tests.py slice_write
    python find_operation_tests.py conv2d --operation-path conv/conv2d
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional


def find_operation_path(operation_name: str, repo_root: Path) -> Optional[str]:
    """Find the operation path in ttnn/cpp/ttnn/operations/"""
    operations_dir = repo_root / "ttnn" / "cpp" / "ttnn" / "operations"
    if not operations_dir.exists():
        return None

    # Search for directories or files containing the operation name
    for root, dirs, files in os.walk(operations_dir):
        # Check directory names
        for dir_name in dirs:
            if operation_name.lower() in dir_name.lower():
                rel_path = Path(root).relative_to(operations_dir) / dir_name
                return str(rel_path)

        # Check file names
        for file_name in files:
            if operation_name.lower() in file_name.lower() and file_name.endswith((".cpp", ".hpp")):
                rel_path = Path(root).relative_to(operations_dir)
                return str(rel_path)

    return None


def find_test_files(operation_name: str, repo_root: Path) -> List[Path]:
    """Find test files that test the given operation"""
    tests_dir = repo_root / "tests" / "ttnn" / "unit_tests" / "operations"
    if not tests_dir.exists():
        return []

    test_files = []
    operation_lower = operation_name.lower()

    # Search for test files
    for root, dirs, files in os.walk(tests_dir):
        for file_name in files:
            if file_name.startswith("test_") and file_name.endswith(".py"):
                file_path = Path(root) / file_name

                # Check if operation name appears in filename
                if operation_lower in file_name.lower():
                    test_files.append(file_path)
                    continue

                # Check if operation name appears in file content
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Look for operation name in imports, function calls, etc.
                        if (
                            operation_lower in content.lower()
                            or f"ttnn.{operation_name}" in content
                            or f"ttnn.experimental.{operation_name}" in content
                            or f"{operation_name}(" in content
                        ):
                            test_files.append(file_path)
                except Exception:
                    pass

    return test_files


def get_test_directory(test_file: Path, repo_root: Path) -> str:
    """Get the test directory category from a test file path"""
    tests_dir = repo_root / "tests" / "ttnn" / "unit_tests" / "operations"
    try:
        rel_path = test_file.relative_to(tests_dir)
        # Get the first directory component
        if len(rel_path.parts) > 1:
            return rel_path.parts[0]
    except ValueError:
        pass
    return "unknown"


def find_pipelines_for_test_directory(test_dir: str, repo_root: Path) -> List[Dict]:
    """Find CI/CD pipelines that run tests in the given directory"""
    workflows_dir = repo_root / ".github" / "workflows"
    if not workflows_dir.exists():
        return []

    pipelines = []
    test_path = f"tests/ttnn/unit_tests/operations/{test_dir}"

    # Key workflow files to check
    workflow_files = [
        "ttnn-post-commit.yaml",
        "tt-metal-l2-nightly-impl.yaml",
        "blackhole-multi-card-unit-tests-impl.yaml",
        "all-post-commit-workflows.yaml",
    ]

    for workflow_file in workflow_files:
        workflow_path = workflows_dir / workflow_file
        if not workflow_path.exists():
            continue

        try:
            with open(workflow_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Find all cmd patterns in the entire file that contain our test directory
            cmd_pattern = r'cmd:\s*(["\'])((?:[^\\]|\\.)*?)\1'
            all_cmd_matches = list(re.finditer(cmd_pattern, content))

            # Find all name patterns in the entire file
            name_pattern = r'name:\s*(["\'])((?:[^\\]|\\.)*?)\1'
            all_name_matches = list(re.finditer(name_pattern, content))

            # For each cmd that contains our test directory, find its corresponding name
            for cmd_match in all_cmd_matches:
                cmd_content = cmd_match.group(2)

                # Only process cmds that contain our test directory
                if test_dir not in cmd_content:
                    continue

                # Find the name that's closest before this cmd
                cmd_start = cmd_match.start()
                test_group_name = "Unknown"
                for name_match in reversed(all_name_matches):
                    if name_match.end() < cmd_start:
                        test_group_name = name_match.group(2)
                        break

                # Only add if we found a valid name (not "Unknown")
                if test_group_name != "Unknown":
                    pipelines.append(
                        {
                            "workflow": workflow_file,
                            "test_group": test_group_name,
                            "command": cmd_content,
                            "test_directory": test_dir,
                        }
                    )
        except Exception as e:
            print(f"Error reading {workflow_file}: {e}", file=sys.stderr)

    # Also check for patterns that might include this test directory
    # e.g., "data_movement/ccl" groups include data_movement tests
    if test_dir == "data_movement":
        # Check for data_movement/ccl groups
        try:
            with open(workflows_dir / "ttnn-post-commit.yaml", "r") as f:
                content = f.read()
            if "data_movement/ccl" in content:
                # Find all data_movement/ccl groups
                pattern = r'name:\s*["\']([^"\']*data_movement/ccl[^"\']*)["\']'
                for match in re.finditer(pattern, content):
                    test_group_name = match.group(1)
                    # Find corresponding command
                    start = match.start()
                    end = min(len(content), match.end() + 1000)
                    context = content[start:end]
                    cmd_match = re.search(r'cmd:\s*(["\'])((?:[^\\]|\\.)*?)\1', context)
                    if cmd_match:
                        pipelines.append(
                            {
                                "workflow": "ttnn-post-commit.yaml",
                                "test_group": test_group_name,
                                "command": cmd_match.group(2),  # group(2) is the content inside quotes
                                "test_directory": test_dir,
                            }
                        )
        except Exception:
            pass

    return pipelines


def main():
    parser = argparse.ArgumentParser(description="Find unit tests for an operation and identify CI/CD pipelines")
    parser.add_argument("operation_name", help="Name of the operation (e.g., 'slice_write', 'conv2d', 'matmul')")
    parser.add_argument(
        "--operation-path", help="Path to the operation (e.g., 'experimental/slice_write', 'conv/conv2d')", default=None
    )
    parser.add_argument(
        "--repo-root", help="Root directory of the repository", default=Path(__file__).parent.parent.parent.parent
    )

    args = parser.parse_args()
    repo_root = Path(args.repo_root)

    # Find operation path if not provided
    operation_path = args.operation_path
    if not operation_path:
        operation_path = find_operation_path(args.operation_name, repo_root)
        if operation_path:
            print(f"Found operation path: {operation_path}")
        else:
            print(f"Warning: Could not find operation path for '{args.operation_name}'")

    # Find test files
    print(f"\nSearching for tests for operation: {args.operation_name}")
    test_files = find_test_files(args.operation_name, repo_root)

    if not test_files:
        print(f"No test files found for operation '{args.operation_name}'")
        print("\nTrying to find tests in common directories...")
        # Try common test directories
        common_dirs = ["data_movement", "eltwise", "conv", "matmul", "pool", "fused", "reduce"]
        for test_dir in common_dirs:
            test_path = repo_root / "tests" / "ttnn" / "unit_tests" / "operations" / test_dir
            if test_path.exists():
                for test_file in test_path.glob(f"test_*{args.operation_name}*.py"):
                    test_files.append(test_file)
                for test_file in test_path.glob(f"*{args.operation_name}*.py"):
                    if test_file not in test_files:
                        test_files.append(test_file)

    if not test_files:
        print(f"\n❌ No test files found for operation '{args.operation_name}'")
        print("\nSuggestions:")
        print("1. Check if the operation name is spelled correctly")
        print("2. Try searching with a partial name")
        print("3. Check if tests might be in a different location")
        return 1

    # Group test files by directory
    test_directories: Dict[str, List[Path]] = {}
    for test_file in test_files:
        test_dir = get_test_directory(test_file, repo_root)
        if test_dir not in test_directories:
            test_directories[test_dir] = []
        test_directories[test_dir].append(test_file)

    # Output results
    print(f"\n{'='*80}")
    print(f"Operation: {args.operation_name}")
    if operation_path:
        print(f"Operation Path: {operation_path}")
    print(f"{'='*80}\n")

    print("Test Files Found:")
    for test_dir, files in test_directories.items():
        print(f"\n  Test Directory: {test_dir}/")
        for test_file in files:
            rel_path = test_file.relative_to(repo_root)
            print(f"    - {rel_path}")

    # Find pipelines
    print(f"\n{'='*80}")
    print("Pipelines That Run These Tests:")
    print(f"{'='*80}\n")

    all_pipelines = []
    for test_dir in test_directories.keys():
        pipelines = find_pipelines_for_test_directory(test_dir, repo_root)
        all_pipelines.extend(pipelines)

    # Remove duplicates
    seen = set()
    unique_pipelines = []
    for pipeline in all_pipelines:
        key = (pipeline["workflow"], pipeline["test_group"])
        if key not in seen:
            seen.add(key)
            unique_pipelines.append(pipeline)

    if not unique_pipelines:
        print("❌ No pipelines found that run these tests")
        print("\nThis could mean:")
        print("1. Tests are run manually or in custom workflows")
        print("2. Tests are in nightly-only pipelines")
        print("3. Test directory mapping needs to be updated")
    else:
        for i, pipeline in enumerate(unique_pipelines, 1):
            print(f"{i}. Pipeline: {pipeline['test_group']}")
            print(f"   Workflow: .github/workflows/{pipeline['workflow']}")
            print(f"   Command: {pipeline['command']}")
            print(f"   Test Directory: {pipeline['test_directory']}/")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
