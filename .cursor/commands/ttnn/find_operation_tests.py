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
from typing import List, Dict, Optional

import yaml


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


TEST_ROOTS = (
    Path("tests") / "ttnn" / "unit_tests" / "operations",
    Path("tests") / "ttnn" / "nightly" / "unit_tests" / "operations",
)

# CI test matrix -> (impl workflow, calling workflows)
TEST_MATRICES = {
    "ttnn_sanity_tests.yaml": ("ttnn-sanity-tests-impl.yaml", "sanity-tests.yaml"),
    "ttnn_merge_gate_tests.yaml": ("ttnn-smoke-tests-impl.yaml", "merge-gate.yaml"),
    "ops_unit_tests.yaml": ("ops-unit-tests-impl.yaml", "tt-metal-l2-nightly.yaml"),
    "blackhole_multi_card_sanity_tests.yaml": ("blackhole-multi-card-sanity-tests-impl.yaml", "sanity-tests.yaml"),
}


def find_test_files(operation_name: str, repo_root: Path) -> List[Path]:
    """Find test files that test the given operation"""
    test_files = []
    operation_lower = operation_name.lower()

    for test_root in TEST_ROOTS:
        tests_dir = repo_root / test_root
        if not tests_dir.exists():
            continue
        for root, dirs, files in os.walk(tests_dir):
            for file_name in files:
                if not (file_name.startswith("test_") and file_name.endswith(".py")):
                    continue
                file_path = Path(root) / file_name

                # Check if operation name appears in filename
                if operation_lower in file_name.lower():
                    test_files.append(file_path)
                    continue

                # Check if operation name appears in file content
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        if operation_lower in f.read().lower():
                            test_files.append(file_path)
                except Exception:
                    pass

    return test_files


def get_test_directory(test_file: Path, repo_root: Path) -> str:
    """Get the test directory category (e.g. tests/ttnn/unit_tests/operations/pool) from a test file path"""
    for test_root in TEST_ROOTS:
        try:
            rel_path = test_file.relative_to(repo_root / test_root)
        except ValueError:
            continue
        if len(rel_path.parts) > 1:
            return str(test_root / rel_path.parts[0])
    return "unknown"


def _cmd_runs_path(cmd: str, path: str) -> bool:
    """True if cmd passes path itself (a file, file::test or a whole directory) to pytest"""
    return re.search(rf"(^|[\s'\"]){re.escape(path)}(::|/?($|[\s'\"]))", cmd) is not None


def find_pipelines_for_test_file(test_file: Path, repo_root: Path) -> List[Dict]:
    """Find CI test matrix entries whose cmd runs the test file or one of its parent directories"""
    rel_file = test_file.relative_to(repo_root)
    candidates = [str(rel_file)] + [str(parent) for parent in rel_file.parents if len(parent.parts) >= 4]

    pipelines = []
    for matrix_file, (impl_workflow, caller_workflow) in TEST_MATRICES.items():
        matrix_path = repo_root / "tests" / "pipeline_reorg" / matrix_file
        if not matrix_path.exists():
            continue
        try:
            with open(matrix_path, "r", encoding="utf-8") as f:
                entries = yaml.safe_load(f) or []
        except Exception as e:
            print(f"Error reading {matrix_path}: {e}", file=sys.stderr)
            continue

        for entry in entries:
            cmd = str(entry.get("cmd", ""))
            if any(_cmd_runs_path(cmd, path) for path in candidates):
                pipelines.append(
                    {
                        "matrix": f"tests/pipeline_reorg/{matrix_file}",
                        "workflow": f"{impl_workflow} (from {caller_workflow})",
                        "test_group": entry.get("name", "Unknown"),
                        "command": cmd,
                        "test_file": str(rel_file),
                    }
                )

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

    unique_pipelines = {}
    for test_file in test_files:
        for pipeline in find_pipelines_for_test_file(test_file, repo_root):
            key = (pipeline["matrix"], pipeline["test_group"])
            if key in unique_pipelines:
                unique_pipelines[key]["test_files"].append(pipeline["test_file"])
            else:
                unique_pipelines[key] = {**pipeline, "test_files": [pipeline["test_file"]]}
    unique_pipelines = list(unique_pipelines.values())

    if not unique_pipelines:
        print("❌ No pipelines found that run these tests")
        print("\nThis could mean:")
        print("1. Tests are run manually or in custom workflows")
        print("2. Tests are only run from a test matrix other than " + ", ".join(TEST_MATRICES))
    else:
        for i, pipeline in enumerate(unique_pipelines, 1):
            print(f"{i}. Pipeline: {pipeline['test_group']}")
            print(f"   Matrix: {pipeline['matrix']}")
            print(f"   Workflow: .github/workflows/{pipeline['workflow']}")
            print(f"   Test Files: {', '.join(pipeline['test_files'])}")
            print(f"   Command: {pipeline['command']}")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
