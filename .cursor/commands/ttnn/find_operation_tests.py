#!/usr/bin/env python3
"""
Find unit tests for a given operation and identify which CI/CD pipelines run those tests.

Usage:
    python find_operation_tests.py <operation_name> [--operation-path <path>]

Example:
    python find_operation_tests.py slice_write
    python find_operation_tests.py conv2d --operation-path conv/conv2d
    python find_operation_tests.py transpose
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional


# APC workflow configuration
APC_WORKFLOW_URL = "https://github.com/tenstorrent/tt-metal/actions/workflows/apc-select-tests.yaml"
APC_JSON_TEMPLATE = {
    "sd-unit-tests": False,
    "fast-dispatch-unit-tests": False,
    "fabric-unit-tests": False,
    "cpp-unit-tests": False,
    "ttnn-unit-tests": False,
    "models-unit-tests": False,
    "tt-train-cpp-unit-tests": False,
    "run-profiler-regression": False,
    "t3000-apc-fast-tests": False,
    "test-ttnn-tutorials": False,
    "triage-tests": False,
}


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


def find_test_files(operation_name: str, repo_root: Path, search_dirs: List[str] = None) -> List[Path]:
    """Find test files that test the given operation"""
    if search_dirs is None:
        search_dirs = [
            "tests/ttnn/unit_tests/operations",
            "tests/ttnn/unit_tests/base_functionality",
            "tests/tt_eager/python_api_testing/unit_testing",
            "tests/tt_eager/python_api_testing/sweep_tests/pytests",
        ]

    test_files = []
    operation_lower = operation_name.lower()

    for search_dir in search_dirs:
        tests_dir = repo_root / search_dir
        if not tests_dir.exists():
            continue

        # Search for test files
        for root, dirs, files in os.walk(tests_dir):
            for file_name in files:
                if file_name.startswith("test_") and file_name.endswith(".py"):
                    file_path = Path(root) / file_name

                    # Check if operation name appears in filename
                    if operation_lower in file_name.lower():
                        test_files.append(file_path)
                        continue

                    # Check if operation name appears in file content (for operations dir only)
                    if "operations" in str(root) or "base_functionality" in str(root):
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                # Look for operation name in imports, function calls, etc.
                                if (
                                    f"ttnn.{operation_name}" in content
                                    or f"ttnn.experimental.{operation_name}" in content
                                ):
                                    test_files.append(file_path)
                        except Exception:
                            pass

    return test_files


def categorize_test_files(test_files: List[Path], repo_root: Path, operation_name: str) -> Dict[str, List[Path]]:
    """Categorize test files by type (primary, related, legacy)"""
    categories = {
        "primary": [],
        "related": [],
        "legacy": [],
    }

    operation_lower = operation_name.lower()

    for test_file in test_files:
        rel_path = str(test_file.relative_to(repo_root))
        file_name_lower = test_file.name.lower()

        if "tt_eager" in rel_path:
            categories["legacy"].append(test_file)
        # Check if operation name is in the filename (primary test)
        elif operation_lower in file_name_lower:
            categories["primary"].append(test_file)
        else:
            categories["related"].append(test_file)

    return categories


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

    # Check base_functionality
    base_func_dir = repo_root / "tests" / "ttnn" / "unit_tests" / "base_functionality"
    try:
        test_file.relative_to(base_func_dir)
        return "base_functionality"
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


def generate_apc_config(test_dirs: Set[str]) -> Dict:
    """Generate APC JSON configuration based on test directories"""
    config = APC_JSON_TEMPLATE.copy()

    # Map test directories to APC config options
    ttnn_dirs = {
        "data_movement",
        "eltwise",
        "conv",
        "matmul",
        "pool",
        "fused",
        "reduce",
        "ccl",
        "transformers",
        "rand",
        "debug",
        "ssm",
        "base_functionality",
    }

    if any(d in ttnn_dirs for d in test_dirs):
        config["ttnn-unit-tests"] = True

    return config


def print_local_testing_commands(test_files: List[Path], repo_root: Path, categories: Dict[str, List[Path]]):
    """Print local testing commands"""
    print("\n" + "=" * 80)
    print("Local Testing Commands")
    print("=" * 80)

    print("\n# Activate environment and run tests")
    print("source python_env/bin/activate\n")

    if categories["primary"]:
        print("# Primary tests:")
        for test_file in categories["primary"][:3]:  # Limit to first 3
            rel_path = test_file.relative_to(repo_root)
            print(f"pytest {rel_path} -v")

    if categories["related"]:
        print("\n# Related tests (use ttnn.{operation}):")
        for test_file in categories["related"][:3]:
            rel_path = test_file.relative_to(repo_root)
            print(f"pytest {rel_path} -v")

    if categories["legacy"]:
        print("\n# Legacy tests:")
        for test_file in categories["legacy"][:2]:
            rel_path = test_file.relative_to(repo_root)
            print(f"pytest {rel_path} -v")


def print_apc_info(test_dirs: Set[str]):
    """Print APC workflow information"""
    print("\n" + "=" * 80)
    print("CI Testing via APC (Automated Pre-Commit)")
    print("=" * 80)

    print(f"\nWorkflow URL: {APC_WORKFLOW_URL}")

    config = generate_apc_config(test_dirs)
    config_json = json.dumps(config, separators=(",", ":"))

    print("\nJSON Configuration (copy-paste ready):")
    print(config_json)

    print("\nPretty-printed config:")
    print(json.dumps(config, indent=2))


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

    # Categorize test files
    categories = categorize_test_files(test_files, repo_root, args.operation_name)

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

    if categories["primary"]:
        print("\n  Primary Tests:")
        for test_file in categories["primary"]:
            rel_path = test_file.relative_to(repo_root)
            print(f"    - {rel_path}")

    if categories["related"]:
        print(f"\n  Tests that use ttnn.{args.operation_name}:")
        for test_file in categories["related"]:
            rel_path = test_file.relative_to(repo_root)
            print(f"    - {rel_path}")

    if categories["legacy"]:
        print("\n  Legacy Tests:")
        for test_file in categories["legacy"]:
            rel_path = test_file.relative_to(repo_root)
            print(f"    - {rel_path}")

    # Print local testing commands
    print_local_testing_commands(test_files, repo_root, categories)

    # Find pipelines
    print(f"\n{'='*80}")
    print("Pipelines That Run These Tests (Post-Commit)")
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

    # Print APC info
    print_apc_info(set(test_directories.keys()))

    return 0


if __name__ == "__main__":
    sys.exit(main())
