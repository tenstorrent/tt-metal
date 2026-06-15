# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys


def strip_test_parameters(test_name):
    """Strip parameters from test name.

    Examples:
        'test_file.py::test_func[param1-param2]' -> 'test_file.py::test_func'
        'test_file.py::test_func' -> 'test_file.py::test_func'
    """
    if "[" in test_name:
        return test_name.split("[")[0]
    return test_name


def process_worker(worker_id, tests):
    """Process a single worker's test results.

    Returns:
        dict with keys: 'worker_id', 'unique_statuses', 'total_tests'
    """
    unique_tests_stats = list()

    unique_names = []

    for test_entry in tests:
        status = test_entry.get("status", "")
        test_name = test_entry.get("test", "")

        base_test_name = strip_test_parameters(test_name)

        if base_test_name not in unique_names:
            unique_names.append(base_test_name)
            unique_tests_stats.append([base_test_name, ""])

        if status == "failed":
            unique_tests_stats[len(unique_tests_stats) - 1][1] = "=> SOME FAILED"
        elif status == "passed":
            if unique_tests_stats[len(unique_tests_stats) - 1][1] != "=> SOME FAILED":
                unique_tests_stats[len(unique_tests_stats) - 1][1] = ""

    return {
        "worker_id": worker_id,
        "unique_statuses": unique_tests_stats,
        "total_tests": len(tests),
    }


def process_json_file(json_file_path):
    """Load and process pytest run order JSON file.

    Returns:
        list of worker results
    """
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {json_file_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, dict):
        print(
            "Error: JSON file must contain a dictionary with worker IDs as keys",
            file=sys.stderr,
        )
        sys.exit(1)

    results = []
    for worker_id, tests in data.items():
        if not isinstance(tests, list):
            print(
                f"Warning: Worker {worker_id} does not contain a list of tests, skipping",
                file=sys.stderr,
            )
            continue

        result = process_worker(worker_id, tests)
        results.append(result)

    return results


def print_results(results):
    """Print results to console."""
    print(f"\n\nTest execution order and status (Per Worker)\n{'=' * 80}\n")

    for result in results:
        worker_id = result["worker_id"]
        total_tests = result["total_tests"]
        unique_statuses = result["unique_statuses"]

        print(f"Worker name: '{worker_id}' | Total tests: {total_tests}\n")
        for idx, entry in enumerate(unique_statuses):
            status_marker = "⚠ SOME FAILED" if entry[1] else "✓"
            print(f"{idx:>2}. {entry[0]:<100} {status_marker}")

        print(f"\n{'-' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract unique pytest tests that ran before the first failure for each worker"
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the JSON file containing pytest run order data",
    )

    args = parser.parse_args()

    # Process the JSON file
    results = process_json_file(args.json_file)

    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
