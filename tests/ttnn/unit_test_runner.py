# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Created to model sweeps_runner.py for unit tests to export results to postgres

import argparse
import sys
import pathlib
import datetime as dt
import os
import json
import subprocess
import traceback
import pytest
from enum import Enum
from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger
from tests.sweep_framework.framework.database import (
    postgres_connection,
    initialize_postgres_database,
    push_run,
    update_run,
    generate_error_signature,
    map_test_status_to_run_status,
)


# --- Status Enum ---
class TestCaseStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skipped"
    XFAIL = "xfail"  # Expected failure
    XPASS = "xpass"  # Unexpected pass (when an xfail test passes)


# --- Git and System Info Helpers ---
def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception as e:
        return "Couldn't get git hash!"


def get_hostname():
    return subprocess.check_output(["uname", "-n"]).decode("ascii").strip()


def get_username():
    return os.environ.get("USER", "unknown")


def get_git_author():
    try:
        return subprocess.check_output(["git", "config", "user.name"]).decode("ascii").strip()
    except Exception as e:
        return "Unknown"


def get_git_branch():
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("ascii").strip()
    except Exception as e:
        return "Unknown"


def get_initiated_by():
    ci_pipeline = os.getenv("GITHUB_WORKFLOW") or os.getenv("CI_PIPELINE_NAME")
    if ci_pipeline:
        return ci_pipeline
    else:
        return get_username()


def get_device_arch_name():
    """Get the real device architecture name using ttnn, with fallback to environment variable."""
    try:
        import ttnn

        return ttnn.get_arch_name()
    except ImportError:
        logger.warning("ttnn not available, falling back to ARCH_NAME environment variable")
        return os.getenv("ARCH_NAME", "unknown")
    except Exception as e:
        logger.warning(f"Failed to get device arch from ttnn ({e}), falling back to ARCH_NAME environment variable")
        return os.getenv("ARCH_NAME", "unknown")


# --- Database Operations ---
def create_unit_test_run(start_time_ts, status="success", run_contents=None):
    """Create a new run record in the database for unit tests."""
    return push_run(
        initiated_by=get_initiated_by(),
        host=get_hostname(),
        git_author=get_git_author(),
        git_branch_name=get_git_branch(),
        git_commit_hash=git_hash(),
        start_time_ts=start_time_ts,
        status=status,
        run_contents=run_contents,
        device=get_device_arch_name(),
        run_type="unit_test",
    )


def push_test_and_cases(run_id, file_path, test_cases_results):
    """Push a test (file) and its test cases to the database."""
    test_id = None
    try:
        with postgres_connection() as (conn, cursor):
            # 1. Create a record for the test file
            test_start_time = min(tc["start_time_ts"] for tc in test_cases_results)
            test_end_time = max(tc["end_time_ts"] for tc in test_cases_results)

            test_insert_query = """
            INSERT INTO tests (run_id, name, start_time_ts, end_time_ts, status)
            VALUES (%s, %s, %s, %s, %s) RETURNING id
            """
            # Placeholder status, will be updated later
            cursor.execute(test_insert_query, (run_id, str(file_path), test_start_time, test_end_time, "success"))
            test_id = cursor.fetchone()[0]

            # 2. Insert test cases in batch
            case_statuses = []
            testcase_insert_query = """
            INSERT INTO unit_testcases (test_id, name, start_time_ts, end_time_ts, status, suite_name, test_vector, message, exception, e2e_perf, device_perf, error_signature)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            # Prepare all case values for batch insert
            batch_values = []
            for case in test_cases_results:
                db_status = case["status"].value
                case_statuses.append(db_status)
                exception_text = case["exception"]
                error_sig = generate_error_signature(exception_text)
                case_values = (
                    test_id,
                    case["name"],
                    case["start_time_ts"],
                    case["end_time_ts"],
                    db_status,
                    case.get("suite_name", "default"),  # Default suite name for unit tests
                    json.dumps(case["test_parameters"]) if case.get("test_parameters") else None,
                    case["message"],
                    exception_text,
                    case.get("e2e_perf"),  # Performance data if available
                    json.dumps(case.get("device_perf")) if case.get("device_perf") else None,
                    error_sig,
                )
                batch_values.append(case_values)

            # Execute batch insert
            if batch_values:
                cursor.executemany(testcase_insert_query, batch_values)

            # 3. Determine and update the overall test status
            test_status = map_test_status_to_run_status(case_statuses)
            cursor.execute("UPDATE tests SET status = %s WHERE id = %s", (test_status, test_id))

            logger.info(f"Successfully pushed results for test file: {file_path}")
            return test_status
    except Exception as e:
        logger.error(f"Failed to push test results for {file_path} to PostgreSQL: {e}")
        # If we have a test_id and the error happened after test creation,
        # try to mark the test as 'error' in a separate transaction
        if test_id:
            try:
                with postgres_connection() as (conn, cursor):
                    cursor.execute("UPDATE tests SET status = 'error' WHERE id = %s", (test_id,))
                    logger.info(f"Marked test {test_id} as error after failure")
            except Exception as e2:
                logger.error(f"Could not mark test as error after another failure: {e2}")
        return "error"


# --- Pytest Result Collector ---
class ResultCollector:
    def __init__(self):
        self.test_results = []
        self.test_params = {}
        self.test_start_times = {}

    def pytest_runtest_setup(self, item):
        self.test_start_times[item.nodeid] = dt.datetime.now()
        if hasattr(item, "callspec"):
            self.test_params[item.nodeid] = item.callspec.params

    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            return

        end_time = dt.datetime.now()
        start_time = self.test_start_times.get(report.nodeid, end_time)

        # Default values
        exception_str = ""

        # --- Revised Status Logic (cleaner approach) ---
        if hasattr(report, "wasxfail"):
            # Test was marked with xfail
            if report.skipped:
                # It failed as expected (XFAIL)
                status = TestCaseStatus.XFAIL
                exception_str = f"XFAIL: {report.wasxfail}"
            elif report.passed:
                # It passed unexpectedly (XPASS)
                status = TestCaseStatus.XPASS
                exception_str = f"XPASS: {report.wasxfail}"
            else:
                # This case should not typically happen for xfail, but we handle it
                status = TestCaseStatus.ERROR
                exception_str = report.longreprtext or "Unexpected xfail state"
        elif report.skipped:
            # Regular skip
            status = TestCaseStatus.SKIP
            # report.longrepr is a tuple: (file, line, reason)
            if report.longrepr and len(report.longrepr) >= 3:
                exception_str = report.longrepr[2]  # Extract just the reason string
            else:
                exception_str = str(report.longrepr)
        elif report.failed:
            exception_str = report.longreprtext or "Test failed"
            if "AssertionError" in exception_str:
                status = TestCaseStatus.FAIL
            else:
                status = TestCaseStatus.ERROR
        elif report.passed:
            status = TestCaseStatus.PASS
        else:
            # Catch any other unforeseen outcome
            status = TestCaseStatus.ERROR
            exception_str = "Unknown test outcome"

        message = report.capstdout

        params = self.test_params.get(report.nodeid, {})
        serializable_params = {k: str(v) for k, v in params.items()} if params else None

        full_name = report.nodeid.split("::")[-1]
        # Remove parameters from the test name (they're stored separately in test_parameters)
        if "[" in full_name:
            full_name = full_name.split("[")[0]
        if len(full_name) > 255:
            full_name = full_name[:252] + "..."
            logger.info(f"Full name has been truncated to first 252 characters: {full_name}")

        # Extract suite name from the test file path or use "default"
        suite_name = "default"
        if "::" in report.nodeid:
            test_file = report.nodeid.split("::")[0]
            # Use the test file name (without .py extension) as suite name
            suite_name = test_file.split("/")[-1].replace(".py", "")

        self.test_results.append(
            {
                "name": full_name,
                "status": status,
                "start_time_ts": start_time,
                "end_time_ts": end_time,
                "message": message,
                "exception": exception_str,
                "test_parameters": serializable_params,
                "suite_name": suite_name,
                "e2e_perf": None,  # Can be populated if performance measurement is added
                "device_perf": None,  # Can be populated if device performance measurement is added
            }
        )

    def pytest_sessionfinish(self, session):
        # Clean up dictionaries
        self.test_params.clear()
        self.test_start_times.clear()


# --- Test Discovery and Execution ---
def discover_and_run_tests(test_path: pathlib.Path):
    """Discover and run tests in a given file or directory."""
    if test_path.is_file():
        test_files = [test_path]
    elif test_path.is_dir():
        # Find all python files that start with test_ recursively
        test_files = sorted(test_path.rglob("test_*.py"))
        if not test_files:
            logger.warning(f"No test files found in directory: {test_path}")
            return {}
        logger.info(f"Found {len(test_files)} test files in directory: {test_path}")
        for file in test_files:
            logger.info(f"  - {file}")
    else:
        logger.error(f"Test path {test_path} is not a valid file or directory.")
        return {}

    results_by_file = {}
    for file in test_files:
        logger.info(f"Running tests in: {file}")
        results_by_file[file] = run_tests_in_file(file)

    return results_by_file


def run_tests_in_file(file_path: pathlib.Path):
    """Run all tests in a file using pytest."""
    collector = ResultCollector()
    try:
        exit_code = pytest.main(["-v", str(file_path)], plugins=[collector])
        if exit_code != 0:
            print(f"Error: pytest encountered issues while running tests in {file_path}. Exit code: {exit_code}")
    except Exception as e:
        print(f"Exception occurred while running tests in {file_path}: {e}")
        traceback.print_exc()
        return {}
    return collector.test_results


def collect_tests_in_file(file_path: pathlib.Path):
    """Collect test cases from a file without running them."""

    class TestCollector:
        def __init__(self):
            self.collected_tests = []

        def pytest_collection_modifyitems(self, config, items):
            for item in items:
                params = {}
                if hasattr(item, "callspec"):
                    params = item.callspec.params

                full_name = item.nodeid.split("::")[-1]
                if len(full_name) > 255:
                    full_name = full_name[:252] + "..."

                self.collected_tests.append(
                    {
                        "name": full_name,
                        "file": str(file_path),
                        "test_parameters": {k: str(v) for k, v in params.items()} if params else None,
                    }
                )

    collector = TestCollector()
    pytest.main(["--collect-only", str(file_path)], plugins=[collector])
    return collector.collected_tests


def discover_and_collect_tests(test_path: pathlib.Path):
    """Discover and collect test information without running them."""
    if test_path.is_file():
        test_files = [test_path]
    elif test_path.is_dir():
        test_files = sorted(test_path.rglob("test_*.py"))
        if not test_files:
            logger.warning(f"No test files found in directory: {test_path}")
            return {}
        logger.info(f"Found {len(test_files)} test files in directory: {test_path}")
        for file in test_files:
            logger.info(f"  - {file}")
    else:
        logger.error(f"Test path {test_path} is not a valid file or directory.")
        return {}

    collected_tests_by_file = {}
    for file in test_files:
        logger.info(f"Collecting tests from: {file}")
        collected_tests_by_file[file] = collect_tests_in_file(file)

    return collected_tests_by_file


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unit Test Runner with PostgreSQL reporting.")
    parser.add_argument(
        "test_paths",
        type=str,
        help="A comma-separated list of paths to test files or directories to run. Directories will be searched recursively for files starting with 'test_'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run to count test cases without executing them or connecting to the database.",
    )
    args = parser.parse_args()

    # Handle dry-run mode
    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
        logger.info("Collecting test information without executing tests...")

        # 3. Discover and collect tests
        test_paths_list = [path.strip() for path in args.test_paths.split(",")]
        all_collected_tests = {}
        total_test_cases = 0

        for test_path_str in test_paths_list:
            test_path = pathlib.Path(test_path_str)
            collected_tests_for_path = discover_and_collect_tests(test_path)
            all_collected_tests.update(collected_tests_for_path)

        # Print summary
        logger.info("\n=== DRY RUN SUMMARY ===")
        if not all_collected_tests:
            logger.info("No test files found to analyze.")
        else:
            max_test_cases_per_file = 0
            max_test_cases_file = None

            for file_path, test_cases in all_collected_tests.items():
                test_count = len(test_cases)
                total_test_cases += test_count

                # Track the file with the most test cases
                if test_count > max_test_cases_per_file:
                    max_test_cases_per_file = test_count
                    max_test_cases_file = file_path

                logger.info(f"File: {file_path}")
                logger.info(f"  Test cases: {test_count}")
                if test_cases:
                    for test_case in test_cases[:3]:  # Show first 3 test cases as examples
                        params_str = ""
                        if test_case.get("test_parameters"):
                            params_str = f" (params: {test_case['test_parameters']})"
                        logger.info(f"    - {test_case['name']}{params_str}")
                    if len(test_cases) > 3:
                        logger.info(f"    ... and {len(test_cases) - 3} more test cases")
                logger.info("")

            logger.info(f"Total test files: {len(all_collected_tests)}")
            logger.info(f"Total test cases that would be executed: {total_test_cases}")
            logger.info(f"Maximum test cases per file: {max_test_cases_per_file} (in {max_test_cases_file})")

        logger.info("=== END DRY RUN ===")
        sys.exit(0)

    # Normal execution mode (existing code)
    # 1. Initialize DB
    initialize_postgres_database()

    # 2. Create a new run
    run_start_time = dt.datetime.now()
    test_paths_list = [path.strip() for path in args.test_paths.split(",")]
    run_contents = ", ".join([path.removeprefix("tests/ttnn/unit_tests/") for path in test_paths_list])
    run_id = create_unit_test_run(run_start_time, run_contents=run_contents)

    # 3. Discover and run tests
    all_results = {}
    for test_path_str in test_paths_list:
        test_path = pathlib.Path(test_path_str)
        results_for_path = discover_and_run_tests(test_path)
        all_results.update(results_for_path)

    # 4. Push results to the database
    overall_statuses = []
    if not all_results:
        logger.info("No test results to report.")
        run_status = "success"  # or 'error' if no tests found is an error
    else:
        for file_path, results in all_results.items():
            if results:
                test_status = push_test_and_cases(run_id, file_path, results)
                overall_statuses.append(test_status)
        run_status = map_test_status_to_run_status(overall_statuses)

    # 5. Finalize the run
    run_end_time = dt.datetime.now()
    update_run(run_id, run_end_time, run_status)

    logger.info(f"\nRun completed with status: {run_status.upper()}")
    if run_status == "failure" or run_status == "error":
        sys.exit(1)
