# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Created to model sweeps_runner.py for unit tests to export results to postgres

import argparse
import sys
import pathlib
import importlib.util
import datetime as dt
import os
import json
import subprocess
from enum import Enum
from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger
import pytest

try:
    import psycopg2
except ImportError as e:
    raise RuntimeError(
        "The psycopg2 library is required but not installed. Please install it using 'pip install psycopg2'."
    ) from e
from contextlib import contextmanager


@contextmanager
def postgres_connection():
    """
    Context manager for PostgreSQL database connections.
    Handles connection setup, commit/rollback, and cleanup automatically.

    Usage:
        with postgres_connection() as (conn, cursor):
            cursor.execute("SELECT * FROM table")
            # Connection automatically committed on success or rolled back on error
    """
    pg_config = get_postgres_config()
    conn = None
    cursor = None

    try:
        conn = psycopg2.connect(**pg_config)
        cursor = conn.cursor()
        yield conn, cursor
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


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


def get_postgres_config():
    config = {
        "host": os.getenv("POSTGRES_HOST"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "database": os.getenv("POSTGRES_DATABASE"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }

    required_vars = ["host", "database", "user", "password"]
    missing_keys = [key for key in required_vars if config[key] is None]

    if missing_keys:
        env_vars_to_set = [f"POSTGRES_{key.upper()}" for key in missing_keys]
        raise ValueError(f"Missing required PostgreSQL environment variables: {', '.join(env_vars_to_set)}")

    config["port"] = int(config["port"])
    return config


# --- Database Operations ---
def initialize_postgres_database():
    """Initialize PostgreSQL database with required tables for unit testing."""
    try:
        with postgres_connection() as (conn, cursor):
            # Create tables if they don't exist
            create_run_table_query = """
            CREATE TABLE IF NOT EXISTS runs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                initiated_by VARCHAR(255) NOT NULL,
                host VARCHAR(255),
                device VARCHAR(255),
                type VARCHAR(255),
                run_contents VARCHAR(1024),
                git_author VARCHAR(255),
                git_branch_name VARCHAR(255),
                git_commit_hash VARCHAR(50),
                start_time_ts TIMESTAMP NOT NULL,
                end_time_ts TIMESTAMP,
                status VARCHAR(20) NOT NULL CHECK (status IN ('success', 'failure', 'error', 'cancelled')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """

            create_test_table_query = """
            CREATE TABLE IF NOT EXISTS tests (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                name VARCHAR(255) NOT NULL,
                start_time_ts TIMESTAMP NOT NULL,
                end_time_ts TIMESTAMP,
                status VARCHAR(20) NOT NULL CHECK (status IN ('success', 'failure', 'error', 'cancelled', 'skipped')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """

            create_testcase_table_query = """
            CREATE TABLE IF NOT EXISTS unit_testcases (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                test_id UUID NOT NULL REFERENCES tests(id) ON DELETE CASCADE,
                name VARCHAR(255) NOT NULL,
                start_time_ts TIMESTAMP NOT NULL,
                end_time_ts TIMESTAMP,
                status VARCHAR(100) NOT NULL,
                suite_name VARCHAR(255) NOT NULL,
                test_vector JSONB,
                message TEXT,
                exception TEXT,
                e2e_perf FLOAT,
                device_perf JSONB,
                error_signature VARCHAR(255)
            );
            """

            cursor.execute(create_run_table_query)
            cursor.execute(create_test_table_query)
            cursor.execute(create_testcase_table_query)

            logger.info("Successfully initialized PostgreSQL database.")

    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL database: {e}")
        raise


def push_run(start_time_ts, status="success", run_contents=None):
    """Create a new run record in the database."""
    try:
        with postgres_connection() as (conn, cursor):
            insert_run_query = """
            INSERT INTO runs (initiated_by, host, device, type, run_contents, git_author, git_branch_name, git_commit_hash, start_time_ts, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            run_data = (
                get_initiated_by(),
                get_hostname(),
                get_device_arch_name(),
                "unit_test",
                run_contents,
                get_git_author(),
                get_git_branch(),
                git_hash(),
                start_time_ts,
                status,
            )
            cursor.execute(insert_run_query, run_data)
            run_id = cursor.fetchone()[0]
            logger.info(f"Successfully created run with ID: {run_id}")
            return run_id
    except Exception as e:
        logger.error(f"Failed to push run to PostgreSQL: {e}")
        raise


def update_run(run_id, end_time_ts, status):
    """Update an existing run's status and end time."""
    try:
        with postgres_connection() as (conn, cursor):
            update_run_query = "UPDATE runs SET status = %s, end_time_ts = %s WHERE id = %s"
            cursor.execute(update_run_query, (status, end_time_ts, run_id))
            logger.info(f"Successfully updated run {run_id} with status {status}.")
    except Exception as e:
        logger.error(f"Failed to update run in PostgreSQL: {e}")
        raise


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


def map_test_status_to_run_status(statuses):
    """Aggregate test case statuses to a single test/run status."""
    if not statuses:
        return "error"
    # Hard failures (error, fail, xpass) indicate overall failure
    # XPASS is treated as failure because it's unexpected behavior
    if any(s in ["error", "fail", "xpass"] for s in statuses):
        return "failure"
    if any(s == "cancelled" for s in statuses):
        return "cancelled"
    # If all tests are skipped or expected failures, consider it skipped
    if all(s in ["skipped", "xfail"] for s in statuses):
        return "skipped"
    # If we have any passes (even with some skips/xfails), it's success
    return "success"


def generate_error_signature(exception_message):
    """Generate a concise error signature from an exception message."""
    if not exception_message:
        return None
    # Take the first line of the exception as the signature, capped at 255 chars
    return exception_message.splitlines()[0][:255]


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
    pytest.main(["-v", str(file_path)], plugins=[collector])
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
    run_id = push_run(run_start_time, run_contents=run_contents)

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
