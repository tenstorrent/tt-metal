# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

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
import inspect
import traceback
from enum import Enum
from contextlib import redirect_stdout
import io

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("Please install psycopg2-binary: pip install psycopg2-binary")
    sys.exit(1)

try:
    import pytest
except ImportError:
    print("Please install pytest: pip install pytest")
    sys.exit(1)


# --- Status Enum ---
class TestCaseStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skipped"


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


def get_postgres_config(env="prod"):
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
def initialize_postgres_database(pg_config):
    """Initialize PostgreSQL database with required tables for unit testing."""
    conn = None
    try:
        conn = psycopg2.connect(**pg_config)
        cursor = conn.cursor()

        # Create tables if they don't exist
        create_run_table_query = """
        CREATE TABLE IF NOT EXISTS runs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            initiated_by VARCHAR(255) NOT NULL,
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
            name VARCHAR(1024) NOT NULL,
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
            device VARCHAR(255),
            host VARCHAR(255) NOT NULL,
            start_time_ts TIMESTAMP NOT NULL,
            end_time_ts TIMESTAMP,
            status VARCHAR(100) NOT NULL,
            test_parameters JSONB,
            message TEXT,
            exception TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        cursor.execute(create_run_table_query)
        cursor.execute(create_test_table_query)
        cursor.execute(create_testcase_table_query)

        conn.commit()
        print("Successfully initialized PostgreSQL database.")

    except Exception as e:
        print(f"Failed to initialize PostgreSQL database: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def push_run(pg_config, start_time_ts, status="success"):
    """Create a new run record in the database."""
    conn = None
    try:
        conn = psycopg2.connect(**pg_config)
        cursor = conn.cursor()
        insert_run_query = """
        INSERT INTO runs (initiated_by, git_author, git_branch_name, git_commit_hash, start_time_ts, status)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        run_data = (
            get_initiated_by(),
            get_git_author(),
            get_git_branch(),
            git_hash(),
            start_time_ts,
            status,
        )
        cursor.execute(insert_run_query, run_data)
        run_id = cursor.fetchone()[0]
        conn.commit()
        print(f"Successfully created run with ID: {run_id}")
        return run_id
    except Exception as e:
        print(f"Failed to push run to PostgreSQL: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def update_run(pg_config, run_id, end_time_ts, status):
    """Update an existing run's status and end time."""
    conn = None
    try:
        conn = psycopg2.connect(**pg_config)
        cursor = conn.cursor()
        update_run_query = "UPDATE runs SET status = %s, end_time_ts = %s WHERE id = %s"
        cursor.execute(update_run_query, (status, end_time_ts, run_id))
        conn.commit()
        print(f"Successfully updated run {run_id} with status {status}.")
    except Exception as e:
        print(f"Failed to update run in PostgreSQL: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def push_test_and_cases(pg_config, run_id, file_path, test_cases_results):
    """Push a test (file) and its test cases to the database."""
    conn = None
    test_id = None
    try:
        conn = psycopg2.connect(**pg_config)
        cursor = conn.cursor()

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

        # 2. Insert each test case
        case_statuses = []
        testcase_insert_query = """
        INSERT INTO unit_testcases (test_id, name, device, host, start_time_ts, end_time_ts, status, test_parameters, message, exception)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        for case in test_cases_results:
            db_status = case["status"].value
            case_statuses.append(db_status)
            case_values = (
                test_id,
                case["name"],
                case["device"],
                get_hostname(),
                case["start_time_ts"],
                case["end_time_ts"],
                db_status,
                json.dumps(case["test_parameters"]) if case.get("test_parameters") else None,
                case["message"],
                case["exception"],
            )
            cursor.execute(testcase_insert_query, case_values)

        # 3. Determine and update the overall test status
        test_status = map_test_status_to_run_status(case_statuses)
        cursor.execute("UPDATE tests SET status = %s WHERE id = %s", (test_status, test_id))

        conn.commit()
        print(f"Successfully pushed results for test file: {file_path}")
        return test_status
    except Exception as e:
        print(f"Failed to push test results for {file_path} to PostgreSQL: {e}")
        if conn:
            conn.rollback()
            if test_id:
                # If testcases failed, still try to mark the parent test as 'error'
                try:
                    cursor.execute("UPDATE tests SET status = 'error' WHERE id = %s", (test_id,))
                    conn.commit()
                except Exception as e2:
                    print(f"Could not mark test as error after another failure: {e2}")
                    conn.rollback()
        return "error"
    finally:
        if conn:
            conn.close()


def map_test_status_to_run_status(statuses):
    """Aggregate test case statuses to a single test/run status."""
    if not statuses:
        return "error"
    if any(s in ["error", "fail"] for s in statuses):
        return "failure"
    if any(s == "cancelled" for s in statuses):
        return "cancelled"
    if all(s == "skipped" for s in statuses):
        return "skipped"
    return "success"


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

        exception_str = report.longreprtext
        status = TestCaseStatus.PASS
        if report.skipped:
            status = TestCaseStatus.SKIP
            # For skipped tests, longrepr is a tuple (file, line, reason)
            exception_str = str(report.longrepr)
        elif report.failed:
            if "AssertionError" in exception_str:
                status = TestCaseStatus.FAIL
            else:
                status = TestCaseStatus.ERROR
        elif report.passed:
            status = TestCaseStatus.PASS

        message = report.capstdout

        params = self.test_params.get(report.nodeid, {})
        serializable_params = {k: str(v) for k, v in params.items()} if params else None

        device_name = os.getenv("ARCH_NAME")

        full_name = report.nodeid.split("::")[-1]
        if len(full_name) > 255:
            full_name = full_name[:252] + "..."

        self.test_results.append(
            {
                "name": full_name,
                "status": status,
                "start_time_ts": start_time,
                "end_time_ts": end_time,
                "message": message,
                "exception": exception_str,
                "device": device_name,
                "test_parameters": serializable_params,
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
        # Find all python files that start with test_
        test_files = sorted(test_path.rglob("test_*.py"))
    else:
        print(f"Error: Test path {test_path} is not a valid file or directory.")
        return {}

    results_by_file = {}
    for file in test_files:
        print(f"Running tests in: {file}")
        results_by_file[file] = run_tests_in_file(file)

    return results_by_file


def run_tests_in_file(file_path: pathlib.Path):
    """Run all tests in a file using pytest."""
    collector = ResultCollector()
    pytest.main(["-v", str(file_path)], plugins=[collector])
    return collector.test_results


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unit Test Runner with PostgreSQL reporting.")
    parser.add_argument(
        "test_paths", type=str, help="A comma-separated list of paths to test files or directories to run."
    )
    parser.add_argument(
        "--postgres-env",
        default="dev",
        choices=["dev", "prod"],
        help="PostgreSQL environment to use ('dev' or 'prod').",
    )
    args = parser.parse_args()

    pg_config = get_postgres_config(args.postgres_env)

    # 1. Initialize DB
    initialize_postgres_database(pg_config)

    # 2. Create a new run
    run_start_time = dt.datetime.now()
    run_id = push_run(pg_config, run_start_time)

    # 3. Discover and run tests
    test_paths_list = [path.strip() for path in args.test_paths.split(",")]
    all_results = {}
    for test_path_str in test_paths_list:
        test_path = pathlib.Path(test_path_str)
        results_for_path = discover_and_run_tests(test_path)
        all_results.update(results_for_path)

    # 4. Push results to the database
    overall_statuses = []
    if not all_results:
        print("No test results to report.")
        run_status = "success"  # or 'error' if no tests found is an error
    else:
        for file_path, results in all_results.items():
            if results:
                test_status = push_test_and_cases(pg_config, run_id, file_path, results)
                overall_statuses.append(test_status)
        run_status = map_test_status_to_run_status(overall_statuses)

    # 5. Finalize the run
    run_end_time = dt.datetime.now()
    update_run(pg_config, run_id, run_end_time, run_status)

    print(f"\nRun completed with status: {run_status.upper()}")
    if run_status == "failure" or run_status == "error":
        sys.exit(1)
