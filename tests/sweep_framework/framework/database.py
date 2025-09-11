# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Common database module for PostgreSQL operations.
Shared by sweeps_runner.py and unit_test_runner.py to avoid code duplication.
"""

import os
import datetime as dt
import json
from contextlib import contextmanager
from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger
import hashlib

try:
    import psycopg2

    PSYCOPG2_AVAILABLE = True
except ImportError as e:
    PSYCOPG2_AVAILABLE = False
    logger.warning(
        "PostgreSQL dependencies not available. If you plan to use database features, "
        "please install psycopg2 using 'pip install psycopg2' or 'pip install psycopg2-binary'"
    )

    # Create mock objects that will raise errors if used
    def _raise_db_error(*args, **kwargs):
        raise RuntimeError(
            "The psycopg2 library is required but not installed. "
            "Please install it using 'pip install psycopg2' or 'pip install psycopg2-binary'."
        ) from e

    class MockPsycopg2:
        connect = _raise_db_error

    psycopg2 = MockPsycopg2()


def get_postgres_config():
    """Get PostgreSQL configuration from environment variables."""
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


@contextmanager
def postgres_connection():
    """
    Context manager for PostgreSQL database connections.
    Handles connection setup, commit/rollback, and cleanup automatically.

    Benefits:
    - Eliminates repetitive try/except/finally blocks
    - Ensures proper resource cleanup (connections, cursors)
    - Automatic transaction management (commit on success, rollback on error)
    - Consistent error handling across all database operations

    Usage:
        with postgres_connection() as (conn, cursor):
            cursor.execute("SELECT * FROM table")
            # Connection automatically committed on success or rolled back on error
    """
    if not PSYCOPG2_AVAILABLE:
        raise RuntimeError(
            "The psycopg2 library is required but not installed. "
            "Please install it using 'pip install psycopg2' or 'pip install psycopg2-binary'."
        )

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


def initialize_postgres_database():
    """Initialize PostgreSQL database with required tables for both sweeps and unit tests."""
    try:
        with postgres_connection() as (conn, cursor):
            # Check if tables already exist
            check_tables_query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('runs', 'tests', 'sweep_testcases', 'unit_testcases')
            """
            cursor.execute(check_tables_query)
            existing_tables = {row[0] for row in cursor.fetchall()}

            if len(existing_tables) >= 3:  # At least runs, tests, and one testcases table
                logger.info("PostgreSQL database already initialized - required tables exist")
                return

            # Create the Run table (shared by both sweeps and unit tests)
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

            # Create the Test table (shared by both sweeps and unit tests)
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

            # Create the Sweep Testcase table (for sweep framework)
            create_sweep_testcase_table_query = """
            CREATE TABLE IF NOT EXISTS sweep_testcases (
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

            # Create the Unit Testcase table (for unit test framework)
            create_unit_testcase_table_query = """
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

            # Execute table creation queries
            cursor.execute(create_run_table_query)
            cursor.execute(create_test_table_query)
            cursor.execute(create_sweep_testcase_table_query)
            cursor.execute(create_unit_testcase_table_query)

            # Create indexes for better query performance
            create_indexes_query = """
            CREATE INDEX IF NOT EXISTS idx_runs_initiated_by ON runs(initiated_by);
            CREATE INDEX IF NOT EXISTS idx_runs_host ON runs(host);
            CREATE INDEX IF NOT EXISTS idx_runs_device ON runs(device);
            CREATE INDEX IF NOT EXISTS idx_runs_git_commit_hash ON runs(git_commit_hash);
            CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
            CREATE INDEX IF NOT EXISTS idx_runs_start_time_ts ON runs(start_time_ts);

            CREATE INDEX IF NOT EXISTS idx_tests_run_id ON tests(run_id);
            CREATE INDEX IF NOT EXISTS idx_tests_name ON tests(name);
            CREATE INDEX IF NOT EXISTS idx_tests_status ON tests(status);
            CREATE INDEX IF NOT EXISTS idx_tests_start_time_ts ON tests(start_time_ts);

            CREATE INDEX IF NOT EXISTS idx_sweep_testcases_test_id ON sweep_testcases(test_id);
            CREATE INDEX IF NOT EXISTS idx_sweep_testcases_name ON sweep_testcases(name);
            CREATE INDEX IF NOT EXISTS idx_sweep_testcases_suite_name ON sweep_testcases(suite_name);
            CREATE INDEX IF NOT EXISTS idx_sweep_testcases_status ON sweep_testcases(status);
            CREATE INDEX IF NOT EXISTS idx_sweep_testcases_start_time_ts ON sweep_testcases(start_time_ts);
            CREATE INDEX IF NOT EXISTS idx_sweep_testcases_error_signature ON sweep_testcases(error_signature);

            CREATE INDEX IF NOT EXISTS idx_unit_testcases_test_id ON unit_testcases(test_id);
            CREATE INDEX IF NOT EXISTS idx_unit_testcases_name ON unit_testcases(name);
            CREATE INDEX IF NOT EXISTS idx_unit_testcases_suite_name ON unit_testcases(suite_name);
            CREATE INDEX IF NOT EXISTS idx_unit_testcases_status ON unit_testcases(status);
            CREATE INDEX IF NOT EXISTS idx_unit_testcases_start_time_ts ON unit_testcases(start_time_ts);
            CREATE INDEX IF NOT EXISTS idx_unit_testcases_error_signature ON unit_testcases(error_signature);
            """

            cursor.execute(create_indexes_query)

            logger.info(
                "Successfully initialized PostgreSQL database with runs, tests, sweep_testcases, and unit_testcases tables"
            )

    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL database: {e}")
        raise


def generate_error_signature(exception_message):
    """Generate a concise error signature from an exception message."""
    if not exception_message:
        return None
    # Take the first line of the exception as the signature, capped at 255 chars
    return exception_message.splitlines()[0][:255]


def generate_error_hash(error_message):
    """Generate SHA-256 hash of error message for grouping and URL filtering."""
    if not error_message:
        return None
    return hashlib.sha256(error_message.encode("utf-8")).hexdigest()


def map_test_status_to_run_status(statuses):
    """
    Aggregate test case statuses to a single test/run status.

    Args:
        statuses: List of status strings

    Returns:
        One of: "success", "failure", "error", "cancelled", "skipped"
    """
    if not statuses:
        return "error"

    # Hard failures (error, fail, xpass) indicate overall failure
    # XPASS is treated as failure because it's unexpected behavior
    failure_statuses = [
        "error",
        "fail",
        "xpass",
        "fail_assert_exception",
        "fail_l1_out_of_mem",
        "fail_watcher",
        "fail_crash_hang",
        "fail_unsupported_device_perf",
    ]
    if any(s in failure_statuses or s.startswith("fail") for s in statuses):
        return "failure"

    # Any cancellation => overall cancelled
    if any(s == "cancelled" for s in statuses):
        return "cancelled"

    # If all tests are skipped or expected failures, consider it skipped
    if all(s in ["skipped", "xfail"] for s in statuses):
        return "skipped"

    # If we have any passes (even with some skips/xfails), it's success
    return "success"


def push_run(
    initiated_by,
    host,
    git_author,
    git_branch_name,
    git_commit_hash,
    start_time_ts,
    status,
    run_contents=None,
    device=None,
    run_type="sweep",
):
    """Create a new run record in the database."""
    try:
        with postgres_connection() as (conn, cursor):
            insert_run_query = """
            INSERT INTO runs (initiated_by, host, device, type, run_contents, git_author, git_branch_name, git_commit_hash, start_time_ts, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            run_data = (
                initiated_by,
                host,
                device,
                run_type,
                run_contents,
                git_author,
                git_branch_name,
                git_commit_hash,
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


def push_test(run_id, name, start_time_ts, end_time_ts, status="success"):
    """Create a new test record in the database."""
    try:
        with postgres_connection() as (conn, cursor):
            insert_test_query = """
            INSERT INTO tests (run_id, name, start_time_ts, end_time_ts, status)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """
            cursor.execute(insert_test_query, (run_id, name, start_time_ts, end_time_ts, status))
            test_id = cursor.fetchone()[0]
            logger.info(f"Successfully created test with ID: {test_id}")
            return test_id
    except Exception as e:
        logger.error(f"Failed to push test to PostgreSQL: {e}")
        raise


def update_test_status(test_id, status):
    """Update a test's status."""
    try:
        with postgres_connection() as (conn, cursor):
            update_test_query = "UPDATE tests SET status = %s WHERE id = %s"
            cursor.execute(update_test_query, (status, test_id))
            logger.info(f"Successfully updated test {test_id} with status {status}.")
    except Exception as e:
        logger.error(f"Failed to update test status in PostgreSQL: {e}")
        raise
