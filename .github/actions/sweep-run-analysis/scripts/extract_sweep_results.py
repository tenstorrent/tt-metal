#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Extract sweep results from PostgreSQL and detect regressions."""

import json
import os
import sys
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

# Output file for results
RESULTS_FILE = os.environ.get("RESULTS_FILE", "sweep_results.json")


def get_connection():
    """Get database connection from environment variable."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    return psycopg2.connect(database_url)


def get_current_run(conn, github_run_id: int) -> Optional[dict]:
    """Find current run by GitHub pipeline ID."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT run_id, run_contents, card_type, git_sha, git_branch, run_start_ts,
                   test_count, pass_count, fail_count,
                   ROUND(pass_count * 100.0 / NULLIF(test_count, 0), 2) AS pass_pct
            FROM sweep_run
            WHERE github_pipeline_id = %s
            """,
            (github_run_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def get_previous_run(conn, run_contents: str, card_type: str, git_branch: str, current_run_id: int) -> Optional[dict]:
    """Find previous run of same type for comparison."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT run_id, run_start_ts,
                   test_count, pass_count, fail_count,
                   ROUND(pass_count * 100.0 / NULLIF(test_count, 0), 2) AS pass_pct
            FROM sweep_run
            WHERE run_contents = %s
              AND card_type = %s
              AND git_branch = %s
              AND run_id < %s
            ORDER BY run_id DESC
            LIMIT 1
            """,
            (run_contents, card_type, git_branch, current_run_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def detect_pass_rate_regressions(conn, current_run_id: int, prev_run_id: int) -> list[dict]:
    """Detect modules with decreased pass rate."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            WITH current_modules AS (
                SELECT
                    SPLIT_PART(full_test_name, '.', 1) || '.' || SPLIT_PART(full_test_name, '.', 2) AS module_name,
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE status = 'pass') AS passed,
                    COUNT(*) FILTER (WHERE status LIKE 'fail%%') AS failed
                FROM sweep_test
                WHERE run_id = %s
                GROUP BY module_name
            ),
            previous_modules AS (
                SELECT
                    SPLIT_PART(full_test_name, '.', 1) || '.' || SPLIT_PART(full_test_name, '.', 2) AS module_name,
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE status = 'pass') AS passed
                FROM sweep_test
                WHERE run_id = %s
                GROUP BY module_name
            )
            SELECT
                c.module_name AS module,
                ROUND(p.passed * 100.0 / NULLIF(p.total, 0), 2) AS prev,
                ROUND(c.passed * 100.0 / NULLIF(c.total, 0), 2) AS current,
                ROUND(c.passed * 100.0 / NULLIF(c.total, 0), 2) -
                    ROUND(p.passed * 100.0 / NULLIF(p.total, 0), 2) AS delta
            FROM current_modules c
            LEFT JOIN previous_modules p ON c.module_name = p.module_name
            WHERE ROUND(c.passed * 100.0 / NULLIF(c.total, 0), 2) <
                  COALESCE(ROUND(p.passed * 100.0 / NULLIF(p.total, 0), 2), 100)
            ORDER BY delta ASC
            """,
            (current_run_id, prev_run_id),
        )
        return [dict(row) for row in cur.fetchall()]


def detect_perf_regressions_by_op(conn, current_run_id: int, prev_run_id: int) -> list[dict]:
    """Detect operations with >15% performance regression (by average)."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            WITH current_perf AS (
                SELECT op_name, AVG(device_fw_duration_ns) AS avg_duration
                FROM sweep_test
                WHERE run_id = %s
                  AND device_fw_duration_ns IS NOT NULL
                GROUP BY op_name
            ),
            previous_perf AS (
                SELECT op_name, AVG(device_fw_duration_ns) AS avg_duration
                FROM sweep_test
                WHERE run_id = %s
                  AND device_fw_duration_ns IS NOT NULL
                GROUP BY op_name
            )
            SELECT
                c.op_name,
                p.avg_duration AS prev_ns,
                c.avg_duration AS current_ns,
                ROUND(((c.avg_duration - p.avg_duration) / NULLIF(p.avg_duration, 0) * 100)::numeric, 2) AS pct_change
            FROM current_perf c
            JOIN previous_perf p ON c.op_name = p.op_name
            WHERE (c.avg_duration - p.avg_duration) / NULLIF(p.avg_duration, 0) > 0.15
            ORDER BY pct_change DESC
            """,
            (current_run_id, prev_run_id),
        )
        return [dict(row) for row in cur.fetchall()]


def detect_perf_regressions_by_test(conn, current_run_id: int, prev_run_id: int, limit: int = 20) -> list[dict]:
    """Detect individual tests with >15% performance regression."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
                c.full_test_name,
                c.input_hash,
                c.op_name,
                c.model_name,
                p.device_fw_duration_ns AS prev_ns,
                c.device_fw_duration_ns AS current_ns,
                ROUND(((c.device_fw_duration_ns - p.device_fw_duration_ns)
                      / NULLIF(p.device_fw_duration_ns, 0) * 100)::numeric, 2) AS pct_change
            FROM sweep_test c
            JOIN sweep_test p
                ON c.full_test_name = p.full_test_name
                AND c.input_hash = p.input_hash
            WHERE c.run_id = %s
              AND p.run_id = %s
              AND c.device_fw_duration_ns IS NOT NULL
              AND p.device_fw_duration_ns IS NOT NULL
              AND (c.device_fw_duration_ns - p.device_fw_duration_ns)
                  / NULLIF(p.device_fw_duration_ns, 0) > 0.15
            ORDER BY pct_change DESC
            LIMIT %s
            """,
            (current_run_id, prev_run_id, limit),
        )
        return [dict(row) for row in cur.fetchall()]


def get_models_affected(conn, current_run_id: int, prev_run_id: Optional[int]) -> list[dict]:
    """Get models affected by failures and/or performance regressions."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Get failure counts per model
        cur.execute(
            """
            SELECT model_name, COUNT(*) AS failure_count
            FROM sweep_test
            WHERE run_id = %s
              AND status LIKE 'fail%%'
              AND model_name IS NOT NULL
            GROUP BY model_name
            ORDER BY failure_count DESC
            """,
            (current_run_id,),
        )
        failure_counts = {row["model_name"]: row["failure_count"] for row in cur.fetchall()}

        # Get new failures (tests that passed before but fail now)
        new_failure_counts = {}
        if prev_run_id:
            cur.execute(
                """
                SELECT c.model_name, COUNT(*) AS new_failures
                FROM sweep_test c
                JOIN sweep_test p
                    ON c.full_test_name = p.full_test_name
                    AND c.input_hash = p.input_hash
                WHERE c.run_id = %s
                  AND p.run_id = %s
                  AND c.status LIKE 'fail%%'
                  AND p.status = 'pass'
                  AND c.model_name IS NOT NULL
                GROUP BY c.model_name
                """,
                (current_run_id, prev_run_id),
            )
            new_failure_counts = {row["model_name"]: row["new_failures"] for row in cur.fetchall()}

        # Get perf regression counts per model
        perf_regression_counts = {}
        if prev_run_id:
            cur.execute(
                """
                SELECT c.model_name, COUNT(*) AS perf_regressions
                FROM sweep_test c
                JOIN sweep_test p
                    ON c.full_test_name = p.full_test_name
                    AND c.input_hash = p.input_hash
                WHERE c.run_id = %s
                  AND p.run_id = %s
                  AND c.device_fw_duration_ns IS NOT NULL
                  AND p.device_fw_duration_ns IS NOT NULL
                  AND (c.device_fw_duration_ns - p.device_fw_duration_ns)
                      / NULLIF(p.device_fw_duration_ns, 0) > 0.15
                  AND c.model_name IS NOT NULL
                GROUP BY c.model_name
                """,
                (current_run_id, prev_run_id),
            )
            perf_regression_counts = {row["model_name"]: row["perf_regressions"] for row in cur.fetchall()}

        # Combine into unified list
        all_models = set(failure_counts.keys()) | set(new_failure_counts.keys()) | set(perf_regression_counts.keys())
        result = []
        for model in all_models:
            entry = {"model_name": model}
            if model in new_failure_counts:
                entry["new_failures"] = new_failure_counts[model]
            if model in perf_regression_counts:
                entry["perf_regressions"] = perf_regression_counts[model]
            # Only include if there's something to report
            if "new_failures" in entry or "perf_regressions" in entry:
                result.append(entry)

        # Sort by total impact (new_failures + perf_regressions)
        result.sort(
            key=lambda x: (x.get("new_failures", 0) + x.get("perf_regressions", 0)),
            reverse=True,
        )
        return result


def get_models_tested(conn, current_run_id: int) -> list[str]:
    """Get list of all models tested in this run."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT model_name
            FROM sweep_test
            WHERE run_id = %s
              AND model_name IS NOT NULL
            ORDER BY model_name
            """,
            (current_run_id,),
        )
        return [row[0] for row in cur.fetchall()]


def main():
    github_run_id = int(os.environ.get("SOURCE_GITHUB_RUN_ID", os.environ.get("GITHUB_RUN_ID", 0)))
    if not github_run_id:
        print("ERROR: SOURCE_GITHUB_RUN_ID environment variable not set", file=sys.stderr)
        sys.exit(1)

    try:
        conn = get_connection()
    except Exception as e:
        print(f"ERROR: Failed to connect to database: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # Get current run by source workflow run id.
        # This must match the exact run pushed by ttnn-run-sweeps.
        current_run = get_current_run(conn, github_run_id)
        if not current_run:
            print(f"ERROR: No run found for github_pipeline_id={github_run_id}", file=sys.stderr)
            sys.exit(1)

        current_run_id = current_run["run_id"]
        print(f"Found current run: run_id={current_run_id}")

        # Get previous run for comparison
        prev_run = get_previous_run(
            conn,
            current_run["run_contents"],
            current_run["card_type"],
            current_run["git_branch"],
            current_run_id,
        )

        comparison_available = prev_run is not None
        prev_run_id = prev_run["run_id"] if prev_run else None

        if prev_run:
            print(f"Found previous run for comparison: run_id={prev_run_id}")
        else:
            print("No previous run found - this is the first run of this type")

        # Build results
        results = {
            "run_id": current_run_id,
            "run_summary": {
                "test_count": current_run["test_count"],
                "pass_count": current_run["pass_count"],
                "fail_count": current_run["fail_count"],
                "pass_pct": float(current_run["pass_pct"]) if current_run["pass_pct"] else 0,
                "prev_pass_pct": float(prev_run["pass_pct"]) if prev_run and prev_run["pass_pct"] else None,
                "card_type": current_run["card_type"],
                "git_sha": current_run["git_sha"],
                "git_branch": current_run["git_branch"],
            },
            "pass_rate_regressions": [],
            "perf_regressions_by_op": [],
            "perf_regressions_by_test": [],
            "models_affected": [],
            "models_tested": get_models_tested(conn, current_run_id),
            "comparison_available": comparison_available,
        }

        # Detect regressions if we have a comparison baseline
        if comparison_available:
            results["pass_rate_regressions"] = detect_pass_rate_regressions(conn, current_run_id, prev_run_id)
            results["perf_regressions_by_op"] = detect_perf_regressions_by_op(conn, current_run_id, prev_run_id)
            results["perf_regressions_by_test"] = detect_perf_regressions_by_test(conn, current_run_id, prev_run_id)
            results["models_affected"] = get_models_affected(conn, current_run_id, prev_run_id)

            print(f"Detected {len(results['pass_rate_regressions'])} pass rate regressions")
            print(f"Detected {len(results['perf_regressions_by_op'])} op-level perf regressions")
            print(f"Detected {len(results['perf_regressions_by_test'])} test-level perf regressions")
            print(f"Models affected: {len(results['models_affected'])}")

        # Write results to file
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results written to {RESULTS_FILE}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
