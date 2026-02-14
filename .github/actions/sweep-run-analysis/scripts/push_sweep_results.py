#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Push lead_models sweep results to the minimal schema database."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg2


def get_connection():
    """Get database connection from environment variable."""
    database_url = os.environ.get("TTNN_OPS_DATABASE_URL")
    if not database_url:
        raise ValueError("TTNN_OPS_DATABASE_URL environment variable not set")
    return psycopg2.connect(database_url)


def collect_test_results(results_dir: str) -> list[dict]:
    """Collect all test results from JSON files in the results directory."""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        return []

    # Prefer consolidated run-level exports when available to avoid double-counting.
    # In current sweep workflow, oprun_*.json already contains all per-test records.
    oprun_files = sorted(results_path.glob("oprun_*.json"))
    if oprun_files:
        json_files = oprun_files
        print(f"Found {len(oprun_files)} run-level files (oprun_*.json); using only these for ingestion")
    else:
        # Fallback for older/non-consolidated layouts: ingest non-oprun JSON result files.
        json_files = sorted(p for p in results_path.glob("*.json") if not p.name.startswith("oprun_"))
        print(f"No run-level files found; using {len(json_files)} non-oprun JSON files")

    tests = []
    for json_file in json_files:
        print(f"Reading {json_file}")
        try:
            with open(json_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    tests.extend(data)
                elif isinstance(data, dict):
                    if "tests" in data:
                        tests.extend(data["tests"])
                    else:
                        # Single test result
                        tests.append(data)
        except json.JSONDecodeError as e:
            print(f"WARNING: Failed to parse {json_file}: {e}")
            continue

    print(f"Collected {len(tests)} test results from {results_dir}")
    return tests


def push_results(
    results_dir: str,
    github_pipeline_id: int,
    run_contents: str,
    card_type: str,
    git_sha: str,
    git_branch: str,
) -> int:
    """Push sweep results to database.

    Args:
        results_dir: Directory containing JSON result files
        github_pipeline_id: GitHub Actions run ID
        run_contents: Type of run (e.g., "lead models")
        card_type: Hardware type (e.g., "wormhole_b0", "blackhole")
        git_sha: Git commit SHA
        git_branch: Git branch name

    Returns:
        The run_id of the inserted/updated run
    """
    conn = get_connection()
    cur = conn.cursor()

    try:
        # Collect all test results from JSON files
        tests = collect_test_results(results_dir)

        if not tests:
            print("WARNING: No test results found, inserting empty run")

        # Calculate counts
        pass_count = sum(1 for t in tests if t.get("status") == "pass")
        fail_count = sum(1 for t in tests if str(t.get("status", "")).startswith("fail"))

        # Insert run metadata (upsert)
        cur.execute(
            """
            INSERT INTO sweep_run (
                github_pipeline_id, run_contents, card_type,
                git_sha, git_branch, run_start_ts,
                test_count, pass_count, fail_count
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (github_pipeline_id) DO UPDATE SET
                test_count = EXCLUDED.test_count,
                pass_count = EXCLUDED.pass_count,
                fail_count = EXCLUDED.fail_count,
                run_start_ts = EXCLUDED.run_start_ts
            RETURNING run_id
            """,
            (
                github_pipeline_id,
                run_contents,
                card_type,
                git_sha,
                git_branch,
                datetime.now(timezone.utc),
                len(tests),
                pass_count,
                fail_count,
            ),
        )

        run_id = cur.fetchone()[0]
        print(f"Created/updated run_id={run_id}")

        # Delete existing tests for this run (in case of re-run)
        cur.execute("DELETE FROM sweep_test WHERE run_id = %s", (run_id,))
        deleted = cur.rowcount
        if deleted > 0:
            print(f"Deleted {deleted} existing test records (re-run)")

        # Insert test results in batches for efficiency
        batch_size = 1000
        inserted = 0

        for i in range(0, len(tests), batch_size):
            batch = tests[i : i + batch_size]

            # Build values for batch insert
            values = []
            for t in batch:
                # Extract device_fw_duration from metrics array
                # Structure: metrics: [{"metric_name": "device_DEVICE FW DURATION [ns]", "metric_value": 18068.0}, ...]
                duration = None
                metrics = t.get("metrics") or []
                for m in metrics:
                    if m.get("metric_name") == "device_DEVICE FW DURATION [ns]":
                        duration = m.get("metric_value")
                        break

                values.append(
                    cur.mogrify(
                        "(%s, %s, %s, %s, %s, %s, %s)",
                        (
                            run_id,
                            t.get("full_test_name"),
                            t.get("input_hash"),
                            t.get("op_name"),
                            t.get("model_name"),
                            t.get("status"),
                            duration,
                        ),
                    ).decode("utf-8")
                )

            if values:
                insert_query = """
                    INSERT INTO sweep_test (
                        run_id, full_test_name, input_hash, op_name,
                        model_name, status, device_fw_duration_ns
                    )
                    VALUES """ + ", ".join(
                    values
                )

                cur.execute(insert_query)
                inserted += len(batch)
                print(f"Inserted {inserted}/{len(tests)} test records...")

        conn.commit()
        print(f"Successfully pushed run_id={run_id} with {len(tests)} tests to database")

        # Print summary
        print(f"\nSummary:")
        print(f"  Run ID: {run_id}")
        print(f"  GitHub Pipeline ID: {github_pipeline_id}")
        print(f"  Run Contents: {run_contents}")
        print(f"  Card Type: {card_type}")
        print(f"  Total Tests: {len(tests)}")
        print(f"  Pass Count: {pass_count}")
        print(f"  Fail Count: {fail_count}")
        print(f"  Pass Rate: {pass_count * 100.0 / len(tests):.2f}%" if tests else "N/A")

        return run_id

    except Exception as e:
        conn.rollback()
        print(f"ERROR: Failed to push results: {e}", file=sys.stderr)
        raise
    finally:
        cur.close()
        conn.close()


def main():
    if len(sys.argv) < 3:
        print("Usage: push_sweep_results.py <results_dir> <run_contents>", file=sys.stderr)
        print("  results_dir: Directory containing JSON result files")
        print("  run_contents: Type of run (e.g., 'lead models')")
        print("")
        print("Required environment variables:")
        print("  TTNN_OPS_DATABASE_URL: PostgreSQL connection string")
        print("  GITHUB_RUN_ID: GitHub Actions run ID")
        print("  ARCH_NAME: Hardware architecture (e.g., 'wormhole_b0')")
        print("")
        print("Optional environment variables:")
        print("  GITHUB_SHA: Git commit SHA")
        print("  GITHUB_REF_NAME: Git branch name")
        sys.exit(1)

    results_dir = sys.argv[1]
    run_contents = sys.argv[2]

    # Get required environment variables
    github_pipeline_id = int(os.environ.get("GITHUB_RUN_ID", 0))
    if not github_pipeline_id:
        print("ERROR: GITHUB_RUN_ID environment variable not set", file=sys.stderr)
        sys.exit(1)

    card_type = os.environ.get("ARCH_NAME", "unknown")
    git_sha = os.environ.get("GITHUB_SHA", "")[:8] if os.environ.get("GITHUB_SHA") else ""
    git_branch = os.environ.get("GITHUB_REF_NAME", "")

    print(f"Pushing results to database...")
    print(f"  Results dir: {results_dir}")
    print(f"  Run contents: {run_contents}")
    print(f"  GitHub run ID: {github_pipeline_id}")
    print(f"  Card type: {card_type}")
    print(f"  Git SHA: {git_sha}")
    print(f"  Git branch: {git_branch}")

    try:
        run_id = push_results(
            results_dir=results_dir,
            github_pipeline_id=github_pipeline_id,
            run_contents=run_contents,
            card_type=card_type,
            git_sha=git_sha,
            git_branch=git_branch,
        )
        print(f"\nDone! Run ID: {run_id}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
