"""Quick one-command ingest: run golden tests and record to dashboard.

Works for any operation, regardless of how it was created (create-op,
interactive session, or run_eval.sh). Run from the repo root.

Usage:
    python3 -m eval.quick_ingest <op_name> [--skip-tests] [--db PATH]

Examples:
    # Run golden tests for layer_norm_rm and ingest results
    python3 -m eval.quick_ingest layer_norm_rm

    # Ingest code only (skip running tests, e.g. if you already have results)
    python3 -m eval.quick_ingest layer_norm_rm --skip-tests

    # Ingest from a specific junit.xml (skip running tests)
    python3 -m eval.quick_ingest layer_norm_rm --junit-xml /tmp/junit.xml
"""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from eval import db
from eval.classify_failures import parse_junit_xml
from eval.ingest import (
    _collect_breadcrumbs,
    _collect_host_code,
    _collect_kernels,
    _collect_self_reflection,
    _collect_tdd_state,
    _find_op_dir,
)


def _git(cmd: str) -> str:
    """Run a git command and return stdout, stripped."""
    result = subprocess.run(["git"] + cmd.split(), capture_output=True, text=True, check=False)
    return result.stdout.strip()


def quick_ingest(
    op_name: str,
    skip_tests: bool = False,
    junit_xml: Path | None = None,
    db_path: Path | None = None,
) -> int:
    """Run golden tests for op_name and ingest into dashboard. Returns run ID."""
    db_path = db_path or db.DEFAULT_DB_PATH
    repo_root = Path(_git("rev-parse --show-toplevel"))
    branch = _git("rev-parse --abbrev-ref HEAD")
    commit = _git("rev-parse --short HEAD")

    golden_dir = repo_root / "eval" / "golden_tests" / op_name
    op_dir = _find_op_dir(repo_root, op_name)

    # --- Run or load test results ---
    test_results = []
    if junit_xml and junit_xml.exists():
        # User provided a pre-existing junit.xml
        print(f"Classifying provided {junit_xml}")
        test_results = parse_junit_xml(junit_xml)
    elif not skip_tests:
        if not golden_dir.is_dir():
            print(f"No golden tests at {golden_dir}, ingesting code only")
        else:
            with tempfile.TemporaryDirectory(prefix="eval_qi_") as tmpdir:
                tmpdir = Path(tmpdir)
                runner = repo_root / "eval" / "eval_test_runner.sh"
                print(f"Running golden tests: {golden_dir}")
                result = subprocess.run(
                    [str(runner), str(golden_dir), str(tmpdir)],
                    cwd=str(repo_root),
                    capture_output=False,
                )
                results_json = tmpdir / "test_results.json"
                if results_json.exists():
                    test_results = json.loads(results_json.read_text())
                else:
                    print("WARNING: No test_results.json produced", file=sys.stderr)

    # --- Compute golden stats ---
    golden_total = len([r for r in test_results if r["status"] != "skipped"]) if test_results else None
    golden_passed = sum(1 for r in test_results if r["status"] == "passed") if test_results else None

    # --- Next run number for this op ---
    conn = db.connect(db_path)
    row = conn.execute("SELECT MAX(run_number) as n FROM runs WHERE prompt_name = ?", (op_name,)).fetchone()
    next_run = (row["n"] or 0) + 1

    # --- Insert run ---
    run_id = db.insert_run(
        conn,
        timestamp=datetime.now().isoformat(),
        prompt_name=op_name,
        run_number=next_run,
        starting_branch=branch,
        starting_commit=commit,
        created_branch=branch,
        score_total=None,
        score_grade=None,
        golden_passed=golden_passed,
        golden_total=golden_total,
    )

    if test_results:
        db.insert_test_results_batch(conn, run_id, test_results)

    # --- Collect source code ---
    if op_dir:
        kernels = _collect_kernels(op_dir)
        if kernels:
            db.insert_kernels(conn, run_id, kernels)

        host_code = _collect_host_code(op_dir, op_name)
        if host_code:
            db.insert_host_code(conn, run_id, host_code)

        reflection = _collect_self_reflection(op_dir)
        if reflection:
            db.insert_artifact(conn, run_id, "self_reflection", reflection)

        tdd_state = _collect_tdd_state(op_dir)
        if tdd_state:
            db.insert_tdd_state(conn, run_id, tdd_state)

        breadcrumbs = _collect_breadcrumbs(op_dir)
        if breadcrumbs:
            db.insert_kw_breadcrumbs(conn, run_id, breadcrumbs)

    conn.commit()
    conn.close()

    # --- Summary ---
    if test_results:
        print(f"Tests: {golden_passed}/{golden_total} passed")
    if op_dir:
        print(f"Op dir: {op_dir}")
    print(f"Ingested as run #{run_id} (run_number={next_run})")
    return run_id


def main():
    parser = argparse.ArgumentParser(description="Quick ingest: run golden tests + record to dashboard")
    parser.add_argument("op_name", help="Operation name (must match golden_tests/<op_name>/ and/or op directory)")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests, ingest code only")
    parser.add_argument("--junit-xml", type=Path, help="Use existing JUnit XML instead of running tests")
    parser.add_argument("--db", type=Path, default=None, help="Database path (default: auto)")
    args = parser.parse_args()

    quick_ingest(
        op_name=args.op_name,
        skip_tests=args.skip_tests,
        junit_xml=args.junit_xml,
        db_path=args.db,
    )


if __name__ == "__main__":
    main()
