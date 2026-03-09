"""Ingest eval run results into the SQLite database.

Called at the end of each run in run_eval.sh to persist results.

Usage:
    python3 -m eval.ingest \
        --prompt-name layer_norm_rm --run-number 1 \
        --starting-branch mare/eval --starting-commit abc123 \
        --created-branch 2026_03_09_1430_run1_layer_norm_rm \
        --test-results /path/to/test_results.json \
        [--score-json /path/to/score.json] \
        [--db /path/to/eval_runs.db]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from eval import db


def ingest_run(
    db_path: Path,
    prompt_name: str,
    run_number: int,
    starting_branch: str,
    starting_commit: str,
    created_branch: str,
    test_results_path: Path = None,
    score_json_path: Path = None,
) -> int:
    """Ingest a single run into the database. Returns run ID."""
    conn = db.connect(db_path)

    timestamp = datetime.now().isoformat()

    # Read score if available
    score_total = None
    score_grade = None
    criteria = []
    if score_json_path and score_json_path.exists():
        score_data = json.loads(score_json_path.read_text())
        score_total = score_data.get("total_score")
        score_grade = score_data.get("grade")
        criteria = score_data.get("criteria", [])

    # Read test results if available
    test_results = []
    golden_passed = None
    golden_total = None
    if test_results_path and test_results_path.exists():
        test_results = json.loads(test_results_path.read_text())
        golden_total = len([r for r in test_results if r["status"] != "skipped"])
        golden_passed = sum(1 for r in test_results if r["status"] == "passed")

    run_id = db.insert_run(
        conn,
        timestamp=timestamp,
        prompt_name=prompt_name,
        run_number=run_number,
        starting_branch=starting_branch,
        starting_commit=starting_commit,
        created_branch=created_branch,
        score_total=score_total,
        score_grade=score_grade,
        golden_passed=golden_passed,
        golden_total=golden_total,
    )

    if test_results:
        db.insert_test_results_batch(conn, run_id, test_results)

    if criteria:
        db.insert_score_criteria(conn, run_id, criteria)

    conn.close()
    return run_id


def main():
    parser = argparse.ArgumentParser(description="Ingest eval run into database")
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH), help="Database path")
    parser.add_argument("--prompt-name", required=True)
    parser.add_argument("--run-number", type=int, required=True)
    parser.add_argument("--starting-branch", required=True)
    parser.add_argument("--starting-commit", required=True)
    parser.add_argument("--created-branch", required=True)
    parser.add_argument("--score-json", help="Path to score.py JSON output")
    parser.add_argument("--test-results", help="Path to test_results.json")
    args = parser.parse_args()

    run_id = ingest_run(
        db_path=Path(args.db),
        prompt_name=args.prompt_name,
        run_number=args.run_number,
        starting_branch=args.starting_branch,
        starting_commit=args.starting_commit,
        created_branch=args.created_branch,
        test_results_path=Path(args.test_results) if args.test_results else None,
        score_json_path=Path(args.score_json) if args.score_json else None,
    )

    print(f"Ingested run {run_id}")


if __name__ == "__main__":
    main()
