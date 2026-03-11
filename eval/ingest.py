"""Ingest eval run results into the SQLite database.

Called at the end of each run in run_eval.sh to persist results.

Usage:
    python3 -m eval.ingest \
        --prompt-name layer_norm_rm --run-number 1 \
        --starting-branch mare/eval --starting-commit abc123 \
        --created-branch 2026_03_09_1430_run1_layer_norm_rm \
        --test-results /path/to/test_results.json \
        [--score-json /path/to/score.json] \
        [--clone-dir /path/to/clone --op-name layer_norm_rm] \
        [--db /path/to/eval_runs.db]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from eval import db

# Common locations where ops are generated
OP_SEARCH_PATHS = [
    "ttnn/ttnn/operations/{op_name}",
    "ttnn/cpp/ttnn/operations/{op_name}",
]

KERNEL_EXTENSIONS = ("*.cpp", "*.hpp")


def _find_op_dir(clone_dir: Path, op_name: str) -> Path | None:
    """Find the operation directory in the clone."""
    for pattern in OP_SEARCH_PATHS:
        candidate = clone_dir / pattern.format(op_name=op_name)
        if candidate.is_dir():
            return candidate
    return None


def _read_source(path: Path) -> str | None:
    """Read a source file, returning None if empty or unreadable."""
    try:
        content = path.read_text()
        return content if content.strip() else None
    except (OSError, UnicodeDecodeError):
        return None


def _collect_kernels(op_dir: Path) -> list:
    """Collect kernel C++ files from kernels/ subdirectory."""
    kernels = []
    kernel_dir = op_dir / "kernels"
    if not kernel_dir.is_dir():
        return kernels

    for ext in KERNEL_EXTENSIONS:
        for path in sorted(kernel_dir.glob(ext)):
            source = _read_source(path)
            if source:
                kernels.append({"filename": path.name, "source_code": source})
    return kernels


def _collect_host_code(op_dir: Path, op_name: str) -> list:
    """Collect host-side Python files (program descriptor + entry point)."""
    files = []
    for name in (f"{op_name}_program_descriptor.py", f"{op_name}.py"):
        path = op_dir / name
        if path.is_file():
            source = _read_source(path)
            if source:
                files.append({"filename": path.name, "source_code": source})
    return files


def _collect_self_reflection(op_dir: Path) -> str | None:
    """Read self_reflection.md if it exists."""
    for name in ("self_reflection.md", "self-reflection.md"):
        path = op_dir / name
        if path.is_file():
            try:
                return path.read_text()
            except (OSError, UnicodeDecodeError):
                pass
    return None


def ingest_run(
    db_path: Path,
    prompt_name: str,
    run_number: int,
    starting_branch: str,
    starting_commit: str,
    created_branch: str,
    test_results_path: Path = None,
    score_json_path: Path = None,
    clone_dir: Path = None,
    op_name: str = None,
    golden_name: str = None,
) -> int:
    """Ingest a single run into the database. Returns run ID."""
    conn = db.connect(db_path)

    timestamp = datetime.now().isoformat()

    # Read score if available
    score_total = None
    score_grade = None
    criteria = []
    duration_seconds = None
    if score_json_path and score_json_path.exists():
        score_data = json.loads(score_json_path.read_text())
        score_total = score_data.get("total_score")
        score_grade = score_data.get("grade")
        criteria = score_data.get("criteria", [])
        # Extract duration from execution_time criterion
        for c in criteria:
            if c.get("name") == "execution_time":
                sub = c.get("sub_scores", {})
                duration_seconds = sub.get("overall_duration_s")
                break

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
        golden_name=golden_name,
        duration_seconds=duration_seconds,
    )

    if test_results:
        db.insert_test_results_batch(conn, run_id, test_results)

    if criteria:
        db.insert_score_criteria(conn, run_id, criteria)

    # Collect kernels and artifacts from clone
    if clone_dir and op_name:
        op_dir = _find_op_dir(clone_dir, op_name)
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

    conn.commit()
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
    parser.add_argument("--clone-dir", help="Path to the clone directory")
    parser.add_argument("--op-name", help="Operation name (for kernel/artifact collection)")
    parser.add_argument("--golden-name", help="Golden test suite name (from prompt '# golden:' tag)")
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
        clone_dir=Path(args.clone_dir) if args.clone_dir else None,
        op_name=args.op_name,
        golden_name=args.golden_name,
    )

    print(f"Ingested run {run_id}")


if __name__ == "__main__":
    main()
