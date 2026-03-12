"""Re-score existing eval runs with the current score.py formula.

Walks eval session directories, finds clone+results pairs, optionally
re-runs the self-reflection agent (which feeds the score function),
then re-runs score.py and updates the database.

Usage:
    # Re-score all runs under a session directory
    python3 -m eval.rescore /localdev/$USER/2026_03_10/1430_mybranch

    # Re-score all sessions under a date directory
    python3 -m eval.rescore /localdev/$USER/2026_03_10

    # Re-score everything, scanning all date dirs
    python3 -m eval.rescore /localdev/$USER --all

    # Re-run self-reflection THEN re-score (needed when self-reflection
    # template/agent changed — it produces the helper compliance table,
    # phase timeline, etc. that score.py reads)
    python3 -m eval.rescore /localdev/$USER/2026_03_10/1430_mybranch --re-reflect

    # Dry run — show what would be re-scored without updating DB
    python3 -m eval.rescore /localdev/$USER/2026_03_10 --dry-run

    # Re-score a specific run by DB run ID
    python3 -m eval.rescore --run-id 42 --session-dir /localdev/$USER/2026_03_10/1430_mybranch
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from eval import db

REPO_ROOT = Path(__file__).parent.parent.resolve()
SCORE_SCRIPT = REPO_ROOT / ".claude" / "scripts" / "tdd-pipeline" / "score.py"

# Common locations where ops are generated
OP_SEARCH_PATHS = [
    "ttnn/ttnn/operations/{op_name}",
    "ttnn/cpp/ttnn/operations/{op_name}",
]


def find_op_dir(clone_dir: Path, op_name: str) -> Path | None:
    """Find the operation directory in a clone."""
    for pattern in OP_SEARCH_PATHS:
        candidate = clone_dir / pattern.format(op_name=op_name)
        if candidate.is_dir():
            return candidate
    return None


def find_runs_in_session(session_dir: Path) -> list[dict]:
    """Discover all (clone_dir, results_dir, prompt_name, run_id) tuples in a session."""
    runs = []
    clones_dir = session_dir / "clones"
    results_dir = session_dir / "results"

    if not clones_dir.is_dir() or not results_dir.is_dir():
        return runs

    for clone_entry in sorted(clones_dir.iterdir()):
        if not clone_entry.is_dir():
            continue

        # Parse: {prompt_name}_run{N}
        m = re.match(r"(.+)_run(\d+)$", clone_entry.name)
        if not m:
            continue

        prompt_name = m.group(1)
        run_number = int(m.group(2))
        clone_dir = clone_entry / "tt-metal"
        log_dir = results_dir / prompt_name / f"run_{run_number}"

        if not clone_dir.is_dir():
            continue

        # Determine op_name from the prompt's golden tag
        op_name = None
        for prompt_path in [
            REPO_ROOT / "eval" / "prompts" / f"{prompt_name}.txt",
            clone_dir / "eval" / "prompts" / f"{prompt_name}.txt",
        ]:
            if prompt_path.is_file():
                for line in prompt_path.read_text().splitlines():
                    gm = re.match(r"^# golden: (\S+)", line)
                    if gm:
                        op_name = gm.group(1)
                        break
                if op_name:
                    break

        # Try to infer op_name from results if not found in prompt
        if not op_name:
            # Check if there's a score.json that references the op
            score_json = log_dir / "score.json"
            if score_json.is_file():
                try:
                    data = json.loads(score_json.read_text())
                    op_name = data.get("op_name")
                except (json.JSONDecodeError, OSError):
                    pass

        if not op_name:
            continue

        runs.append(
            {
                "prompt_name": prompt_name,
                "run_number": run_number,
                "clone_dir": clone_dir,
                "log_dir": log_dir,
                "op_name": op_name,
            }
        )

    return runs


def find_sessions(base_dir: Path, scan_all: bool = False) -> list[Path]:
    """Find session directories under a base path.

    A session dir has clones/ and results/ subdirectories.
    If scan_all, walks all date dirs. Otherwise treats base_dir as the session.
    """
    sessions = []

    if (base_dir / "clones").is_dir() and (base_dir / "results").is_dir():
        # base_dir is itself a session
        return [base_dir]

    # Walk one or two levels looking for session dirs
    max_depth = 3 if scan_all else 2
    for depth in range(1, max_depth + 1):
        pattern = "/".join(["*"] * depth)
        for candidate in sorted(base_dir.glob(pattern)):
            if candidate.is_dir() and (candidate / "clones").is_dir():
                sessions.append(candidate)

    return sessions


def re_reflect_run(run_info: dict, dry_run: bool = False) -> bool:
    """Re-run the self-reflection agent on a clone to regenerate self_reflection.md.

    This is needed when the self-reflection agent prompt or template has changed,
    since self_reflection.md contains the helper compliance table, phase timeline
    total, and other data that score.py reads.

    Uses `claude -p` with the ttnn-self-reflection agent prompt against the clone.
    Returns True on success.
    """
    clone_dir = run_info["clone_dir"]
    op_name = run_info["op_name"]
    label = f"{run_info['prompt_name']}:run{run_info['run_number']}"

    op_dir = find_op_dir(clone_dir, op_name)
    if not op_dir:
        print(f"  [SKIP] {label}: op dir not found for re-reflect", file=sys.stderr)
        return False

    if dry_run:
        print(f"  [DRY RUN] {label}: would re-run self-reflection on {op_dir}")
        return True

    prompt = (
        f"Run self-reflection analysis on the operation at {op_dir}. "
        f"The operation name is {op_name}. "
        f"Write the completed report to {op_dir}/self_reflection.md. "
        f"Do NOT update pipeline-improvements.md. "
        f"Do NOT commit anything."
    )

    cmd = [
        "claude",
        "-p",
        "--dangerously-skip-permissions",
        "--max-turns",
        "30",
        "--model",
        "sonnet",
        prompt,
    ]

    print(f"  [REFLECT] {label}: re-running self-reflection agent...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(clone_dir),
        )
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] {label}: self-reflection timed out (10m)", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"  [ERROR] {label}: 'claude' CLI not found", file=sys.stderr)
        return False

    # Check if self_reflection.md was regenerated
    reflection_path = op_dir / "self_reflection.md"
    if reflection_path.is_file():
        print(f"  [REFLECT OK] {label}: self_reflection.md regenerated")

        # Update the artifact in DB if we have a connection
        return True
    else:
        print(f"  [REFLECT WARN] {label}: self_reflection.md not produced (exit={result.returncode})")
        if result.stderr:
            print(f"    stderr: {result.stderr[:300]}", file=sys.stderr)
        return False


def rescore_run(run_info: dict, db_conn, dry_run: bool = False) -> dict | None:
    """Re-score a single run. Returns the new score report or None on failure."""
    clone_dir = run_info["clone_dir"]
    log_dir = run_info["log_dir"]
    op_name = run_info["op_name"]

    op_dir = find_op_dir(clone_dir, op_name)
    if not op_dir:
        return None

    # Check required artifacts exist
    tdd_state = op_dir / ".tdd_state.json"
    if not tdd_state.is_file():
        return None

    # Build score.py command
    cmd = [sys.executable, str(SCORE_SCRIPT), str(op_dir), "--json"]

    golden_results = log_dir / "golden_results.txt"
    if golden_results.is_file():
        cmd += ["--golden-results", str(golden_results)]

    label = f"{run_info['prompt_name']}:run{run_info['run_number']}"

    if dry_run:
        golden_tag = " (with golden)" if golden_results.is_file() else " (no golden)"
        print(f"  [DRY RUN] {label}: would re-score {op_dir}{golden_tag}")
        return {"dry_run": True}

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(clone_dir),
        )
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] {label}: score.py timed out", file=sys.stderr)
        return None

    if result.returncode not in (0, 1):  # 1 = score < 50, still valid
        print(f"  [ERROR] {label}: score.py failed: {result.stderr[:200]}", file=sys.stderr)
        return None

    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  [ERROR] {label}: invalid JSON from score.py", file=sys.stderr)
        return None

    # Write updated score.json to log_dir
    score_json_path = log_dir / "score.json"
    score_json_path.write_text(json.dumps(report, indent=2))

    # Update DB: find matching run and update score
    new_score = report.get("total_score")
    new_grade = report.get("grade")
    criteria = report.get("criteria", [])

    # Extract duration from execution_time criterion
    duration_seconds = None
    for c in criteria:
        if c.get("name") == "execution_time":
            sub = c.get("sub_scores", {})
            duration_seconds = sub.get("overall_duration_s")
            break

    # Find the run in the DB by matching prompt_name + run_number + created_branch
    rows = db_conn.execute(
        "SELECT id FROM runs WHERE prompt_name = ? AND run_number = ?",
        (run_info["prompt_name"], run_info["run_number"]),
    ).fetchall()

    if rows:
        for row in rows:
            run_id = row["id"]
            # Update score
            db_conn.execute(
                "UPDATE runs SET score_total = ?, score_grade = ?, duration_seconds = ? WHERE id = ?",
                (new_score, new_grade, duration_seconds, run_id),
            )
            # Delete old criteria and insert new
            db_conn.execute("DELETE FROM score_criteria WHERE run_id = ?", (run_id,))
            if criteria:
                db.insert_score_criteria(db_conn, run_id, criteria)

        db_conn.commit()
        print(f"  [OK] {label}: {new_score:.1f} ({new_grade}) — updated {len(rows)} DB row(s)")
    else:
        print(f"  [OK] {label}: {new_score:.1f} ({new_grade}) — no matching DB row found, score.json written")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Re-score existing eval runs with the current score.py formula",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Session dir, date dir, or base dir to scan",
    )
    parser.add_argument("--all", action="store_true", help="Scan all date dirs under path")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be re-scored")
    parser.add_argument(
        "--re-reflect",
        action="store_true",
        help="Re-run self-reflection agent before re-scoring (needed when "
        "self-reflection template changed — regenerates helper compliance "
        "table, phase timeline, etc. that score.py reads)",
    )
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH), help="Database path")
    parser.add_argument(
        "--run-id",
        type=int,
        help="Re-score a specific DB run ID (requires --session-dir)",
    )
    parser.add_argument(
        "--session-dir",
        help="Session dir for --run-id mode",
    )

    args = parser.parse_args()

    if not args.path and not args.run_id:
        parser.error("Either a path or --run-id is required")

    db_path = Path(args.db)
    conn = db.connect(db_path) if not args.dry_run else None

    total_rescored = 0
    total_found = 0
    total_failed = 0

    if args.run_id:
        # Single run mode
        if not args.session_dir:
            parser.error("--session-dir is required with --run-id")
        session = Path(args.session_dir)
        all_runs = find_runs_in_session(session)
        # Match by run_id from DB
        if conn:
            row = db.get_run(conn, args.run_id)
            if not row:
                print(f"Run ID {args.run_id} not found in DB", file=sys.stderr)
                sys.exit(1)
            matching = [
                r for r in all_runs if r["prompt_name"] == row["prompt_name"] and r["run_number"] == row["run_number"]
            ]
            if not matching:
                print(f"No matching clone found for run {args.run_id}", file=sys.stderr)
                sys.exit(1)
            for run_info in matching:
                if args.re_reflect:
                    re_reflect_run(run_info, dry_run=args.dry_run)
                result = rescore_run(run_info, conn, dry_run=args.dry_run)
                if result:
                    total_rescored += 1
                else:
                    total_failed += 1
    else:
        base_path = Path(args.path)
        sessions = find_sessions(base_path, scan_all=args.all)

        if not sessions:
            print(f"No eval sessions found under {base_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(sessions)} session(s) to scan\n")

        for session in sessions:
            print(f"Session: {session}")
            runs = find_runs_in_session(session)
            total_found += len(runs)

            if not runs:
                print("  (no runs found)\n")
                continue

            for run_info in runs:
                if args.re_reflect:
                    re_reflect_run(run_info, dry_run=args.dry_run)
                result = rescore_run(run_info, conn, dry_run=args.dry_run)
                if result:
                    total_rescored += 1
                else:
                    total_failed += 1
            print()

    if conn:
        conn.close()

    print(
        f"Done. Rescored: {total_rescored}, Failed: {total_failed}, Total found: {total_found or total_rescored + total_failed}"
    )


if __name__ == "__main__":
    main()
