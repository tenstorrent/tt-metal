#!/usr/bin/env python3
"""List eval runs with clickable clone paths.

Queries the eval SQLite database and resolves filesystem paths for each run.

Usage:
    python3 -m eval.list_runs                         # last 10 runs
    python3 -m eval.list_runs --last 20               # last 20 runs
    python3 -m eval.list_runs --date today             # today's runs
    python3 -m eval.list_runs --date yesterday         # yesterday's runs
    python3 -m eval.list_runs --date 2026_03_11        # specific date
    python3 -m eval.list_runs --prompt softmax         # filter by prompt
    python3 -m eval.list_runs --branch NewOrchestrator # filter by branch (substring)
    python3 -m eval.list_runs --session 1701           # filter by session time
    python3 -m eval.list_runs --paths-only             # just print paths
"""

import argparse
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

from eval import db


BASE_DIR = Path(f"/localdev/{os.environ.get('USER', 'unknown')}")


def resolve_clone_path(run: dict) -> str | None:
    """Reconstruct the clone directory path from DB fields."""
    branch = run["created_branch"]

    # Parse created_branch: <DATE>_<TIME>_run<N>_<PROMPT>
    # e.g. 2026_03_11_1701_run1_layer_norm_rm
    m = re.match(r"^(\d{4}_\d{2}_\d{2})_(\d{4})_run(\d+)_(.+)$", branch)
    if not m:
        return None

    date_stamp = m.group(1)
    time_stamp = m.group(2)
    run_number = m.group(3)
    prompt_name = m.group(4)

    # Build branch slug from starting_branch
    branch_slug = run["starting_branch"].replace("/", "_")

    session_dir = BASE_DIR / date_stamp / f"{time_stamp}_{branch_slug}"
    clone_dir = session_dir / "clones" / f"{prompt_name}_run{run_number}" / "tt-metal"

    if clone_dir.is_dir():
        return str(clone_dir)

    # Fallback: glob for matching paths if exact match fails
    pattern = str(BASE_DIR / date_stamp / f"{time_stamp}_*" / "clones" / f"{prompt_name}_run{run_number}" / "tt-metal")
    from glob import glob

    matches = glob(pattern)
    if matches:
        return matches[0]

    return str(clone_dir)  # Return reconstructed path even if not found


def parse_date(date_str: str) -> str:
    """Convert date string to YYYY_MM_DD format."""
    if date_str == "today":
        return datetime.now().strftime("%Y_%m_%d")
    elif date_str == "yesterday":
        return (datetime.now() - timedelta(days=1)).strftime("%Y_%m_%d")
    elif re.match(r"^\d{4}[_-]\d{2}[_-]\d{2}$", date_str):
        return date_str.replace("-", "_")
    else:
        raise ValueError(f"Invalid date format: {date_str}. Use 'today', 'yesterday', or YYYY_MM_DD")


def query_runs(
    conn,
    last: int | None = None,
    date: str | None = None,
    prompt: str | None = None,
    branch: str | None = None,
    session: str | None = None,
    grade: str | None = None,
) -> list[dict]:
    """Query runs with optional filters."""
    conditions = []
    params = []

    if date:
        date_stamp = parse_date(date)
        conditions.append("created_branch LIKE ?")
        params.append(f"{date_stamp}%")

    if prompt:
        conditions.append("prompt_name LIKE ?")
        params.append(f"%{prompt}%")

    if branch:
        conditions.append("starting_branch LIKE ?")
        params.append(f"%{branch}%")

    if session:
        # Session time filter: match the HHMM part in created_branch
        conditions.append("created_branch LIKE ?")
        params.append(f"%_{session}_%")

    if grade:
        conditions.append("score_grade = ?")
        params.append(grade.upper())

    where = " AND ".join(conditions) if conditions else "1=1"
    limit = f"LIMIT {last}" if last else ""

    query = f"SELECT * FROM runs WHERE {where} ORDER BY timestamp DESC {limit}"
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "--"
    m, s = divmod(seconds, 60)
    return f"{m}m{s:02d}s"


def format_golden(run: dict) -> str:
    if run["golden_total"] and run["golden_total"] > 0:
        return f"{run['golden_passed'] or 0}/{run['golden_total']}"
    return "--"


def print_runs(runs: list[dict], paths_only: bool = False):
    """Print runs grouped by session."""
    if not runs:
        print("No runs found.")
        return

    if paths_only:
        for run in runs:
            path = resolve_clone_path(run)
            if path:
                print(path)
        return

    # Group by session (date + time)
    sessions = {}
    for run in runs:
        m = re.match(r"^(\d{4}_\d{2}_\d{2}_\d{4})_", run["created_branch"])
        session_key = m.group(1) if m else "unknown"
        branch_key = run["starting_branch"]
        key = f"{session_key} ({branch_key})"
        sessions.setdefault(key, []).append(run)

    for session_key, session_runs in sessions.items():
        print(f"\n{'='*80}")
        print(f"Session: {session_key}  [{len(session_runs)} run(s)]")
        print(f"{'='*80}")

        for run in session_runs:
            path = resolve_clone_path(run)
            exists = Path(path).is_dir() if path else False
            grade = run["score_grade"] or "--"
            golden = format_golden(run)
            dur = format_duration(run.get("duration_seconds"))

            print(f"\n  [{run['id']:>3}] {run['prompt_name']} (run {run['run_number']})")
            print(f"       Grade: {grade}  Golden: {golden}  Duration: {dur}")
            if path:
                marker = "" if exists else " [NOT FOUND]"
                print(f"       Path: {path}{marker}")

    print()


def main():
    parser = argparse.ArgumentParser(description="List eval runs with clone paths")
    parser.add_argument("--last", type=int, default=None, help="Show last N runs (default: all matching)")
    parser.add_argument("--date", help="Filter by date: 'today', 'yesterday', or YYYY_MM_DD")
    parser.add_argument("--prompt", help="Filter by prompt name (substring match)")
    parser.add_argument("--branch", help="Filter by starting branch (substring match)")
    parser.add_argument("--session", help="Filter by session time (HHMM)")
    parser.add_argument("--grade", help="Filter by score grade (A/B/C/D/F)")
    parser.add_argument("--paths-only", action="store_true", help="Print only clone paths")
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH), help="Database path")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # Default to last 10 if no filters specified
    if not any([args.date, args.prompt, args.branch, args.session, args.grade, args.last]):
        args.last = 10

    conn = db.connect(Path(args.db))
    runs = query_runs(
        conn,
        last=args.last,
        date=args.date,
        prompt=args.prompt,
        branch=args.branch,
        session=args.session,
        grade=args.grade,
    )
    conn.close()

    if args.json:
        import json

        enriched = []
        for run in runs:
            run["clone_path"] = resolve_clone_path(run)
            enriched.append(run)
        print(json.dumps(enriched, indent=2))
    else:
        print_runs(runs, paths_only=args.paths_only)


if __name__ == "__main__":
    main()
