"""CLI tool to annotate eval runs with satisfaction scores.

Usage:
    python3 -m eval.annotate <run_id> --score 4 --notes "clean but slow"
    python3 -m eval.annotate <run_id> -s 3
"""

import argparse
import sys
from pathlib import Path

from eval import db


def main():
    parser = argparse.ArgumentParser(description="Annotate an eval run")
    parser.add_argument("run_id", type=int, help="Run ID to annotate")
    parser.add_argument(
        "--score",
        "-s",
        type=int,
        required=True,
        choices=range(1, 6),
        help="Satisfaction score (1-5)",
    )
    parser.add_argument("--notes", "-n", default="", help="Optional notes")
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH), help="Database path")
    args = parser.parse_args()

    conn = db.connect(Path(args.db))

    run = db.get_run(conn, args.run_id)
    if not run:
        print(f"Error: Run {args.run_id} not found", file=sys.stderr)
        conn.close()
        return 1

    db.annotate_run(conn, args.run_id, args.score, args.notes)

    print(f"Annotated run {args.run_id} ({run['prompt_name']}): score={args.score}")
    if args.notes:
        print(f"  Notes: {args.notes}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
