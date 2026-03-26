"""CLI helper for incremental run updates from run_eval.sh.

Usage:
    python3 -m eval.update_run create --prompt-name softmax --run-number 1 \
        --starting-branch main --starting-commit abc123 --created-branch run1_softmax
    # prints: RUN_ID=42

    python3 -m eval.update_run status 42 building
    python3 -m eval.update_run phase 42 "tdd_stage_1"
    python3 -m eval.update_run score 42 85.5 B
    python3 -m eval.update_run golden 42 8 10
    python3 -m eval.update_run duration 42 3600
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from eval import db


def main():
    parser = argparse.ArgumentParser(description="Update eval run incrementally")
    parser.add_argument("--db", default=str(db.DEFAULT_SQLITE_PATH))
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = sub.add_parser("create")
    p_create.add_argument("--prompt-name", required=True)
    p_create.add_argument("--run-number", type=int, required=True)
    p_create.add_argument("--starting-branch", required=True)
    p_create.add_argument("--starting-commit", required=True)
    p_create.add_argument("--created-branch", required=True)
    p_create.add_argument("--golden-name")

    # status <run_id> <status>
    p_status = sub.add_parser("status")
    p_status.add_argument("run_id", type=int)
    p_status.add_argument("new_status")

    # phase <run_id> <phase>
    p_phase = sub.add_parser("phase")
    p_phase.add_argument("run_id", type=int)
    p_phase.add_argument("new_phase")

    # score <run_id> <total> <grade>
    p_score = sub.add_parser("score")
    p_score.add_argument("run_id", type=int)
    p_score.add_argument("total", type=float)
    p_score.add_argument("grade")

    # golden <run_id> <passed> <total>
    p_golden = sub.add_parser("golden")
    p_golden.add_argument("run_id", type=int)
    p_golden.add_argument("passed", type=int)
    p_golden.add_argument("total", type=int)

    # duration <run_id> <seconds>
    p_dur = sub.add_parser("duration")
    p_dur.add_argument("run_id", type=int)
    p_dur.add_argument("seconds", type=int)

    args = parser.parse_args()
    db_path = Path(args.db)

    conn = db.connect(db_path)

    if args.command == "create":
        run_id = db.insert_run(
            conn,
            timestamp=datetime.now().isoformat(),
            prompt_name=args.prompt_name,
            run_number=args.run_number,
            starting_branch=args.starting_branch,
            starting_commit=args.starting_commit,
            created_branch=args.created_branch,
            status="queued",
            golden_name=args.golden_name,
        )
        conn.commit()
        # Output in a format bash can eval
        print(f"RUN_ID={run_id}")
    elif args.command == "status":
        db.update_run_status(conn, args.run_id, args.new_status)
        conn.commit()
    elif args.command == "phase":
        db.update_run_phase(conn, args.run_id, args.new_phase)
        conn.commit()
    elif args.command == "score":
        db.update_run_score(conn, args.run_id, args.total, args.grade)
        conn.commit()
    elif args.command == "golden":
        db.update_run_golden(conn, args.run_id, args.passed, args.total)
        conn.commit()
    elif args.command == "duration":
        db.update_run_duration(conn, args.run_id, args.seconds)
        conn.commit()

    conn.close()


if __name__ == "__main__":
    main()
