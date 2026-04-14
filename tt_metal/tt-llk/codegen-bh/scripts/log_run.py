#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Log a Blackhole issue solver run to the shared runs directory.

Creates a versioned run directory, parses CLI JSON for tokens/cost,
captures git state, and writes run.json + appends to runs.jsonl.

Usage:
    python scripts/log_run.py \
        --issue 1153 --title "Fix unpack_reduce" \
        --branch $(whoami)/issue-1153-codegen-v1 \
        --status completed --start 2026-04-07T10:00:00Z --end 2026-04-07T11:00:00Z \
        --log-dir /tmp/codegen_bh_logs_20260407/ \
        --model claude-opus-4-6 --repo-root /proj_sw/user_dev/$(whoami)/tt-llk \
        [--batch-id 2026-04-07_bh_batch] \
        [--issues-json artifacts/bh_p2_issues.json] \
        [--evaluation /tmp/eval.json] \
        [--review /tmp/review.json]
"""

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODEGEN_DIR = SCRIPT_DIR.parent
REPO_ROOT_DEFAULT = CODEGEN_DIR.parent
RUNS_BASE_DEFAULT = (
    REPO_ROOT_DEFAULT / "../../llk_code_gen/blackhole_issue_solver"
).resolve()
EXTRACT_SCRIPT = SCRIPT_DIR / "extract_conversation.py"


def parse_cli_json(cli_json_path: Path) -> dict:
    """Parse Claude CLI JSON output for tokens, cost, turns, duration.

    The CLI outputs a JSON array of conversation events. The last entry
    carries aggregated modelUsage and total_cost_usd.
    """
    empty = {"cost_usd": 0, "tokens": {}, "num_turns": 0, "duration_seconds": 0}
    if not cli_json_path.exists():
        return empty

    try:
        data = json.loads(cli_json_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(
            f"  Warning: could not read CLI JSON {cli_json_path}: {e}", file=sys.stderr
        )
        return empty

    if not isinstance(data, list) or len(data) == 0:
        print(f"  Warning: CLI JSON is empty or not a list", file=sys.stderr)
        return empty

    last = data[-1]
    if not isinstance(last, dict):
        return empty

    result = {
        "num_turns": last.get("num_turns", 0),
        "cost_usd": round(last.get("total_cost_usd", 0), 4),
        "duration_seconds": int(last.get("duration_ms", 0) / 1000),
    }

    model_usage = last.get("modelUsage", {})
    if model_usage:
        tokens = {}
        for model, usage in model_usage.items():
            tokens[model] = {
                "input": usage.get("inputTokens", 0),
                "output": usage.get("outputTokens", 0),
                "cache_read": usage.get("cacheReadInputTokens", 0),
                "cache_creation": usage.get("cacheCreationInputTokens", 0),
                "cost_usd": round(usage.get("costUSD", 0), 4),
            }
        # Compute totals across all models
        model_entries = [v for v in tokens.values() if isinstance(v, dict)]
        tokens["total"] = {
            "input": sum(t.get("input", 0) for t in model_entries),
            "output": sum(t.get("output", 0) for t in model_entries),
            "cache_read": sum(t.get("cache_read", 0) for t in model_entries),
            "cache_creation": sum(t.get("cache_creation", 0) for t in model_entries),
            "cost_usd": result["cost_usd"],
        }
        result["tokens"] = tokens
    else:
        result["tokens"] = {}

    return result


def get_changed_files(repo_root: Path) -> list[str]:
    """Get files changed between origin/main and HEAD."""
    try:
        proc = subprocess.run(
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=10,
        )
        return [f.strip() for f in proc.stdout.strip().splitlines() if f.strip()]
    except Exception as e:
        print(f"  Warning: git diff failed: {e}", file=sys.stderr)
        return []


def get_git_commit(repo_root: Path) -> str:
    """Get short HEAD commit hash."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=5,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


def create_run_dir(issue_num: int, runs_base: Path) -> tuple[str, Path]:
    """Create versioned run directory: blackhole_issue_{num}_v{N}."""
    existing = sorted(glob(str(runs_base / f"blackhole_issue_{issue_num}_v*")))
    version = len(existing) + 1
    run_id = f"blackhole_issue_{issue_num}_v{version}"
    run_dir = runs_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def log_run(
    issue_num: int,
    title: str,
    branch: str,
    status: str,
    start_time: str,
    end_time: str,
    tmp_log_dir: Path,
    model: str,
    repo_root: Path,
    runs_base: Path,
    batch_id: str | None = None,
    issues_json: Path | None = None,
    evaluation: dict | None = None,
    review: dict | None = None,
) -> dict:
    """Create run directory, parse metrics, write run.json + runs.jsonl.

    Returns the run entry dict.
    """
    runs_jsonl = runs_base / "runs.jsonl"
    run_id, run_dir = create_run_dir(issue_num, runs_base)

    # Copy logs from temp dir into run dir
    for pattern in [f"issue_{issue_num}.json", f"issue_{issue_num}.log"]:
        src = tmp_log_dir / pattern
        if src.is_file():
            shutil.copy2(src, run_dir)

    # Also copy as cli_output.json for dashboard backfill compatibility
    cli_src = run_dir / f"issue_{issue_num}.json"
    if cli_src.is_file():
        shutil.copy2(cli_src, run_dir / "cli_output.json")

    # Parse CLI JSON for tokens/cost/turns
    cli_json = run_dir / f"issue_{issue_num}.json"
    metrics = parse_cli_json(cli_json)

    # Validate token capture
    if not metrics["tokens"]:
        print(
            "  Warning: NO TOKEN DATA CAPTURED — CLI JSON may be missing or malformed",
            file=sys.stderr,
        )

    # Get git info
    changed_files = get_changed_files(repo_root)
    # Filter out infrastructure files (not agent output)
    INFRA_PREFIXES = (
        ".claude/",
        "codegen-bh/",
        "codegen/",
        "docs/superpowers/",
        "CLAUDE.md",
    )
    changed_files = [
        f
        for f in changed_files
        if not any(f.startswith(p) or f == p for p in INFRA_PREFIXES)
    ]
    git_commit = get_git_commit(repo_root)

    # Copy changed files as snapshots (path separators flattened to underscores)
    # Also save the base (origin/main) version with a "base_" prefix for diffs.
    for fpath in changed_files:
        full = repo_root / fpath
        flat_name = fpath.replace("/", "_")
        if full.is_file():
            try:
                shutil.copy2(full, run_dir / flat_name)
            except OSError:
                pass
        # Save base version from origin/main
        try:
            proc = subprocess.run(
                ["git", "show", f"origin/main:{fpath}"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=repo_root,
            )
            if proc.returncode == 0:
                (run_dir / f"base_{flat_name}").write_text(proc.stdout)
        except Exception:
            pass

    # Fetch issue metadata from cached issues JSON
    issue_meta = {
        "number": issue_num,
        "title": title,
        "url": f"https://github.com/tenstorrent/tt-llk/issues/{issue_num}",
        "labels": [],
    }
    if issues_json and issues_json.exists():
        try:
            data = json.loads(issues_json.read_text())
            for iss in data.get("issues", []):
                if iss["number"] == issue_num:
                    issue_meta["labels"] = iss.get("labels", [])
                    issue_meta["url"] = iss.get("url", issue_meta["url"])
                    break
        except (json.JSONDecodeError, KeyError):
            pass

    # Read test results if the agent wrote them
    tests_total = 0
    tests_passed = 0
    tests_details = []
    test_results_file = run_dir / "test_results.json"
    if test_results_file.exists():
        try:
            tr = json.loads(test_results_file.read_text())
            tests_total = tr.get("tests_total", 0)
            tests_passed = tr.get("tests_passed", 0)
            tests_details = tr.get("tests_details", [])
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: could not read test_results.json: {e}", file=sys.stderr)

    # Build the run entry
    entry = {
        "run_id": run_id,
        "arch": "blackhole",
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": metrics["duration_seconds"],
        "num_turns": metrics["num_turns"],
        "status": status,
        "model": model,
        "run_type": "ci" if batch_id else "manual",
        "cost_usd": metrics["cost_usd"],
        "tokens": metrics["tokens"],
        "issue": issue_meta,
        "changed_files": changed_files,
        "tests_total": tests_total,
        "tests_passed": tests_passed,
        "git_branch": branch,
        "git_commit": git_commit,
        "batch_id": batch_id or None,
        "log_dir": run_id,
    }

    if tests_details:
        entry["tests_details"] = tests_details

    if evaluation:
        entry["evaluation"] = evaluation

    if review:
        entry["review"] = review

    # Write run.json into run directory
    (run_dir / "run.json").write_text(json.dumps(entry, indent=2) + "\n")

    # Append to runs.jsonl (file-locked for parallel safety)
    runs_base.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    with open(runs_jsonl, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        fcntl.flock(f, fcntl.LOCK_UN)

    # Extract readable conversation from CLI JSON
    if EXTRACT_SCRIPT.exists() and cli_json.exists():
        try:
            proc = subprocess.run(
                [sys.executable, str(EXTRACT_SCRIPT), str(run_dir)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode == 0:
                print(proc.stdout.strip())
            else:
                print(
                    f"  Warning: extract_conversation failed: {proc.stderr[:200]}",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"  Warning: could not extract conversation: {e}", file=sys.stderr)

    # Print summary
    cost = metrics["cost_usd"]
    turns = metrics["num_turns"]
    n_files = len(changed_files)
    print(f"  Logged to {run_dir}/")
    print(f"  cost=${cost}  turns={turns}  changed={n_files} files")
    if evaluation:
        print(f"  evaluation: {evaluation.get('overall', '?')}")
    if review:
        print(
            f"  review: {review.get('verdict', '?')} ({len(review.get('comments', []))} comments)"
        )

    return entry


def main():
    parser = argparse.ArgumentParser(description="Log a BH issue solver run")
    parser.add_argument("--issue", type=int, required=True, help="GitHub issue number")
    parser.add_argument("--title", required=True, help="Issue title")
    parser.add_argument("--branch", required=True, help="Git branch name")
    parser.add_argument(
        "--status",
        required=True,
        choices=["completed", "crashed"],
        help="Run exit status",
    )
    parser.add_argument("--start", required=True, help="Start time (ISO8601)")
    parser.add_argument("--end", required=True, help="End time (ISO8601)")
    parser.add_argument(
        "--log-dir", type=Path, required=True, help="Temp log directory"
    )
    parser.add_argument("--model", required=True, help="Model used")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--runs-base", type=Path, default=RUNS_BASE_DEFAULT)
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--issues-json", type=Path, default=None)
    parser.add_argument(
        "--evaluation", type=Path, default=None, help="Path to evaluation JSON file"
    )
    parser.add_argument(
        "--review", type=Path, default=None, help="Path to review JSON file"
    )
    args = parser.parse_args()

    eval_data = None
    if args.evaluation and args.evaluation.exists():
        try:
            eval_data = json.loads(args.evaluation.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    review_data = None
    if args.review and args.review.exists():
        try:
            review_data = json.loads(args.review.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    entry = log_run(
        issue_num=args.issue,
        title=args.title,
        branch=args.branch,
        status=args.status,
        start_time=args.start,
        end_time=args.end,
        tmp_log_dir=args.log_dir,
        model=args.model,
        repo_root=args.repo_root,
        runs_base=args.runs_base,
        batch_id=args.batch_id,
        issues_json=args.issues_json,
        evaluation=eval_data,
        review=review_data,
    )

    # Print the entry for the caller
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
