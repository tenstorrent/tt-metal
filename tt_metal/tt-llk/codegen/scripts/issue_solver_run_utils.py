#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Small utilities for issue-solver runs.

These helpers stay intentionally boring: prompts can call them from Bash, and
tests can exercise the behavior without involving Claude.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def cmd_upsert_runs_jsonl(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    runs_jsonl = Path(args.runs_jsonl)
    run_path = log_dir / "run.json"
    if not run_path.exists():
        raise SystemExit(f"run.json not found: {run_path}")
    run = json.loads(run_path.read_text())
    run_id = run.get("run_id")
    if not run_id:
        raise SystemExit(f"run.json missing run_id: {run_path}")

    lock_path = runs_jsonl.parent / ".runs.jsonl.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            os.chmod(lock_path, 0o664)
        except OSError:
            pass
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = _read_jsonl(runs_jsonl)
        replaced = False
        for i, row in enumerate(rows):
            if row.get("run_id") == run_id:
                rows[i] = run
                replaced = True
                break
        if not replaced:
            rows.append(run)

        payload = "".join(json.dumps(row) + "\n" for row in rows)
        _atomic_write(runs_jsonl, payload)
    action = "updated" if replaced else "appended"
    print(f"runs-jsonl-{action}: {run_id}")


def cmd_autodebug(args: argparse.Namespace) -> None:
    """Run the installed inspection-only launcher within the current worker retry."""
    log_dir = Path(args.log_dir).resolve()
    run = json.loads((log_dir / "run.json").read_text())
    package = (run.get("solver_plugins") or {}).get("tt-autodebug")
    if not package:
        raise SystemExit("tt-autodebug is not configured for this run")
    launcher = Path(package["path"]) / "skills/autodebug/scripts/autodebug.sh"
    worktree = Path(args.worktree).resolve()
    report = worktree / "AUTODEBUG.md"
    if report.exists() or report.is_symlink():
        raise SystemExit("refusing to overwrite pre-existing AUTODEBUG.md")
    directory = Path(tempfile.mkdtemp(prefix="autodebug-", dir=log_dir))
    env = dict(os.environ)
    # A fresh Claude process must not inherit the parent's nesting sentinel.
    env.pop("CLAUDECODE", None)
    env.pop("AUTODEBUG_CLAUDE_MODEL", None)
    command = ["bash", str(launcher), "--agent", "claude", "--effort", "high"]
    if env.get("CODEGEN_MODEL") and not env.get("ANTHROPIC_BASE_URL"):
        command += ["--model", env["CODEGEN_MODEL"]]
    command += ["--", args.problem]
    with (directory / "launcher.log").open("w") as output:
        proc = subprocess.Popen(
            command,
            cwd=worktree,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = proc.wait(timeout=args.timeout)
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            if report.is_file() and not report.is_symlink():
                shutil.move(str(report), str(directory / "AUTODEBUG.md"))
            print(directory)
    if code or not (directory / "AUTODEBUG.md").is_file():
        raise SystemExit(f"AutoDebug did not produce a successful report: {directory}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    upsert = sub.add_parser(
        "upsert-runs-jsonl", help="Insert or update a run in runs.jsonl"
    )
    upsert.add_argument("--log-dir", required=True)
    upsert.add_argument("--runs-jsonl", required=True)
    upsert.set_defaults(func=cmd_upsert_runs_jsonl)

    debug = sub.add_parser(
        "autodebug", help="Bounded installed AutoDebug investigation"
    )
    debug.add_argument("--log-dir", required=True)
    debug.add_argument("--worktree", required=True)
    debug.add_argument("--problem", required=True)
    debug.add_argument("--timeout", type=int, default=1800)
    debug.set_defaults(func=cmd_autodebug)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
