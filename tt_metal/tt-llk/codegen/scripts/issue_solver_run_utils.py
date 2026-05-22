#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Small guardrails for issue-solver runs.

These helpers stay intentionally boring: prompts can call them from Bash, and
tests can exercise the behavior without involving Claude.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or proc.stdout.strip() or proc.returncode)
    return proc.stdout


def _changed_files(worktree: Path) -> list[str]:
    outputs = [
        _run(["git", "-C", str(worktree), "diff", "--name-only"]),
        _run(["git", "-C", str(worktree), "diff", "--name-only", "--cached"]),
        _run(
            ["git", "-C", str(worktree), "ls-files", "--others", "--exclude-standard"]
        ),
    ]
    changed: list[str] = []
    seen: set[str] = set()
    for out in outputs:
        for line in out.splitlines():
            path = line.strip()
            if path and path not in seen:
                changed.append(path)
                seen.add(path)
    return changed


def cmd_check_scope(args: argparse.Namespace) -> None:
    worktree = Path(args.worktree).resolve()
    allowed_prefixes = tuple(p.rstrip("/") + "/" for p in args.allowed_prefix)
    changed = _changed_files(worktree)
    out_of_scope = [path for path in changed if not path.startswith(allowed_prefixes)]
    if out_of_scope:
        print(
            "SCOPE_VIOLATION: changed files outside allowed prefixes:", file=sys.stderr
        )
        for path in out_of_scope:
            print(f"  {path}", file=sys.stderr)
        raise SystemExit(2)
    print(f"scope-ok: {len(changed)} changed file(s)")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp, 0o664)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    scope = sub.add_parser(
        "check-scope", help="Reject changed files outside allowed prefixes"
    )
    scope.add_argument("--worktree", required=True)
    scope.add_argument(
        "--allowed-prefix",
        action="append",
        default=["tt_metal/tt-llk"],
        help="Allowed git diff path prefix. May be repeated.",
    )
    scope.set_defaults(func=cmd_check_scope)

    upsert = sub.add_parser(
        "upsert-runs-jsonl", help="Insert or update a run in runs.jsonl"
    )
    upsert.add_argument("--log-dir", required=True)
    upsert.add_argument("--runs-jsonl", required=True)
    upsert.set_defaults(func=cmd_upsert_runs_jsonl)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
