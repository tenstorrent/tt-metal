from __future__ import annotations

import subprocess
import sys
from pathlib import Path


_TOOL_PATH_PREFIX = "scripts/tt_hw_planner/"


def cmd_commit_tool(args) -> int:
    repo_root = _repo_root()
    changed = _list_changed_files(repo_root)
    tool_changed = [f for f in changed if f.startswith(_TOOL_PATH_PREFIX)]
    non_tool_changed = [f for f in changed if not f.startswith(_TOOL_PATH_PREFIX)]

    if not tool_changed:
        print("  no tt_hw_planner tool files changed")
        return 1

    print(f"  staging {len(tool_changed)} tool file(s):")
    for f in tool_changed:
        print(f"    + {f}")
    if non_tool_changed:
        print(f"  ignoring {len(non_tool_changed)} non-tool file(s) (treated as learnings, not part of the commit):")
        for f in non_tool_changed[:10]:
            print(f"    - {f}")
        if len(non_tool_changed) > 10:
            print(f"    ... ({len(non_tool_changed) - 10} more)")

    add_proc = subprocess.run(
        ["git", "add", *tool_changed],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if add_proc.returncode != 0:
        print(f"  git add failed: {add_proc.stderr.strip()}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("  --dry-run: stopping before commit")
        return 0

    message = args.message
    commit_proc = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(commit_proc.stdout)
    sys.stderr.write(commit_proc.stderr)
    return commit_proc.returncode


def _repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(out.stdout.strip())


def _list_changed_files(repo_root: Path) -> list:
    out = subprocess.run(
        ["git", "status", "--porcelain", "-uall"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    files = []
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        files.append(path)
    return files
