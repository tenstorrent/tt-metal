#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Run test functions a PR adds or edits under matmul/fused/reduce.

The merge-gate ``-m merge_gate`` entry does not run unmarked tests. This script
is the 5-minute backstop that would have caught a PR whose own new test was red.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from merge_gate_process import (
    ensure_revision,
    git_name_status,
    git_unified_diff,
    resolve_diff_base,
    select_pytest_targets,
)


def _read_files(repo: Path, paths: list[str]) -> dict[str, str]:
    contents: dict[str, str] = {}
    for path in paths:
        full = repo / path
        if full.is_file():
            contents[path] = full.read_text()
    return contents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=None, help="git revision to diff against (default: PR base or origin/main)")
    parser.add_argument("--dry-run", action="store_true", help="print selected pytest targets and exit")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    base = ensure_revision(resolve_diff_base(args.base))
    name_status = git_name_status(base)
    diff_text = git_unified_diff(base)
    contents = _read_files(repo, [path for _, path in name_status])
    targets = select_pytest_targets(name_status, diff_text, contents)
    if not targets:
        print(f"changed-test gate: no matmul/fused/reduce test functions changed vs {base}")
        return 0
    print("changed-test gate: running")
    for target in targets:
        print(f"  {target}")
    if args.dry_run:
        return 0
    cmd = [sys.executable, "-m", "pytest", "--timeout", "60", "-xv", *targets]
    return subprocess.run(cmd, cwd=repo, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
