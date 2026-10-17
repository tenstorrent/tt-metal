#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Fail if ``pytest -m merge_gate`` under matmul/fused/reduce collects a different count.

Used as the first command of the ``ttnn matmul fused reduction ops`` merge-gate
entry so a parametrize edit cannot quietly drop marks.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from merge_gate_process import COUNT_FILE, FAMILY_DIRS, parse_collected_count, parse_expected_count


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    count_path = repo / COUNT_FILE
    expected = parse_expected_count(count_path.read_text())
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-o",
        "addopts=",
        "-m",
        "merge_gate",
        *FAMILY_DIRS,
    ]
    result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        sys.stderr.write(output)
        sys.stderr.write("collection-count guard: pytest --collect-only failed\n")
        return result.returncode
    collected = parse_collected_count(output)
    if collected != expected:
        sys.stderr.write(
            f"collection-count guard: collected {collected} merge_gate tests, expected {expected} "
            f"(from {COUNT_FILE}). Update the count file when you add or remove marks.\n"
        )
        return 1
    print(f"collection-count guard: {collected} merge_gate tests (matches {COUNT_FILE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
