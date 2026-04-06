#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
KB_ROOT = REPO_ROOT / "agent_kb"


def run_rg(query: str, roots: list[Path]) -> int:
    rg = shutil.which("rg")
    if rg is None:
        return -1

    cmd = [rg, "-n", "-S", query]
    cmd.extend(str(root) for root in roots if root.exists())
    return subprocess.run(cmd, check=False).returncode


def fallback_search(query: str, roots: list[Path]) -> int:
    found = False
    lowered = query.lower()
    for root in roots:
        if not root.exists():
            continue
        files = [root] if root.is_file() else sorted(root.rglob("*.md"))
        for path in files:
            if path.name.startswith("."):
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            matches = []
            for index, line in enumerate(lines, start=1):
                if lowered in line.lower():
                    matches.append(f"{path.relative_to(REPO_ROOT)}:{index}:{line}")
            if matches:
                found = True
                print("\n".join(matches))
    return 0 if found else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Search the tt-metal agent knowledge base.")
    parser.add_argument("query", help="Literal query string to search for.")
    parser.add_argument(
        "--sources",
        action="store_true",
        help="Also search high-value raw sources alongside the KB.",
    )
    args = parser.parse_args()

    roots = [KB_ROOT]
    if args.sources:
        roots.extend(
            [
                REPO_ROOT / "METALIUM_GUIDE.md",
                REPO_ROOT / "CONTRIBUTING.md",
                REPO_ROOT / "docs/source/tt-metalium",
                REPO_ROOT / "tech_reports/op_kernel_dev",
                REPO_ROOT / "tt_metal/programming_examples",
            ]
        )

    rc = run_rg(args.query, roots)
    if rc == -1:
        return fallback_search(args.query, roots)
    return rc


if __name__ == "__main__":
    sys.exit(main())
