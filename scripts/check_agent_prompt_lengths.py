#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""pre-commit hook: block oversized agent prompts."""

import argparse
from pathlib import Path
import sys


DEFAULT_PROMPT_ROOT = Path(".agents/prompts")


def _prompt_files(filenames: list[str]) -> list[Path]:
    if filenames:
        return [Path(filename) for filename in filenames]

    if not DEFAULT_PROMPT_ROOT.exists():
        return []

    return sorted(path for path in DEFAULT_PROMPT_ROOT.rglob("*") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="*", help="Prompt files to check.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4096,
        help="Maximum prompt length in Unicode characters (default: 4096).",
    )
    args = parser.parse_args()

    retval = 0
    for path in _prompt_files(args.filenames):
        if not path.is_file():
            continue
        try:
            prompt = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            print(f"{path}: failed to decode as UTF-8: {exc}")
            retval = 1
            continue

        length = len(prompt)
        if length > args.max_chars:
            print(f"{path}: {length} characters exceeds the {args.max_chars} character limit.")
            retval = 1

    return retval


if __name__ == "__main__":
    sys.exit(main())
