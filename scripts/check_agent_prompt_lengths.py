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


# Codex app-server rejects persisted goal objectives over 4000 characters
# (multigoal MAX_GOAL_OBJECTIVE_CHARS). The objective is the prompt after
# HF_MODEL substitution with the leading "/goal" stripped, so measure that,
# substituting a deliberately long model name for safety margin.
APP_SERVER_OBJECTIVE_LIMIT = 4_000
LONG_MODEL_NAME_PLACEHOLDER = "organization-name/Some-Model-99B-A9B-Instruct-v0.1"


def objective_length(prompt: str) -> int:
    text = prompt.replace("HF_MODEL", LONG_MODEL_NAME_PLACEHOLDER).lstrip()
    if text.startswith("/goal"):
        text = text[len("/goal") :]
    return len(text.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="*", help="Prompt files to check.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=APP_SERVER_OBJECTIVE_LIMIT,
        help=(
            "Maximum goal objective length in Unicode characters after HF_MODEL "
            f"substitution (default: {APP_SERVER_OBJECTIVE_LIMIT})."
        ),
    )
    args = parser.parse_args()

    retval = 0
    for path in _prompt_files(args.filenames):
        if not path.is_file():
            continue
        if path.suffix not in ("", ".txt", ".md"):
            continue
        try:
            prompt = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            print(f"{path}: failed to decode as UTF-8: {exc}")
            retval = 1
            continue

        length = objective_length(prompt)
        if length > args.max_chars:
            print(
                f"{path}: objective is {length} characters with a "
                f"{len(LONG_MODEL_NAME_PLACEHOLDER)}-char model name substituted, "
                f"over the {args.max_chars} app-server goal limit."
            )
            retval = 1

    return retval


if __name__ == "__main__":
    sys.exit(main())
