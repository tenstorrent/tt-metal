#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Mirror selected skills from .claude/skills into .cursor/skills.

.claude/skills is the source. Each mirrored skill is copied file for file; the
only change is that SKILL.md drops its `user_invocable:` frontmatter line, which
Cursor does not use.

    sync_cursor_skills.py           write the mirror
    sync_cursor_skills.py --check   report drift and exit 1 if there is any
"""

import argparse
import sys
from pathlib import Path

LLK_ROOT = Path(__file__).resolve().parents[2]
SOURCE = LLK_ROOT / ".claude" / "skills"
MIRROR = LLK_ROOT / ".cursor" / "skills"

MIRRORED_SKILLS = (
    "perf-coverage",
    "perf-parameter-impact",
    "perf-regression-check",
    "perf-report",
    "perf-test",
    "quasar-perf-test",
    "quasar-test-coverage",
)


def mirrored_text(path: Path) -> str:
    text = path.read_text()
    if path.name != "SKILL.md":
        return text
    return "".join(
        line
        for line in text.splitlines(keepends=True)
        if not line.startswith("user_invocable:")
    )


def expected_files() -> dict[Path, str]:
    files = {}
    for skill in MIRRORED_SKILLS:
        root = SOURCE / skill
        if not (root / "SKILL.md").is_file():
            raise SystemExit(f"{root / 'SKILL.md'} does not exist")
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            files[MIRROR / path.relative_to(SOURCE)] = mirrored_text(path)
    return files


def drift() -> list[str]:
    """Every difference between the mirror and what sync would write."""
    expected = expected_files()
    problems = []
    for path, text in expected.items():
        rel = path.relative_to(LLK_ROOT)
        if not path.is_file():
            problems.append(f"missing: {rel}")
        elif path.read_text() != text:
            problems.append(f"differs: {rel}")
    for path in sorted(p for p in MIRROR.rglob("*") if p.is_file()):
        skill = path.relative_to(MIRROR).parts[0]
        if skill in MIRRORED_SKILLS and path not in expected:
            problems.append(f"extra: {path.relative_to(LLK_ROOT)}")
    for entry in sorted(MIRROR.iterdir()):
        if entry.is_dir() and entry.name not in MIRRORED_SKILLS:
            problems.append(
                f"unlisted: {entry.relative_to(LLK_ROOT)} is not in MIRRORED_SKILLS"
            )
    return problems


def sync() -> None:
    expected = expected_files()
    for path in sorted(p for p in MIRROR.rglob("*") if p.is_file()):
        if (
            path.relative_to(MIRROR).parts[0] in MIRRORED_SKILLS
            and path not in expected
        ):
            path.unlink()
    for path, text in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report drift only")
    args = parser.parse_args()
    if not args.check:
        sync()
    problems = drift()
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems and args.check:
        print(
            "Run .claude/scripts/sync_cursor_skills.py to update the mirror.",
            file=sys.stderr,
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
