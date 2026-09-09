#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check API header guards and literal include boundaries without a build.

This is a lexical check of direct includes, not a C++ dependency resolver.
See tt_metal/api/README.md for its scope and the compiler-based follow-up.
"""

import argparse
import json
import re
from pathlib import Path

API_ROOT = Path("tt_metal/api")
EXCEPTIONS = Path("scripts/validate_api/header_hygiene_exceptions.json")
HEADER_SUFFIXES = {".h", ".hpp", ".hh", ".hxx"}
ALLOWED_DEPENDENCIES = {
    "stable": {"stable"},
    "experimental": {"stable", "experimental"},
    "internal": {"stable", "experimental", "internal"},
}
# Preserve ordinary string/character literals so comment markers in them are not
# interpreted as comments. Mask raw strings too: they can contain fake directives.
COMMENTS_AND_LITERALS = re.compile(
    r'R"(?P<delimiter>[^\s()\\]{0,16})\(.*?\)(?P=delimiter)"'
    r'|"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'"
    r"|//[^\n]*|/\*.*?\*/",
    re.DOTALL,
)
INCLUDE = re.compile(r'^\s*#\s*include\s*(?:<([^>]+)>|"([^"]+)")\s*$')
PRAGMA_ONCE = re.compile(r"^\s*#\s*pragma\s+once\s*$")


def source_lines(text: str) -> list[tuple[int, str]]:
    """Splice continued lines, then mask comments while retaining source locations."""
    lines = []
    pending = []
    start = 1
    for number, line in enumerate(text.splitlines(), 1):
        if not pending:
            start = number
        if line.endswith("\\"):
            pending.append(line[:-1])
            continue
        lines.append((start, "".join(pending) + line))
        pending = []
    if pending:
        lines.append((start, "".join(pending)))

    def mask(match: re.Match) -> str:
        value = match.group()
        if value.startswith(("//", "/*", 'R"')):
            return "".join("\n" if c == "\n" else " " for c in value)
        return value

    cleaned = COMMENTS_AND_LITERALS.sub(mask, "\n".join(line for _, line in lines))
    return [(number, line) for (number, _), line in zip(lines, cleaned.split("\n"))]


def tier(path: str) -> str | None:
    if path.startswith("tt-metalium/internal/"):
        return None  # Internal headers belong in api/internal/.
    if path.startswith("tt-metalium/experimental/"):
        return "experimental"
    if path.startswith("tt-metalium/"):
        return "stable"
    if path.startswith("internal/"):
        return "internal"
    return None


def include_target(api_root: Path, source: Path, name: str, quoted: bool) -> str | None:
    """Recognize API-root includes and quoted relative includes, normalizing '..'."""
    candidates = [api_root / name]
    if quoted:
        candidates.insert(0, source.parent / name)
    # Prefer an existing relative header, as the preprocessor would. A missing
    # canonical API include still has a tier; compilation checks its existence.
    target = next((p for p in candidates if p.is_file()), api_root / name)
    try:
        return target.resolve().relative_to(api_root.resolve()).as_posix()
    except ValueError:
        return None


def load_exceptions(path: Path) -> set[tuple[str, str]]:
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError("include exceptions must be a JSON array")
    exceptions = set()
    for entry in entries:
        fields = {"source", "include", "owner", "reason", "remove_when"}
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError(f"each exception must contain exactly {sorted(fields)}")
        if any(not isinstance(value, str) or not value.strip() for value in entry.values()):
            raise ValueError("exception fields must be nonempty strings")
        source, target = entry["source"], entry["include"]
        for name in (source, target):
            if Path(name).as_posix() != name or any(part in name for part in ("..", "*", "?", "[", "\\")):
                raise ValueError(f"exception paths must be exact canonical API-relative paths: {name}")
        source_tier, target_tier = tier(source), tier(target)
        if not source_tier or not target_tier or target_tier in ALLOWED_DEPENDENCIES[source_tier]:
            raise ValueError(f"exception is not a forbidden API include: {source} -> {target}")
        key = (source, target)
        if key in exceptions:
            raise ValueError(f"duplicate include exception: {source} -> {target}")
        exceptions.add(key)
    return exceptions


def validate(api_root: Path, exceptions: set[tuple[str, str]]) -> tuple[list[str], int]:
    errors = []
    used_exceptions = set()
    headers = sorted(p for p in api_root.rglob("*") if p.is_file() and p.suffix in HEADER_SUFFIXES)
    if not headers:
        return [f"{api_root}: no API headers found; check the repository root"], 0
    for header in headers:
        source = header.relative_to(api_root).as_posix()
        source_tier = tier(source)
        if source_tier is None:
            errors.append(
                f"{header}:1: unsupported API location; use tt-metalium/, tt-metalium/experimental/, or internal/"
            )
            continue
        lines = source_lines(header.read_text(encoding="utf-8"))
        substantive = [(number, line) for number, line in lines if line.strip()]
        if not substantive or not PRAGMA_ONCE.fullmatch(substantive[0][1]):
            errors.append(f"{header}:1: place unconditional #pragma once before declarations and other directives")
        for number, line in lines:
            if not re.match(r"^\s*#\s*include\b", line):
                continue
            match = INCLUDE.fullmatch(line)
            if not match:
                errors.append(f"{header}:{number}: use a literal #include so the API boundary can be checked")
                continue
            name = match[1] or match[2]
            target = include_target(api_root, header, name, quoted=match[2] is not None)
            target_tier = tier(target) if target else None
            if target and target.startswith("tt-metalium/internal/"):
                errors.append(f"{header}:{number}: internal headers belong in api/internal/, not tt-metalium/internal/")
            elif target_tier and target_tier not in ALLOWED_DEPENDENCIES[source_tier]:
                key = (source, target)
                if key in exceptions:
                    used_exceptions.add(key)
                else:
                    errors.append(
                        f"{header}:{number}: {source_tier} API must not include {target_tier} header <{target}>; "
                        "move the dependency into a source file or the interface into the appropriate API tier"
                    )
    for source, target in sorted(exceptions - used_exceptions):
        errors.append(f"{EXCEPTIONS}: stale exception {source} -> {target}; remove it with the repaired include")
    return errors, len(headers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        exceptions = load_exceptions(args.repo_root / EXCEPTIONS)
        errors, count = validate(args.repo_root / API_ROOT, exceptions)
    except (OSError, ValueError) as error:
        print(f"API header hygiene: {error}")
        return 1
    for error in errors:
        print(error)
    print(f"API header hygiene: {count} headers, {len(exceptions)} recorded include exceptions, {len(errors)} errors")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
