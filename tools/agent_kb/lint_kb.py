#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
KB_ROOT = REPO_ROOT / "agent_kb"
SPECIAL_FILES = {
    KB_ROOT / "README.md",
    KB_ROOT / "AGENTS.md",
    KB_ROOT / "index.md",
    KB_ROOT / "log.md",
}
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def iter_md_files() -> list[Path]:
    return sorted(path for path in KB_ROOT.rglob("*.md") if path.is_file())


def resolve_link(base: Path, link: str) -> Path | None:
    if link.startswith(("http://", "https://", "mailto:", "#")):
        return None
    target = link.split("#", 1)[0]
    if not target:
        return None
    return (base.parent / target).resolve()


def main() -> int:
    files = iter_md_files()
    broken_links: list[str] = []
    inbound_links: dict[Path, int] = defaultdict(int)
    missing_sources: list[str] = []

    index_text = (KB_ROOT / "index.md").read_text(encoding="utf-8")
    index_targets = set(LINK_RE.findall(index_text))

    for path in files:
        text = path.read_text(encoding="utf-8")
        if path not in SPECIAL_FILES and "## Sources" not in text:
            missing_sources.append(str(path.relative_to(REPO_ROOT)))

        for link in LINK_RE.findall(text):
            resolved = resolve_link(path, link)
            if resolved is None:
                continue
            if not resolved.exists():
                broken_links.append(f"{path.relative_to(REPO_ROOT)} -> {link}")
                continue
            try:
                resolved.relative_to(KB_ROOT)
            except ValueError:
                continue
            inbound_links[resolved] += 1

    unindexed = []
    for path in files:
        if path in SPECIAL_FILES:
            continue
        rel = path.relative_to(KB_ROOT).as_posix()
        if rel not in index_targets:
            unindexed.append(str(path.relative_to(REPO_ROOT)))

    orphans = []
    for path in files:
        if path in SPECIAL_FILES:
            continue
        if inbound_links[path] == 0:
            orphans.append(str(path.relative_to(REPO_ROOT)))

    problems = False
    if broken_links:
        problems = True
        print("Broken links:")
        for item in broken_links:
            print(f"  - {item}")

    if unindexed:
        problems = True
        print("Unindexed pages:")
        for item in unindexed:
            print(f"  - {item}")

    if orphans:
        problems = True
        print("Orphan pages:")
        for item in orphans:
            print(f"  - {item}")

    if missing_sources:
        problems = True
        print("Pages missing a Sources section:")
        for item in missing_sources:
            print(f"  - {item}")

    if not problems:
        print("KB lint passed.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
