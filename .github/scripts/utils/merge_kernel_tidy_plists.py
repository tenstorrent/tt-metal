#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Deduplicate kernel clang-tidy plists ahead of ``CodeChecker parse``.

Every translation unit in a leg is a firmware wrapper with the kernel #included
into it, so the same header defect is re-reported by every TU that pulls the
header in, and again by every leg. Measured on two legs: 370,060 raw
diagnostics for 19,863 distinct ones, an 18.7x redundancy.

``CodeChecker parse`` is single-threaded with no --jobs and its cost scales with
raw diagnostic count, so it spends ~95% of its runtime rediscovering findings it
has already seen. This drops the duplicates first, in parallel, and hands
CodeChecker only the distinct ones. No finding is lost: a defect reported at the
same file, line, column, checker and message is the same defect, and it is the
one an agent would fix once.
"""

from __future__ import annotations

import argparse
import multiprocessing
import pathlib
import plistlib
import shutil
import sys
from typing import Any

# A finding is identified by where it is, what fired, and what it said. Message
# is included because a handful of checkers report distinct problems at one
# location (measured: 28 of 19,863 on the two-leg sample).
Key = tuple[Any, ...]


def _resolve(diag: dict, files: list[str]) -> Key:
    loc = diag.get("location", {})
    idx = loc.get("file")
    path = files[idx] if isinstance(idx, int) and 0 <= idx < len(files) else idx
    return (path, loc.get("line"), loc.get("col"), diag.get("check_name"), diag.get("description"))


def _scan(args: tuple[pathlib.Path, pathlib.Path, pathlib.Path]) -> tuple[str, list[Key]]:
    """Parse one plist, drop within-file duplicates, stage the remainder."""
    src, root, stage = args
    try:
        data = plistlib.loads(src.read_bytes())
    except Exception as exc:  # a truncated plist must not sink the whole merge
        print(f"warning: skipping unreadable {src.name}: {exc}", file=sys.stderr)
        return (str(src), [])

    files = data.get("files", [])
    seen: dict[Key, None] = {}
    kept = []
    for diag in data.get("diagnostics", []):
        key = _resolve(diag, files)
        if key not in seen:
            seen[key] = None
            kept.append(diag)

    # Flatten to a unique name: leg directories collide on TU filenames.
    rel = src.relative_to(root)
    out = stage / ("__".join(rel.parts[:-1] + (rel.name,)) if len(rel.parts) > 1 else rel.name)
    data["diagnostics"] = kept
    out.write_bytes(plistlib.dumps(data))
    return (str(out), list(seen))


def _prune(args: tuple[pathlib.Path, pathlib.Path, set[Key]]) -> int:
    """Keep only the diagnostics this plist owns after global deduplication."""
    staged, final, owned = args
    data = plistlib.loads(staged.read_bytes())
    files = data.get("files", [])
    kept = [d for d in data.get("diagnostics", []) if _resolve(d, files) in owned]
    if not kept:
        return 0
    data["diagnostics"] = kept
    (final / staged.name).write_bytes(plistlib.dumps(data))
    return len(kept)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=pathlib.Path, help="directory tree of per-leg plists")
    ap.add_argument("output", type=pathlib.Path, help="directory to write deduplicated plists into")
    ap.add_argument("-j", "--jobs", type=int, default=0, help="worker processes (default: all cores)")
    args = ap.parse_args()

    jobs = args.jobs or multiprocessing.cpu_count()
    sources = sorted(args.input.rglob("*.plist"))
    if not sources:
        print(f"no plists under {args.input}", file=sys.stderr)
        return 1

    stage = args.output.with_name(args.output.name + ".stage")
    for d in (stage, args.output):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool(jobs) as pool:
        scanned = pool.map(_scan, [(s, args.input, stage) for s in sources], chunksize=1)

        # First plist to report a finding owns it; the rest drop their copies.
        owner: dict[Key, str] = {}
        raw = 0
        for path, keys in scanned:
            raw += len(keys)
            for key in keys:
                owner.setdefault(key, path)

        by_plist: dict[str, set[Key]] = {}
        for key, path in owner.items():
            by_plist.setdefault(path, set()).add(key)

        written = pool.map(
            _prune, [(pathlib.Path(p), args.output, k) for p, k in by_plist.items()], chunksize=1
        )

    shutil.rmtree(stage, ignore_errors=True)
    print(
        f"{len(sources)} plists in, {sum(1 for w in written if w)} out; "
        f"{raw} findings after per-file dedup -> {sum(written)} distinct "
        f"({jobs} workers)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
