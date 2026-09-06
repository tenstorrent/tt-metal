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
import os
import pathlib
import plistlib
import shutil
import sys
from typing import Any


def usable_cpus() -> int:
    """CPUs this process may actually use.

    ``cpu_count()`` reports the host's cores, which in a CPU-limited container
    means oversubscribing by an order of magnitude and multiplying peak memory
    by the same factor. Honour CPU affinity and the cgroup v2 quota instead.
    """
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        n = os.cpu_count() or 1
    try:
        quota, period = pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)

# A finding is identified by where it is, what fired, and what it said. Message
# is included because a handful of checkers report distinct problems at one
# location (measured: 28 of 19,863 on the two-leg sample).
Key = tuple[Any, ...]


def _resolve(diag: dict, files: list[str]) -> Key:
    loc = diag.get("location", {})
    idx = loc.get("file")
    path = files[idx] if isinstance(idx, int) and 0 <= idx < len(files) else idx
    return (path, loc.get("line"), loc.get("col"), diag.get("check_name"), diag.get("description"))


def _scan(src: pathlib.Path) -> tuple[str, list[Key]]:
    """Report which distinct findings one plist contains. Reads only."""
    try:
        data = plistlib.loads(src.read_bytes())
    except Exception as exc:  # a truncated plist must not sink the whole merge
        print(f"warning: skipping unreadable {src.name}: {exc}", file=sys.stderr)
        return (str(src), [])
    files = data.get("files", [])
    return (str(src), list(dict.fromkeys(_resolve(d, files) for d in data.get("diagnostics", []))))


def _prune(args: tuple[pathlib.Path, pathlib.Path, pathlib.Path, set[Key]]) -> int:
    """Write out only the diagnostics this plist owns after global dedup."""
    src, root, final, owned = args
    data = plistlib.loads(src.read_bytes())
    files = data.get("files", [])
    kept = [d for d in data.get("diagnostics", []) if _resolve(d, files) in owned]
    if not kept:
        return 0
    data["diagnostics"] = kept
    # Flatten to a unique name: leg directories collide on TU filenames.
    rel = src.relative_to(root)
    name = "__".join(rel.parts[:-1] + (rel.name,)) if len(rel.parts) > 1 else rel.name
    (final / name).write_bytes(plistlib.dumps(data))
    return len(kept)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=pathlib.Path, help="directory tree of per-leg plists")
    ap.add_argument("output", type=pathlib.Path, help="directory to write deduplicated plists into")
    ap.add_argument("-j", "--jobs", type=int, default=0, help="worker processes (default: usable cores)")
    args = ap.parse_args()

    sources = sorted(args.input.rglob("*.plist"))
    # No point starting more workers than there are plists; each holds one
    # parsed plist in memory (~70 MB for the largest observed).
    jobs = max(1, min(args.jobs or usable_cpus(), len(sources) or 1))
    if not sources:
        print(f"no plists under {args.input}", file=sys.stderr)
        return 1

    shutil.rmtree(args.output, ignore_errors=True)
    args.output.mkdir(parents=True, exist_ok=True)

    # Two read passes over the inputs rather than one plus a staged rewrite:
    # duplicates are almost entirely cross-plist, so a per-plist intermediate
    # would be a byte-for-byte copy of the input (~4 GB at 15 legs) for nothing.
    with multiprocessing.Pool(jobs) as pool:
        scanned = pool.map(_scan, sources, chunksize=1)

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
            _prune,
            [(pathlib.Path(p), args.input, args.output, k) for p, k in by_plist.items()],
            chunksize=1,
        )

    print(
        f"{len(sources)} plists in, {sum(1 for w in written if w)} out; "
        f"{raw} findings -> {sum(written)} distinct ({jobs} workers)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
