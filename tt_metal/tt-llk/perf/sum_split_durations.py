# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Sum the per-leg test durations of one run into one table per architecture.

A test that ran in more than one leg adds up, so the table holds its whole cost.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ARCHES = ("wormhole", "blackhole")
# The smoke legs store pytest-split's own file; the perf legs store leg seconds.
DURATION_FILES = (".test_durations", ".perf_split_durations")


def sum_durations(download: Path, arch: str) -> dict:
    summed: dict = {}
    for name in DURATION_FILES:
        for leg in sorted(download.glob(f"test-durations-{arch}-*/{name}")):
            for test, seconds in json.loads(leg.read_text()).items():
                summed[test] = summed.get(test, 0.0) + seconds
    return summed


def main(argv=None) -> int:
    download, out = (Path(p) for p in (argv or sys.argv[1:])[:2])
    out.mkdir(parents=True, exist_ok=True)
    found = False
    for arch in ARCHES:
        summed = sum_durations(download, arch)
        print(f"{arch}: durations for {len(summed)} tests")
        if summed:
            (out / f"{arch}.json").write_text(json.dumps(summed))
            found = True
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"found={str(found).lower()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
