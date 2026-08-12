# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-derive the "watcher clean" claim from a committed watcher log.

A watcher-clean run still contains attach/detach lines, kernel-id lines and
periodic dumps with stack-usage summaries; what it must not contain is a fatal
watcher exception or a sanitize/assert/out-of-bounds/hardware-fault message.
This script counts both sides so the claim in ``README.md`` is a command rather
than a sentence.

Usage::

    python check_watcher.py                       # the committed watcher run
    python check_watcher.py ../watcher_reverify/watcher.log.gz
"""

from __future__ import annotations

import argparse
import gzip
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # doc/functional_decoder/
DEFAULT_LOG = ROOT / "watcher/watcher.log.gz"

#: any hit here fails the run
FATAL = re.compile(
    r"Watcher detected|tripped|sanitize|TT_ASSERT|DEBUG_ASSERT|"
    r"out-of-bounds|Stack overflow|hardware fault|HARDWARE FAULT|"
    r"Invalid NOC|noc_addr|watcher_assert",
    re.IGNORECASE,
)
#: expected, benign structure -> minimum count for the log to count as a real
#: watcher run.  Absence of fatal messages is not enough on its own: an empty or
#: truncated log has none either, so the shape of a live run is asserted too.
BENIGN = {
    "dump boundaries": (re.compile(r"Dump #\d+"), 2),
    "kernel id lines": (re.compile(r"^\s*k_ids:", re.MULTILINE), 100),
    "stack usage rows": (re.compile(r"highest stack usage: \d+ bytes free"), 10),
    "device attach": (re.compile(r"^At [0-9.]+s attach device \d+", re.MULTILINE), 1),
    "device detach": (re.compile(r"^At [0-9.]+s detach device \d+", re.MULTILINE), 1),
}
#: a real watcher run over this stage's tests is thousands of lines long
MIN_LINES = 1000


def read(path: pathlib.Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="ignore") as f:
            return f.read()
    return path.read_text(errors="ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=str(DEFAULT_LOG))
    args = ap.parse_args()

    path = pathlib.Path(args.log)
    if not path.is_file():
        print(f"missing watcher log {path}", file=sys.stderr)
        return 2
    text = read(path)
    lines = text.count("\n")
    fatal = FATAL.findall(text)

    print(f"{path}: {lines} lines")
    missing = [] if lines >= MIN_LINES else [f"only {lines} lines, expected >= {MIN_LINES}"]
    for name, (pattern, minimum) in BENIGN.items():
        count = len(pattern.findall(text))
        print(f"  {name}: {count} (min {minimum})")
        if count < minimum:
            missing.append(f"{name}: {count} < {minimum}")
    print(f"  fatal watcher messages: {len(fatal)}")

    if fatal:
        for hit in sorted(set(fatal))[:10]:
            print(f"    {hit}", file=sys.stderr)
        print("WATCHER_NOT_CLEAN", file=sys.stderr)
        return 1
    if missing:
        # A truncated or stubbed log has no fatal messages either, so refuse to
        # certify it as clean.
        for item in missing:
            print(f"    missing expected watcher structure: {item}", file=sys.stderr)
        print("WATCHER_LOG_NOT_A_REAL_RUN", file=sys.stderr)
        return 1
    print("WATCHER_CLEAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
