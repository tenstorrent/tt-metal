#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
pre-commit hook: block files that exceed a size threshold.

Unlike the built-in check-added-large-files hook (pre-commit-hooks), this
script operates on the filenames passed by pre-commit rather than on the git
staging area.  That makes it effective in both local-commit mode and in CI
(--from-ref / --to-ref mode), where nothing is ever staged.
"""

import argparse
import math
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="*", help="Files to check.")
    parser.add_argument(
        "--maxkb",
        type=int,
        default=500,
        help="Maximum allowable file size in KB (default: 500).",
    )
    args = parser.parse_args()

    retval = 0
    for filename in args.filenames:
        if not os.path.isfile(filename):
            continue
        kb = int(math.ceil(os.stat(filename).st_size / 1024))
        if kb > args.maxkb:
            print(f"{filename} ({kb} KB) exceeds the {args.maxkb} KB limit.")
            retval = 1

    return retval


if __name__ == "__main__":
    sys.exit(main())
