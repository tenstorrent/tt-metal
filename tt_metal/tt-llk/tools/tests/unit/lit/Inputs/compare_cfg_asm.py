# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compare actual_CASE and expected_CASE assembly, preserving every instruction.

References use hardware intrinsics or MMIO, independently of CFG planning and
emission. Ignore only source comments, whitespace, and generated local-label IDs.
Register allocation and instruction order must match within each pair.
"""

import difflib
import re
import sys
from pathlib import Path


def functions(assembly):
    result = {}
    pattern = r"^((?:actual|expected)_\w+):\n(.*?)^\s*\.size\s+\1,"
    for match in re.finditer(pattern, assembly, re.MULTILINE | re.DOTALL):
        body = re.sub(r"^\s*#.*$", "", match[2], flags=re.MULTILINE)
        labels = {}
        body = re.sub(
            r"\.L\w+",
            lambda label: labels.setdefault(label[0], f".L{len(labels)}"),
            body,
        )
        result[match[1]] = [line.strip() for line in body.splitlines() if line.strip()]
    return result


def main():
    bodies = functions(Path(sys.argv[1]).read_text())
    actual = {
        name.removeprefix("actual_") for name in bodies if name.startswith("actual_")
    }
    expected = {
        name.removeprefix("expected_")
        for name in bodies
        if name.startswith("expected_")
    }
    if not actual or actual != expected:
        sys.exit(
            f"Missing assembly pairs: actual={sorted(actual)}, expected={sorted(expected)}"
        )
    failed = False
    for case in sorted(actual):
        left, right = bodies["expected_" + case], bodies["actual_" + case]
        if left != right:
            failed = True
            print(
                "\n".join(
                    difflib.unified_diff(
                        left,
                        right,
                        fromfile="expected_" + case,
                        tofile="actual_" + case,
                        lineterm="",
                    )
                )
            )
    if failed:
        sys.exit(1)
    print(f"{len(actual)} CFG assembly pairs match")


if __name__ == "__main__":
    main()
