#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate the includes of the public host/device headers in tt_metal/hw/inc/hostdev/api.

The device compiler and host compilers, inside and outside tt-metal, all compile these headers, so whatever one of
them includes every device kernel that uses it pays for. The rules:

- A header includes only other headers of the folder, spelled hostdev/api/<name>.
- base.h is the one exception: it brings in the standard headers the folder needs, and only those in BASE_INCLUDES.
  That list is closed; growing it is a change to this script, reviewed as such.
- No preprocessor conditional on the side being compiled (__riscv, COMPILE_FOR_*, ...): a header is the same text
  on the device and the host, so a host-only section, and a heavy include inside it, cannot hide in one.

This is a lexical check, like validate_api.py, whose comment masking it shares.
"""

import argparse
import re
from pathlib import Path

from validate_api import INCLUDE, PRAGMA_ONCE, source_lines

BASE = "base.h"
PREFIX = "hostdev/api/"
# The standard headers base.h may include: the C fixed-width types and the type traits the shared field walk needs.
BASE_INCLUDES = {"stddef.h", "stdint.h", "type_traits", "utility"}
HEADER_SUFFIXES = {".h", ".hpp"}
CONDITIONAL = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|elifdef|elifndef)\b(.*)$")
SIDE_MACROS = re.compile(
    r"\b(__riscv\w*|COMPILE_FOR_\w+|KERNEL_BUILD|FW_BUILD|TENSIX_FIRMWARE|ARCH_\w+|__x86_64__|__aarch64__)\b"
)


def validate(root: Path) -> tuple[list[str], int]:
    errors = []
    headers = sorted(p for p in root.iterdir() if p.is_file())
    if not any(p.name == BASE for p in headers):
        return [f"{root}: no {BASE}; check the directory"], 0
    names = {p.name for p in headers}
    for header in headers:
        where = header.as_posix()
        if header.suffix not in HEADER_SUFFIXES:
            errors.append(f"{where}: only headers belong in {PREFIX}")
            continue
        lines = source_lines(header.read_text(encoding="utf-8"))
        substantive = [(n, line) for n, line in lines if line.strip()]
        if not substantive or not PRAGMA_ONCE.fullmatch(substantive[0][1]):
            errors.append(f"{where}: #pragma once must be the first directive")
        for number, line in substantive:
            if match := INCLUDE.fullmatch(line):
                path = match[1] or match[2]
                if header.name == BASE:
                    if path not in BASE_INCLUDES:
                        errors.append(
                            f"{where}:{number}: {BASE} may include only {', '.join(sorted(BASE_INCLUDES))}, "
                            f"not {path}"
                        )
                elif not path.startswith(PREFIX) or path[len(PREFIX) :] not in names:
                    errors.append(
                        f"{where}:{number}: include only headers of {PREFIX} (standard headers come through "
                        f"{PREFIX}{BASE}), not {path}"
                    )
            elif (match := CONDITIONAL.fullmatch(line)) and (macro := SIDE_MACROS.search(match[2])):
                errors.append(
                    f"{where}:{number}: no conditional on the side being compiled ({macro[0]}): a {PREFIX} header is "
                    f"the same on the device and the host"
                )
    return errors, len(headers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="normally tt_metal/hw/inc/hostdev/api")
    args = parser.parse_args()
    errors, count = validate(args.directory)
    for error in errors:
        print(error)
    print(f"hostdev/api validation: {count} headers, {len(errors)} errors")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
