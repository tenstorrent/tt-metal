#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate the includes of the observability/api headers the host and the device share.

The host's public folder, tt_metal/observability/api, is compiled by host compilers inside and outside tt-metal and,
through the copy the JIT build stages, by the device compiler. The device's folder, tt_metal/hw/observability/api,
holds device versions that replace host headers of the same name in that copy, and device-only headers. Whatever a
header includes every device kernel that uses it pays for. The rules, for both folders:

- A header includes only other observability/api headers, spelled observability/api/<name>: a host header those of
  the host folder, a device header those of the copy the device sees, the two folders merged.
- base.h is the one exception: it brings in the standard headers the folder needs, and only those in BASE_INCLUDES.
  That list is closed; growing it is a change to this script, reviewed as such.
- No preprocessor conditional on the side being compiled (__riscv, COMPILE_FOR_*, ...): a host header is the same
  text on the device and the host, so a host-only section, and a heavy include inside it, cannot hide in one.

This is a lexical check, like validate_api.py, whose comment masking it shares.
"""

import argparse
import re
from pathlib import Path
from typing import Optional

from validate_api import INCLUDE, PRAGMA_ONCE, source_lines

BASE = "base.h"
PREFIX = "observability/api/"
# The standard headers base.h may include: the C fixed-width types and the type traits the shared field walk needs.
BASE_INCLUDES = {"stddef.h", "stdint.h", "type_traits", "utility"}
HEADER_SUFFIXES = {".h", ".hpp"}
# The device folder's README explains the folder; it is not staged.
DEVICE_DOCS = {"README.md"}
CONDITIONAL = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|elifdef|elifndef)\b(.*)$")
SIDE_MACROS = re.compile(
    r"\b(__riscv\w*|COMPILE_FOR_\w+|KERNEL_BUILD|FW_BUILD|TENSIX_FIRMWARE|ARCH_\w+|__x86_64__|__aarch64__)\b"
)


def files_of(root: Optional[Path]) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file()) if root and root.is_dir() else []


def check_header(header: Path, visible: set[str]) -> list[str]:
    errors = []
    where = header.as_posix()
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
                        f"{where}:{number}: {BASE} may include only {', '.join(sorted(BASE_INCLUDES))}, not {path}"
                    )
            elif not path.startswith(PREFIX) or path[len(PREFIX) :] not in visible:
                errors.append(
                    f"{where}:{number}: include only headers of {PREFIX} (standard headers come through "
                    f"{PREFIX}{BASE}), not {path}"
                )
        elif (match := CONDITIONAL.fullmatch(line)) and (macro := SIDE_MACROS.search(match[2])):
            errors.append(
                f"{where}:{number}: no conditional on the side being compiled ({macro[0]}): a {PREFIX} header is "
                f"the same on the device and the host"
            )
    return errors


def validate(host: Path, device: Optional[Path] = None) -> tuple[list[str], int]:
    host_files, device_files = files_of(host), files_of(device)
    if not any(p.name == BASE for p in host_files):
        return [f"{host}: no {BASE}; check the directory"], 0
    host_names = {p.name for p in host_files}
    device_view = host_names | {p.name for p in device_files}
    errors = []
    count = 0
    for files, visible, docs in ((host_files, host_names, set()), (device_files, device_view, DEVICE_DOCS)):
        for path in files:
            if path.name in docs:
                continue
            if path.suffix not in HEADER_SUFFIXES:
                errors.append(f"{path.as_posix()}: only headers belong in {path.parent.as_posix()}")
                continue
            count += 1
            errors += check_header(path, visible)
    return errors, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("host", type=Path, help="normally tt_metal/observability/api")
    parser.add_argument("device", type=Path, nargs="?", help="normally tt_metal/hw/observability/api")
    args = parser.parse_args()
    errors, count = validate(args.host, args.device)
    for error in errors:
        print(error)
    print(f"observability/api validation: {count} headers, {len(errors)} errors")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
