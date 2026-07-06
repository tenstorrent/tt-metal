#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pre-commit guard: forbid smuggled buffer-address runtime args in descriptor factories.

A "smuggled RTA" is a raw buffer base address (``tensor.buffer()->address()``) pushed into a
kernel's descriptor runtime args as a plain ``uint32_t`` instead of being declared as a ``Buffer*``
binding via ``KernelDescriptor::emplace_runtime_args(core, {buffer, ...})``. The framework can't tell
such a slot holds a buffer address, so it can't patch it on a program-cache hit (and falls back to a
full descriptor rebuild every dispatch).

This is a BUILD-TIME text check (no runtime cost). It only inspects ProgramDescriptor-based factories
(files that mention ``CoreRuntimeArgs`` / ``emplace_runtime_args`` / ``ProgramDescriptor``); legacy
``SetRuntimeArgs`` ops are out of scope. Intentional exceptions (e.g. ops parked pending
``get_dynamic_runtime_args``) suppress a line with a trailing ``// smuggled-rta-ok: <reason>``.
"""

import bisect
import re
import sys

DESCRIPTOR_MARKERS = ("CoreRuntimeArgs", "emplace_runtime_args", "ProgramDescriptor", "RTArgList")
# Start of a runtime-arg construction whose contents must not contain a raw address.
SINK_START = re.compile(
    r"(runtime_args\.emplace_back|emplace_runtime_args|emplace_common_runtime_args"
    r"|CoreRuntimeArgs\s*[\{(]|common_runtime_args\s*=)"
)
ADDRESS = re.compile(r"->\s*address\s*\(\s*\)|\.\s*address\s*\(\s*\)")
# Local: `<...> foo = <...>address();`  -> foo becomes an "address variable".
ADDR_VAR = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*[^;]*?address\s*\(\s*\)\s*;")
OK = "smuggled-rta-ok"


def _sink_regions(text):
    """Yield (start_off, end_off) char spans of each runtime-arg construction."""
    for m in SINK_START.finditer(text):
        i = m.end()
        # advance to the first opening bracket of the construction
        while i < len(text) and text[i] not in "{(":
            if text[i] == ";":
                break
            i += 1
        if i >= len(text) or text[i] not in "{(":
            continue
        depth, j = 0, i
        while j < len(text):
            c = text[j]
            if c in "{(":
                depth += 1
            elif c in "})":
                depth -= 1
                if depth == 0:
                    yield (m.start(), j + 1)
                    break
            j += 1


def check_file(path):
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return []
    if not any(mark in text for mark in DESCRIPTOR_MARKERS):
        return []  # not a descriptor factory

    lines = text.splitlines()
    starts = [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln) + 1)

    def line_of2(off):
        return bisect.bisect_right(starts, off)

    def suppressed(lineno):
        return OK in lines[lineno - 1] if 1 <= lineno <= len(lines) else False

    addr_vars = {m.group(1) for m in ADDR_VAR.finditer(text)}
    findings = []
    for s, e in _sink_regions(text):
        region = text[s:e]
        # (a) direct raw address inside the runtime-arg construction
        for am in ADDRESS.finditer(region):
            ln = line_of2(s + am.start())
            if not suppressed(ln):
                findings.append((ln, "raw buffer address pushed into runtime args"))
        # (b) an address-variable referenced inside the construction
        for var in addr_vars:
            for vm in re.finditer(r"\b" + re.escape(var) + r"\b", region):
                ln = line_of2(s + vm.start())
                if not suppressed(ln):
                    findings.append((ln, f"address-derived local '{var}' pushed into runtime args"))
                break
    # de-dup by line
    seen, out = set(), []
    for ln, msg in sorted(findings):
        if ln not in seen:
            seen.add(ln)
            out.append((ln, msg))
    return out


def main(argv):
    bad = False
    for path in argv[1:]:
        if not re.search(r"\.(cpp|hpp)$", path):
            continue
        for ln, msg in check_file(path):
            bad = True
            print(f"{path}:{ln}: smuggled RTA — {msg}")
    if bad:
        print(
            "\nDeclare the buffer as a binding: KernelDescriptor::emplace_runtime_args(core, {buffer, ...})\n"
            "so the framework patches the address on cache hits instead of rebuilding.\n"
            "If this is intentional (e.g. parked pending get_dynamic_runtime_args), append "
            "'// smuggled-rta-ok: <reason>' to the line.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
