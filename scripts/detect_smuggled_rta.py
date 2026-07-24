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
``SetRuntimeArgs`` ops are out of scope. A file that mixes both paths (e.g. a factory ported to Metal
2.0 that still keeps a classic ``Program`` + ``SetRuntimeArgs`` + ``override_runtime_arguments`` path)
is common, so the check is per-container: an ``*args`` builder vector whose only runtime-arg consumer is
classic ``SetRuntimeArgs`` is left alone — its addresses are re-applied every cache hit by
``override_runtime_arguments`` (the sanctioned classic pattern), not smuggled onto a descriptor. Only
addresses flowing into a descriptor sink (``emplace_runtime_args`` / ``emplace_common_runtime_args`` /
``KernelDescriptor::runtime_args.emplace_back``) are flagged. Intentional exceptions (e.g. ops parked
pending ``get_dynamic_runtime_args``) suppress a line with a trailing ``// smuggled-rta-ok: <reason>``.
"""

import bisect
import re
import sys

DESCRIPTOR_MARKERS = ("CoreRuntimeArgs", "emplace_runtime_args", "ProgramDescriptor", "RTArgList")
# Start of a runtime-arg construction whose contents must not contain a raw address.
SINK_START = re.compile(
    r"(runtime_args\.emplace_back"
    r"|emplace_runtime_args|emplace_common_runtime_args"
    r"|CoreRuntimeArgs"  # bare: let _sink_regions find the following { or ( (do not consume it)
    r"|common_runtime_args\s*="
    # a builder container whose name ends in *args (reader_args/writer_runtime_args/…) being appended to
    r"|(?P<container>[A-Za-z_]\w*args)\s*\.\s*(?:push_back|append|emplace_back)"
    r")"
)
# Container passed as the last argument to a classic (non-descriptor) SetRuntimeArgs call. Such a
# container is the legacy path — its addresses are re-applied by override_runtime_arguments, not
# smuggled onto a descriptor — so pushes into it are out of scope (unless it also feeds a descriptor sink).
CLASSIC_CONSUMER = re.compile(r"\bSetRuntimeArgs\s*\([^;]*?,\s*([A-Za-z_]\w*)\s*\)", re.DOTALL)
# Container passed by name to a descriptor sink (emplace_runtime_args(core, NAME) /
# emplace_common_runtime_args(NAME) / runtime_args.emplace_back(core, NAME)).
DESCRIPTOR_CONSUMER = re.compile(
    r"(?:emplace_runtime_args|runtime_args\s*\.\s*emplace_back)\s*\([^;{]*?,\s*([A-Za-z_]\w*)\s*\)"
    r"|emplace_common_runtime_args\s*\(\s*([A-Za-z_]\w*)\s*\)",
    re.DOTALL,
)
ADDRESS = re.compile(r"->\s*address\s*\(\s*\)|\.\s*address\s*\(\s*\)")
# Local: `<...> foo = <...>address();`  -> foo becomes an "address variable".
ADDR_VAR = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*[^;]*?address\s*\(\s*\)\s*;")
OK = "smuggled-rta-ok"


def _strip_comments(text):
    """Blank out // and /* */ comments and string/char literals with spaces of equal
    length (newlines preserved) so scanning sees only code while byte offsets/line numbers
    stay valid. Prevents matching address()/emplace_runtime_args inside a comment or string."""
    out = list(text)
    n = len(text)
    i = 0
    CODE, LINE, BLOCK, STR = 0, 1, 2, 3
    state, quote = CODE, ""
    while i < n:
        c = text[i]
        pair = text[i : i + 2]
        if state == CODE:
            if pair == "//":
                out[i] = out[i + 1] = " "
                i += 2
                state = LINE
            elif pair == "/*":
                out[i] = out[i + 1] = " "
                i += 2
                state = BLOCK
            elif c in "\"'":
                quote = c
                i += 1
                state = STR
            else:
                i += 1
        elif state == LINE:
            if c == "\n":
                state = CODE
            else:
                out[i] = " "
            i += 1
        elif state == BLOCK:
            if pair == "*/":
                out[i] = out[i + 1] = " "
                i += 2
                state = CODE
            else:
                if c != "\n":
                    out[i] = " "
                i += 1
        else:  # STR — leave literal text intact but consume it so // inside isn't a comment
            if c == "\\":
                i += 2
            elif c == quote:
                i += 1
                state = CODE
            else:
                i += 1
    return "".join(out)


def _sink_regions(text):
    """Yield (start_off, end_off, container) char spans of each runtime-arg construction.

    ``container`` is the builder-vector name for the ``*args.push_back(...)`` form (used to gate out
    classic ``SetRuntimeArgs`` builders), else ``None`` for a direct descriptor sink."""
    for m in SINK_START.finditer(text):
        container = m.groupdict().get("container")
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
                    yield (m.start(), j + 1, container)
                    break
            j += 1


def check_file(path):
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return []
    if not any(mark in text for mark in DESCRIPTOR_MARKERS):
        return []  # not a descriptor factory

    code = _strip_comments(text)  # scan code only; comments/strings blanked (offsets preserved)
    lines = text.splitlines()
    starts = [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln) + 1)

    def line_of2(off):
        return bisect.bisect_right(starts, off)

    def suppressed(lineno):
        return OK in lines[lineno - 1] if 1 <= lineno <= len(lines) else False

    addr_vars = {m.group(1) for m in ADDR_VAR.finditer(code)}
    # Containers consumed by classic SetRuntimeArgs vs by a descriptor sink. A builder consumed only by
    # SetRuntimeArgs is the legacy path (out of scope); if it also feeds a descriptor sink, keep checking it.
    classic_containers = {m.group(1) for m in CLASSIC_CONSUMER.finditer(code)}
    descriptor_containers = {g for m in DESCRIPTOR_CONSUMER.finditer(code) for g in m.groups() if g}
    classic_only = classic_containers - descriptor_containers
    findings = []
    for s, e, container in _sink_regions(code):
        if container is not None and container in classic_only:
            continue  # legacy SetRuntimeArgs builder — out of scope (re-applied via override, not smuggled)
        region = code[s:e]
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
