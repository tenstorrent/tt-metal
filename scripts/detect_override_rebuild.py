#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pre-commit guard: forbid rebuilding a ProgramDescriptor from a cache-hit hook.

``override_runtime_arguments()`` runs on EVERY program-cache hit. Calling ``create_descriptor()``
there — usually paired with ``apply_descriptor_runtime_args()``, and usually justified as "re-derive
from the single source of truth so the re-applied args can't drift" — pays the entire cache-MISS host
cost on every hit: the work split, ``CoreRangeSet`` construction, arch queries via
``get_compute_kernel_config_args``, ``TensorAccessorArgs`` building, kernel-source strings,
compile-time arg vectors, and a freshly heap-allocated runtime-arg vector for every core, followed by
a walk over every kernel x core x arg plus every CB. It is strictly slower than the
``get_dynamic_runtime_args`` the hook replaced, and it is the slow-path-rebuild-on-hit anti-pattern
behind the ResNet50 regressions (#46506, #50347).

The drift concern is real but has a cheaper answer: express each kernel's arg order ONCE in an
emitter templated on its sink, and instantiate it with ``KernelDescriptor::RTArgList`` on the miss
path and ``ttnn::RuntimeArgPatcher`` (``ttnn/api/ttnn/descriptor_arg_patcher.hpp``) on the hit path.
No arg index is written by hand, nothing is allocated on a hit, and the result is byte-identical to a
rebuild by construction.

This is a BUILD-TIME text check (no runtime cost). It scans the body of every
``override_runtime_arguments`` definition and rejects a descriptor rebuild inside it. Ops that
legitimately delegate to the framework's cheap binding path (``descriptor_adapter_t::apply_descriptor``)
are unaffected. A genuinely unavoidable case suppresses a line with a trailing
``// override-rebuild-ok: <reason>``.
"""

import os
import re
import sys

BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detect_override_rebuild_baseline.txt")


def _baseline():
    """Repo-relative paths that still contain the anti-pattern.

    Pre-existing violations are listed so this hook blocks NEW ones without obstructing unrelated
    edits to those files. Fixing an op means deleting its line from the baseline.
    """
    try:
        with open(BASELINE_PATH) as f:
            return {line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")}
    except OSError:
        return set()


HOOK = re.compile(r"\boverride_runtime_arguments\s*\(")
# A descriptor rebuild, or the framework's bulk copy-everything applier (which only makes sense when
# handed a freshly built descriptor, so it is the same defect wearing a different hat).
REBUILD = re.compile(r"\b(?:[A-Za-z_]\w*::)?create_(?:\w+_)?descriptor\s*\(|\bapply_descriptor_runtime_args\s*\(")
SUPPRESS = re.compile(r"//\s*override-rebuild-ok\b")


def _mask_comments(text):
    """Blank out comment and string-literal contents, preserving length and newlines, so a rebuild
    merely *described* in a comment is not flagged."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        two = text[i : i + 2]
        if two == "//":
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif two == "/*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif text[i] == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            for k in range(i, min(j + 1, n)):
                if out[k] != "\n":
                    out[k] = " "
            i = j + 1
        else:
            i += 1
    return "".join(out)


def _body_span(text, open_paren_idx):
    """Return (start, end) of the brace-delimited body following a declaration, or None if it is a
    declaration only (ends in ``;``)."""
    depth = 0
    i = open_paren_idx
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    # Skip trailing qualifiers/whitespace to the first '{' or ';'.
    j = i + 1
    while j < len(text) and text[j] not in "{;":
        j += 1
    if j >= len(text) or text[j] == ";":
        return None  # declaration, no body
    start = j
    depth = 0
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return (start, j)
        j += 1
    return (start, len(text))


def check(path):
    try:
        with open(path, errors="ignore") as f:
            text = f.read()
    except OSError:
        return []
    if "override_runtime_arguments" not in text:
        return []
    # Match against a comment-free view; report against the original.
    scan = _mask_comments(text)

    line_starts = [0]
    for m in re.finditer(r"\n", text):
        line_starts.append(m.end())

    def line_of(pos):
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    findings = []
    for hook in HOOK.finditer(scan):
        span = _body_span(scan, scan.index("(", hook.end() - 1))
        if span is None:
            continue
        body = scan[span[0] : span[1]]
        for bad in REBUILD.finditer(body):
            pos = span[0] + bad.start()
            lineno = line_of(pos)
            line = text[line_starts[lineno - 1] : (line_starts[lineno] if lineno < len(line_starts) else len(text))]
            if SUPPRESS.search(line):
                continue
            findings.append((lineno, bad.group(0).rstrip("("), line.strip()))
    return findings


def main(argv):
    failed = False
    baseline = _baseline()
    for path in argv[1:]:
        if os.path.normpath(path) in baseline:
            continue
        for lineno, what, line in check(path):
            failed = True
            print(f"{path}:{lineno}: error: '{what}' called inside override_runtime_arguments")
            print(f"    {line}")
    if failed:
        print()
        print("override_runtime_arguments runs on EVERY program-cache hit; rebuilding the descriptor there pays the")
        print("full cache-MISS host cost per dispatch. Express each kernel's arg order once in an emitter templated")
        print("on its sink: KernelDescriptor::RTArgList on the miss path, ttnn::RuntimeArgPatcher on the hit path")
        print("(ttnn/api/ttnn/descriptor_arg_patcher.hpp). No hand-written arg indices, no allocation on a hit.")
        print("Re-point globally-allocated CB addresses with UpdateDynamicCircularBufferAddress, matched by CBIndex.")
        print("Verify with -DENABLE_DESCRIPTOR_PATCHING_PARITY_CHECK=ON.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
