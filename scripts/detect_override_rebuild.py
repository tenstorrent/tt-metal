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

The drift concern is real but has a cheaper answer, used by the merged reference fixes
(#50351 sparse_sdpa, #50894 slice): on a cache hit, patch only the slots that actually vary. Overwrite
scalar/address runtime args in place with ``tt::tt_metal::GetRuntimeArgs(program, kernel_idx, core)``,
and re-point CB-backed addresses through a MINIMAL (CB-only) ``ProgramDescriptor`` +
``apply_descriptor_runtime_args`` (mark that one line ``// override-rebuild-ok: cb-addr-only`` — it is
O(1), not a rebuild). Nothing is re-split, no per-core arg vector is reallocated, and the hit stays O(1).

This is a BUILD-TIME text check (no runtime cost). It scans the body of every
``override_runtime_arguments`` definition and rejects a descriptor rebuild inside it. Ops that
legitimately delegate to the framework's cheap binding path (``descriptor_adapter_t::apply_descriptor``)
are unaffected. A genuinely unavoidable case suppresses a line with a trailing
``// override-rebuild-ok: <reason>``.
"""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detect_override_rebuild_baseline.txt")


def _safe_path(path):
    """Resolve a caller-supplied path under the repo root and reject anything that escapes it.

    pre-commit passes repo-relative paths from a trusted, git-tracked file list, but the path is still
    dynamic input, so normalize it and bound the result to the tree before opening. Returns the resolved
    absolute path, or ``None`` if it points outside the repo (which the caller then skips)."""
    joined = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    resolved = os.path.realpath(joined)
    if resolved != REPO_ROOT and not resolved.startswith(REPO_ROOT + os.sep):
        return None
    return resolved


def _baseline():
    """Pre-existing violations, keyed per offender so the guard is never disabled for a whole file.

    Each entry grandfathers ONE known hit as a ``<repo-relative-path>\\t<symbol>\\t<source-line>``
    fingerprint (line-number independent, so unrelated edits above it don't churn the baseline). A
    baselined file is still fully scanned: any NEW or additional violation whose fingerprint is not
    listed is blocked. Fixing an op means deleting its line(s) here (or ``--update-baseline``).

    Returns ``{normpath: {(symbol, source_line), ...}}``.
    """
    result = {}
    try:
        with open(BASELINE_PATH) as f:
            for raw in f:
                stripped = raw.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = raw.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                path, what, src = parts
                result.setdefault(os.path.normpath(path), set()).add((what, src))
    except OSError:
        pass
    return result


HOOK = re.compile(r"\boverride_runtime_arguments\s*\(")
# A descriptor rebuild, or the framework's bulk copy-everything applier (which only makes sense when
# handed a freshly built descriptor, so it is the same defect wearing a different hat).
REBUILD = re.compile(r"\b(?:[A-Za-z_]\w*::)?create_(?:\w+_)?descriptor\s*\(|\bapply_descriptor_runtime_args\s*\(")
SUPPRESS = re.compile(r"//\s*override-rebuild-ok\b")


def _mask_comments(text):
    """Blank out comment, string-literal and char-literal contents, preserving length and newlines, so
    a rebuild merely *described* in a comment — or a brace/quote inside a literal such as ``'}'`` or
    ``'"'`` — cannot corrupt brace/parse tracking or be flagged."""
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
        elif text[i] == '"' or (
            # A char literal opener, e.g. ' } ' or '\''. Guard against the C++14 digit separator
            # (1'000, 0xFF'FF): a separator quote directly follows an alphanumeric, a literal never does.
            text[i] == "'"
            and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_"))
        ):
            quote = text[i]
            j = i + 1
            while j < n and text[j] != quote:
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
    safe = _safe_path(path)
    if safe is None:
        return []
    try:
        with open(safe, errors="ignore") as f:
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


def _selftest():
    """Fast, self-contained checks of the detector's own behavior (run: --selftest)."""
    import tempfile

    ok = True

    def expect(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print(f"SELFTEST FAIL: {msg}")

    # --- masker: char literals must not corrupt brace tracking, digit separators must survive ---
    masked = _mask_comments("void override_runtime_arguments() { char c = '}'; create_descriptor(x); }")
    expect("create_descriptor" in masked, "char literal '}' truncated the body before create_descriptor")
    expect("1'000'000" in _mask_comments("int n = 1'000'000; create_descriptor(y);"), "digit separator mis-masked")
    expect(
        "create_descriptor" in _mask_comments('const char* s = "}"; create_descriptor(z);'),
        "string literal masking regressed",
    )

    # --- path guard: traversal is rejected, an in-repo path resolves ---
    expect(_safe_path("../../../../etc/passwd") is None, "path traversal not rejected")
    expect(_safe_path("scripts/detect_override_rebuild.py") is not None, "in-repo path wrongly rejected")

    # A body that rebuilds, plus a legitimately-suppressed line and a bare comment mention.
    good = (
        "void Op::override_runtime_arguments(Program& p) {\n"
        "    // create_descriptor(a) mentioned in a comment must NOT flag\n"
        "    auto& r = GetRuntimeArgs(p, 0, core);\n"
        "    create_descriptor(b);\n"
        "    apply_descriptor_runtime_args(p, cb);  // override-rebuild-ok: cb-addr-only\n"
        "}\n"
    )
    with tempfile.NamedTemporaryFile("w", dir=REPO_ROOT, suffix=".cpp", delete=False) as tf:
        tf.write(good)
        tmp = tf.name
    try:
        findings = check(tmp)
        symbols = sorted(w for _l, w, _line in findings)
        expect(symbols == ["create_descriptor"], f"expected only create_descriptor flagged, got {symbols}")

        # (c) grandfathering the known offender must NOT hide a NEW, differently-worded violation.
        rel = os.path.relpath(tmp, REPO_ROOT)
        base = {os.path.normpath(rel): {("create_descriptor", "create_descriptor(b);")}}
        new_hits = [(w, line) for _l, w, line in check(tmp) if (w, line) not in base[os.path.normpath(rel)]]
        expect(new_hits == [], "baseline fingerprint failed to grandfather the exact known offender")

        with open(tmp, "a") as f:
            f.write("void Op2::override_runtime_arguments(Program& p) { create_descriptor(NEW); }\n")
        new_hits = [(w, line) for _l, w, line in check(tmp) if (w, line) not in base[os.path.normpath(rel)]]
        expect(any("NEW" in line for _w, line in new_hits), "baselined file did not catch a NEW violation")
    finally:
        os.unlink(tmp)

    if ok:
        print("SELFTEST OK")
    return 0 if ok else 1


BASELINE_HEADER = """\
# Baseline for scripts/detect_override_rebuild.py
#
# Pre-existing rebuilds of a ProgramDescriptor inside override_runtime_arguments, grandfathered one
# offender at a time. Every listed file is still FULLY scanned: any new or additional violation not
# already fingerprinted below is blocked, so a baselined file keeps its guard.
#
# Each entry is a live per-dispatch cost: override_runtime_arguments runs on EVERY program-cache hit,
# so calling create_descriptor() there pays the full cache-MISS host cost (work split, CoreRangeSet
# construction, arch queries, TensorAccessorArgs, kernel-source strings, compile-time arg vectors, a
# heap-allocated arg vector per core) on every dispatch. Fixing an op means deleting its line(s) here.
#
# For reference, the equivalent fix measured on the ops migrated in #49889 / #50338:
#   rotary_embedding cache hit  35.1 -> 27.2 us
#   uniform          cache hit  23.2 -> 13.8 us
#
# Format: one tab-separated <repo-relative-path>\\t<symbol>\\t<source-line> per offender. Regenerate
# with:  python3 scripts/detect_override_rebuild.py --update-baseline <files...>
# Comments (#) and blank lines are ignored.
"""


def _write_baseline(baseline):
    entries = []
    for path in sorted(baseline):
        for what, src in sorted(baseline[path]):
            entries.append(f"{path}\t{what}\t{src}\n")
    with open(BASELINE_PATH, "w") as f:
        f.write(BASELINE_HEADER)
        if entries:
            f.write("\n")
            f.writelines(entries)


def _update_baseline(paths):
    """Refresh baseline fingerprints for the given files, preserving entries for all others."""
    baseline = _baseline()
    for path in paths:
        np = os.path.normpath(path)
        found = {(what, line) for _lineno, what, line in check(path)}
        if found:
            baseline[np] = found
        else:
            baseline.pop(np, None)
    _write_baseline(baseline)
    return 0


def main(argv):
    if argv[1:2] == ["--update-baseline"]:
        return _update_baseline(argv[2:])
    if argv[1:2] == ["--selftest"]:
        return _selftest()
    failed = False
    baseline = _baseline()
    for path in argv[1:]:
        known = baseline.get(os.path.normpath(path), set())
        for lineno, what, line in check(path):
            if (what, line) in known:
                continue  # grandfathered pre-existing offender; new hits in this file still fail
            failed = True
            print(f"{path}:{lineno}: error: '{what}' called inside override_runtime_arguments")
            print(f"    {line}")
    if failed:
        print()
        print("override_runtime_arguments runs on EVERY program-cache hit; rebuilding the descriptor there pays the")
        print("full cache-MISS host cost per dispatch. Patch only what varies instead (see #50351 sparse_sdpa and")
        print("#50894 slice): overwrite scalar/address args in place with")
        print("tt::tt_metal::GetRuntimeArgs(program, kernel_idx, core), and re-point CB-backed addresses via a")
        print("minimal CB-only ProgramDescriptor + apply_descriptor_runtime_args (mark that one line")
        print("'// override-rebuild-ok: cb-addr-only' -- it is O(1), not a full rebuild), or with")
        print("UpdateDynamicCircularBufferAddress matched by CBIndex. No per-core reallocation on a hit.")
        print("Verify parity with -DENABLE_DESCRIPTOR_PATCHING_PARITY_CHECK=ON.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
