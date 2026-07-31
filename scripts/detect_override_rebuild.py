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

Every hit is a live per-dispatch regression, so there is no baseline and no suppression marker: the
only way to satisfy this check is to fix the op. The drift concern is real but has a cheaper answer,
used by the merged reference fixes (#50351 sparse_sdpa, #50894 slice): on a cache hit, patch only the
slots that actually vary. Overwrite scalar/address runtime args in place with
``tt::tt_metal::GetRuntimeArgs(program, kernel_idx, core)``, and re-point CB-backed addresses by
building a CB-only ``ProgramDescriptor`` inline — no ``create_descriptor`` — and handing that to
``apply_descriptor_runtime_args``. Nothing is re-split, no per-core arg vector is reallocated, and the
hit stays O(1). Verify parity with ``-DENABLE_DESCRIPTOR_PATCHING_PARITY_CHECK=ON``.

This is a BUILD-TIME text check (no runtime cost). It scans the body of every
``override_runtime_arguments`` definition and rejects a descriptor rebuild inside it. pre-commit passes
the staged files, like the other local hooks; run with no arguments it sweeps all of ttnn (~1s), which
is the audit mode. Ops that legitimately delegate to the framework's cheap binding path
(``descriptor_adapter_t::apply_descriptor``) are unaffected.
"""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_ROOT = os.path.join(REPO_ROOT, "ttnn")
SCAN_SUFFIXES = (".cpp", ".hpp")


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


def _all_sources():
    """Every ttnn source, for the default whole-tree run."""
    for dirpath, dirnames, filenames in os.walk(SCAN_ROOT):
        dirnames[:] = [d for d in dirnames if d != "build" and not d.startswith(".")]
        for name in sorted(filenames):
            if name.endswith(SCAN_SUFFIXES):
                yield os.path.relpath(os.path.join(dirpath, name), REPO_ROOT)


HOOK = re.compile(r"\boverride_runtime_arguments\s*\(")
# The rebuild itself. A CB-only ProgramDescriptor assembled inline is the sanctioned O(1) patch and
# does not match; only re-deriving a full descriptor does.
REBUILD = re.compile(r"\b(?:[A-Za-z_]\w*::)?create_(?:\w+_)?descriptor\s*\(")


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

    src = (
        "void Op::override_runtime_arguments(Program& p) {\n"
        "    // create_descriptor(a) mentioned in a comment must NOT flag\n"
        "    auto& r = GetRuntimeArgs(p, 0, core);\n"
        "    r[3] = addr;\n"
        "    ProgramDescriptor cb_only;\n"
        "    cb_only.cbs.push_back(CBDescriptor{.buffer = t.buffer()});\n"
        "    apply_descriptor_runtime_args(p, cb_only);\n"
        "}\n"
        "void Op2::override_runtime_arguments(Program& p) {\n"
        "    auto desc = Factory::create_descriptor(attrs, args, out);\n"
        "    apply_descriptor_runtime_args(p, desc);\n"
        "}\n"
    )
    with tempfile.NamedTemporaryFile("w", dir=REPO_ROOT, suffix=".cpp", delete=False) as tf:
        tf.write(src)
        tmp = tf.name
    try:
        # The sanctioned O(1) CB-only patch stays clean; only the rebuild is flagged.
        findings = check(tmp)
        expect(len(findings) == 1, f"expected exactly 1 finding, got {[f[1] for f in findings]}")
        expect(findings and findings[0][1] == "Factory::create_descriptor", f"wrong symbol: {findings}")

        # A trailing marker no longer buys an exemption.
        with open(tmp, "a") as f:
            f.write(
                "void Op3::override_runtime_arguments(Program& p) {\n"
                "    create_descriptor(x);  // override-rebuild-ok: allowed before this change\n"
                "}\n"
            )
        expect(len(check(tmp)) == 2, "a suppression comment still exempted a rebuild")
    finally:
        os.unlink(tmp)

    if ok:
        print("SELFTEST OK")
    return 0 if ok else 1


def main(argv):
    if argv[1:2] == ["--selftest"]:
        return _selftest()
    failed = False
    for path in argv[1:] or _all_sources():
        for lineno, what, line in check(path):
            failed = True
            print(f"{path}:{lineno}: error: '{what}' called inside override_runtime_arguments")
            print(f"    {line}")
    if failed:
        print()
        print("override_runtime_arguments runs on EVERY program-cache hit; rebuilding the descriptor there pays the")
        print("full cache-MISS host cost per dispatch. There is no baseline and no suppression -- patch only what")
        print("varies instead (see #50351 sparse_sdpa and #50894 slice): overwrite scalar/address args in place with")
        print("tt::tt_metal::GetRuntimeArgs(program, kernel_idx, core), and re-point CB-backed addresses by building")
        print("a CB-only ProgramDescriptor inline (no create_descriptor) and passing that to")
        print("apply_descriptor_runtime_args, or with UpdateDynamicCircularBufferAddress matched by CBIndex.")
        print("No per-core reallocation on a hit. Verify parity with -DENABLE_DESCRIPTOR_PATCHING_PARITY_CHECK=ON.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
