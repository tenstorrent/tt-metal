#!/usr/bin/env python3
"""Flag a function whose Tensix mutex acquires and releases do not balance.

An unbalanced `t6_mutex_acquire` / `t6_mutex_release` (or raw `ATGETM` / `ATRELM`) is a lock leak
or a double release: the thread either never frees the mutex -- wedging every other thread that
takes it -- or frees one it does not hold.

Scope is one function body, matched by brace depth. Two constructs are legitimately unbalanced
within a single body and are NOT flagged:

  * the definitions of the wrappers themselves (`t6_mutex_acquire` / `t6_mutex_release` in
    ckernel.h), which contain one half each by definition;
  * an RAII guard (`T6MutexLockGuard`), which acquires in its constructor and releases in its
    destructor -- balanced at the object level, never within one function.

Known limit, inherited from the llk-audit checker this is derived from: an acquire or release
placed under a branch can look unbalanced when it is not. Such a site should be restructured or
baselined with a reason rather than silently tolerated -- on the tree this was written against the
count is zero, so any finding is new.
"""
import argparse
import os
import re
import sys

ACQUIRE = re.compile(r"\b(?:t6_mutex_acquire|TTI?_ATGETM)\s*\(")
RELEASE = re.compile(r"\b(?:t6_mutex_release|TTI?_ATRELM)\s*\(")
# The wrapper definitions and the RAII guard's ctor/dtor each hold one half by construction.
EXEMPT_DEFN = re.compile(
    r"\b(?:inline\s+void\s+)?t6_mutex_(?:acquire|release)\s*\(\s*const\b"
)
GUARD_CLASS = re.compile(r"\bclass\b[^;{]*\bT6MutexLockGuard\b")


def functions(lines):
    """Yield (start_line_index, end_line_index, text) for each brace-delimited body.

    Keyed on position, not name: two overloads with the same name in one file would otherwise
    share a single balance count and mask each other.
    """
    depth, start, buf = 0, None, []
    for i, raw in enumerate(lines):
        line = re.sub(r"//.*", "", raw)
        if depth == 0 and "{" in line:
            start, buf = i, []
        if start is not None:
            buf.append(raw)
        depth += line.count("{") - line.count("}")
        if start is not None and depth <= 0:
            # The opening brace is usually on its own line; the signature is above it. Report the
            # signature, not "{" -- it is what a reader recognises and what the baseline keys on.
            sig = start
            while sig > 0:
                prev = re.sub(r"//.*", "", lines[sig - 1]).strip()
                if not prev or prev.startswith(("*", "/*", "#")):
                    sig -= 1
                    continue
                if lines[start].strip() == "{":
                    sig -= 1
                break
            yield sig, i, "\n".join(buf)
            start, buf, depth = None, [], 0


def scan(path):
    src = open(path, errors="ignore").read()
    lines = src.split("\n")
    if GUARD_CLASS.search(src):
        return  # RAII guard: acquire in ctor, release in dtor, balanced at the object level
    for start, end, body in functions(lines):
        head = lines[start] if start < len(lines) else ""
        if EXEMPT_DEFN.search(head):
            continue
        body_nc = re.sub(r"//.*", "", body)
        acq = len(ACQUIRE.findall(body_nc))
        rel = len(RELEASE.findall(body_nc))
        if acq != rel:
            yield start + 1, acq, rel, head.strip()[:76]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="*")
    ap.add_argument("--baseline")
    args = ap.parse_args()

    accepted = set()
    if args.baseline:
        try:
            accepted = {
                l.split("#")[0].strip()
                for l in open(args.baseline)
                if l.strip() and not l.startswith("#")
            }
        except FileNotFoundError:
            pass

    n = 0
    for path in args.files:
        for line, acq, rel, head in scan(path):
            key = f"{os.path.relpath(path)}:{head}"
            if key in accepted:
                continue
            n += 1
            what = (
                "acquired but never released"
                if acq > rel
                else "released without acquiring"
            )
            print(f"{path}:{line}: mutex {what} ({acq} acquire vs {rel} release)")
            print(f"  in: {head}")
            print(
                "  fix: pair every t6_mutex_acquire with a t6_mutex_release on every path, or"
            )
            print(
                "       use the T6MutexLockGuard RAII guard so the release cannot be missed.\n"
            )
    if n:
        print(f"{n} function(s) with unbalanced Tensix mutex use.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
