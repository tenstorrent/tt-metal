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

# A call may carry template arguments before its `(`: `t6_mutex_acquire<mutex::REG_RMW>()`.
_CALL = r"\s*(?:<[^;{}()]*>)?\s*\("
ACQUIRE = re.compile(r"\b(?:t6_mutex_acquire|TTI?_ATGETM)" + _CALL)
RELEASE = re.compile(r"\b(?:t6_mutex_release|TTI?_ATRELM)" + _CALL)
# The wrapper definitions and the RAII guard's ctor/dtor each hold one half by construction.
EXEMPT_DEFN = re.compile(
    r"\b(?:inline\s+void\s+)?t6_mutex_(?:acquire|release)\s*\(\s*const\b"
)
GUARD_CLASS = re.compile(r"\b(?:class|struct)\b[^;{]*\bT6MutexLockGuard\b")
# A brace opened by one of these declares a scope, not a function body.
CONTAINER = re.compile(r"\b(?:namespace|class|struct|union|enum|extern)\b")
_TEMPLATE = re.compile(r"\btemplate\s*<")
# A constructor's member-initializer list: a `:` (not `::`) after the declarator's `)`.
_CTOR_INIT = re.compile(r"\)\s*(?:noexcept\b[^:{]*)?:(?!:)")
# The end of a lambda declarator: `[...](params) mutable -> T` right before its body.
_LAMBDA_DECL = re.compile(
    r"\]\s*(?:\((?:[^()]|\([^()]*\))*\))?\s*(?:(?:mutable|constexpr|noexcept)\s*)*(?:->[^{};]*)?$"
)
_PP_LINE = re.compile(r"(?m)^[ \t]*#(?:[^\n]*\\\n)*[^\n]*")


def _scope_words(head):
    """head without its template parameter lists and parenthesised text.

    Only what is left says which scope a brace opens: `template <class T>` and an `enum`
    parameter type name a container keyword without declaring one.
    """
    out, i, n, paren = [], 0, len(head), 0
    while i < n:
        m = _TEMPLATE.match(head, i) if paren == 0 else None
        if m:
            depth, i = 1, m.end()
            while i < n and depth:
                depth += {"<": 1, ">": -1}.get(head[i], 0)
                i += 1
            continue
        c = head[i]
        if c == "(":
            paren += 1
        elif c == ")":
            paren = max(paren - 1, 0)
        elif paren == 0:
            out.append(c)
        i += 1
    return "".join(out)


def _blank_noncode(src):
    """Return src with comment and string-literal bodies blanked, offsets and lines preserved.

    Brace counting has to see only real braces: one inside a comment or a string literal
    unbalances every scope after it.
    """
    out, i, n = [], 0, len(src)
    while i < n:
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
        elif src[i] == "'" and i and src[i - 1].isalnum():
            # a digit separator (`0x8000'0000`), not a character literal
            out.append(src[i])
            i += 1
            continue
        elif src[i] in "\"'":
            q, j = src[i], i + 1
            while j < n and src[j] != q and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
        else:
            out.append(src[i])
            i += 1
            continue
        out.append("".join(c if c == "\n" else " " for c in src[i:j]))
        i = j
    return "".join(out)


def _signature(lines, brace_line):
    """The declarator line for a body, which is what a reader recognises and the baseline keys on."""
    if (
        lines[brace_line].strip().rstrip("\\").strip() != "{"
    ):  # `{ \` inside a macro is bare too
        return brace_line
    sig = brace_line - 1
    while sig > 0:
        prev = re.sub(r"//.*", "", lines[sig]).strip()
        if not prev or prev.startswith(("*", "/*", "//", "#")):
            sig -= 1
            continue
        break
    return max(sig, 0)


def functions(src):
    """Yield (signature_line_index, end_line_index, text, in_guard) for each function body.

    Brace kind decides what a body is. A `namespace`, `class`, `struct`, `union`, `enum` or
    `extern` brace opens a container; only a declarator carrying a parameter list opens a
    function. Every LLK header wraps its functions in `namespace ckernel`, so counting the
    namespace brace as a body would merge the whole file into one balance and let a leak in
    one function cancel against a release in another. A keyword inside a template parameter
    list or the parameter list (`template <class T>`, an `enum` parameter) does not make a
    container, and a brace inside the declarator -- a `= {}` default argument, a member
    brace-init `: m{0}` -- is part of the head; a lambda passed as an argument is a body.

    Keyed on position, not name: two overloads with the same name in one file would otherwise
    share a single balance count and mask each other.

    `in_guard` marks a body lexically inside `T6MutexLockGuard`, whose ctor acquires and dtor
    releases -- balanced per object, never within one function.
    """
    code = _blank_noncode(src)
    line_of, ln = [], 0
    for c in code:
        line_of.append(ln)
        if c == "\n":
            ln += 1
    line_of.append(ln)
    # A preprocessor line's parentheses need not balance, so they never move `paren`; its
    # braces still count, since a macro can define a whole function.
    in_pp = bytearray(len(code))
    for m in _PP_LINE.finditer(code):
        in_pp[m.start() : m.end()] = b"\x01" * (m.end() - m.start())

    stack, decl_start, fn, paren = [], 0, None, 0
    for k, c in enumerate(code):
        if c in "()" and in_pp[k]:
            pass
        elif c == "(":
            paren += 1
        elif c == ")":
            paren = max(paren - 1, 0)
        elif c == "{":
            head = code[decl_start:k]
            if fn is None and paren and _LAMBDA_DECL.search(head):
                kind, fn = "fn", k  # a lambda passed as an argument at namespace scope
            # A brace inside the declarator -- a `= {}` default argument or a member
            # brace-init `: m{0}` -- is part of the head, not a body.
            elif paren or (
                fn is None and _CTOR_INIT.search(head) and re.search(r"[\w>]\s*$", head)
            ):
                kind = "declarator"
            elif (
                fn is None and "(" in head and not CONTAINER.search(_scope_words(head))
            ):
                kind, fn = "fn", k
            elif GUARD_CLASS.search(head):
                kind = "guard"
            else:
                kind = "other"
            stack.append(kind)
            if kind != "declarator":
                decl_start = k + 1
        elif c == "}":
            kind = stack.pop() if stack else "other"
            if kind != "declarator":
                decl_start = k + 1
            if kind == "fn" and fn is not None:
                yield line_of[fn], line_of[k], code[fn : k + 1], "guard" in stack
                fn = None
        elif c == ";":
            decl_start = k + 1
            if fn is None:
                paren = 0  # outside a body, no statement spans a `;`


def scan(path):
    src = open(path, errors="ignore").read()
    lines = src.split("\n")
    for brace_line, end, body, in_guard in functions(src):
        if in_guard:
            continue  # RAII guard: acquire in ctor, release in dtor, balanced at the object level
        start = _signature(lines, brace_line)
        head = lines[start] if start < len(lines) else ""
        if EXEMPT_DEFN.search(head):
            continue
        acq = len(ACQUIRE.findall(body))
        rel = len(RELEASE.findall(body))
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
