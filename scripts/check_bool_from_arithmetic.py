#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pre-commit guard: forbid a ``bool`` initialized from a multiplicative expression.

``bool n = (a + b - 1) / b;`` stores a quotient in a ``bool``, which keeps only zero / non-zero:
every tile count, size or index collapses to 1. No compiler covers this: GCC's
``-Wint-in-bool-context`` fires only for ``*`` and the host build disables it; neither GCC nor Clang
warns for ``/`` or ``%``.

Flagged: a declaration of a ``bool`` (``const`` / ``constexpr`` / ``static`` allowed before or, for
cv-qualifiers, after ``bool``; ``= init``, ``{init}`` or a default argument) whose initializer's
top-level operator, after stripping enclosing parentheses, is a binary ``*``, ``/`` or ``%``. Every
declarator of a multi-declaration is checked (``bool a, b = x / y;`` flags ``b``). Anything
else at the top level -- a call, a comparison, a logical, bitwise, additive or shift operator, a
ternary -- makes the declaration out of scope. The check prefers missing a case to flagging correct
code, so anything it cannot parse unambiguously (for example a ``<`` that may be a comparison) is
skipped. Direct-initialization ``bool x(a / b);`` is not checked: textually it is a function
declaration (``bool f(T* p);``), so accepting it would flag every such declaration with a pointer
parameter.

Only files passed on the command line are checked, so a commit is blocked only by what it touches.
An optional ``--baseline`` file grandfathers known sites, one ``<repo-relative-path>\t<source line>``
per line (line-number independent); none exist today.
"""

import argparse
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DECL = re.compile(
    r"\b(?:(?:const|constexpr|static|inline|volatile|thread_local)\s+)*bool\s+(?:(?:const|volatile)\s+)*"
    r"(?=[A-Za-z_])"
)
NAME = re.compile(r"([A-Za-z_]\w*)\s*")
RAW_START = re.compile(r'(?:u8|[uUL])?R"([^ ()\\\t\n]{0,16})\(')
# A `'` right after a number literal is a digit separator, not a character literal.
NUMBER_BEFORE = re.compile(r"(?<![\w.])\d[\w.']*$")
TOKEN = re.compile(
    r"\s*(?:(?P<num>\.?\d(?:[\w.']|[eEpP][+-])*)|(?P<id>[A-Za-z_]\w*)|(?P<op>::|->\*?|\.\*|<=>|<<=|>>=|\+\+|--|&&|\|\||"
    r"[=!<>+\-*/%&|^]=|<<|>>|[-+*/%&|^~!<>=?:,.;()\[\]{}#]))"
)
# Binary operators below the multiplicative ones: if one is at the top level, the top-level operator
# is not multiplicative.
LOWER = {
    "+",
    "-",
    "<<",
    ">>",
    "<",
    ">",
    "<=",
    ">=",
    "<=>",
    "==",
    "!=",
    "&",
    "^",
    "|",
    "&&",
    "||",
    "?",
    ":",
    "=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "<<=",
    ">>=",
    "&=",
    "|=",
    "^=",
    ",",
}
MULT = {"*", "/", "%"}
OPEN = {"(": ")", "[": "]", "{": "}"}
CLOSE = {")", "]", "}"}


def blank_noncode(src):
    """Blank comments, string / char literals and preprocessor lines, preserving offsets and newlines."""
    out, i, n = [], 0, len(src)
    at_line_start = True
    while i < n:
        c = src[i]
        if at_line_start and c in " \t":
            out.append(c)
            i += 1
            continue
        if at_line_start and c == "#":
            j = i
            while j < n:
                k = src.find("\n", j)
                if k < 0:
                    j = n
                    break
                if src[k - 1] == "\\":
                    j = k + 1
                    continue
                j = k
                break
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
            continue
        at_line_start = c == "\n"
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
        elif RAW_START.match(src, i) and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            m = RAW_START.match(src, i)
            if not m:
                out.append(c)
                i += 1
                continue
            end = src.find(")" + m.group(1) + '"', m.end())
            j = n if end < 0 else end + len(m.group(1)) + 2
        elif c == "'" and NUMBER_BEFORE.search(src, max(0, i - 64), i):
            out.append(c)  # digit separator
            i += 1
            continue
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
        else:
            out.append(c)
            i += 1
            continue
        out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
        i = j
    return "".join(out)


def tokens(text):
    pos, out = 0, []
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m or m.end() == pos:
            if text[pos:].strip() == "":
                break
            return None  # unknown character: refuse to judge
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
        pos = m.end()
    return out


def initializer(code, start, brace):
    """The initializer text from start to its top-level end, as (text, index of the terminator), or
    None if it is not one expression.

    `= init` ends at `;`, `,` or the `)` closing a parameter list; `{init}` ends at its `}` and must
    hold a single element.
    """
    depth = 0
    for i in range(start, len(code)):
        c = code[i]
        if c in OPEN:
            depth += 1
        elif c in CLOSE:
            if depth == 0:
                return (code[start:i], i) if c == ("}" if brace else ")") else None
            depth -= 1
        elif depth == 0 and c == ";":
            return None if brace else (code[start:i], i)
        elif depth == 0 and c == ",":
            return None if brace else (code[start:i], i)
    return None


def declarators(code, pos):
    """Yield (name offset, initializer text or None) for each declarator of the declaration at pos.

    `bool a = 1, b{2}, c;` yields a, b and c. The walk follows top-level commas only while the next
    declarator is a bare name followed by `=`, `{`, `,`, `;` or `)`, so a parameter list
    (`void f(bool a, int b = x / y)`) or a template parameter list stops after the bool.
    """
    while True:
        m = NAME.match(code, pos)
        if not m:
            return
        name_at, pos = m.start(1), m.end()
        c = code[pos : pos + 1]
        expr = None
        if c == "=" and code[pos + 1 : pos + 2] != "=":
            found = initializer(code, pos + 1, False)
            if found is None:
                return
            expr, pos = found
        elif c == "{":
            found = initializer(code, pos + 1, True)
            if found is None:
                return
            expr, pos = found
            pos += 1
            while pos < len(code) and code[pos].isspace():
                pos += 1
        elif c not in (",", ";", ")"):
            return
        yield name_at, expr
        if code[pos : pos + 1] != ",":
            return
        pos += 1
        while pos < len(code) and code[pos].isspace():
            pos += 1


def _skip_template_args(toks, i):
    """If toks[i] is a `<` opening template arguments, return the index after its `>`; else None.

    Only plain type-like contents (names, ::, numbers, commas, trailing * or &, nested <>) count,
    and the `>` must be followed by `(`, `{` or `::` -- so a comparison is never mistaken for one.
    """
    if i == 0 or toks[i - 1][0] != "id":
        return None
    depth, j = 0, i
    while j < len(toks):
        kind, t = toks[j]
        if t == "<":
            depth += 1
        elif t == ">" or t == ">>":
            depth -= 1 if t == ">" else 2
            if depth <= 0:
                if depth < 0:
                    return None
                nxt = toks[j + 1][1] if j + 1 < len(toks) else None
                return j + 1 if nxt in ("(", "{", "::") else None
        elif not (kind in ("id", "num") or t in ("::", ",", "*", "&")):
            return None
        elif t in ("*", "&") and (j + 1 >= len(toks) or toks[j + 1][1] not in (">", ",")):
            return None
        j += 1
    return None


def _strip_parens(toks):
    while len(toks) >= 2 and toks[0][1] == "(" and toks[-1][1] == ")":
        depth = 0
        for k, (_, t) in enumerate(toks):
            if t in OPEN:
                depth += 1
            elif t in CLOSE:
                depth -= 1
                if depth == 0 and k != len(toks) - 1:
                    return toks
        toks = toks[1:-1]
    return toks


def top_level_is_multiplicative(expr):
    toks = tokens(expr)
    if not toks:
        return False
    toks = _strip_parens(toks)
    depth, seen_mult, prev_operand, i = 0, False, False, 0
    while i < len(toks):
        kind, t = toks[i]
        if t in OPEN:
            depth += 1
            prev_operand = False
        elif t in CLOSE:
            depth -= 1
            if depth < 0:
                return False
            prev_operand = True
        elif depth == 0:
            if t == "<":
                after = _skip_template_args(toks, i)
                if after is None:
                    return False  # a comparison, or something not parsed with certainty
                i = after
                prev_operand = False
                continue
            if kind in ("id", "num"):
                if kind == "id" and t in ("new", "delete", "throw", "co_await", "co_yield"):
                    return False
                prev_operand = True
            elif t in MULT:
                if prev_operand:
                    seen_mult = True
                elif t != "*":
                    return False
                prev_operand = False
            elif t in ("+", "-") and not prev_operand:
                prev_operand = False  # unary sign binds tighter
            elif t in LOWER:
                return False
            elif t in ("!", "~", "&", "++", "--"):
                prev_operand = t in ("++", "--") and prev_operand
            elif t in ("::", ".", "->", ".*", "->*"):
                prev_operand = False
            else:
                return False
        i += 1
    return seen_mult


def scan(path):
    """Yield (line_no, source_line) for each flagged declaration."""
    try:
        src = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return
    if "bool" not in src:
        return
    code = blank_noncode(src)
    lines = src.split("\n")
    for m in DECL.finditer(code):
        for name_at, expr in declarators(code, m.end()):
            if expr is not None and top_level_is_multiplicative(expr):
                line_no = code.count("\n", 0, name_at) + 1
                yield line_no, lines[line_no - 1].strip()


def load_baseline(path):
    entries = set()
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                if raw.strip() and not raw.lstrip().startswith("#"):
                    rel, _, line = raw.rstrip("\n").partition("\t")
                    entries.add((rel, line.strip()))
    except FileNotFoundError:
        pass
    return entries


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--baseline")
    args = ap.parse_args(argv)

    baseline = load_baseline(args.baseline) if args.baseline else set()
    used, bad = set(), 0
    checked = set()
    for path in args.files:
        abspath = os.path.realpath(path if os.path.isabs(path) else os.path.join(os.getcwd(), path))
        rel = os.path.relpath(abspath, REPO_ROOT)
        checked.add(rel)
        for line_no, line in scan(abspath):
            if (rel, line) in baseline:
                used.add((rel, line))
                continue
            bad += 1
            print(f"{rel}:{line_no}: bool initialized from a multiplicative expression: {line}")
    for rel, line in sorted(baseline - used):
        if rel in checked:
            print(f"note: baseline entry no longer found, remove it from {args.baseline}: {rel}\t{line}")
    if bad:
        print(
            f"{bad} finding(s). A bool keeps only zero / non-zero, so a count or quotient collapses to 1.\n"
            "Use an integer type, or compare explicitly (e.g. `(a / b) != 0`) if a truth value is meant."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
