#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reading the LLK source as text: the helpers the contract tools share.

The tools read C++ headers without compiling them. They strip comments and strings, keeping lines so
that positions map back to source lines; find function definitions and the calls in them (matched
by argument count); and collect a function's text together with every LLK helper it calls. They
over-approximate: every branch counts, including template and #if branches a build would not take.
"""

import os
import re
from pathlib import Path

HDR_EXT = {".h", ".hpp", ".inl"}
SKIP_DIRS = {
    ".git",
    "python_env",
    ".cache",
    "_deps",
    ".cpmcache",
    "node_modules",
    "generated",
    "third_party",
    "runtime",
    "docs",
    "infra",
    "scripts",
    "data",
    "__pycache__",
    "tt-train",
}
TOP_DIRS = ("ttnn", "tt_metal", "tests", "models")
CALL = re.compile(r"\b([A-Za-z_]\w*)\s*(?:<[^;{}()]*?>)?\s*\(")


KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "catch",
    "static_assert",
    "decltype",
    "alignof",
    "noexcept",
    "requires",
    "constexpr",
    "defined",
    "__attribute__",
    "SAN_HOOK",
    "LLK_SAN_FUNCTION",
    "else",
    "do",
    "case",
    "new",
    "delete",
    "throw",
    "co_await",
}


def walk_files(repo, exts):
    """Files under repo with one of exts, without following symlinks or entering build trees."""
    if (repo / "tt_metal").is_dir() and (
        repo / "ttnn"
    ).is_dir():  # the repo root: only kernel trees
        for top in TOP_DIRS:
            if (repo / top).is_dir():
                yield from walk_files(repo / top, exts)
        return
    for root, dirs, files in os.walk(repo, followlinks=False):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith("build")]
        for name in files:
            if os.path.splitext(name)[1] in exts:
                yield Path(root) / name


def strip(text):
    """Remove comments and string/char literals, keep line structure."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            seg = text[i : (n if j < 0 else j + 2)]
            out.append("\n" * seg.count("\n"))
            i = n if j < 0 else j + 2
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            out.append(c + c)
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def match(text, i, open_, close):
    """Index just past the bracket that closes text[i] (== open_)."""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == open_:
            depth += 1
        elif text[j] == close:
            depth -= 1
            if depth == 0:
                return j + 1
    return len(text)


def count_args(text, open_paren):
    """Number of top-level arguments inside text[open_paren] == '(' ... ')'."""
    depth, commas, any_tok = 0, 0, False
    for j in range(open_paren, len(text)):
        ch = text[j]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return commas + 1 if any_tok else 0
        elif depth == 1:
            if ch == ",":
                commas += 1
            elif not ch.isspace():
                any_tok = True
    return commas + 1 if any_tok else 0


def param_range(params):
    """(min, max) arguments a parameter list accepts (defaults and ... handled)."""
    params = params.strip()
    if params in ("", "void"):
        return 0, 0
    parts, depth, cur = [], 0, []
    for ch in params:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    if any("..." in x for x in parts):
        return max(0, len(parts) - 1), 10**6
    required = sum(1 for x in parts if "=" not in x)
    return required, len(parts)


def calls_with_arity(body):
    """[(name, argc)] for every call in body."""
    out = []
    for m in CALL.finditer(body):
        if m.group(1) not in KEYWORDS:
            out.append((m.group(1), count_args(body, m.end() - 1)))
    return out


QUAL = re.compile(
    r"\s*(?:const|noexcept(?:\s*\([^)]*\))?|override|final|&|&&|->\s*[\w:<>,\s*&]+?|__attribute__\s*\(\(.*?\)\))*\s*\{",
    re.S,
)


def functions(text):
    """(name, (min, max) arguments, body) for function definitions found in stripped text."""
    i, n = 0, len(text)
    while i < n:
        m = re.compile(r"\b([A-Za-z_]\w*)\s*(?:<[^;{}()]*?>)?\s*\(").search(text, i)
        if not m:
            return
        name = m.group(1)
        close = match(text, m.end() - 1, "(", ")")
        if name in KEYWORDS:
            i = m.end()
            continue
        q = QUAL.match(text, close)
        if q:
            brace = q.end() - 1
            end = match(text, brace, "{", "}")
            yield name, param_range(text[m.end() : close - 1]), text[brace:end]
            i = end
        else:
            i = m.end()


OUTSIDE = [
    (
        "MOP program",
        r"\bckernel_(?:unpack_)?template\b|\.program\s*\(|\bmop_run\b|TTI_MOP\b|TT_MOP\b",
    ),
    (
        "replay buffer",
        r"\blltt::(?:record|replay)\b|TTI_REPLAY\b|TT_REPLAY\b|\bload_replay_buf\b",
    ),
    ("address modifiers", r"\baddr_mod_t\b|\baddr_mod_pack_t\b|ADDR_MOD_\d"),
    (
        "address counters",
        r"T{1,2}I?_SETADC\w*|T{1,2}I?_INCADC\w*|T{1,2}I?_ADDRCRXY\w*|T{1,2}I?_ADDRCRZW\w*|\breset_counters\b",
    ),
    ("GPRs", r"T{1,2}I?_SETDMAREG\b|\bp_gpr(?:_\w+)?::"),
    (
        "semaphores between threads",
        r"\bt6_semaphore\w*|T{1,2}I?_SEMINIT\b|T{1,2}I?_SEMPOST\b|T{1,2}I?_SEMGET\b|T{1,2}I?_SEMWAIT\b|\bsemaphore::",
    ),
    (
        "DEST handoff and sync",
        r"\bdest_section_flip\b|\bset_math_semaphores\b|\bmath_dest_wait\b|\bdest_dvalid\w*|\bcfg_state_id\b|\bpacker_dest_\w+",
    ),
    ("stalls and waits", r"T{1,2}I?_STALLWAIT\b|\btensix_sync\b|\bwait_for_\w+"),
    (
        "SFPU state",
        r"T{1,2}I?_SFPCONFIG\b|T{1,2}I?_SFPLOADI\b|\bsfpi::|\b_sfpu_\w+|\bLREG\d",
    ),
]


OUTSIDE_RX = [(name, re.compile(rx)) for name, rx in OUTSIDE]


def bodies_of(defs, name, memo, stack=()):
    """The text of `name` and every LLK helper it calls, transitively."""
    if name in memo:
        return memo[name]
    if name in stack:
        return ""
    parts = []
    for rng, params, body, rel in defs.get(name, []):
        parts.append(body)
        for c, argc in calls_with_arity(body):
            if c in defs and c != name:
                parts.append(bodies_of(defs, c, memo, stack + (name,)))
    memo[name] = "\n".join(parts)
    return memo[name]


IF_CONSTEXPR = re.compile(r"\bif\s+constexpr\s*\(")
ELSE = re.compile(r"\s*else\b")
CPP_WORDS = {"and", "or", "not", "True", "False"}


def safe_eval(expr, names=None):
    """Value of a small condition: literals, names from `names`, not/and/or and comparisons.
    Anything else (calls, attributes, subscripts ...) raises ValueError. Used instead of eval(),
    since the text comes from source files."""
    import ast

    names = names or {}
    cmp = {
        ast.Eq: lambda a, b: a == b,
        ast.NotEq: lambda a, b: a != b,
        ast.Lt: lambda a, b: a < b,
        ast.LtE: lambda a, b: a <= b,
        ast.Gt: lambda a, b: a > b,
        ast.GtE: lambda a, b: a >= b,
    }

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (str, int, bool)):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in names:
                return names[n.id]
            raise ValueError(n.id)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return not ev(n.operand)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            for op, right in zip(n.ops, n.comparators):
                r = ev(right)
                if type(op) not in cmp or not cmp[type(op)](left, r):
                    return False
                left = r
            return True
        raise ValueError(ast.dump(n))

    return ev(ast.parse(expr.strip(), mode="eval").body)


def _decide(cond, args):
    """True/False for a constexpr condition over the given template arguments, None if undecidable.
    Other names are allowed only as booleans (`flag && x == A::B`); the condition is decided when it
    has the same value for every value of those."""
    import itertools
    import warnings

    expr = re.sub(
        r"\b([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)", lambda m: repr(m.group(1)), cond
    )
    for name, value in args.items():
        expr = re.sub(rf"(?<!['\w:]){re.escape(name)}(?![\w'])", repr(str(value)), expr)
    expr = expr.replace("&&", " and ").replace("||", " or ")
    expr = re.sub(r"!(?!=)", " not ", expr)
    names = set(re.findall(r"(?<!['\w])([A-Za-z_]\w*)(?!['\w])", expr)) - CPP_WORDS
    for n in names:  # a name compared with something is not a boolean: undecidable
        if re.search(
            rf"(?<!['\w]){n}(?!['\w])\s*[=!<>]=|[=!<>]=\s*(?<!['\w]){n}(?!['\w])", expr
        ):
            return None
    if len(names) > 4:
        return None
    results = set()
    for vals in itertools.product((True, False), repeat=len(names)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results.add(bool(safe_eval(expr, dict(zip(sorted(names), vals)))))
        except Exception:
            return None
    return results.pop() if len(results) == 1 else None


def _branch(text, k):
    """(start, end) of the statement or block starting at k (after whitespace)."""
    while k < len(text) and text[k].isspace():
        k += 1
    if k < len(text) and text[k] == "{":
        return k, match(text, k, "{", "}")
    semi = text.find(";", k)
    return k, (len(text) if semi < 0 else semi + 1)


def _blank(text, a, b):
    return re.sub(r"[^\n]", " ", text[a:b])


def _chain_end(text, j):
    """End of an else part starting at j: a block, a statement, or an else-if chain."""
    s0 = len(text) - len(text[j:].lstrip())
    if text.startswith("if", s0) and re.match(r"if\b", text[s0:]):
        c = match(text, text.index("(", s0), "(", ")")
        _, b1 = _branch(text, c)
        e = ELSE.match(text, b1)
        return _chain_end(text, e.end()) if e else b1
    return _branch(text, j)[1]


def specialize(text, args):
    """Blank out the branches of `if constexpr (...)` that the template arguments `args` rule out, e.g.
    {"src_b_bcast_type": "BroadcastType::NONE"}. Conditions that mention anything else are kept.
    The text keeps its length and lines, so positions still map to source lines."""
    if not args:
        return text
    out, i = [], 0
    while True:
        m = IF_CONSTEXPR.search(text, i)
        if not m:
            out.append(text[i:])
            return "".join(out)
        close = match(text, m.end() - 1, "(", ")")
        decided = _decide(text[m.end() : close - 1], args)
        t0, t1 = _branch(text, close)
        if decided is None:
            out.append(text[i:close])  # keep the condition; look inside its branches
            i = close
            continue
        e = ELSE.match(text, t1)
        out.append(text[i : m.start()])
        if decided:
            out.append(_blank(text, m.start(), t0))
            out.append(specialize(text[t0:t1], args))
            if e:
                end = _chain_end(text, e.end())
                out.append(_blank(text, t1, end))
                t1 = end
            i = t1
        else:
            if e:
                out.append(_blank(text, m.start(), e.end()))
                i = e.end()  # the else part is read next, as text
            else:
                out.append(_blank(text, m.start(), t1))
                i = t1
