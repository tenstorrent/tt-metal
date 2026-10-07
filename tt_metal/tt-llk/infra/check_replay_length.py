#!/usr/bin/env python3
"""Fail when a replay recording's declared length does not match what it records.

A recording names a length N and captures the next N Tensix instructions the thread issues:

  * `lltt::record(start, N)` and a raw `REPLAY` with load mode 1 record the instructions that
    follow them;
  * `load_replay_buf(...)` records what its callable emits, and its contract is that N matches
    that expansion exactly.

A short recording silently captures whatever the thread issues next -- after a short
`lltt::record`, that is the caller's instructions once the function returns; an over-long
callable issues its extra instructions immediately instead of recording them.

Only decidable sites are checked. N must be a literal or a same-file `constexpr` over literals,
and every statement up to the point of decision must have a known instruction count: one per
`TTI_<op>` / `TT_<op>` / `TTI_INSN` / `TT_INSN`, file-local macros made of those, constant-bound
`for` loops and integer declarations. Anything else -- a call, a branch, sfpi code, `#if`, a
template-dependent length -- makes the site undecidable and it is skipped, never guessed.
"""

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_mutex_balance import _blank_noncode, functions  # noqa: E402

LLK = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
METAL_HW = os.path.normpath(os.path.join(LLK, "..", "hw", "ckernels"))
ARCHES = ("wormhole_b0", "blackhole", "quasar")


class Undecidable(Exception):
    pass


def _ops_header(arch):
    return os.path.join(LLK, f"tt_llk_{arch}", "common", "inc", "ckernel_ops.h")


def _opcodes(arch):
    """Mnemonics with a `TT_OP_<X>`: each `TTI_<X>` / `TT_<X>` emits exactly one instruction."""
    try:
        src = open(_ops_header(arch)).read()
    except OSError:
        return set()
    return set(re.findall(r"#define\s+TT_OP_(\w+)", src)) - {""}


OPCODES = {a: _opcodes(a) for a in ARCHES}


def arch_of(path):
    p = os.path.abspath(path).replace(os.sep, "/")
    for a in ARCHES:
        if f"/tt_llk_{a}/" in p or f"/ckernels/{a}/" in p:
            return a
    return None


# ---------------------------------------------------------------- integer evaluation
_INT = re.compile(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]*\b")
_IDENT = re.compile(r"\b[A-Za-z_]\w*(?:::\w+)*\b")
_SAFE = re.compile(r"^[\d\s+\-*/%()<>]*$")


def _defs(text, name):
    pat = re.compile(
        r"\bconstexpr\s+(?:static\s+)?(?:const\s+)?[\w:]+\s+"
        + re.escape(name)
        + r"\s*(?:=|\{)\s*([^;{}]+?)\s*[;}]"
    )
    return [m.group(1) for m in pat.finditer(text)]


def evaluate(expr, scopes, depth=0):
    """Integer value of expr, resolving names to a unique constexpr in the nearest scope."""
    if depth > 8:
        raise Undecidable("length not constant")
    expr = expr.strip()

    def name(m):
        n = m.group(0)
        if re.fullmatch(r"\d\w*", n):
            return n
        # One definition across every visible scope; a shadowed or branch-dependent name is skipped.
        found = {d for scope in scopes for d in _defs(scope, n.split("::")[-1])}
        if len(found) != 1:
            raise Undecidable("length not constant")
        return f"({evaluate(found.pop(), scopes, depth + 1)})"

    out = _IDENT.sub(name, expr)
    out = _INT.sub(lambda m: str(int(m.group(1), 0)), out)
    if not _SAFE.match(out) or not out.strip():
        raise Undecidable("length not constant")
    out = re.sub(r"(?<![/])/(?![/])", "//", out)
    try:
        val = eval(
            out, {"__builtins__": {}}, {}
        )  # noqa: S307 -- digits and operators only
    except Exception:
        raise Undecidable("length not constant")
    if not isinstance(val, int) or val < 0:
        raise Undecidable("length not constant")
    return val


# ---------------------------------------------------------------- statement parsing
def _skip_ws(t, i):
    while i < len(t) and t[i].isspace():
        i += 1
    return i


def _match(t, i, open_c, close_c):
    depth = 0
    for k in range(i, len(t)):
        if t[k] == open_c:
            depth += 1
        elif t[k] == close_c:
            depth -= 1
            if depth == 0:
                return k + 1
    raise Undecidable("unbalanced")


def split_args(s):
    out, depth, cur = [], 0, ""
    for c in s:
        if c in "(<[{":
            depth += 1
        elif c in ")>]}":
            depth -= 1
        if c == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += c
    if cur.strip():
        out.append(cur.strip())
    return out


class Node:
    def __init__(self, kind, start, end, **kw):
        self.kind, self.start, self.end = kind, start, end
        self.__dict__.update(kw)


def parse_block(t, i, end):
    """Parse statements in t[i:end] into a list of Nodes."""
    nodes = []
    while True:
        i = _skip_ws(t, i)
        if i >= end:
            return nodes
        n = parse_stmt(t, i, end)
        nodes.append(n)
        i = n.end


_KW = re.compile(
    r"(for|if|while|do|switch|else|return|break|continue|goto|case|default)\b"
)


def parse_stmt(t, i, end):
    c = t[i]
    if c == "#":
        j = t.find("\n", i)
        j = end if j < 0 or j > end else j
        return Node("pp", i, j, text=t[i:j])
    if c == "{":
        j = _match(t, i, "{", "}")
        return Node("block", i, j, kids=parse_block(t, i + 1, j - 1))
    if c == ";":
        return Node("empty", i, i + 1)
    m = _KW.match(t, i)
    if m and m.group(1) == "for":
        p = _skip_ws(t, m.end())
        if p >= end or t[p] != "(":
            raise Undecidable("unparsed")
        q = _match(t, p, "(", ")")
        body = parse_stmt(t, _skip_ws(t, q), end)
        return Node("for", i, body.end, header=t[p + 1 : q - 1], body=body)
    if m and m.group(1) == "if":
        p = _skip_ws(t, m.end())
        if t.startswith("constexpr", p):
            p = _skip_ws(t, p + len("constexpr"))
        if p >= end or t[p] != "(":
            raise Undecidable("unparsed")
        q = _match(t, p, "(", ")")
        then = parse_stmt(t, _skip_ws(t, q), end)
        j = then.end
        k = _skip_ws(t, j)
        if t.startswith("else", k) and not re.match(r"\w", t[k + 4 : k + 5] or " "):
            other = parse_stmt(t, _skip_ws(t, k + 4), end)
            j = other.end
        return Node("if", i, j, then=then)
    # A simple statement, or a macro invocation that carries no trailing `;`.
    mm = re.match(r"([A-Za-z_]\w*)\s*", t[i:end])
    if mm:
        p = i + mm.end()
        if p < end and t[p] == "(":
            q = _match(t, p, "(", ")")
            r = _skip_ws(t, q)
            if r >= end or t[r] != ";" and not re.match(r"[.\[=+\-*/%&|^<>?:,]", t[r]):
                return Node("simple", i, q, text=t[i:q])
    depth, k = 0, i
    while k < end:
        ch = t[k]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                raise Undecidable("unparsed")
        elif ch == ";" and depth == 0:
            return Node("simple", i, k + 1, text=t[i:k])
        k += 1
    return Node("simple", i, end, text=t[i:end])


# ---------------------------------------------------------------- instruction counting
_INT_DECL = re.compile(
    r"^(?:(?:static|constexpr|const|volatile)\s+)*(?:std::)?(?:u?int(?:8|16|32|64)_t|unsigned|int|uint|bool|size_t)"
    r"\s+\w+\s*(?:=\s*(?P<init>.*)|\{(?P<binit>.*)\})?$",
    re.S,
)
_CALLISH = re.compile(r"\b[A-Za-z_]\w*\s*(?:<[^;{}()]*>)?\s*\(")
_CASTS = re.compile(r"\b(?:static_cast|reinterpret_cast|const_cast)\s*<[^<>]*>\s*\(")


class Counter:
    def __init__(self, arch, macros, scopes):
        self.ops = OPCODES.get(arch) or set().union(*OPCODES.values())
        self.macros = macros
        self.scopes = scopes

    def simple(self, text, depth=0):
        s = " ".join(text.split())
        if not s:
            return 0
        m = re.fullmatch(r"(TTI?)_(\w+)\s*(\(.*\))?", s, re.S)
        if m and m.group(2) == "INSN" and m.group(3):
            return 1
        if m and m.group(2) in self.ops:
            if m.group(2) == "REPLAY":
                raise Undecidable("nested replay")
            if m.group(3) and _match(m.group(3), 0, "(", ")") != len(m.group(3)):
                raise Undecidable("statement")
            return 1
        m = re.fullmatch(r"([A-Za-z_]\w*)\s*(\(.*\))?", s, re.S)
        if m and m.group(1) in self.macros and depth < 8:
            body = self.macros[m.group(1)]
            return self.seq(parse_block(body, 0, len(body)), depth + 1)
        d = _INT_DECL.match(s)
        if d:
            init = d.group("init") or d.group("binit") or ""
            if (
                _CALLISH.search(_CASTS.sub("(", init))
                or "[" in init
                or "instrn_buffer" in init
            ):
                raise Undecidable("statement")
            return 0
        raise Undecidable("call or non-instruction statement")

    def node(self, n, depth=0):
        if n.kind == "empty":
            return 0
        if n.kind == "pp":
            if re.match(r"#\s*pragma\b", n.text):
                return 0  # unrolling changes code size, not the number of instructions issued
            raise Undecidable("preprocessor conditional")
        if n.kind == "block":
            return self.seq(n.kids, depth)
        if n.kind == "for":
            try:
                trips = self.trips(n.header)
            except Undecidable:
                raise Undecidable("loop bound")
            return trips * self.node(n.body, depth)
        if n.kind == "if":
            raise Undecidable("branch")
        return self.simple(n.text, depth)

    def seq(self, nodes, depth=0):
        return sum(self.node(n, depth) for n in nodes)

    def trips(self, header):
        parts = header.split(";")
        if len(parts) != 3:
            raise Undecidable("loop bound")
        init, cond, step = (p.strip() for p in parts)
        m = re.fullmatch(
            r"(?:(?:constexpr|const)\s+)?(?:std::)?(?:u?int(?:8|16|32|64)_t|unsigned|int|uint|size_t|auto)"
            r"\s+(\w+)\s*=\s*(.+)",
            init,
        )
        if not m:
            raise Undecidable("loop bound")
        var, lo = m.group(1), evaluate(m.group(2), self.scopes)
        m = re.fullmatch(re.escape(var) + r"\s*(<|<=|!=)\s*(.+)", cond)
        if not m:
            raise Undecidable("loop bound")
        op, hi = m.group(1), evaluate(m.group(2), self.scopes)
        if re.fullmatch(
            r"\+\+\s*" + re.escape(var) + r"|" + re.escape(var) + r"\s*\+\+", step
        ):
            inc = 1
        else:
            m = re.fullmatch(re.escape(var) + r"\s*\+=\s*(.+)", step)
            if not m:
                raise Undecidable("loop bound")
            inc = evaluate(m.group(1), self.scopes)
        if op == "<=":
            hi += 1
        if inc <= 0 or hi < lo or (op == "!=" and (hi - lo) % inc):
            raise Undecidable("loop bound")
        return (hi - lo + inc - 1) // inc


# ---------------------------------------------------------------- sites
_SITE = re.compile(
    r"\b(?P<fn>lltt::record|load_replay_buf|TTI_REPLAY|TT_REPLAY)\s*(?:<(?P<targs>[^;{}()]*(?:\([^()]*\)[^;{}()]*)*)>)?\s*\("
)
_DEFINE = re.compile(
    r"(?m)^[ \t]*#[ \t]*define[ \t]+(\w+)(\([^)]*\))?[ \t]*((?:[^\n]*\\\n)*[^\n]*)"
)


def file_macros(code):
    return {
        m.group(1): m.group(3).replace("\\\n", " ")
        for m in _DEFINE.finditer(code)
        if not m.group(1).startswith(("TT_OP_", "TTI_", "TT_"))
    }


def _path_to(nodes, pos):
    """Chain of (sibling list, index) from the outermost block down to the statement at pos."""
    for idx, n in enumerate(nodes):
        if n.start <= pos < n.end:
            chain = [(nodes, idx)]
            inner = None
            if n.kind == "block":
                inner = n.kids
            elif n.kind == "if":
                inner = [n.then] if n.then.start <= pos < n.then.end else None
                if inner is None:
                    raise Undecidable("branch")
            elif n.kind == "for":
                raise Undecidable("recording inside a loop")
            if inner is not None:
                sub = _path_to(inner, pos)
                if sub:
                    return chain + sub
            return chain
    return None


def check_site(kind, n_len, body_code, site_pos, lambda_span, counter):
    """Return None when consistent, ('short'|'long', counted) on a mismatch."""
    if n_len == 0:
        raise Undecidable("zero length")
    if kind == "lambda":
        a, b = lambda_span
        got = counter.seq(parse_block(body_code, a, b))
        if got != n_len:
            return ("short" if got < n_len else "long", got)
        return None
    nodes = parse_block(body_code, 0, len(body_code))
    chain = _path_to(nodes, site_pos)
    if not chain:
        raise Undecidable("unparsed")
    got = 0
    # Walk outward: rest of the innermost block, then each enclosing block after its statement.
    for sibs, idx in reversed(chain):
        for n in sibs[idx + 1 :]:
            got += counter.node(n)
            if got >= n_len:
                return None
    return ("short", got)


def scan(path):
    """Yield ('finding', line, msg, src_line) or ('undecidable', line, reason, '') per site."""
    src = open(path, errors="ignore").read()
    if not re.search(r"lltt::record|load_replay_buf|REPLAY", src):
        return
    code = _blank_noncode(src)
    lines = src.split("\n")
    arch = arch_of(path)
    macros = file_macros(code)
    # File scope = everything outside function bodies.
    bodies = list(functions(src))
    line_starts = [0]
    for ln in code.split("\n"):
        line_starts.append(line_starts[-1] + len(ln) + 1)
    spans = []
    for brace_line, end_line, body, _g in bodies:
        s = code.find(body, line_starts[brace_line])
        if s >= 0:
            spans.append((s, s + len(body)))
    file_scope = code
    for s, e in sorted(spans, reverse=True):
        file_scope = file_scope[:s] + " " * (e - s) + file_scope[e:]

    for s, e in spans:
        body = code[s:e]
        inner = body[1:-1]  # strip the function's own braces
        off = s + 1
        for m in _SITE.finditer(inner):
            line = code.count("\n", 0, off + m.start()) + 1
            src_line = lines[line - 1].strip()
            try:
                yield _site(m, inner, arch, macros, file_scope, line, src_line)
            except Undecidable as u:
                yield ("undecidable", line, str(u), src_line)


def _site(m, inner, arch, macros, file_scope, line, src_line):
    fn, targs = m.group("fn"), [a for a in split_args(m.group("targs") or "")]
    p = m.end() - 1
    q = _match(inner, p, "(", ")")
    args = split_args(inner[p + 1 : q - 1])
    scopes = [inner[: m.start()], file_scope]
    counter = Counter(arch, macros, scopes)
    if fn in ("TTI_REPLAY", "TT_REPLAY"):
        load = {6: args[5:6], 4: args[3:4]}.get(len(args), [None])[0]
        if load is None:
            raise Undecidable("unparsed")
        if evaluate(load, scopes) != 1:
            return ("playback", line, "", src_line)
        kind, n_expr, lam = "record", args[1], None
    elif fn == "lltt::record":
        kind, n_expr, lam = "record", args[1] if len(args) == 2 else None, None
    else:
        kind = "lambda"
        if len(args) == 1 and len(targs) >= 2:
            n_expr, lam = targs[1], args[0]  # Quasar template form
        elif len(args) == 6:
            n_expr, lam = args[1], args[5]  # Quasar runtime form
        elif len(args) >= 3:
            n_expr, lam = args[1], args[2]  # WH / BH form
        else:
            raise Undecidable("unparsed")
    if n_expr is None:
        raise Undecidable("unparsed")
    n_len = evaluate(n_expr, scopes)
    span = None
    if kind == "lambda":
        lm = re.fullmatch(
            r"\s*\[[^\]]*\]\s*(?:\(\s*\))?\s*(?:mutable\s*)?(\{.*\})\s*", lam, re.S
        )
        if not lm:
            raise Undecidable("callable is not an inline lambda")
        a = inner.find(lm.group(1), p)
        span = (a + 1, a + len(lm.group(1)) - 1)
    stmt_end = inner.find(";", q)
    res = check_site(
        kind, n_len, inner, stmt_end if kind == "record" else m.start(), span, counter
    )
    if res is None:
        return ("ok", line, "", src_line)
    what, got = res
    if what == "short":
        msg = f"records {n_len} instructions but only {got} follow" + (
            " in the callable" if kind == "lambda" else " before the function ends"
        )
    else:
        msg = f"records {n_len} instructions but the callable emits {got}"
    return ("finding", line, msg, src_line)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "files", nargs="*", help="headers to check (default: every LLK header)"
    )
    ap.add_argument("--baseline")
    ap.add_argument("--stats", action="store_true", help="print site coverage")
    args = ap.parse_args()

    # A change to the checker itself moves the verdict everywhere, so it rescans the whole tree.
    whole = not args.files or any(
        os.path.basename(f) == os.path.basename(__file__) for f in args.files
    )
    files = (args.files if not whole else None) or sorted(
        glob.glob(os.path.join(LLK, "tt_llk_*", "**", "*.h"), recursive=True)
        + glob.glob(os.path.join(METAL_HW, "*", "metal", "**", "*.h"), recursive=True)
    )
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

    stats, used, n = {}, set(), 0
    for path in files:
        if not path.endswith((".h", ".hpp")):
            continue
        for kind, line, msg, src_line in scan(path):
            key = kind if kind != "undecidable" else f"undecidable: {msg}"
            stats[key] = stats.get(key, 0) + 1
            if kind != "finding":
                continue
            bkey = f"{os.path.relpath(path)}:{src_line}"
            if bkey in accepted:
                used.add(bkey)
                continue
            n += 1
            print(f"{path}:{line}: replay {msg}")
            print(f"  at: {src_line[:96]}")
            print("  fix: make the recorded length equal the instructions recorded.\n")
    if args.stats:
        for k in sorted(stats):
            print(f"{stats[k]:5d}  {k}")
    if args.baseline and whole:
        for stale in sorted(accepted - used):
            print(f"note: baseline entry no longer matches: {stale}")
    if n:
        print(f"{n} replay recording(s) whose length does not match what they record.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
