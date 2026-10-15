#!/usr/bin/env python3
"""Flag a RISC-blocking drain (`tensix_sync()` / `mop_sync()`) that orders nothing.

A drain blocks the RISC until the Tensix thread (or its MOP expander) is idle. That only matters
when the RISC itself is about to do something that must see the drained state: a mailbox or
semaphore access, a configuration-register read or write through the memory map, a GPR read, a
volatile load or store, a store through a pointer. Tensix instructions pushed after a drain are
ordered behind the earlier ones by the thread anyway, so a drain followed only by Tensix pushes and
plain code orders nothing, and costs a full pipeline drain every time it runs.

Two findings, per drain, within one function body:

  * REDUNDANT (certain): on every path from the drain, the next thing that could need it is another
    drain that covers it: `mop_sync()` before `tensix_sync()` (thread idle implies the expander is
    idle), or a `tensix_sync()` before another `tensix_sync()`, with only Tensix pushes or plain
    code between them.
  * ADVISORY: nothing RISC-side follows the drain in this function on any path, only Tensix pushes
    and plain code up to the function's end. The drain may still serve a caller; if so the caller
    should drain itself, or the site should be baselined with the reason.

The whole repository is in scope: the Blackhole and Wormhole LLK trees, the per-architecture
`hw/ckernels` wrappers, the compute API, firmware, and every compute kernel under `ttnn`, `models`
and `tests`. A compute kernel is compiled once per TRISC, and `UNPACK(...)` / `MATH(...)` /
`PACK(...)`, `#if defined(TRISC_*)`, `UCK_CHLKC_*` and `LLK_TRISC_*` scope a statement to one
thread, so the tail after a drain is classified per thread the drain runs on, and per architecture
(a kernel's helpers resolve to that architecture's ckernels and LLK trees). A finding is reported
only when every thread, architecture and preprocessor path the drain is compiled on agrees.

A drain inside a loop also meets the loop body before it on the next iteration, and a drain inside
a conditional block, or after a conditional `break`/`continue`/`return`, covers nothing. Anything
the checker cannot classify -- a call to a function it cannot find, a preprocessor condition with
too many unknowns, a statement it does not recognise -- ends the scan with no finding, so a false
flag needs a misclassification, not a gap in the model. A helper called after the drain is
classified by its own body, found by name in the architecture's trees, the compute API, and the
file's own directory.

Quasar is skipped: its drains use a different mechanism and are not modelled here.

Output is advisory. The hook always exits 0; findings are printed so they are seen, and the
baseline lists sites that were reviewed and kept with their reason.
"""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_mutex_balance import _signature, functions  # noqa: E402

DRAIN = re.compile(r"\b(?:ckernel::)?(tensix_sync|mop_sync)\s*\(\s*\)")

# A statement that pushes a Tensix instruction (or is plain code) and touches nothing the RISC
# must wait for. `TT_` covers the runtime-encoded pushes, `TTI_` the immediates.
TENSIX_PUSH = re.compile(
    r"^\s*(?:ckernel::)?(?:TTI?_[A-Z0-9_]+\b|cfg_reg_rmw_tensix\b|lltt::|"
    r"[A-Za-z_][\w:]*\.run\s*\(|t6_mutex_(?:acquire|release)\b)"
)
# A statement the RISC performs against state a drain can order. The names are the LLK's own
# memory-mapped helpers; the generic forms catch raw volatile accesses and pointer stores.
RISC_ACCESS = re.compile(
    r"\b(?:mailbox_(?:write|read|not_empty|fence)|semaphore_(?:read|post|get)|t6_semaphore_\w+|"
    r"reg_(?:read|write)|cfg_(?:read|write|rmw|rmw_gpr)\b|get_cfg(?:16)?_pointer|"
    r"load_blocking|store_blocking|sync_regfile_write|read_(?:pack|unpack)_\w+|dbg_\w+|"
    r"memory_\w+|regfile|cfg_regs|mop_cfg|pc_buf_base|mailbox_base|dest_array|"
    r"RISCV_DEBUG_REG_\w+|RISCV_DEST_START_ADDR|TENSIX_CFG_BASE|TENSIX_MOP_CFG_BASE|"
    r"csr_(?:read|write)|disable_gathering|enable_gathering|volatile|reinterpret_cast|"
    r"__atomic_\w+|invalidate_l1_cache)\b"
    r"|(?<![\w)\]])\*\s*[A-Za-z_(]"  # a load or store through a pointer
    r"|\[[^\]]*\]"  # an indexed load or store: an array, a mapped register, a buffer
    r"|->"
)
# Plain code the drain cannot be for: control-flow heads, declarations and assignments of scalar
# locals and globals, asserts, RISC `nop`s, template helpers that only compute.
PLAIN = re.compile(
    r"^\s*(?:(?:if|else|for|while|do|switch|case|default|return|break|continue|constexpr|static_assert|"
    r"LLK_ASSERT|asm\s+volatile\s*\(\s*\"nop\"|using|typedef)\b|[A-Za-z_][\w:<>,\s*&]*\s+[A-Za-z_]\w*\s*(?:=|\(|;|\{)|"
    r"[A-Za-z_][\w:.]*\s*(?:[-+*/|&^]?=|\+\+|--)(?!=)|\{|\}|$)"
)
# A call to something named; resolved against the index when nothing above matched.
CALL = re.compile(r"\b(?:ckernel::|[a-z_]+::)*([A-Za-z_]\w*)\s*(?:<[^;{}()]*>)?\s*\(")
# `Type name(args);` constructs a local: the type is the call, the name is not.
DECL = re.compile(
    r"^\s*(?:const\s+)?(?!(?:return|else|case|throw|new|delete|typename|using|goto|co_return)\b)"
    r"[A-Za-z_][\w:]*(?:<[^;]*>)?\s*[*&]*\s+([A-Za-z_]\w*)\s*\("
)
CONTROL = {
    "if",
    "else",
    "for",
    "while",
    "do",
    "switch",
    "return",
    "sizeof",
    "static_assert",
    "LLK_ASSERT",
    "constexpr",
}
# A statement wrapped for one thread by the compute API's macros.
WRAPPER = re.compile(
    r"^\s*(?:ckernel::)?(UNPACK|MATH|PACK)\s*\(\s*(.*)\)\s*;?\s*$", re.S
)
JUMP = re.compile(r"\b(return|break|continue|goto)\b")
LOOP_HEAD = re.compile(r"^\s*(?:for|while|do)\b")
SWITCH_HEAD = re.compile(r"^\s*switch\b")
PP_LINE = re.compile(r"^\s*#\s*(\w+)\s*(.*)$", re.S)

THREADS = frozenset(("unpack", "math", "pack"))
ARCHES = ("wormhole_b0", "blackhole")
THREAD_DEFINE = {
    f"{p}{t.upper()}": t
    for p in ("TRISC_", "UCK_CHLKC_", "LLK_TRISC_")
    for t in THREADS
}
ARCH_DEFINE = {
    "ARCH_BLACKHOLE": "blackhole",
    "ARCH_WORMHOLE": "wormhole_b0",
    "ARCH_WORMHOLE_B0": "wormhole_b0",
}
TRUE_DEFINES = {"COMPILE_FOR_TRISC"}
FALSE_DEFINES = {
    "ARCH_QUASAR",
    "ARCH_GRAYSKULL",
    "COMPILE_FOR_BRISC",
    "COMPILE_FOR_NCRISC",
    "COMPILE_FOR_ERISC",
    "COMPILE_FOR_IDLE_ERISC",
    "COMPILE_FOR_AERISC",
    "COMPILE_FOR_DRISC",
    "COMPILE_FOR_DM",
    "COMPILE_FOR_DISPATCH_ENGINE",
}
MAX_PATH_VARS = 4  # preprocessor unknowns enumerated per function before giving up
MAX_CALL_DEPTH = 6

EXTS = (".h", ".hpp", ".cpp", ".cc")
INDEX_EXTS = (".h", ".hpp", ".inl")
SKIP_QUASAR = ("/tt_llk_quasar/", "/ckernels/quasar/", "/tt-2xx/")
SKIP_SILENT = ("/third_party/",)
LLK_TREE = {
    "blackhole": "tt_metal/tt-llk/tt_llk_blackhole",
    "wormhole_b0": "tt_metal/tt-llk/tt_llk_wormhole_b0",
}
CKERNELS = "tt_metal/hw/ckernels/{arch}"
SHARED_ROOTS = (
    "tt_metal/hw/inc",
    "ttnn/cpp/ttnn/kernel_lib",
    "ttnn/cpp/ttnn/kernel",
    "tt_metal/tt-llk/tests/helpers/include",
)
ARCH_OF_DIR = (
    ("/tt_llk_blackhole/", "blackhole"),
    ("/tt_llk_wormhole_b0/", "wormhole_b0"),
    ("/ckernels/blackhole/", "blackhole"),
    ("/ckernels/wormhole_b0/", "wormhole_b0"),
)
FILE_THREAD = re.compile(r"^(?:llk_|c)(?:io_)?(unpack|math|pack)")
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))


def _function_name(lines, sig_line, brace_line):
    """The identifier before the parameter list of the declarator ending at brace_line."""
    head = " ".join(lines[sig_line : brace_line + 1])
    head = re.sub(r"\{.*$", "", head)
    head = re.sub(
        r"\)\s*(?:noexcept\b[^:{]*)?:(?!:).*$", ")", head
    )  # a ctor's init list
    names = re.findall(r"([A-Za-z_]\w*)\s*(?:<[^()]*>)?\s*\(", head)
    return names[-1] if names else ""


# ---------------------------------------------------------------------------------------------
# A function body as a flat list of items: statements, braces and preprocessor directives.


class Item:
    __slots__ = (
        "off",
        "text",
        "kind",
        "threads",
        "pp",
        "depth",
        "mate",
        "block",
        "encl",
    )

    def __init__(self, off, text, kind):
        self.off, self.text, self.kind = off, text, kind
        self.threads = THREADS
        self.pp = ()
        self.depth = 0
        self.mate = None
        self.block = None
        self.encl = ()


class Group:
    """One `#if` ... `#endif`: a list of branch expressions (None for `#else`)."""

    __slots__ = ("branches",)

    def __init__(self):
        self.branches = []


def _split(inner):
    """Items in order: one per `;`, braces as their own items, directives as their own items."""
    items, cur, start, paren, init, i, n = [], [], 0, 0, 0, 0, len(inner)
    line_start = True

    def flush():
        # The offset is that of the first character: it is what the directives order against.
        text = "".join(cur)
        lead = len(text) - len(text.lstrip())
        if text.strip():
            items.append(Item(start + lead, text[lead:], "stmt"))

    while i < n:
        if line_start:
            j = i
            while j < n and inner[j] in " \t":
                j += 1
            if j < n and inner[j] == "#":
                k = j
                while True:
                    e = inner.find("\n", k)
                    e = n if e < 0 else e
                    if e < n and inner[e - 1] == "\\":
                        k = e + 1
                        continue
                    break
                items.append(Item(j, inner[j:e], "pp"))
                cur.append("".join(c if c == "\n" else " " for c in inner[i:e]))
                i = e
                line_start = False
                continue
        c = inner[i]
        line_start = c == "\n"
        if c in "{}" and paren == 0:
            prev = "".join(cur).rstrip()
            if c == "{" and (
                init
                or re.search(r"(?:[=(,\[]|\breturn)\s*$", prev)
                or (
                    re.search(r"\w\s*$", prev)
                    and not re.match(
                        r"\s*(?:else|do|try|struct|class|union|enum|namespace|extern)\b",
                        prev,
                    )
                )
            ):
                init += 1  # a brace-init list, part of the statement
                cur.append(c)
            elif c == "}" and init:
                init -= 1
                cur.append(c)
            else:
                flush()
                cur, start = [], i + 1
                items.append(Item(i, c, "open" if c == "{" else "close"))
        elif c == ";" and paren == 0 and not init:
            cur.append(c)
            flush()
            cur, start = [], i + 1
        else:
            if c == "(":
                paren += 1
            elif c == ")":
                paren = max(paren - 1, 0)
            cur.append(c)
        i += 1
    flush()
    items.sort(key=lambda it: it.off)
    return items


class Unparsable(Exception):
    pass


_COND_TOKEN = re.compile(
    r"\s*(?:defined\s*\(\s*(\w+)\s*\)|defined\s+(\w+)|(&&|\|\||!(?!=)|\(|\))|([A-Za-z_]\w*)|(\d+))"
)


def _parse_cond(text):
    """A tree for an `#if` condition: ('or'|'and', a, b), ('not', a), ('def', name), ('num', n).

    Only the boolean structure over `defined(X)`, identifiers and integers is understood; a
    comparison or arithmetic raises Unparsable and the branch becomes one undecidable condition.
    """
    toks, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = _COND_TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise Unparsable(text)
        pos = m.end()
        if m.group(1) or m.group(2):
            toks.append(("def", m.group(1) or m.group(2)))
        elif m.group(3):
            toks.append(("op", m.group(3)))
        elif m.group(4):
            toks.append(("def", m.group(4)))
        else:
            toks.append(("num", int(m.group(5))))
    at = [0]

    def peek():
        return toks[at[0]] if at[0] < len(toks) else (None, None)

    def take():
        t = peek()
        at[0] += 1
        return t

    def p_or():
        a = p_and()
        while peek() == ("op", "||"):
            take()
            a = ("or", a, p_and())
        return a

    def p_and():
        a = p_not()
        while peek() == ("op", "&&"):
            take()
            a = ("and", a, p_not())
        return a

    def p_not():
        if peek() == ("op", "!"):
            take()
            return ("not", p_not())
        k, v = take()
        if (k, v) == ("op", "("):
            a = p_or()
            if take() != ("op", ")"):
                raise Unparsable(text)
            return a
        if k in ("def", "num"):
            return (k, v)
        raise Unparsable(text)

    tree = p_or()
    if at[0] != len(toks):
        raise Unparsable(text)
    return tree


def _expr(kind, rest):
    """The condition tree of an `#if`-family directive, or '?' when it cannot be parsed."""
    rest = rest.strip()
    if kind in ("ifdef", "ifndef"):
        m = re.match(r"\w+", rest)
        if not m:
            return "?"
        return ("def", m.group()) if kind == "ifdef" else ("not", ("def", m.group()))
    try:
        return _parse_cond(rest)
    except Unparsable:
        return "?"


def _eval(node, D):
    k = node[0]
    if k == "or":
        return _eval(node[1], D) or _eval(node[2], D)
    if k == "and":
        return _eval(node[1], D) and _eval(node[2], D)
    if k == "not":
        return not _eval(node[1], D)
    if k == "def":
        return D(node[1])
    return node[1] != 0


def build(inner):
    """Items of a body with brace structure and preprocessor nesting resolved."""
    items = _split(inner)
    stack, groups, pp = [], [], []
    prev = None
    for i, it in enumerate(items):
        if it.kind == "pp":
            m = PP_LINE.match(it.text)
            kind, rest = (m.group(1), m.group(2)) if m else ("", "")
            if kind in ("if", "ifdef", "ifndef"):
                g = Group()
                g.branches.append(_expr(kind, rest))
                groups.append(g)
                pp.append((g, 0))
            elif kind == "elif" and pp:
                g, _ = pp.pop()
                g.branches.append(_expr("if", rest))
                pp.append((g, len(g.branches) - 1))
            elif kind == "else" and pp:
                g, _ = pp.pop()
                g.branches.append(None)
                pp.append((g, len(g.branches) - 1))
            elif kind == "endif" and pp:
                pp.pop()
            continue
        it.pp = tuple(pp)
        it.encl = tuple(stack)
        it.depth = len(stack)
        if it.kind == "open":
            if (
                prev is not None
                and prev.kind == "stmt"
                and not prev.text.rstrip().endswith(";")
            ):
                head = prev.text
                it.block = (
                    "loop"
                    if LOOP_HEAD.match(head)
                    else "switch" if SWITCH_HEAD.match(head) else "cond"
                )
            else:
                it.block = "plain"
            stack.append(i)
        elif it.kind == "close":
            if stack:
                o = stack.pop()
                it.mate, items[o].mate, it.block = o, i, items[o].block
                it.encl, it.depth = items[o].encl, items[o].depth
        else:
            m = WRAPPER.match(it.text)
            if m:
                it.threads = frozenset((m.group(1).lower(),))
        prev = it
    return items, groups


# ---------------------------------------------------------------------------------------------
# Preprocessor paths: which branch of each group is live for a thread, an architecture and an
# assignment of the conditions the checker cannot decide.


class NeedVar(Exception):
    pass


def _define(thread, arch, assign):
    def D(name):
        if name in THREAD_DEFINE:
            return thread == THREAD_DEFINE[name]
        if name in ARCH_DEFINE:
            return arch == ARCH_DEFINE[name]
        if name in TRUE_DEFINES:
            return True
        if name in FALSE_DEFINES:
            return False
        if name in assign:
            return assign[name]
        raise NeedVar(name)

    return D


def live_branch(group, thread, arch, assign):
    """Index of the branch that is compiled, or -1."""
    D = _define(thread, arch, assign)
    for b, cond in enumerate(group.branches):
        if cond is None:
            return b
        if cond == "?":
            name = f"?{id(group)}:{b}"
            if name not in assign:
                raise NeedVar(name)
            if assign[name]:
                return b
            continue
        if _eval(cond, D):
            return b
    return -1


def path_vars(groups, thread, arch):
    """The undecidable condition names of these groups, or None when there are too many.

    Every assignment of the names found so far is tried, since a true value can expose a further
    name behind a short-circuited `&&` / `||`.
    """
    names = set()
    while True:
        try:
            for assign in assignments(sorted(names)):
                for g in groups:
                    live_branch(g, thread, arch, assign)
            return sorted(names)
        except NeedVar as e:
            names.add(e.args[0])
            if len(names) > MAX_PATH_VARS:
                return None


def assignments(names):
    for values in itertools.product((False, True), repeat=len(names)):
        yield dict(zip(names, values))


def is_live(item, thread, arch, assign):
    return thread in item.threads and all(
        live_branch(g, thread, arch, assign) == b for g, b in item.pp
    )


# ---------------------------------------------------------------------------------------------
# The index of function bodies a called helper is classified from.

_TEXT, _PARSED = {}, {}


def _read(path):
    if path not in _TEXT:
        try:
            _TEXT[path] = open(path, errors="ignore").read()
        except OSError:
            _TEXT[path] = ""
    return _TEXT[path]


def file_threads(path):
    m = FILE_THREAD.match(os.path.basename(path))
    return frozenset((m.group(1),)) if m else THREADS


def _parse(path):
    if path not in _PARSED:
        src = _read(path)
        lines = src.split("\n")
        bodies = {}
        for brace_line, _, body, _ in functions(src):
            name = _function_name(lines, _signature(lines, brace_line), brace_line)
            if name:
                bodies.setdefault(name, []).append(build(body[1:-1]))
        _PARSED[path] = (bodies, file_threads(path))
    return _PARSED[path]


class Index:
    """Function bodies under a set of roots, by name, parsed on first use."""

    _cache = {}

    def __init__(self, roots):
        self.paths = []
        seen = set()
        for root in roots:
            if os.path.isfile(root):
                cands = [root]
            elif os.path.isdir(root):
                cands = [
                    os.path.join(d, f)
                    for d, _, fs in os.walk(root)
                    for f in fs
                    if f.endswith(INDEX_EXTS + EXTS)
                ]
            else:
                continue
            for p in cands:
                norm = os.path.abspath(p).replace(os.sep, "/")
                if any(s in norm for s in SKIP_QUASAR + SKIP_SILENT) or norm in seen:
                    continue
                seen.add(norm)
                self.paths.append(p)
        self.memo = {}

    @classmethod
    def get(cls, roots):
        key = tuple(os.path.abspath(r) for r in roots)
        if key not in cls._cache:
            cls._cache[key] = cls(roots)
        return cls._cache[key]

    def bodies(self, name):
        out = []
        for p in self.paths:
            if name in _read(p):
                bodies, threads = _parse(p)
                out += [(b, threads) for b in bodies.get(name, [])]
        return out

    def classify_call(self, name, thread, arch, depth=0):
        """'tensix', 'risc', 'drain:<name>' or 'unknown' for a call to `name` on this thread."""
        key = (name, thread, arch)
        if key in self.memo:
            return self.memo[key]
        self.memo[key] = "unknown"  # recursion guard
        bodies = self.bodies(name)
        if not bodies or depth > MAX_CALL_DEPTH:
            return "unknown"
        verdicts = []
        for (items, groups), threads in bodies:
            if thread not in threads:
                verdicts.append("end")  # not compiled for this thread: nothing
                continue
            names = path_vars(groups, thread, arch)
            if names is None:
                verdicts.append("unknown")
                continue
            for assign in assignments(names):
                verdicts.append(
                    walk(items, -1, thread, arch, assign, self, depth + 1, None)
                )
        if "risc" in verdicts:
            v = "risc"
        elif "unknown" in verdicts:
            v = "unknown"
        elif verdicts and all(x.startswith("drain:") for x in verdicts):
            v = (
                "drain:tensix_sync"
                if all(x == "drain:tensix_sync" for x in verdicts)
                else "drain:mop_sync"
            )
        else:
            v = "tensix"
        self.memo[key] = v
        return v


# ---------------------------------------------------------------------------------------------
# Classification.


def classify_statement(stmt, thread, arch, index, depth=0):
    """'tensix', 'risc', 'drain:<name>', 'plain' or 'unknown' for one statement."""
    s = stmt.strip()
    if not s or s in "{}":
        return "plain"
    m = WRAPPER.match(s)
    if m:
        s = m.group(2).strip()
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1].strip()
        if not s:
            return "plain"
    m = DRAIN.search(s)
    if m:
        if re.search(r"\b(?:if|while|for)\b", s[: m.start()]):
            return "plain"  # a one-line conditional drain: covers nothing, orders nothing here
        return "drain:" + m.group(1)
    if TENSIX_PUSH.match(s):
        return "tensix"
    if RISC_ACCESS.search(s):
        return "risc"
    names = CALL.findall(s)
    d = DECL.match(s)
    if d:
        names = [n for n in names if n != d.group(1)]
        names.append(
            re.match(r"\s*(?:const\s+)?([A-Za-z_][\w:]*)", s).group(1).split("::")[-1]
        )
    worst = None
    for name in names:
        if name in CONTROL or name.isupper():
            continue
        v = index.classify_call(name, thread, arch, depth)
        if v == "risc":
            return v
        if v == "unknown" or (v.startswith("drain:") and worst != "unknown"):
            worst = v
        elif worst is None:
            worst = "tensix"
    if worst:
        return worst
    if PLAIN.match(s):
        return "plain"
    return "unknown"


def walk(items, start, thread, arch, assign, index, depth, dkind):
    """Verdict for every path from items[start] on: 'risc', 'unknown', 'drain:<name>' or 'end'.

    start is the drain's item, or -1 for a helper body. Items on a dead thread or preprocessor
    branch are skipped. A drain covers only when it is reached on every path: it must sit in no
    block beyond the drain's own ancestors, after no conditional jump, and not in the replay of a
    loop body (the part before the drain that the next iteration runs). A `break` leaves the
    innermost loop (or switch), a `continue` or the end of the loop body replays it and then
    exits, a `return` ends the path.
    """
    ancestors = set(items[start].encl) if start >= 0 else set()
    live = [it.kind == "pp" or is_live(it, thread, arch, assign) for it in items]
    ranges = [(start + 1, len(items), False)]
    diverged = False
    while ranges:
        lo, hi, replay = ranges.pop(0)
        i = lo
        while i < hi:
            it = items[i]
            if it.kind == "pp" or not live[i]:
                i += 1
                continue
            uncond = all(o in ancestors for o in it.encl)
            if it.kind == "open":
                i += 1
                continue
            if it.kind == "close":
                if it.mate in ancestors and it.block == "loop" and not replay:
                    ranges.insert(0, (i + 1, hi, False))
                    ranges.insert(0, (it.mate + 1, start, True))
                    break
                i += 1
                continue
            v = classify_statement(it.text, thread, arch, index, depth)
            if v in ("risc", "unknown"):
                return v
            if v.startswith("drain:"):
                name = v.split(":", 1)[1]
                covers = name == "tensix_sync" or dkind == "mop_sync"
                if uncond and not replay and not diverged and covers:
                    return v
            jm = JUMP.search(it.text)
            if (
                jm
                and not re.match(
                    r"^\s*(?:case\b|default\b)?[^;]*\b(?:if|else|for|while)\b",
                    it.text[: jm.start()],
                )
                and uncond
            ):
                word = jm.group(1)
                if replay:
                    break  # the next iteration leaves the body before reaching the drain again
                if word == "return":
                    break  # this path ends; another range may follow for an outer loop exit
                if word == "goto":
                    return "unknown"
                target = None
                for o in reversed(it.encl):
                    if items[o].block == "loop" or (
                        word == "break" and items[o].block == "switch"
                    ):
                        target = o
                        break
                if target is None:
                    i += 1
                    continue
                c = items[target].mate
                if word == "break":
                    if items[target].block == "loop":
                        ranges.insert(0, (c + 1, hi, False))
                        break
                    i = c + 1
                    continue
                i = c  # continue: handled as the loop's close
                continue
            if jm:
                diverged = True
            i += 1
    return "end"


# ---------------------------------------------------------------------------------------------
# Per-file scan.


def is_quasar(path):
    norm = os.path.abspath(path).replace(os.sep, "/")
    return any(s in norm for s in SKIP_QUASAR)


def is_skipped_silently(path):
    norm = os.path.abspath(path).replace(os.sep, "/")
    return any(s in norm for s in SKIP_SILENT)


def context_for(path, repo):
    """(architectures, threads, roots-by-arch) for a file: what its tails are evaluated against."""
    norm = os.path.abspath(path).replace(os.sep, "/")
    own = os.path.dirname(os.path.abspath(path))
    arch = None
    for key, a in ARCH_OF_DIR:
        i = norm.find(key)
        if i >= 0:
            arch = a
            if key.startswith("/tt_llk_"):
                tree = norm[: i + len(key) - 1]
                return (a,), file_threads(path), {a: [tree, own]}
            break
    arches = (arch,) if arch else ARCHES
    roots = {
        a: [
            os.path.join(repo, LLK_TREE[a]),
            os.path.join(repo, CKERNELS.format(arch=a)),
            *(os.path.join(repo, r) for r in SHARED_ROOTS),
            own,
        ]
        for a in arches
    }
    return arches, file_threads(path), roots


def scan(path, repo):
    text = _read(path)
    if not DRAIN.search(text):
        return
    src = re.sub(r"\bMAIN\s*(?=\{)", "MAIN() ", text)  # a compute kernel's entry point
    lines = src.split("\n")
    orig = text.split("\n")
    arches, threads, roots = context_for(path, repo)
    for brace_line, _, body, _ in functions(src):
        sig = _signature(lines, brace_line)
        head = orig[sig].strip() if sig < len(orig) else lines[sig].strip()
        if re.search(r"\b(?:tensix_sync|mop_sync)\s*\(\s*\)\s*$", lines[sig].strip()):
            continue  # the drain definitions themselves
        inner = body[1:-1]
        items, groups = build(inner)
        stmts = [it for it in items if it.kind == "stmt"]
        if (
            len(stmts) == 1
            and DRAIN.search(stmts[0].text)
            and not any(it.kind in ("open", "close") for it in items)
        ):
            continue  # a named wrapper around the drain: callers resolve it as a drain
        for i, it in enumerate(items):
            if it.kind != "stmt":
                continue
            m = DRAIN.search(it.text)
            if not m or re.search(r"\b(?:if|while|for)\b", it.text[: m.start()]):
                continue
            drain = m.group(1)
            line = brace_line + 1 + inner[: it.off + m.start()].count("\n")
            verdicts = []
            for arch in arches:
                index = Index.get(roots[arch])
                for thread in sorted(threads & it.threads):
                    names = path_vars(groups, thread, arch)
                    if names is None:
                        verdicts.append("unknown")
                        continue
                    for assign in assignments(names):
                        if not is_live(it, thread, arch, assign):
                            continue
                        verdicts.append(
                            walk(items, i, thread, arch, assign, index, 0, drain)
                        )
            if not verdicts or any(v in ("risc", "unknown") for v in verdicts):
                continue
            covering = {v.split(":", 1)[1] for v in verdicts if v.startswith("drain:")}
            if len(covering) == 1 and all(v.startswith("drain:") for v in verdicts):
                nxt = covering.pop()
                yield line, "REDUNDANT", drain, f"only Tensix pushes or plain code before the {nxt}() that covers it", head
            else:
                yield line, "ADVISORY", drain, "nothing RISC-side follows in this function; if a caller needs the drain it should drain itself", head


def load_baseline(path):
    accepted = {}
    if not path:
        return accepted
    try:
        for l in open(path):
            l = l.rstrip("\n")
            if not l.strip() or l.lstrip().startswith("#"):
                continue
            key, _, reason = l.partition("#")
            accepted[key.strip()] = reason.strip()
    except FileNotFoundError:
        pass
    return accepted


# The programs this script may start: a process is started only for a name in this table, with
# its arguments as a list and no shell.
PROGRAMS = {"git": "git"}


def _run(program, args):
    return subprocess.run([PROGRAMS[program], *args], capture_output=True, text=True)


def all_drain_files(repo):
    """Every source file in the repository that calls a drain."""
    try:
        res = _run(
            "git",
            [
                "-C",
                repo,
                "grep",
                "-lE",
                r"\b(tensix_sync|mop_sync)\s*\(",
                "--",
                *[f"*{e}" for e in EXTS],
            ],
        )
        if res.returncode == 0:
            return [os.path.join(repo, l) for l in res.stdout.splitlines() if l]
    except OSError:
        pass
    out = []
    for d, dirs, fs in os.walk(repo):
        dirs[:] = [x for x in dirs if x not in (".git", "build", "build_Release")]
        for f in fs:
            if f.endswith(EXTS):
                p = os.path.join(d, f)
                if DRAIN.search(_read(p)):
                    out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("files", nargs="*", help="source files to check")
    ap.add_argument(
        "--tree", action="append", default=[], help="scan a whole directory"
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="scan every file in the repository that calls a drain",
    )
    ap.add_argument(
        "--repo", default=REPO, help="repository root the helper trees are found under"
    )
    ap.add_argument(
        "--baseline",
        help="reviewed sites to keep, one `path:line:function # reason` per line",
    )
    ap.add_argument("--json", action="store_true", help="one JSON object per finding")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 on any REDUNDANT finding (never used by the hook)",
    )
    args = ap.parse_args()

    files = [f for f in args.files if f.endswith(EXTS)]
    # A change to the checker or the baseline moves the answer on every file: scan them all.
    if args.all or any(
        os.path.basename(f)
        in ("check_redundant_drain.py", "redundant_drain_baseline.txt")
        for f in args.files
    ):
        files += all_drain_files(args.repo)
    for t in args.tree:
        for dirpath, _, names in os.walk(t):
            files += [os.path.join(dirpath, f) for f in names if f.endswith(EXTS)]
    accepted = load_baseline(args.baseline)
    findings, skipped = [], 0
    for path in sorted(set(os.path.abspath(f) for f in files)):
        if is_skipped_silently(path):
            continue
        if is_quasar(path):
            skipped += 1
            continue
        rel = os.path.relpath(path).replace(os.sep, "/")
        for line, kind, drain, reason, head in scan(path, args.repo):
            key = f"{rel}:{head[:60]}"
            if key in accepted:
                continue
            findings.append((rel, line, kind, drain, reason, head))

    if args.json:
        for rel, line, kind, drain, reason, head in findings:
            print(
                json.dumps(
                    {
                        "file": rel,
                        "line": line,
                        "kind": kind,
                        "drain": drain,
                        "reason": reason,
                        "function": head,
                    }
                )
            )
    else:
        for rel, line, kind, drain, reason, head in findings:
            print(f"{rel}:{line}: {kind}: {drain}() {reason}\n    in {head[:100]}")
        if findings:
            n_red = sum(1 for f in findings if f[2] == "REDUNDANT")
            print(
                f"llk-redundant-drain: {n_red} redundant, {len(findings) - n_red} advisory drain site(s); see the docstring for what each means"
            )
        if skipped:
            print(
                f"llk-redundant-drain: {skipped} Quasar file(s) not checked (different drain mechanism)"
            )
    if args.strict and any(f[2] == "REDUNDANT" for f in findings):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
