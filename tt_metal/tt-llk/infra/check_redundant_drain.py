#!/usr/bin/env python3
"""Flag a RISC-blocking drain (`tensix_sync()` / `mop_sync()`) that orders nothing.

A drain blocks the RISC until the Tensix thread (or its MOP expander) is idle. That only matters
when the RISC itself is about to do something that must see the drained state: a mailbox or
semaphore access, a configuration-register read or write through the memory map, a GPR read, a
volatile load or store, a store through a pointer. Tensix instructions pushed after a drain are
ordered behind the earlier ones by the thread anyway, so a drain followed only by Tensix pushes and
plain code orders nothing, and costs a full pipeline drain every time it runs.

Two findings, per drain, within one function body:

  * REDUNDANT (certain): the next thing that could need the drain is another drain that covers it:
    `mop_sync()` directly before `tensix_sync()` (thread idle implies the expander is idle), or a
    `tensix_sync()` directly before another `tensix_sync()`, with only Tensix pushes or plain code
    between them.
  * ADVISORY: nothing RISC-side follows the drain in this function at all, only Tensix pushes and
    plain code up to the function's end. The drain may still serve a caller; if so the caller should
    drain itself, or the site should be baselined with the reason.

Anything the checker cannot classify -- a call to a function it cannot find in the tree, a loop
condition it does not understand -- ends the scan with no finding, so a false flag needs a
misclassification, not a gap in the model. A helper called after the drain is classified by its own
body, found by name in the same architecture tree.

Quasar is skipped: its drains use a different mechanism and are not modelled here.

Output is advisory. The hook always exits 0; findings are printed so they are seen, and the
baseline lists sites that were reviewed and kept with their reason.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    r"csr_(?:read|write)|disable_gathering|enable_gathering|volatile|reinterpret_cast)\b"
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
# A call to something named; resolved against the tree when nothing above matched.
CALL = re.compile(r"\b(?:ckernel::|[a-z_]+::)*([A-Za-z_]\w*)\s*(?:<[^;{}()]*>)?\s*\(")
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

ARCH_DIRS = ("tt_llk_blackhole", "tt_llk_wormhole_b0")
SKIPPED_ARCH = "tt_llk_quasar"


def _function_name(lines, sig_line, brace_line):
    """The identifier before the parameter list of the declarator ending at brace_line."""
    head = " ".join(lines[sig_line : brace_line + 1])
    head = re.sub(r"\{.*$", "", head)
    names = re.findall(r"([A-Za-z_]\w*)\s*(?:<[^()]*>)?\s*\(", head)
    return names[-1] if names else ""


def _statements(text):
    """Split body text into (offset, statement): one per `;`, with `{` and `}` as their own items."""
    out, cur, start, depth = [], [], 0, 0
    for i, c in enumerate(text):
        if c in "{}":
            if "".join(cur).strip():
                out.append((start, "".join(cur)))
            cur, start = [], i + 1
            out.append((i, c))
        elif c == ";" and depth == 0:
            cur.append(c)
            out.append((start, "".join(cur)))
            cur, start = [], i + 1
        else:
            if c == "(":
                depth += 1
            elif c == ")":
                depth = max(depth - 1, 0)
            cur.append(c)
    if "".join(cur).strip():
        out.append((start, "".join(cur)))
    return out


class Tree:
    """Function bodies of one architecture tree, by name, for classifying called helpers."""

    def __init__(self, root):
        self.bodies = {}
        self.memo = {}
        if not root or not os.path.isdir(root):
            return
        for dirpath, _, files in os.walk(root):
            for f in files:
                if not f.endswith((".h", ".hpp")):
                    continue
                path = os.path.join(dirpath, f)
                src = open(path, errors="ignore").read()
                lines = src.split("\n")
                for brace_line, _, body, _ in functions(src):
                    name = _function_name(
                        lines, _signature(lines, brace_line), brace_line
                    )
                    if name:
                        self.bodies.setdefault(name, []).append(body)

    def classify_call(self, name, depth=0):
        """'tensix', 'risc', 'drain' or 'unknown' for a call to `name`."""
        if name in self.memo:
            return self.memo[name]
        self.memo[name] = "unknown"  # recursion guard
        bodies = self.bodies.get(name)
        if not bodies or depth > 6:
            return "unknown"
        verdict = "tensix"
        for body in bodies:
            inner = body[1:-1]
            d = DRAIN.search(inner)
            if d:
                verdict = "drain:" + d.group(1)
                break
            v = classify_tail(_statements(inner), self, depth + 1)[0]
            if v == "risc":
                verdict = "risc"
                break
            if v == "unknown":
                verdict = "unknown"
        self.memo[name] = verdict
        return verdict


def classify_statement(stmt, tree, depth=0):
    """'tensix', 'risc', 'drain', 'plain' or 'unknown' for one statement."""
    s = stmt.strip()
    if not s or s in "{}":
        return "plain"
    m = DRAIN.search(s)
    if m and not re.search(r"\b(?:if|while|for)\b", s[: m.start()]):
        return "drain:" + m.group(1)
    if TENSIX_PUSH.match(s):
        return "tensix"
    if RISC_ACCESS.search(s):
        return "risc"
    for name in CALL.findall(s):
        if name in CONTROL or name.isupper():
            continue
        v = tree.classify_call(name, depth)
        if v != "tensix":
            return v
        return "tensix"
    if PLAIN.match(s):
        return "plain"
    return "unknown"


def classify_tail(stmts, tree, depth=0):
    """Classify the statements after a drain: ('risc'|'drain'|'end'|'unknown', drain name or None)."""
    for _, s in stmts:
        v = classify_statement(s, tree, depth)
        if v.startswith("drain:"):
            return "drain", v.split(":", 1)[1]
        if v in ("risc", "unknown"):
            return v, None
    return "end", None


def scan(path, tree):
    src = open(path, errors="ignore").read()
    lines = src.split("\n")
    for brace_line, _, body, _ in functions(src):
        sig = _signature(lines, brace_line)
        head = lines[sig].strip()
        if re.search(r"\b(?:tensix_sync|mop_sync)\s*\(\s*\)\s*$", head):
            continue  # the drain definitions themselves
        inner = body[1:-1]
        stmts = _statements(inner)
        for i, (off, s) in enumerate(stmts):
            m = DRAIN.search(s)
            if not m or re.search(r"\b(?:if|while|for)\b", s[: m.start()]):
                continue
            drain = m.group(1)
            line = brace_line + 1 + inner[: off + m.start()].count("\n")
            verdict, nxt = classify_tail(stmts[i + 1 :], tree)
            if verdict == "drain" and (nxt == "tensix_sync" or drain == "mop_sync"):
                yield line, "REDUNDANT", drain, f"only Tensix pushes or plain code before the {nxt}() that covers it", head
            elif verdict == "end":
                yield line, "ADVISORY", drain, "nothing RISC-side follows in this function; if a caller needs the drain it should drain itself", head


def tree_root_for(path):
    norm = os.path.abspath(path).replace(os.sep, "/")
    for d in ARCH_DIRS:
        i = norm.find("/" + d + "/")
        if i >= 0:
            return norm[: i + len(d) + 1]
    return None


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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "files", nargs="*", help="headers to check (the tree is found from each path)"
    )
    ap.add_argument(
        "--tree", action="append", default=[], help="scan a whole architecture tree"
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

    files = [f for f in args.files if f.endswith((".h", ".hpp"))]
    # A change to the checker or the baseline moves the answer on every header: scan the trees.
    if any(
        os.path.basename(f)
        in ("check_redundant_drain.py", "redundant_drain_baseline.txt")
        for f in args.files
    ):
        here = os.path.dirname(os.path.abspath(__file__))
        args.tree += [os.path.join(here, "..", d) for d in ARCH_DIRS]
    for t in args.tree:
        for dirpath, _, names in os.walk(t):
            files += [
                os.path.join(dirpath, f) for f in names if f.endswith((".h", ".hpp"))
            ]
    accepted = load_baseline(args.baseline)
    trees, findings, skipped = {}, [], 0
    for path in sorted(set(files)):
        if ("/" + SKIPPED_ARCH + "/") in os.path.abspath(path).replace(os.sep, "/"):
            skipped += 1
            continue
        root = tree_root_for(path)
        if root not in trees:
            trees[root] = Tree(root)
        rel = os.path.relpath(path).replace(os.sep, "/")
        for line, kind, drain, reason, head in scan(path, trees[root]):
            key = f"{rel}:{head[:60]}"
            if key in accepted:
                continue
            findings.append((rel, line, kind, drain, reason, head))

    if args.json:
        import json

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
