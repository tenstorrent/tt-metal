#!/usr/bin/env python3
"""Survey of fabric atomic increments that make the router flush (§59).

Finds every construction of the three fabric command headers whose `flush` field makes the receiving router drain all of
its outstanding local NoC writes before the atomic (`flush_write_to_noc_pipeline`):
  - NocUnicastAtomicIncCommandHeader(noc_address, val, flush = true)            standalone atomic increment
  - NocUnicastAtomicIncFusedCommandHeader(noc_address, sem_address, val, flush = true)   write + atomic
  - NocUnicastScatterAtomicIncFusedCommandHeader(..., flush = true)              scatter write + atomic
and classifies the flush argument: explicit true / false, omitted (= the default, true), or an expression.
Call sites are grouped by op (the path up to `device/` under ttnn/cpp/ttnn/operations, or the file's directory elsewhere).

usage: flush_survey.py [--csv out.csv] [--include-tests]
"""
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = ["ttnn", "models", "tt_metal"]
EXTS = (".cpp", ".hpp", ".h", ".cc")
HEADERS = {
    "NocUnicastAtomicIncCommandHeader": ("atomic", 2),  # (kind, number of args before flush)
    "NocUnicastAtomicIncFusedCommandHeader": ("fused", 3),
    "NocUnicastScatterAtomicIncFusedCommandHeader": ("scatter", 4),  # ({addrs}, sem, {sizes}, val, flush = true)
}
DEFINITION_FILES = ("tt_metal/fabric/fabric_edm_packet_header.hpp",)
PAT = re.compile(r"\b(" + "|".join(HEADERS) + r")\b\s*([({])")


def split_args(s):
    """Split a C++ argument list at top-level commas (brackets ( [ { only; template commas are rare here)."""
    args, depth, cur = [], 0, ""
    for c in s:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += c
    if cur.strip():
        args.append(cur.strip())
    return args


def balanced(text, start):
    """Return the text inside the bracket at text[start] (which is '(' or '{') and the end index."""
    open_c = text[start]
    close_c = ")" if open_c == "(" else "}"
    depth = 0
    for j in range(start, len(text)):
        if text[j] in "({[":
            depth += 1
        elif text[j] in ")}]":
            depth -= 1
            if depth == 0:
                return text[start + 1 : j], j
    return None, start


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: " " * len(m.group(0)), text, flags=re.S)
    return re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)


def classify(kind, n_before, args):
    if len(args) <= n_before:
        return "default(true)", ""
    v = re.sub(r"/\*.*?\*/", "", args[n_before]).strip()
    if v in ("true", "1"):
        return "true", v
    if v in ("false", "0"):
        return "false", v
    return "expr", v


def op_of(rel):
    m = re.match(r"ttnn/cpp/ttnn/operations/(.+?)/device/", rel)
    if m:
        return m.group(1)
    m = re.match(r"ttnn/cpp/ttnn/operations/(.+?)/kernels/", rel)
    if m:
        return m.group(1)
    m = re.match(r"(models/.+?)/(?:kernels|tt|device)/", rel)
    if m:
        return m.group(1)
    return os.path.dirname(rel)


def main():
    include_tests = "--include-tests" in sys.argv
    out_csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
    rows = []
    for d in DIRS:
        for dirpath, _, files in os.walk(os.path.join(ROOT, d)):
            if "/build" in dirpath or "/.git" in dirpath:
                continue
            for fn in files:
                if not fn.endswith(EXTS):
                    continue
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, ROOT)
                if rel in DEFINITION_FILES or (not include_tests and "/tests/" in "/" + rel):
                    continue
                try:
                    raw = open(path, errors="replace").read()
                except OSError:
                    continue
                if not any(h in raw for h in HEADERS):
                    continue
                text = strip_comments(raw)
                for m in PAT.finditer(text):
                    name, br = m.group(1), m.group(2)
                    # skip declarations/definitions: `struct X {`, `X(...) :` constructors, parameter types
                    before = text[max(0, m.start() - 40) : m.start()]
                    if re.search(r"\b(struct|class)\s+$", before):
                        continue
                    inner, end = balanced(text, m.end() - 1)
                    if inner is None:
                        continue
                    args = split_args(inner)
                    if br == "(" and len(args) == 1 and args[0].startswith("{") and args[0].endswith("}"):
                        args = split_args(args[0][1:-1])
                    if len(args) == 0 or (len(args) == 1 and re.fullmatch(r"[A-Za-z_][\w.\->\[\]]*", args[0])):
                        # copy construction / parameter type usage, not a field-by-field construction
                        continue
                    kind, n_before = HEADERS[name]
                    flush, expr = classify(kind, n_before, args)
                    line = text.count("\n", 0, m.start()) + 1
                    rows.append(
                        {"file": rel, "line": line, "kind": kind, "flush": flush, "expr": expr, "op": op_of(rel)}
                    )
    rows.sort(key=lambda r: (r["op"], r["file"], r["line"]))
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["op", "file", "line", "kind", "flush", "expr"])
            w.writeheader()
            w.writerows(rows)
    by_op = {}
    for r in rows:
        by_op.setdefault(r["op"], []).append(r)
    flushing = lambda r: r["flush"] in ("true", "default(true)")
    print(
        f"{len(rows)} call sites in {len(by_op)} ops/dirs; flush true or default: {sum(flushing(r) for r in rows)}, "
        f"false: {sum(r['flush'] == 'false' for r in rows)}, expression: {sum(r['flush'] == 'expr' for r in rows)}\n"
    )
    for op, rs in sorted(by_op.items()):
        tags = []
        for kind in ("fused", "atomic", "scatter"):
            k = [r for r in rs if r["kind"] == kind]
            if not k:
                continue
            c = {f: sum(r["flush"] == f for r in k) for f in ("true", "default(true)", "false", "expr")}
            tags.append(f"{kind}: " + " ".join(f"{f}={n}" for f, n in c.items() if n))
        print(f"{op}\n    " + "; ".join(tags))
        for r in rs:
            if r["flush"] == "expr":
                print(f"      expr at {r['file']}:{r['line']}: {r['expr'][:80]}")


if __name__ == "__main__":
    main()
