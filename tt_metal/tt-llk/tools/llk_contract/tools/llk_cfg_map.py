#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""LLK register-write map for Blackhole, by static reading of tt_llk_blackhole (read-only, no chip).

For every LLK function (_llk_..._) it lists the configuration registers it writes, directly or
through the helpers it calls, with how (field RMW, whole word, 16-bit, HAL) and whether the value
depends on one of the function's parameters or is fixed. contract_tool.py check-code uses it to see
whether the LLK functions named for an entry point write the fields the entry point ensures.

  python3 llk_cfg_map.py [repo]                  summary
  python3 llk_cfg_map.py [repo] --fn NAME        one function's writes
  python3 llk_cfg_map.py [repo] --csv out.csv    the whole map, one row per (function, register)

Approximate: text, not compiled code; every branch counts (see llk_source.py).
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llk_source as sc  # strip, match, calls_with_arity, param_range, functions

LLK_DIRS = [
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc",
]

# register symbols, as written in cfg_defines.h: <NAME>_ADDR32 (word), <NAME>_RMW (field), <NAME>_SHAMT/_MASK
SYM = r"([A-Z][A-Za-z0-9_]*?)_(ADDR32|RMW)\b"
WRITES = [
    ("rmw", re.compile(r"cfg_reg_rmw_tensix\s*<\s*" + SYM)),  # field or part-word RMW
    (
        "word",
        re.compile(r"\bcfg\s*\[\s*" + SYM + r"[^\]]*\]\s*=(?!=)"),
    ),  # cfg[ADDR32 (+i)] = ...
    (
        "wrcfg",
        re.compile(r"\bTTI?_WRCFG\s*\([^;]*?" + SYM),
    ),  # 32/128-bit word from a GPR
    ("setc16", re.compile(r"\bTTI?_SETC16\s*\(\s*" + SYM)),  # 16-bit immediate
    ("rmwcib", re.compile(r"\bTTI?_RMWCIB[0-3]\s*\([^;]*?" + SYM)),  # byte RMW
]
RMW_PART = re.compile(
    r"cfg_reg_rmw_tensix\s*<\s*([A-Z][A-Za-z0-9_]*)_ADDR32\s*,\s*([A-Z][A-Za-z0-9_]*)_SHAMT"
)
HAL_WRITE = re.compile(
    r"\bhal::cfg::\w+\s*<\s*([A-Za-z_][\w:]*)"
)  # typed HAL writes, if any
STATEMENT_END = re.compile(r";")


def param_names(params):
    names = []
    depth, cur = 0, []
    for ch in params + ",":
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == "," and depth == 0:
            p = "".join(cur).split("=")[0].strip()
            m = re.search(r"([A-Za-z_]\w*)\s*(\[[^\]]*\])?\s*$", p)
            if m and m.group(1) not in ("void", "const"):
                names.append(m.group(1))
            cur = []
        else:
            cur.append(ch)
    return names


def functions_with_params(text):
    """(name, params_text, body) for function definitions."""
    i, n = 0, len(text)
    pat = re.compile(r"\b([A-Za-z_]\w*)\s*(?:<[^;{}()]*?>)?\s*\(")
    while i < n:
        m = pat.search(text, i)
        if not m:
            return
        name = m.group(1)
        close = sc.match(text, m.end() - 1, "(", ")")
        if name in sc.KEYWORDS or re.fullmatch(
            r"[A-Z0-9_]+", name
        ):  # a macro use (TTI_PACR(...)) is never a definition
            i = m.end()
            continue
        if False:
            i = m.end()
            continue
        q = sc.QUAL.match(text, close)
        if q:
            brace = q.end() - 1
            end = sc.match(text, brace, "{", "}")
            yield name, text[m.end() : close - 1], text[brace:end]
            i = end
        else:
            i = m.end()


def direct_writes(body, params):
    """[(register, how, value_kind)] written in this body (not through calls)."""
    out = []
    for how, rx in WRITES:
        for m in rx.finditer(body):
            reg = m.group(1)
            also = None
            if how == "rmw" and m.group(2) == "ADDR32":
                p = RMW_PART.match(body, m.start())
                if p:
                    reg = p.group(2)  # the field named by its _SHAMT
                    if p.group(1) != p.group(2):
                        also = p.group(
                            1
                        )  # ADDR32 of another field: the word written is the ADDR32's
            # the statement the write sits in, to see whether the value comes from a parameter
            end = body.find(";", m.end())
            stmt = body[m.start() : (len(body) if end < 0 else end)]
            depends = any(re.search(r"\b" + re.escape(p) + r"\b", stmt) for p in params)
            out.append((reg, how, "param" if depends else "fixed"))
            if also:
                out.append((also, how, "param" if depends else "fixed"))
    for m in HAL_WRITE.finditer(body):
        out.append((m.group(1).split("::")[-1], "hal", "param"))
    return out


def build(repo):
    defs = defaultdict(list)  # name -> [(rng, params, body, file)]
    for d in LLK_DIRS:
        for f in sc.walk_files(repo / d, sc.HDR_EXT):
            raw = f.read_text(errors="replace")
            rel = f.relative_to(repo).as_posix()
            for name, ptxt, body in functions_with_params(sc.strip(raw)):
                defs[name].append((sc.param_range(ptxt), param_names(ptxt), body, rel))
    return defs


def unit_of(rel):
    """Which unit a source file belongs to, by its name: unpack, math or pack (None for shared files)."""
    base = rel.rsplit("/", 1)[-1]
    for u in ("unpack", "math", "pack"):
        if u in base:
            return u
    return None


def writes_of(defs, name, memo, stack=(), unit=None):
    """Union over the definitions of `name` and the helpers it calls (arity-matched). A helper defined in
    several files (set_dst_write_addr exists for unpack, math and pack) is taken from the caller's unit,
    so a math function does not inherit the unpacker's writes."""
    key = (name, unit)
    if key in memo:
        return memo[key]
    if name in stack:
        return set()
    mine = defs.get(name, [])
    if unit:
        same = [d for d in mine if unit_of(d[3]) == unit]
        mine = same or mine
    acc = set()
    for rng, params, body, rel in mine:
        u = unit_of(rel) or unit
        for reg, how, kind in direct_writes(body, params):
            acc.add((reg, how, kind, name))
        for c, argc in sc.calls_with_arity(body):
            if c in defs and c != name:
                fits = [d for d in defs[c] if d[0][0] <= argc <= d[0][1]] or defs[c]
                if fits:
                    for reg, how, kind, via in writes_of(
                        defs, c, memo, stack + (name,), u
                    ):
                        # a helper's write counts as "param" here only if it was "param" there
                        acc.add((reg, how, kind, via))
    memo[key] = acc
    return acc


ROLE_RULES = [
    ("cleanup", re.compile(r"cleanup|canonical")),
    ("uninit", re.compile(r"_uninit")),
    ("configure", re.compile(r"hw_configure|reconfig|_configure_|set_fp32|dest_acc")),
    ("init", re.compile(r"_init|mop_config|_setup_|_program_")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", nargs="?", default=str(Path.home() / "tt-metal"))
    ap.add_argument("--fn")
    ap.add_argument("--csv")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    defs = build(repo)
    memo = {}
    llk = sorted(n for n in defs if n.startswith("_llk_") and n.endswith("_"))
    rows = []
    for n in llk:
        for reg, how, kind, via in sorted(writes_of(defs, n, memo)):
            rows.append((n, reg, how, kind, via))

    if a.fn:
        for n, reg, how, kind, via in rows:
            if n == a.fn:
                print(f"{reg:<55} {how:<7} {kind:<6} via {via}")
        return
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["function", "register", "how", "value", "written_in"])
            w.writerows(rows)
    regs = {r[1] for r in rows}
    fixed = sum(1 for r in rows if r[3] == "fixed")
    print(f"repo {repo}")
    print(
        f"LLK functions {len(llk)}, with a register write {len({r[0] for r in rows})}"
    )
    print(
        f"(function, register) rows {len(rows)}, distinct registers {len(regs)}, fixed value {fixed} ({fixed / max(1, len(rows)):.0%})"
    )


if __name__ == "__main__":
    main()
