#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""From the compute API down to the hardware state, for Blackhole.

1. For each compute API function (tt_metal/hw/inc/api/compute/...), the LLK wrappers it calls on each
   thread: the UNPACK((...)), MATH((...)) and PACK((...)) blocks, with #if ARCH_ branches resolved
   for Blackhole and calls to other compute API functions followed.
2. For each wrapper (llk_api), the _llk_ functions it reaches.
3. The state those calls touch: configuration registers (the register map, llk_cfg_map.py) and state
   outside them (MOP, counters, semaphores..., the patterns in llk_source.py).
4. Candidate abstractions: state that is always written by exactly the same compute API functions is
   grouped, since what always changes together is one thing to the layer above.

Usage: compute_api_map.py [REPO] [--groups tilize reduce ...] [--json FILE]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llk_cfg_map as m
import llk_source as sc
from llk_source import OUTSIDE_RX, bodies_of

API_DIR = "tt_metal/hw/inc/api/compute"
LLK_API_DIR = "tt_metal/hw/ckernels/blackhole/metal/llk_api"
GROUPS = {
    "startup": ["compute_kernel_hw_startup.h"],
    "reconfig": ["reconfig_data_format.h"],
    "tilize": ["tilize.h"],
    "untilize": ["untilize.h", "pack_untilize.h"],
    "reduce": ["reduce.h"],
    "matmul": ["matmul.h"],
    "binary": ["eltwise_binary.h"],
    "copy": ["tile_move_copy.h"],
    "pack": ["pack.h"],
}
ARCH = {
    "ARCH_BLACKHOLE": True,
    "ARCH_WORMHOLE": False,
    "ARCH_QUASAR": False,
    "ARCH_WORMHOLE_B0": False,
}

COND = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")


def eval_cond(kind, expr):
    """True, False, or None (unknown: keep the branch and its alternatives)."""
    expr = expr.strip()
    if kind in ("ifdef", "ifndef"):
        v = ARCH.get(expr.split()[0]) if expr else None
        return None if v is None else (v if kind == "ifdef" else not v)
    e = re.sub(
        r"defined\s*\(\s*(\w+)\s*\)|defined\s+(\w+)",
        lambda mm: {True: " 1 ", False: " 0 "}.get(
            ARCH.get(mm.group(1) or mm.group(2)), " X "
        ),
        expr,
    )
    if "X" in e or re.search(
        r"[A-Za-z_]", e.replace("and", "").replace("or", "").replace("not", "")
    ):
        return None
    e = e.replace("&&", " and ").replace("||", " or ").replace("!", " not ")
    try:
        return bool(sc.safe_eval(e))
    except Exception:
        return None


def for_blackhole(raw):
    """Keep the lines a Blackhole build sees; unknown conditions keep every branch."""
    out, stack = [], []  # entries: [active, taken]  (taken None = unknown condition)
    for line in raw.split("\n"):
        mm = COND.match(line)
        if not mm:
            out.append(line if all(s[0] for s in stack) else "")
            continue
        kind, expr = mm.group(1), mm.group(2)
        if kind in ("if", "ifdef", "ifndef"):
            v = eval_cond("if" if kind == "if" else kind, expr)
            stack.append([v is not False, v])
        elif kind == "elif" and stack:
            top = stack[-1]
            if top[1] is True:
                top[0] = False
            else:
                v = eval_cond("if", expr)
                top[0] = v is not False
                top[1] = True if v is True else top[1]
        elif kind == "else" and stack:
            top = stack[-1]
            top[0] = top[1] is not True
        elif kind == "endif" and stack:
            stack.pop()
        out.append("")
    return "\n".join(out)


def parse_dir_functions(repo, rel_dir, files=None):
    defs = defaultdict(list)
    base = repo / rel_dir
    paths = [base / f for f in files] if files else sorted(base.rglob("*.h"))
    for p in paths:
        if not p.exists():
            continue
        text = sc.strip(for_blackhole(p.read_text(errors="replace")))
        for name, rng, body in sc.functions(text):
            defs[name].append((body, p.relative_to(repo).as_posix()))
    return defs


THREAD = re.compile(
    r"\b(UNPACK|MATH|PACK)\s*\("
)  # MATH((...)) and MATH(SFPU_...(...)) both
LLK_CALL = re.compile(r"\b(llk_\w+)\s*(?:<[^;{}()]*?>)?\s*\(")
UNDER_LLK = re.compile(r"\b(_llk_\w+_)\s*(?:<[^;{}()]*?>)?\s*\(")
# SFPU operations call the LLK through macros (llk_math_eltwise_{unary,binary,ternary}_sfpu_macros.h):
#   SFPU_<KIND>_CALL...  ->  _llk_math_eltwise_<kind>_sfpu_params_   (the execute)
#   SFPU_<KIND>_INIT...  ->  llk_math_eltwise_<kind>_sfpu_init       (the init wrapper)
SFPU_MACRO = re.compile(r"\bSFPU_(UNARY|BINARY|TERNARY)_(CALL|INIT)\w*\s*\(")


def thread_calls(body):
    calls = defaultdict(set)
    for mm in THREAD.finditer(body):
        end = sc.match(body, mm.end() - 1, "(", ")")
        for c in LLK_CALL.finditer(body[mm.end() : end]):
            calls[mm.group(1)].add(c.group(1))
        for c in SFPU_MACRO.finditer(body[mm.end() : end]):
            kind = c.group(1).lower()
            calls[mm.group(1)].add(
                f"_llk_math_eltwise_{kind}_sfpu_params_"
                if c.group(2) == "CALL"
                else f"llk_math_eltwise_{kind}_sfpu_init"
            )
    return calls


def api_calls(api_defs, name, memo, stack=()):
    """thread -> LLK wrappers, for a compute API function and the compute API functions it calls."""
    if name in memo:
        return memo[name]
    if name in stack:
        return {}
    acc = defaultdict(set)
    for body, rel in api_defs.get(name, []):
        for t, cs in thread_calls(body).items():
            acc[t] |= cs
        for c, argc in sc.calls_with_arity(body):
            if c in api_defs and c != name:
                for t, cs in api_calls(api_defs, c, memo, stack + (name,)).items():
                    acc[t] |= cs
    memo[name] = acc
    return acc


def wrapper_llk(wdefs, name, memo, stack=()):
    """_llk_ functions a wrapper reaches, through other wrappers."""
    if name in memo:
        return memo[name]
    if name in stack:
        return set(), ""
    fns, text = set(), []
    for body, rel in wdefs.get(name, []):
        text.append(body)
        fns |= {mm.group(1) for mm in UNDER_LLK.finditer(body)}
        for c in {mm.group(1) for mm in LLK_CALL.finditer(body)}:
            if c in wdefs and c != name:
                f2, t2 = wrapper_llk(wdefs, c, memo, stack + (name,))
                fns |= f2
                text.append(t2)
    memo[name] = (fns, "\n".join(text))
    return memo[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", nargs="?", default=".")
    ap.add_argument("--groups", nargs="*", default=list(GROUPS))
    ap.add_argument(
        "--all",
        action="store_true",
        help="every compute API header (one group per header), not only GROUPS",
    )
    ap.add_argument("--json")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()

    if a.all:
        base = repo / API_DIR
        hdrs = sorted(
            p.relative_to(base).as_posix()
            for p in base.rglob("*.h")
            if not p.relative_to(base).as_posix().startswith(("debug/", "sentinel/"))
        )
        GROUPS.clear()
        for h in hdrs:
            GROUPS[h[:-2]] = [h]
        a.groups = list(GROUPS)
    all_api = parse_dir_functions(
        repo, API_DIR, [f for g in GROUPS.values() for f in g]
    )
    wdefs = parse_dir_functions(repo, LLK_API_DIR)
    ldefs = m.build(repo)
    amemo, wmemo, rmemo, bmemo = {}, {}, {}, {}

    rows = []
    for g in a.groups:
        names = set()
        for f in GROUPS[g]:
            p = repo / API_DIR / f
            if p.exists():
                names |= {
                    n
                    for n, rng, body in sc.functions(
                        sc.strip(for_blackhole(p.read_text(errors="replace")))
                    )
                }
        for api in sorted(names):
            calls = api_calls(all_api, api, amemo)
            if not calls:
                continue
            row = dict(
                group=g,
                api=api,
                threads={},
                regs=defaultdict(set),
                outside=defaultdict(set),
            )
            for t in ("UNPACK", "MATH", "PACK"):
                wrappers = sorted(calls.get(t, ()))
                if not wrappers:
                    continue
                llk, wtext = set(), []
                for w in wrappers:
                    if w.startswith(
                        "_llk_"
                    ):  # an SFPU call macro names the LLK function directly
                        llk.add(w)
                        continue
                    f2, t2 = wrapper_llk(wdefs, w, wmemo)
                    llk |= f2
                    wtext.append(t2)
                row["threads"][t] = dict(wrappers=wrappers, llk=sorted(llk))
                text = "\n".join(wtext + [bodies_of(ldefs, f, bmemo) for f in llk])
                for f in llk:
                    for reg, how, kind, via in m.writes_of(ldefs, f, rmemo):
                        row["regs"][reg].add(t)
                for name, rx in OUTSIDE_RX:
                    if rx.search(text):
                        row["outside"][name].add(t)
            rows.append(row)

    # 1-3: per compute API function
    print(f"{len(rows)} compute API functions in groups: {', '.join(a.groups)}\n")
    for r in rows:
        th = "; ".join(
            f"{t}: {', '.join(v['wrappers'])}" for t, v in r["threads"].items()
        )
        print(f"[{r['group']}] {r['api']}  ->  {th}")
        outs = (
            ", ".join(
                sorted(
                    k + "(" + "/".join(sorted(v)) + ")" for k, v in r["outside"].items()
                )
            )
            or "-"
        )
        print(f"      registers written: {len(r['regs'])}   outside state: {outs}")

    # 4: group state by the exact set of compute API functions that write it
    writers = defaultdict(set)
    for r in rows:
        for reg, ts in r["regs"].items():
            writers[("reg", reg)].add(r["api"])
        for name, ts in r["outside"].items():
            for t in ts:
                writers[("outside", f"{t}:{name}")].add(r["api"])
    clusters = defaultdict(list)
    for item, ws in writers.items():
        clusters[frozenset(ws)].append(item)
    print(
        f"\n{len(writers)} pieces of state, in {len(clusters)} groups that always change together:"
    )
    for ws, items in sorted(
        clusters.items(), key=lambda kv: (-len(kv[1]), sorted(kv[0]))
    ):
        regs = sorted(i[1] for i in items if i[0] == "reg")
        outs = sorted(i[1] for i in items if i[0] == "outside")
        print(
            f"\n  [{len(items)} items] written by {len(ws)}: {', '.join(sorted(ws))[:220]}"
        )
        if regs:
            print(f"     registers: {', '.join(regs)[:600]}")
        if outs:
            print(f"     outside:   {', '.join(outs)[:400]}")

    if a.json:
        out = dict(
            apis=[
                dict(
                    group=r["group"],
                    api=r["api"],
                    threads=r["threads"],
                    registers={k: sorted(v) for k, v in r["regs"].items()},
                    outside={k: sorted(v) for k, v in r["outside"].items()},
                )
                for r in rows
            ],
            clusters=[
                dict(
                    writers=sorted(ws),
                    registers=sorted(i[1] for i in items if i[0] == "reg"),
                    outside=sorted(i[1] for i in items if i[0] == "outside"),
                )
                for ws, items in clusters.items()
            ],
        )
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\nwritten {a.json}")


if __name__ == "__main__":
    main()
