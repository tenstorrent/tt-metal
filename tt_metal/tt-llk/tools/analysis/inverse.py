#!/usr/bin/env python3
"""Classify LLK init/uninit pairs by whether uninit actually reverts init (F3).

For each _llk_<op>_uninit_ in the llk lib, extract its body and its matching
_llk_<op>_init_, strip comments and the llk::san:: hook lines, and check what
HW work each does. Tiers:
  save+restore : init saves cfg to GPR (RDCFG) and uninit writes it back (WRCFG) -> true inverse
  reset/other  : uninit writes HW but without a matching save -> resets to default, not exact revert
  hook-only    : uninit body is only the sanitizer hook / empty -> reverts nothing
The gap the metric wants: init is HW-active but uninit is hook-only (or reset-only).
"""

import collections
import glob
import os
import re

ROOTS = [
    "tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib",
]
import sys

BASE = (
    sys.argv[1] if len(sys.argv) > 1 else "."
)  # repo root; run: python3 tools/analysis/inverse.py <repo-root>


def strip(s):
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    s = re.sub(r"//[^\n]*", " ", s)
    return s


FUNC = re.compile(r"\binline\s+void\s+(_llk_[a-z0-9_]+_)\s*\(")


def bodies(src):
    """name -> body text, by brace scan."""
    out = {}
    for m in FUNC.finditer(src):
        i = src.find("{", m.end())
        if i < 0:
            continue
        depth = 0
        j = i
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out[m.group(1)] = src[i + 1 : j]
    return out


# a "real" (non-hook) statement: ends in ; and isn't a sanitizer call
def real_stmts(body):
    out = []
    for line in body.split(";"):
        t = line.strip()
        if not t:
            continue
        if "llk::san::" in t:
            continue
        if t.startswith("return") or t in ("}", "{"):
            continue
        # must look like a statement (call / assignment / macro)
        if re.search(r"[A-Za-z_]\w*\s*\(", t) or "=" in t:
            out.append(t)
    return out


HAS_RDCFG = re.compile(r"\bTTI?_RDCFG\b")
HAS_WRCFG = re.compile(r"\bTTI?_WRCFG\b")


def base_op(name):  # _llk_unpack_untilize_uninit_ -> unpack_untilize
    n = name.strip("_")
    n = re.sub(r"^llk_", "", n)
    n = re.sub(r"_(uninit|init)$", "", n)
    return n


rows = []
for root in ROOTS:
    arch = "WH" if "wormhole" in root else "BH"
    allfun = {}
    for path in glob.glob(os.path.join(BASE, root, "*.h")):
        allfun.update(
            {
                (arch, k): v
                for k, v in bodies(strip(open(path, errors="replace").read())).items()
            }
        )
    # index inits by base op for this arch
    inits = {base_op(k[1]): v for k, v in allfun.items() if k[1].endswith("_init_")}
    for (a, name), body in allfun.items():
        if not name.endswith("_uninit_"):
            continue
        op = base_op(name)
        u_real = real_stmts(body)
        init_body = inits.get(op, "")
        i_real = real_stmts(init_body)
        init_active = len(i_real) > 0
        if not u_real:
            tier = "hook-only"
        elif HAS_WRCFG.search(body) and init_body and HAS_RDCFG.search(init_body):
            tier = "save+restore"
        else:
            tier = "reset/other"
        rows.append((a, op, tier, len(u_real), len(i_real), init_active))

print(f"LLK-lib init/uninit pairs (WH+BH): {len(rows)}\n")
tier_count = collections.Counter(r[2] for r in rows)
print("uninit tier:")
for t in ("save+restore", "reset/other", "hook-only"):
    print(f"  {t:14} {tier_count[t]}")
print()

# the gap: init changes HW but uninit reverts nothing (hook-only) or only resets
not_inverse = [r for r in rows if r[2] != "save+restore" and r[5]]
hookonly_active = [r for r in rows if r[2] == "hook-only" and r[5]]
print(
    f"init is HW-active but uninit is NOT a true (save+restore) inverse: "
    f"{len(not_inverse)}/{len([r for r in rows if r[5]])} HW-active pairs "
    f"({100*len(not_inverse)/max(len([r for r in rows if r[5]]),1):.0f}%)"
)
print(f"  of which hook-only (reverts nothing): {len(hookonly_active)}\n")

print(f"{'arch':4} {'op':28} {'tier':13} {'u_stmts':7} {'i_stmts':7}")
for a, op, tier, us, is_, act in sorted(
    rows, key=lambda r: (r[2] != "hook-only", r[0], r[1])
):
    print(f"{a:4} {op:28} {tier:13} {us:<7} {is_:<7} {'init-active' if act else ''}")
