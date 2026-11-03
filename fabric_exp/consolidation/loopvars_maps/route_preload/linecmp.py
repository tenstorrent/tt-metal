#!/usr/bin/env python3
"""linecmp.py BASE_BUILD NEW_BUILD BASE_VARIANT NEW_VARIANT IMAGE_GLOB: loop instructions per source line (innermost
router_*.hpp / codec frame), base vs new, with the base's line numbers mapped onto the new sources (difflib over each
router file), so moved lines compare as the same line."""
import collections, difflib, glob, os, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import loopvars as L
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
R = "tt_metal/fabric/hw/inc/edm_fabric/router"
bb, nb, bv, nv, pat = sys.argv[1:6]


def line_map(f):
    a = open(f"{V}/{bv}/{R}/{f}").read().splitlines()
    b = open(f"{V}/{nv}/{R}/{f}").read().splitlines()
    m = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                m[i1 + k + 1] = j1 + k + 1
    return m


maps = {f: line_map(f) for f in os.listdir(f"{V}/{nv}/{R}") if f.endswith(".hpp")}


def lines(build, rel, mapped):
    elf = f"{CB}/{build}/{rel}"
    fns = L.functions(elf)
    names = [k for k in fns if "8receiver3run" in k] or [k for k in fns if "6single13receiver_pass" in k]
    body = fns[names[0]]
    succ, _ = L.cfg(body)
    loop = L.largest_scc([a for a, *_ in body], succ) if "run" in names[0] else {a for a, *_ in body}
    ch = L.chains(elf, sorted(loop))
    c = collections.Counter()
    for a in loop:
        fl = next((fl for f, fl in ch.get(a, []) if fl.startswith("router_")), "?")
        if mapped and fl != "?":
            f, ln = fl.split(":")
            ln = maps.get(f, {}).get(int(ln), f"old{ln}")
            fl = f"{f}:{ln}"
        c[fl] += 1
    return c, len(loop)


rels = [os.path.relpath(p, f"{CB}/{nb}") for p in sorted(glob.glob(f"{CB}/{nb}/{pat}"))]
tot = collections.Counter()
for rel in rels:
    a, na = lines(bb, rel, True)
    b, nn = lines(nb, rel, False)
    print(rel.split("/")[0], "loop", na, "->", nn)
    for k in set(a) | set(b):
        tot[k] += b.get(k, 0) - a.get(k, 0)
src = {}
for k, d in sorted(tot.items(), key=lambda kv: -abs(kv[1])):
    if d == 0:
        continue
    text = ""
    if ":" in k and not k.split(":")[1].startswith("old"):
        f, ln = k.split(":")
        p = f"{V}/{nv}/{R}/{f}"
        if os.path.exists(p):
            text = open(p).read().splitlines()[int(ln) - 1].strip()[:90]
    print(f"{d:+5d}  {k:32s} {text}")
