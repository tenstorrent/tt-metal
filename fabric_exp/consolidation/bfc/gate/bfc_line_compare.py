#!/usr/bin/env python3
"""bfc_line_compare.py BASE.elf NEW.elf [--function F]: line_compare.py (tree's router_bringup tool) per innermost
(function, file:line), with the base image's line numbers mapped onto the branch's sources for the files the branch
changed (difflib over 4db4f45790b's and the worktree's text), so moved lines do not show as changes."""
import collections, difflib, os, subprocess, sys

T = "/home/aagarwal/tt-metal-mainline-bfc"
sys.path.insert(0, T + "/tt_metal/fabric/tools/router_bringup")
import line_compare as lc  # noqa: E402

CHANGED = {
    "router_sender.hpp": "tt_metal/fabric/hw/inc/edm_fabric/router/router_sender.hpp",
    "fabric_erisc_router_ct_args.hpp": "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp",
}


def line_map(rel):
    old = subprocess.run(["git", "-C", T, "show", "4db4f45790b:" + rel], capture_output=True, text=True).stdout.splitlines()
    new = open(os.path.join(T, rel)).read().splitlines()
    m = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                m[i1 + k + 1] = j1 + k + 1
    return m


MAPS = {f: line_map(rel) for f, rel in CHANGED.items()}


def remap(key):
    fn, loc = key.rsplit(" ", 1)
    f, _, ln = loc.partition(":")
    if f in MAPS and ln.isdigit():
        n = MAPS[f].get(int(ln))
        return f"{fn} {f}:{n if n else ln + '(base, changed)'}"
    return key


base, new = sys.argv[1], sys.argv[2]
want = sys.argv[sys.argv.index("--function") + 1] if "--function" in sys.argv else None
na, ca, pa, ma, _ = lc.summarize(base, want, set())
nb, cb, pb, mb, _ = lc.summarize(new, want, set())
pa2 = collections.Counter()
for k, v in pa.items():
    pa2[remap(k)] += v
print(f"base {na}: {ca} instructions; new {nb}: {cb} ({cb - ca:+d})")
print(f"loads/stores: base {dict(ma)}  new {dict(mb)}")
d = {k: pb.get(k, 0) - pa2.get(k, 0) for k in set(pa2) | set(pb)}
rows = sorted(((k, v) for k, v in d.items() if v), key=lambda kv: (-abs(kv[1]), kv[0]))
print(f"changed lines after the line map: {len(rows)}")
for k, v in rows[:20]:
    print(f"   {v:+4d}  {k}  ({pa2.get(k, 0)} -> {pb.get(k, 0)})")
