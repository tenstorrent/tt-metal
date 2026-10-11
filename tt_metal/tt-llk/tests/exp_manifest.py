#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Path-independent manifest of a Wormhole perf producer build (experiment CI legs; stdlib only), so a CI build can be
compared with a local one: per variant (test source | sha256 of build.h), per ELF (elf/, layout_*/elf/), sha256 of the
loaded sections (name, address, bytes; debug sections excluded: they name the build directories) and the layout choices.
usage: exp_manifest.py RUNNER_TEMP OUT.json.gz   |   exp_manifest.py --compare A.json.gz B.json.gz"""
import sys, os, json, gzip, hashlib, struct
from pathlib import Path
def loaded_digest(path):
    d = Path(path).read_bytes()
    assert d[:4] == b"\x7fELF" and d[4] == 1, path  # ELF32
    shoff, = struct.unpack_from("<I", d, 0x20); shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x2E)
    sh = [struct.unpack_from("<IIIIIIIIII", d, shoff + i * shentsize) for i in range(shnum)]
    stro = sh[shstrndx][4]
    name = lambda o: d[stro + o: d.index(b"\0", stro + o)]
    h = hashlib.sha256()
    for s in sorted(sh, key=lambda s: (s[3], name(s[0]))):
        nm, typ, flags, addr, off, size = name(s[0]), s[1], s[2], s[3], s[4], s[5]
        if not flags & 2 or typ == 8: continue  # SHF_ALLOC, not NOBITS
        h.update(nm + struct.pack("<II", addr, size) + d[off:off + size])
    return h.hexdigest()[:24]
def build(rt, out):
    root = Path(rt) / "tt-llk-build" / "sources"; M = {}
    for b in sorted(root.rglob("build.h")):
        vd = b.parent
        key = f"{vd.parent.name}|{hashlib.sha256(b.read_bytes()).hexdigest()[:20]}"
        rec = {"elfs": {}, "choices": {}}
        for sub in [vd / "elf", vd / "init_elf"] + sorted(vd.glob("layout_*/elf")):
            for e in sorted(sub.glob("*.elf")):
                rec["elfs"][f"{sub.parent.name if sub.parent != vd else ''}/{sub.name}/{e.name}".lstrip("/")] = loaded_digest(e)
        for c in sorted((vd / "layout").glob("*.json")) if (vd / "layout").is_dir() else []:
            try: rec["choices"][c.stem] = json.loads(c.read_text())
            except ValueError: pass
        M[key] = rec
    json.dump(M, gzip.open(out, "wt"))
    print(f"{len(M)} variants, {sum(len(r['elfs']) for r in M.values())} ELFs, {sum(len(r['choices']) for r in M.values())} choices -> {out}")
def compare(a, b):
    A, B = json.load(gzip.open(a, "rt")), json.load(gzip.open(b, "rt"))
    common = sorted(set(A) & set(B)); ne = de = nc = dc = 0; ex = []
    for k in common:
        ea, eb = A[k]["elfs"], B[k]["elfs"]
        for e in set(ea) | set(eb):
            ne += 1
            if ea.get(e) != eb.get(e): de += 1; ex.append((k, e))
        ca, cb = A[k]["choices"], B[k]["choices"]
        for c in set(ca) & set(cb):
            nc += 1; dc += ca[c] != cb[c]
    print(f"variants A {len(A)} B {len(B)} common {len(common)}; ELFs {de} of {ne} differ (loaded sections); layout choices {dc} of {nc} differ")
    for x in ex[:10]: print("  ", x)
if __name__ == "__main__":
    compare(sys.argv[2], sys.argv[3]) if sys.argv[1] == "--compare" else build(sys.argv[1], sys.argv[2])
