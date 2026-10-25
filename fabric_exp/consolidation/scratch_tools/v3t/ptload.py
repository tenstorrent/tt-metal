#!/usr/bin/env python3
"""ptload.py A B: compare PT_LOAD segments (vaddr, filesz, memsz, bytes) of every ELF in builds A and B, per proc
(active_erisc = ERISC0 / MY_ERISC_ID 0, subordinate_active_erisc = ERISC1). Prints counts and the differing hashes."""
import glob, os, struct, sys, collections, concurrent.futures
def loads(path):
    d = open(path, "rb").read()
    assert d[:4] == b"\x7fELF" and d[4] == 1
    phoff, = struct.unpack_from("<I", d, 28); phentsz, phnum = struct.unpack_from("<HH", d, 42)
    segs = []
    for i in range(phnum):
        t, off, va, pa, fsz, msz, fl, al = struct.unpack_from("<8I", d, phoff + i * phentsz)
        if t == 1: segs.append((va, fsz, msz, d[off:off + fsz]))
    return segs
a, b = sys.argv[1], sys.argv[2]
keys = sorted(os.path.relpath(e, b) for e in glob.glob(f"{b}/*/*/*.elf") if not e.endswith(".xip.elf"))
def cmp(k):
    pa = os.path.join(a, k)
    if not os.path.exists(pa): return k, None
    return k, loads(pa) == loads(os.path.join(b, k))
with concurrent.futures.ThreadPoolExecutor(16) as ex: res = dict(ex.map(cmp, keys))
c = collections.Counter(); diff = collections.defaultdict(list)
for k, v in res.items():
    p = k.split("/")[1]; c[(p, {True: "same", False: "diff", None: "missing"}[v])] += 1
    if v is False: diff[p].append(k.split("/")[0])
print(dict(sorted(c.items())))
if len(sys.argv) > 3:
    import json; json.dump({p: sorted(v) for p, v in diff.items()}, open(sys.argv[3], "w"))
