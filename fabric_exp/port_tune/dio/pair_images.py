#!/usr/bin/env python3
"""pair_images.py CACHE_A CACHE_B [--json OUT]: pair the router ELFs of two JIT caches by processor and the build's named
compile-time args (named_ct_arg_map_generated.h; the JIT's hash directories differ between runtime roots), and report
per pair whether the loadable bytes are identical, with both images' loadable sizes (text + data)."""
import argparse, collections, glob, hashlib, json, os, struct, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/port_tune")
import elf_load_cmp as e


def load_size(p):
    b = open(p, "rb").read()
    phoff, = struct.unpack_from("<I", b, 28); pe, pn = struct.unpack_from("<HH", b, 42)
    return sum(struct.unpack_from("<8I", b, phoff + i * pe)[4] for i in range(pn)
               if struct.unpack_from("<8I", b, phoff + i * pe)[0] == 1)


def builds(cache):
    out = collections.defaultdict(list)
    for d in glob.glob(f"{cache}/*/kernels/fabric_erisc_router/*/"):
        m = os.path.join(d, "named_ct_arg_map_generated.h")
        if not os.path.exists(m):
            continue
        key = hashlib.md5(open(m, "rb").read()).hexdigest()[:12]
        for proc in ("active_erisc", "subordinate_active_erisc"):
            elfs = [p for p in glob.glob(os.path.join(d, proc, "*.elf")) if not p.endswith(".xip.elf")]
            if elfs:
                out[(proc, key)].append(elfs[0])
    return out


ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--json", default="")
a = ap.parse_args()
A, B = builds(a.a), builds(a.b)
pairs, summ = [], collections.defaultdict(collections.Counter)
for k in sorted(set(A) | set(B)):
    pa, pb = A.get(k), B.get(k)
    if not pa or not pb:
        summ[k[0]]["unpaired"] += 1
        continue
    same = e.load_hash(pa[0]) == e.load_hash(pb[0])
    sa, sb = load_size(pa[0]), load_size(pb[0])
    summ[k[0]]["identical" if same else "different"] += 1
    pairs.append({"proc": k[0], "key": k[1], "a": pa[0], "b": pb[0], "same": same, "size_a": sa, "size_b": sb})
for proc, c in sorted(summ.items()):
    diff = [p for p in pairs if p["proc"] == proc and not p["same"]]
    big = max(pairs, key=lambda p: p["size_b"] if p["proc"] == proc else -1)
    print(f"{proc:26s} {dict(c)}; largest A {max(p['size_a'] for p in pairs if p['proc'] == proc)} B "
          f"{max(p['size_b'] for p in pairs if p['proc'] == proc)} B" +
          (f"; changed images B - A: {min(p['size_b'] - p['size_a'] for p in diff):+d} to "
           f"{max(p['size_b'] - p['size_a'] for p in diff):+d} B" if diff else ""))
if a.json:
    json.dump(pairs, open(a.json, "w"), indent=1)
