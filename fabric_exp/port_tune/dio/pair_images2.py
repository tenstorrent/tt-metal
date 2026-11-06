#!/usr/bin/env python3
"""pair_images2.py CACHE_A CACHE_B [--ignore-args REGEX] [--json OUT]: pair_images.py with named compile-time args
matching REGEX left out of the pairing key (e.g. the streaming profiler's LINK_SYNC_* args, which only one arm has);
per pair, loadable-byte identity and loadable sizes, and per processor a summary."""
import argparse, collections, glob, hashlib, json, os, re, struct, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/port_tune")
import elf_load_cmp as e


def load_size(p):
    b = open(p, "rb").read()
    phoff, = struct.unpack_from("<I", b, 28); pe, pn = struct.unpack_from("<HH", b, 42)
    segs = [struct.unpack_from("<8I", b, phoff + i * pe) for i in range(pn)]
    return sum(s[4] for s in segs if s[0] == 1)


def builds(cache, ignore):
    out = collections.defaultdict(list)
    for d in glob.glob(f"{cache}/*/kernels/fabric_erisc_router/*/"):
        m = os.path.join(d, "named_ct_arg_map_generated.h")
        if not os.path.exists(m):
            continue
        args = re.findall(r'\{"([A-Z_0-9]+)",(-?\d+)\}', open(m).read())
        key = hashlib.md5(repr(sorted(a for a in args if not (ignore and re.match(ignore, a[0])))).encode()).hexdigest()[:12]
        role = dict(args).get("LINK_SYNC_ROLE")
        for proc in ("active_erisc", "subordinate_active_erisc"):
            elfs = [p for p in glob.glob(os.path.join(d, proc, "*.elf")) if not p.endswith(".xip.elf")]
            if elfs:
                out[(proc, key)].append((elfs[0], role))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--ignore-args", default=""); ap.add_argument("--json", default="")
a = ap.parse_args()
A, B = builds(a.a, a.ignore_args), builds(a.b, a.ignore_args)
pairs, summ = [], collections.defaultdict(collections.Counter)
for k in sorted(set(A) | set(B)):
    if k not in A or k not in B:
        summ[k[0]]["unpaired"] += 1
        continue
    (fa, ra), (fb, rb) = A[k][0], B[k][0]
    same = e.load_hash(fa) == e.load_hash(fb)
    sa, sb = load_size(fa), load_size(fb)
    summ[k[0]]["identical" if same else "different"] += 1
    pairs.append({"proc": k[0], "key": k[1], "a": fa, "b": fb, "same": same, "size_a": sa, "size_b": sb,
                  "role_a": ra, "role_b": rb})
for proc, c in sorted(summ.items()):
    pp = [p for p in pairs if p["proc"] == proc]
    diff = [p for p in pp if not p["same"]]
    roles = collections.Counter((p["role_b"]) for p in pp)
    print(f"{proc:26s} {dict(c)}; largest A {max(p['size_a'] for p in pp)} B {max(p['size_b'] for p in pp)} B"
          + (f"; changed B - A {min(p['size_b'] - p['size_a'] for p in diff):+d} to {max(p['size_b'] - p['size_a'] for p in diff):+d} B" if diff else "")
          + f"; B LINK_SYNC_ROLE counts {dict(roles)}")
if a.json:
    json.dump(pairs, open(a.json, "w"), indent=1)
