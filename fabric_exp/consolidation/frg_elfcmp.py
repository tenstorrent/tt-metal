#!/usr/bin/env python3
"""frg_elfcmp.py DIR_A DIR_B: byte-identity of the fabric router ELFs two mock (or device) runs compiled, for the
FULLRING_MCAST_GAP.md variants (v4rxg3bd at delays 0, v4rxg3sl with its flag off) against v4rxg3. The live-tree offline
build (coverage.py) is not used: other jobs swap variants into the live tree, so its headers race.

DIR_A / DIR_B: directories searched recursively for kernels/fabric_erisc_router_v2/<hash>/<proc>/<proc>.elf (e.g. a
bd_mock_vet.sh SCR, whose galaxy/cache_<test> dirs hold one cache per test). Builds are paired by (test cache name,
proc, the named CT arg map, the chlkc descriptors), which fixes the config; the ELFs are compared on every loadable
section (bnd_loadcmp.loadable: SHF_ALLOC sections by name, address and bytes; NOBITS by size).
Prints pairs same / differ / unpaired and the first differing configs."""
import collections
import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bnd_loadcmp import loadable  # noqa: E402


def builds(root):
    out = {}
    for elf in glob.glob(f"{root}/**/kernels/fabric_erisc_router_v2/*/*/*.elf", recursive=True):
        if elf.endswith(".xip.elf"):
            continue
        proc_dir = os.path.dirname(elf)
        bdir = os.path.dirname(proc_dir)
        cache = next((p for p in elf.split("/") if p.startswith("cache_")), "")
        h = hashlib.md5()
        for f in ("named_ct_arg_map_generated.h", "chlkc_descriptors.h"):
            p = os.path.join(bdir, f)
            if os.path.exists(p):
                h.update(open(p, "rb").read())
        out[(cache, os.path.basename(proc_dir), h.hexdigest())] = elf
    return out


def main():
    a, b = builds(sys.argv[1]), builds(sys.argv[2])
    keys = sorted(set(a) | set(b))
    res = collections.Counter()
    diff = []
    for k in keys:
        if k not in a or k not in b:
            res["unpaired"] += 1
            continue
        if loadable(a[k]) == loadable(b[k]):
            res["same"] += 1
        else:
            res["differ"] += 1
            diff.append(k)
    per = collections.Counter((k[0], k in a and k in b and k not in diff) for k in keys)
    print(f"{sys.argv[1]} vs {sys.argv[2]}: {len(a)} / {len(b)} ELFs; same {res['same']}, differ {res['differ']}, "
          f"unpaired {res['unpaired']}")
    for (cache, ok), n in sorted(per.items()):
        print(f"   {cache}: {'same' if ok else 'differ/unpaired'} {n}")
    for k in diff[:5]:
        print("   differs:", k[0], k[1], a[k])


if __name__ == "__main__":
    main()
