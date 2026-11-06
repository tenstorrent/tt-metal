#!/usr/bin/env python3
"""covcmp_sig.py TREE_A COV_A TREE_B COV_B CASE...: compare two coverage runs' JIT router (and other kernel) images
when they were mocked under different runtime roots, so the JIT's hash directories differ. Builds are matched by a
signature: the compile command with the runtime root, the cache path and the hash directory replaced by tokens, plus
the contents of the build's generated headers (named compile args, defines). Matched builds are compared by loadable
bytes (cov.image). A build with no match on the other side means its inputs differ (host-side args or defines).
ROUTER_ONLY=1 limits to fabric_erisc_router builds."""
import collections, glob, hashlib, importlib.util, json, os, re, sys

ARGS = sys.argv[1:]


def load(tree, cov):
    os.environ["TT_COV_TREE"], os.environ["TT_COV_DIR"] = tree, cov
    sys.argv = ["cov"]
    spec = importlib.util.spec_from_file_location("cov_" + hashlib.md5(cov.encode()).hexdigest()[:8],
                                                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "cov.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def builds(m, case):
    db = json.load(open(os.path.join(m.COV, "cases", case, "cmds.json")))
    out = collections.defaultdict(list)
    cache = os.path.join(m.COV, "cases", case, "cache")
    for key, cmds in db.items():
        if os.environ.get("ROUTER_ONLY") and "/fabric_erisc_router/" not in key:
            continue
        mm = re.search(r"\s-o\s+(\S+\.elf)", cmds["link"][0])
        if not mm:
            continue
        elf = os.path.join(os.path.dirname(mm.group(1)), os.path.basename(mm.group(1)).split(".")[0] + ".elf")
        if not os.path.exists(elf):
            continue
        c = cmds["compile"][0]
        c = c.replace(m.ROOT, "<ROOT>").replace(m.TREE, "<TREE>")
        c = re.sub(r"\S*/kernels/([^/\s]+)/\d+/", r"<CACHE>/kernels/\1/<H>/", c)
        c = re.sub(r"\S*/firmware/", "<CACHE>/firmware/", c)
        c = re.sub(r"\S*/pch/[0-9a-f]+/", "<PCH>/", c)
        c = re.sub(r'(-DFULL_KERNEL_NAME="[^/"]+/)\d+/', r"\1<H>/", c)
        c = re.sub(r"-o \S+", "-o <O>", c)
        c = re.sub(r"-MF \S+", "-MF <D>", c)
        bdir = os.path.dirname(os.path.dirname(elf))  # kernels/<name>/<hash>
        gen = b"".join(open(f, "rb").read() for f in sorted(glob.glob(bdir + "/*.h") + glob.glob(bdir + "/*.hpp")))
        gen = gen.replace(m.ROOT.encode(), b"<ROOT>")
        sig = hashlib.md5(c.encode() + gen).hexdigest()
        out[(key.split("/")[1] if key.startswith("kernels/") else key, key.split("/")[-1], sig)].append(elf)
    return out


a = load(ARGS[0], ARGS[1])
b = load(ARGS[2], ARGS[3])
tot = collections.Counter()
for case in ARGS[4:]:
    try:
        ba, bb = builds(a, case), builds(b, case)
    except FileNotFoundError as e:
        print(f"{case:24s} MISSING ({e.filename})")
        tot["missing"] += 1
        continue
    shared = sorted(set(ba) & set(bb))
    differ = [s for s in shared if a.image(ba[s][0]) != b.image(bb[s][0])]
    oa, ob = set(ba) - set(bb), set(bb) - set(ba)
    kinds = collections.Counter(s[0] for s in differ)
    status = "IDENTICAL" if not differ and not oa and not ob else "DIFFER"
    tot[status] += 1
    tot["builds"] += len(shared)
    tot["differ"] += len(differ)
    print(f"{case:24s} {status:9s} matched {len(shared):4d}, differ {len(differ):3d}, unmatched {len(oa)}/{len(ob)}"
          + (f"  differ by kernel: {dict(kinds)}" if kinds else "")
          + (f"  unmatched kernels: {sorted({s[0] for s in oa | ob})}" if oa or ob else ""))
print("TOTAL", dict(tot))
