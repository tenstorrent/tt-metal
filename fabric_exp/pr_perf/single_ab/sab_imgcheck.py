#!/usr/bin/env python3
"""sab_imgcheck.py RESULTS_DIR: the router images the device job built (single_ab_runs.sh pass a,
results/sab_routers_<arm>.tar.zst) against each other and against the offline mock builds (offline.sh; local, never
committed). Builds are paired by (ERISC, generated headers with the runtime root replaced); images are compared by
loadable bytes (port_tune/cov/cov.py image()).
  device A vs device B        every pair should differ on one ERISC (two-ERISC builds: none in these suites)
  device A vs offline A, device B vs offline B
                              the pairs found (configs both built) should be identical: the device ran the images the
                              offline gate checked"""
import collections, glob, hashlib, importlib.util, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TT_COV_TREE", "/home/aagarwal/tt-metal-single-ab")
os.environ.setdefault("TT_COV_DIR", "/home/aagarwal/.tt-metal-sab-A-cov")
sys.argv, args = ["cov"], sys.argv[1:]
spec = importlib.util.spec_from_file_location("cov", os.path.join(HERE, "..", "..", "port_tune", "cov", "cov.py"))
cov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cov)
ROOTS = {"A": "/home/aagarwal/tt-metal-single-ab", "B": "/home/aagarwal/tt-metal-single-in",
         "main": "/home/aagarwal/tt-metal-main"}
COVS = {"A": "/home/aagarwal/.tt-metal-sab-A-cov", "B": "/home/aagarwal/.tt-metal-sab-B-cov"}


def sigs(elfs, roots):
    out = collections.defaultdict(list)
    for e in elfs:
        b = os.path.dirname(os.path.dirname(e))
        gen = b"".join(open(f, "rb").read() for f in sorted(glob.glob(b + "/*.h") + glob.glob(b + "/*.hpp")))
        for r in roots:
            gen = gen.replace(r.encode(), b"<ROOT>")
        out[(os.path.basename(os.path.dirname(e)), hashlib.md5(gen).hexdigest())].append(e)
    return out


def elfs_in(d):
    return [e for e in glob.glob(d + "/**/kernels/fabric_erisc_router*/*/*/*.elf", recursive=True)
            if not e.endswith(".xip.elf")]


def cmp(name, a, b):
    shared = sorted(set(a) & set(b))
    same = sum(1 for k in shared if {cov.image(e) for e in a[k]} == {cov.image(e) for e in b[k]}
               and len({cov.image(e) for e in a[k]}) == 1)
    procs = collections.Counter(k[0] for k in shared)
    print(f"{name:34} paired {len(shared):4d} ({dict(procs)}): identical {same}, differ {len(shared) - same}; "
          f"unpaired {len(set(a) - set(b))} / {len(set(b) - set(a))}")


def main():
    res = args[0]
    tmp = tempfile.mkdtemp(prefix="sab_img_", dir=os.environ.get("TMPDIR"))
    dev = {}
    for arm in ("main", "A", "B"):
        t = os.path.join(res, f"sab_routers_{arm}.tar.zst")
        if not os.path.exists(t):
            print(f"{arm}: no {t}")
            continue
        d = os.path.join(tmp, arm)
        os.makedirs(d)
        subprocess.run(["tar", "-I", "zstd", "-xf", t, "-C", d], check=True)
        dev[arm] = sigs(elfs_in(d), [ROOTS[arm] + "/", ROOTS[arm]])
        print(f"device {arm}: {sum(map(len, dev[arm].values()))} router ELFs, {len(dev[arm])} distinct configs")
    off = {arm: sigs(elfs_in(c + "/cases"), [c + "/root/", c + "/root"]) for arm, c in COVS.items()}
    for arm in off:
        print(f"offline {arm}: {sum(map(len, off[arm].values()))} router ELFs, {len(off[arm])} distinct configs")
    if "A" in dev and "B" in dev:
        cmp("device A vs device B", dev["A"], dev["B"])
    for arm in ("A", "B"):
        if arm in dev:
            cmp(f"device {arm} vs offline {arm}", dev[arm], off[arm])
    if "A" in dev:
        cmp("device A vs offline B (should differ)", dev["A"], off["B"])
    subprocess.run(["rm", "-rf", tmp])


if __name__ == "__main__":
    main()
