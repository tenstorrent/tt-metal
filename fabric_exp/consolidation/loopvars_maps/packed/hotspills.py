"""hotspills.py BUILD...: per coverage build and config family, the E1 images with a hot value on the stack in the
receiver loop (loopvars.stack_classes), and which values."""
import sys, glob, collections, concurrent.futures
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import loopvars as L
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
FAMS = ("bh_1d", "bh_ring", "bh_2d", "bh_torus", "bh_2d_vc2", "bh_torus_vc2", "bh_vc1")
def one(p):
    names = L.sh([L.B + "nm", p])
    if "8receiver3run" not in names:
        return p, None
    r = L.analyse(p, detail=True)
    return p, {v: c for v, c in L.stack_classes(r).items() if c == "hot"}
for b in sys.argv[1:]:
    paths = [p for p in glob.glob(f"{CB}/{b}/*/subordinate_active_erisc/subordinate_active_erisc.elf")
             if p.split(f"/{b}/")[1].split("__")[0] in FAMS]
    with concurrent.futures.ProcessPoolExecutor(16) as ex:
        res = [x for x in ex.map(one, paths) if x[1] is not None]
    fam = collections.defaultdict(list)
    for p, hot in res:
        fam[p.split(f"/{b}/")[1].split("__")[0]].append(hot)
    print("==", b)
    for f in FAMS:
        hs = fam.get(f, [])
        c = collections.Counter(v for h in hs for v in h)
        print(f"  {f:14s} {sum(1 for h in hs if h)}/{len(hs)} images with hot values on the stack: {dict(c.most_common(5))}")
