#!/usr/bin/env python3
"""covcmp_rebuild_subst.py CASE OUT [OLD=NEW ...]: G6 check of run-order artifacts. Rebuilds every logged kernel build of
CASE in the A cov (TT_COV_DIR/TT_COV_TREE) with (1) its streaming-profiler TT_PROFILER_TU_ID remapped to the id the B
cov's run assigned the same source+target (the registry hands ids out first-come across parallel builds), and (2) each
OLD=NEW token substitution in the compile-time args (a run's random seed), then compares the per-kernel image multiset
with the JIT's own ELFs of the same case in the B cov (CMP_B)."""
import collections, hashlib, importlib.util, json, os, re, sys
case, out, subs = sys.argv[1], sys.argv[2], [a.split("=", 1) for a in sys.argv[3:]]
B = os.environ["CMP_B"]
_argv, sys.argv = sys.argv, ["cov"]
spec = importlib.util.spec_from_file_location("cov", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cov.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def registry(cov):
    p = os.path.join(cov, "cases", case, "cache", "tt-metal-cache.observability_event_tu_ids")
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p):
        sid, i = line.rstrip("\n").rsplit("\t", 1)
        out[re.sub(r"/home/aagarwal/[^/]*-cov/", "COV/", sid)] = i
    return out
ra, rb = registry(m.COV), registry(B)
tumap = {ra[s]: rb[s] for s in ra if s in rb}
print("tu id map A->B:", {k: v for k, v in tumap.items() if k != v})
cmds = json.load(open(os.path.join(m.COV, "cases", case, "cmds.json")))
got = collections.Counter()
import concurrent.futures
def one(item):
    key, c = item
    c = dict(c)
    new = []
    for cmd in c.get("compile", []):
        cmd = re.sub(r"-DTT_PROFILER_TU_ID=(\d+)", lambda x: "-DTT_PROFILER_TU_ID=" + tumap.get(x.group(1), x.group(1)), cmd)
        for old, nw in subs:
            cmd = cmd.replace(old, nw)
        new.append(cmd)
    c["compile"] = new
    wd, err = m.build_one_(case, key, c, out)
    if err:
        return key, None, err
    elf = [f for f in os.listdir(wd) if f.endswith(".elf")][0]
    return key, hashlib.md5(m.image(os.path.join(wd, elf))).hexdigest(), None
kname = lambda key: key.split("kernels/")[1].split("/")[0] if "kernels/" in key else key.split("/")[0]
with concurrent.futures.ThreadPoolExecutor(16) as ex:
    for key, dig, err in ex.map(one, [kv for kv in cmds.items() if "kernels/" in kv[0]]):
        if err:
            print("FAILED", key, err[:400])
        else:
            got[(kname(key), dig)] += 1
# B: the JIT's own ELFs
exp = collections.Counter()
for dp, _, fs in os.walk(os.path.join(B, "cases", case, "cache")):
    if "/kernels/" in dp + "/":
        for f in fs:
            if f == os.path.basename(dp) + ".elf":
                exp[(dp.split("/kernels/")[1].split("/")[0], hashlib.md5(m.image(os.path.join(dp, f))).hexdigest())] += 1
for k in sorted({k for k, _ in exp} | {k for k, _ in got}):
    a = collections.Counter({d: n for (kk, d), n in got.items() if kk == k})
    b = collections.Counter({d: n for (kk, d), n in exp.items() if kk == k})
    print(f"{case} {k:28s} rebuilt {sum(a.values())} vs B {sum(b.values())}: "
          + ("IDENTICAL" if a == b else f"only rebuilt {sum((a - b).values())}, only B {sum((b - a).values())}"))
