#!/usr/bin/env python3
"""w_loops.py BASE ARM...: for every E1 image whose loadable contents differ between BASE and the first ARM (cov_m
builds), the receiver loop (loopvars: instructions, stack loads / stores, and loads / stores through other registers)
in BASE and each ARM; prints the per-case deltas against BASE. Study analysis only (§6.12)."""
import collections, glob, multiprocessing, os, re, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import bnd_loadcmp as LC  # noqa: E402
import loopvars as L  # noqa: E402

CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
MEM = re.compile(r"\w+,(-?\d+)\((\w+)\)")


def stats(elf):
    try:
        a = L.analyse(elf, "receiver", detail=True)
    except BaseException as e:  # loopvars exits on images with no receiver loop
        return None
    loop = set(a["loop_addrs"])
    c = collections.Counter(n=a["loop_insns"])
    for addr, op, raw in a["body"]:
        m = MEM.fullmatch(raw)
        if addr in loop and m and (op in L.LOADS or op in L.STORES):
            c[("s" if m.group(2) == "sp" else "m") + ("L" if op in L.LOADS else "S")] += 1
    return c


def job(rel):
    out = {}
    for b in [sys.argv[1]] + sys.argv[2:]:
        out[b] = stats(f"{CB}/{b}/{rel}")
    return rel.split("__")[0], out


if __name__ == "__main__":
    base, arms = sys.argv[1], sys.argv[2:]
    rels = sorted(os.path.relpath(p, f"{CB}/{arms[0]}") for p in glob.glob(f"{CB}/{arms[0]}/*/subordinate_active_erisc/*.elf")
                  if not p.endswith(".xip.elf"))
    rels = [r for r in rels if os.path.exists(f"{CB}/{base}/{r}") and LC.loadable(f"{CB}/{base}/{r}") != LC.loadable(f"{CB}/{arms[0]}/{r}")]
    with multiprocessing.Pool(24) as pool:
        res = pool.map(job, rels)
    per_case = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for case, out in res:
        if any(v is None for v in out.values()):
            continue
        for b in arms:
            d = tuple(out[b][k] - out[base][k] for k in ("n", "sL", "sS", "mL", "mS"))
            per_case[case][b][d] += 1
    print(f"{len(res)} differing E1 images ({base} vs {arms[0]}); receiver-loop deltas against {base}: "
          "(insns, stack L, stack S, other L, other S): count")
    for case in sorted(per_case):
        print(f"  {case:24s} " + " | ".join(f"{b}: " + ", ".join(f"{d}x{n}" for d, n in per_case[case][b].most_common(3))
                                           for b in arms))
