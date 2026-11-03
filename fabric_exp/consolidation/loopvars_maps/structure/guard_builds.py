#!/usr/bin/env python3
"""guard_builds.py BASE NEW: the register-map guard between two coverage builds (cov_m builds/<name>): for every router
image whose .text differs, loopvars' loop analysis of both (each role the image runs), and the changes in loop size,
stack stores / loads and spilled values. Images with equal .text have equal maps by construction (names aside)."""
import collections, glob, hashlib, multiprocessing, os, subprocess, sys
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import loopvars as L
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"


def text(p):
    return subprocess.run([L.B + "objcopy", "-O", "binary", "-j", ".text", p, "/dev/stdout"], capture_output=True).stdout


def job(j):
    rel, role = j
    a = L.analyse(f"{CB}/{sys.argv[1]}/{rel}", role)
    b = L.analyse(f"{CB}/{sys.argv[2]}/{rel}", role)
    return rel, role, a, b


if __name__ == "__main__":
    base, new = sys.argv[1], sys.argv[2]
    jobs = []
    for p in sorted(glob.glob(f"{CB}/{new}/*/*/*.elf")):
        rel = os.path.relpath(p, f"{CB}/{new}")
        q = f"{CB}/{base}/{rel}"
        if os.path.exists(q) and text(p) != text(q):
            for role in L.roles_of(p):
                jobs.append((rel, role))
    with multiprocessing.Pool(16) as pool:
        res = pool.map(job, jobs)
    fam = collections.defaultdict(collections.Counter)
    for rel, role, a, b in res:
        f = rel.split("__")[0] + " " + role
        if "error" in a or "error" in b:
            fam[f]["error"] += 1
            continue
        d = (b["loop_insns"] - a["loop_insns"], b["accesses"]["stores"] - a["accesses"]["stores"],
             b["accesses"]["loads"] - a["accesses"]["loads"])
        extra = sorted(set(b["stored"]) - set(a["stored"]))
        gone = sorted(set(a["stored"]) - set(b["stored"]))
        fam[f][f"loop {d[0]:+d} sp S{d[1]:+d} L{d[2]:+d}" + (f" +{extra}" if extra else "") + (f" -{gone}" if gone else "")] += 1
    print(f"{base} -> {new}: {len(set(r for r, _ in jobs))} images with different .text, {len(jobs)} role analyses")
    for f in sorted(fam):
        for k, n in fam[f].most_common():
            print(f"  {f:40s} {n:4d}  {k}")
