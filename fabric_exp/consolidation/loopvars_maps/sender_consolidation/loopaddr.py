#!/usr/bin/env python3
"""loopaddr.py BASE NEW [LIST]: per image (every shared ELF, or those in LIST), the start address of each loop function
(sender::run, single::sender_pass, receiver::run, single::receiver_pass, single::run) in two cov_m builds, and whether
it moved; moves summarised per case and function, with the address delta and whether the 64 B line offset changed."""
import collections, concurrent.futures as cf, glob, os, re, subprocess, sys
NM = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-nm"
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
LOOPS = {"fabric_router::sender::run()": "sender::run", "fabric_router::single::sender_pass": "single::sender_pass",
         "fabric_router::receiver::run": "receiver::run", "fabric_router::single::receiver_pass": "single::receiver_pass",
         "fabric_router::single::run": "single::run"}


def addrs(elf):
    out = subprocess.run([NM, "-C", elf], capture_output=True, text=True).stdout
    res = {}
    for l in out.splitlines():
        p = l.split(" ", 2)
        if len(p) == 3 and p[1] in "tT":
            for k, short in LOOPS.items():
                if p[2].startswith(k):
                    res[short] = int(p[0], 16)
    return res


base, new = sys.argv[1:3]
if len(sys.argv) > 3:
    rels = open(sys.argv[3]).read().split()
else:
    rels = sorted({os.path.relpath(e, f"{CB}/{new}") for e in glob.glob(f"{CB}/{new}/*/*/*.elf")} &
                  {os.path.relpath(e, f"{CB}/{base}") for e in glob.glob(f"{CB}/{base}/*/*/*.elf")})


def one(rel):
    return rel, addrs(f"{CB}/{base}/{rel}"), addrs(f"{CB}/{new}/{rel}")


moved = collections.Counter()
same = collections.Counter()
with cf.ThreadPoolExecutor(32) as ex:
    for rel, a, b in ex.map(one, rels):
        case = rel.split("__")[0]
        for f in sorted(set(a) | set(b)):
            if a.get(f) == b.get(f):
                same[(case, f)] += 1
            else:
                d = (b.get(f, 0) - a.get(f, 0))
                line = "64B offset changed" if a.get(f, 0) % 64 != b.get(f, 0) % 64 else "same 64B offset"
                moved[(case, f, d, line)] += 1
print(f"{base} -> {new}: {len(rels)} images; loop functions at the same address: {sum(same.values())}, moved: "
      f"{sum(moved.values())}")
for (case, f, d, line), n in sorted(moved.items()):
    print(f"  {case:28s} {f:22s} {d:+6d} B  {line}  x{n}")
