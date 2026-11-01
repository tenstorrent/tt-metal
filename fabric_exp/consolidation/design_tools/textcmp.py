#!/usr/bin/env python3
"""textcmp.py A B: per case of two coverage builds, how many ELFs have byte-identical .text."""
import collections, concurrent.futures, glob, os, subprocess, sys
COV = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
OC = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy"
def text(e):
    return subprocess.run([OC, "-O", "binary", "--only-section=.text", e, "/dev/stdout"], capture_output=True).stdout
a, b = sys.argv[1:3]
pairs = []
for e in glob.glob(f"{COV}/{a}/*/*/*.elf"):
    rel = os.path.relpath(e, f"{COV}/{a}")
    if os.path.exists(f"{COV}/{b}/{rel}"):
        pairs.append((rel.split("__")[0], e, f"{COV}/{b}/{rel}"))
with concurrent.futures.ThreadPoolExecutor(32) as ex:
    res = list(ex.map(lambda p: (p[0], text(p[1]) == text(p[2])), pairs))
c = collections.defaultdict(lambda: [0, 0])
for case, same in res:
    c[case][0 if same else 1] += 1
for case in sorted(c):
    print(f"{case:28s} same {c[case][0]:4d}  differ {c[case][1]:4d}")
