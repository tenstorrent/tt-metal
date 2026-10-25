#!/usr/bin/env python3
# sizes.py BUILD [BUILD...]: largest E0 and E1 loadable bytes (sum of LOAD filesz) over the VC1 configs, and their sum.
import glob, os, subprocess, sys
RE = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-readelf"
vc1 = set(open("/home/aagarwal/tt-metal/fabric_exp/consolidation/offline/vc1_hashes.txt").read().split())
def load(e):
    out = subprocess.run([RE, "-lW", e], capture_output=True, text=True).stdout
    return sum(int(l.split()[4], 16) for l in out.splitlines() if l.strip().startswith("LOAD"))
for d in sys.argv[1:]:
    m = {}
    for e in glob.glob(f"{d}/*/*/*.elf"):
        h = e.split("/")[-3]; p = e.split("/")[-2]
        if h in vc1:
            m.setdefault(p, []).append((load(e), h))
    e0 = max(m.get("active_erisc", [(0, "")])); e1 = max(m.get("subordinate_active_erisc", [(0, "")]))
    print(f"{d}: E0 {e0[0]} ({e0[1]}), E1 {e1[0]} ({e1[1]}), sum {e0[0] + e1[0]} (limit 26624: {'fits' if e0[0] + e1[0] <= 26624 else 'over by ' + str(e0[0] + e1[0] - 26624)})")
