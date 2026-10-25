import json, re, subprocess, sys, glob
RE = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-readelf"
cmds = json.load(open("/home/aagarwal/tt-metal/fabric_exp/consolidation/offline/cmds.json"))["O2"]
vc1 = sorted(set(open("/home/aagarwal/tt-metal/fabric_exp/consolidation/offline/vc1_hashes.txt").read().split()))
def load(e):
    try:
        out = subprocess.run([RE, "-lW", e], capture_output=True, text=True).stdout
    except Exception:
        return 0
    return sum(int(l.split()[4], 16) for l in out.splitlines() if l.strip().startswith("LOAD"))
builds = sys.argv[1:]
rows = []
for h in vc1:
    for p in ("active_erisc", "subordinate_active_erisc"):
        if p not in cmds.get(h, {}): continue
        c = cmds[h][p].get("compile", "")
        tag = ("xover" if "CROSSOVER" in c else "") + ("svc" if "VC1_SERVICED" in c else "")
        sz = [load(f"{b}/{h}/{p}/{p}.elf") for b in builds]
        rows.append((p[:3], tag, h, sz))
import collections
for p in ("act", "sub"):
    for tag in sorted(set(r[1] for r in rows if r[0] == p)):
        rs = [r for r in rows if r[0] == p and r[1] == tag]
        mx = [max(r[3][i] for r in rs) for i in range(len(builds))]
        print(p, tag or "-", len(rs), "max", mx)
