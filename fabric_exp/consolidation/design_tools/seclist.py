import sys, re
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import pcs_sections as ps
path, pat, thr = sys.argv[1], sys.argv[2], float(sys.argv[3])
lines = open(path).read().splitlines(); twin = lines[0].split()[1]
rows = [(int(l[:8], 16), float(l.split()[1]), l) for l in lines[1:]]
loc = ps.a2l(twin, [a for a, _, _ in rows]); tot = 0
for a, c, l in rows:
    fns = " ".join(f for f, _, _ in loc.get(a, []))
    if re.search(pat, fns):
        tot += c
        if c >= thr:
            chain = " < ".join(re.sub(r"\(.*", "", f).split("::")[-1][:28] for f, _, _ in loc.get(a, [])[:4])
            print(f"  {l[:100]}  [{chain}]")
print(f"  total {tot:.1f}")
