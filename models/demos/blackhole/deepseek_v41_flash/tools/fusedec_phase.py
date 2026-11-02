"""python3 -I fusedec_phase.py <tab.json> <phase> [trace] -- every op group of one phase (us on the mean chip), sorted"""
import json
import sys

res = json.load(open(sys.argv[1]))
ph = sys.argv[2]
tr = sys.argv[3] if len(sys.argv) > 3 else "M"
subs = res[tr]["phase_subs"][ph]
tot = sum(subs.values())
print(f"{ph}: {tot/1e3:.1f} us, {len(subs)} groups")
for k, v in sorted(subs.items(), key=lambda kv: -kv[1]):
    print(f"{v/1e3:7.1f}  {k[:150]}")
