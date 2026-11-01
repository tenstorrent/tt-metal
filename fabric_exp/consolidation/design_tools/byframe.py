import sys, re, collections
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import pcs_sections as ps
def load(path):
    lines = open(path).read().splitlines(); twin = lines[0].split()[1]
    rows = [(int(l[:8], 16), float(l.split()[1]), l) for l in lines[1:]]
    loc = ps.a2l(twin, [a for a, _, _ in rows])
    c = collections.Counter(); hot = []
    for a, v, l in rows:
        fr = loc.get(a, [])
        nm0 = fr[0][0] if fr else "?"; base = nm0.split("<")[0].split("::")[-1].split("(")[0]
        t = re.search(r"<(\d+)u", nm0); name = base + (f"<{t.group(1)}>" if t else "")
        c[name] += v
        if v >= 1.5: hot.append(f"    {l[:96]}  [{name}]")
    return c, hot
A, ha = load(sys.argv[1]); B, hb = load(sys.argv[2])
print(f"total {sum(A.values()):.1f} -> {sum(B.values()):.1f}")
for k in sorted(set(A) | set(B), key=lambda k: -abs(B[k] - A[k]))[:16]:
    print(f"  {A[k]:6.1f} -> {B[k]:6.1f}  {B[k]-A[k]:+6.1f}  {k}")
print("hot A:"); print("\n".join(ha)); print("hot B:"); print("\n".join(hb))
