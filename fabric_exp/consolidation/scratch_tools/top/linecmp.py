import sys
sys.path.insert(0, sys.argv[1])
from linemap import parse, fn_ranges
from collections import Counter
h = sys.argv[2]
V = sys.argv[3].split(",")
fn = sys.argv[4] if len(sys.argv) > 4 else "try_send"
rows = {}
for v in V:
    src = f"fabric_exp/variants/{v}/fabric_erisc_router_v2.cpp"
    lines = open(src).read().split("\n")
    a, b = fn_ranges(src)[fn]
    ins = parse(f"fabric_exp/variants/{v}/elf/{h}/active_erisc.elf")
    c = Counter(x[3][1] for x in ins if x[3][0] == "fabric_erisc_router_v2.cpp" and a <= x[3][1] <= b)
    for ln, n in c.items():
        rows.setdefault(lines[ln - 1].strip()[:90], {})[v] = rows.get(lines[ln - 1].strip()[:90], {}).get(v, 0) + n
for t, d in rows.items():
    if len(set(d.get(v, 0) for v in V)) > 1:
        print("  ".join(f"{d.get(v, 0):3d}" for v in V), "|", t)
