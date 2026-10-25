import sys
sys.path.insert(0, sys.argv[1])
from linemap import parse
# print the instruction stream between the first and last instruction attributed to [a,b] of the router, per contiguous block
v, h, pat_a, pat_b = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
src = f"fabric_exp/variants/{v}/fabric_erisc_router_v2.cpp"
L = open(src).read().split("\n")
a = next(i + 1 for i, l in enumerate(L) if pat_a in l)
b = next(i + 1 for i, l in enumerate(L) if i + 1 > a and pat_b in l)
ins = parse(f"fabric_exp/variants/{v}/elf/{h}/active_erisc.elf")
hit = [i for i, x in enumerate(ins) if x[3][0] == "fabric_erisc_router_v2.cpp" and a <= x[3][1] <= b]
print(f"{v}: lines {a}-{b}, {len(hit)} insns, span idx {hit[0]}..{hit[-1]}")
lo = int(sys.argv[6], 16) if len(sys.argv) > 6 else 0
hi = int(sys.argv[7], 16) if len(sys.argv) > 7 else 1 << 32
last = None
for x in ins:
    if not (lo <= x[0] < hi):
        continue
    f, ln = x[3]
    tag = f"{f[:22]}:{ln}"
    print(f"{x[0]:06x}  {x[1]:7s} {x[2]:36s} {tag}")
