import json
import statistics as st
import sys

tag = sys.argv[1]
lo, hi = int(sys.argv[2]), int(sys.argv[3])
ops = json.load(open(f"logs/{tag}_ops.json"))["rows"]
name = {}
for x in ops:
    for o in range(x["id0"], x["id1"]):
        name[o] = (x["op"].split(".")[-1][:20], x["caller"].split(":")[0][-18:])
go = json.load(open(f"data/{tag}_go.json"))
raw = json.load(open(f"data/{tag}_raw.json"))
print("op name caller cores | kernel(mean) kmin | A(done->go) B(go->K0) gap")
tk = tg = tA = tB = 0
stall = 0
ns = 0
out = []
for op in range(lo, hi + 1):
    L = raw[str(op)]
    k = st.mean(r["kdur"] for r in L)
    kmin = min(r["kdur"] for r in L)
    n = round(st.mean(r["n"] for r in L))
    a, amin, amax, b, gap, nc = go[str(op)]
    print(op, name[op][0], name[op][1], n, "| %.1f %.1f | %.2f %.2f %.2f" % (k, kmin, a, b, gap))
    tk += k
    tg += gap
    tA += a
    tB += b
    if a > 1.5:
        stall += a
        ns += 1
print(
    "TOTAL ops %d kernel %.1f gap %.1f (A %.1f B %.1f) stalled n=%d A=%.1f ; wall %.1f"
    % (hi - lo + 1, tk, tg, tA, tB, ns, stall, tk + tg)
)
