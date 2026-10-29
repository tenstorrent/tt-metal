import csv, re, sys, collections
csv.field_size_limit(10**9)
path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
def dims(r, p):
    def g(k):
        v = r.get(f"{p}_{k}_PAD[LOGICAL]", "")
        m = re.match(r"\s*(\d+)\s*\[", v) or re.match(r"\s*(\d+)", v)
        return int(m.group(1)) if m else None
    return g("W"), g("Z"), g("Y"), g("X")
agg = collections.defaultdict(list)
for r in rows:
    if r["DEVICE ID"] != "0" or "Matmul" not in r["OP CODE"]:
        continue
    w0,z0,y0,x0 = dims(r, "INPUT_0"); w1,z1,y1,x1 = dims(r, "INPUT_1"); ow,oz,oy,ox = dims(r, "OUTPUT_0")
    if None in (y0,x0,y1,x1,oy,ox): continue
    k = x0 if x0 in (y1, x1) else y0
    m, n = oy, ox
    batch = (w0 or 1)*(z0 or 1)
    fl = 2*m*n*k*batch
    ns = float(r["DEVICE KERNEL DURATION [ns]"])
    if ns <= 0: continue
    key = (m, n, k, batch, int(r["CORE COUNT"]), r["MATH FIDELITY"])
    agg[key].append(fl/ns/1e3)  # TFLOP/s
items = sorted(agg.items(), key=lambda kv: -kv[0][0]*kv[0][1]*kv[0][2]*kv[0][3])
print("M N K batch cores fidelity  count  TFLOP/s(median)  %of148.5")
import statistics
for (m,n,k,b,c,f), v in items[:25]:
    med = statistics.median(v)
    print(f"{m:6d} {n:6d} {k:6d} {b:3d} {c:4d} {f:14s} {len(v):4d} {med:8.1f} {100*med/148.5:6.1f}%")
