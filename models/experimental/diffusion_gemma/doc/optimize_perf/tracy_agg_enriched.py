import csv, collections, glob, os

R = sorted(
    glob.glob("/home/zni/tt-metal-tracy/generated/profiler/reports/**/ops_perf_results*.csv", recursive=True)
    + glob.glob("/home/zni/tt-metal-tracy/generated/profiler/**/ops_perf_results*.csv", recursive=True),
    key=os.path.getmtime,
)[-1]
rows = list(csv.DictReader(open(R)))
h = rows[0]


def find(*ts):
    for c in h:
        cl = c.lower()
        if all(t in cl for t in ts):
            return c


K_OP = find("op", "code")
K_ID = find("device", "id")
K_FW = find("device", "fw", "dur")
K_KN = find("device", "kernel", "dur")
K_GAP = "OP TO OP LATENCY [ns]" if "OP TO OP LATENCY [ns]" in h else find("op to op")
K_CORE = find("core", "count")
d0 = [r for r in rows if (r.get(K_ID, "") or "").strip() in ("0", "")]


def f(r, k):
    try:
        return float(r[k])
    except:
        return 0.0


agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
for r in d0:
    op = (r[K_OP] or "").strip() or "(blank)"
    a = agg[op]
    a[0] += 1
    a[1] += f(r, K_FW)
    a[2] += f(r, K_KN)
    a[3] += f(r, K_GAP)
    a[4] += f(r, K_CORE)
tot_fw = sum(a[1] for a in agg.values())
tot_gap = sum(a[3] for a in agg.values())
tot_kn = sum(a[2] for a in agg.values())
print("report:", R.split("reports/")[-1] if "reports/" in R else R)
print(f"device0 rows={len(d0)}  op_types={len(agg)}  cols: OP={K_OP} FW={K_FW} GAP={K_GAP}")
print(
    f"SUM device_fw={tot_fw/1e6:.1f}ms  kernel={tot_kn/1e6:.1f}ms  op2op_gap={tot_gap/1e6:.1f}ms  GAP_FRACTION={tot_gap/(tot_fw+tot_gap)*100:.1f}%"
)
print(
    f"{'FW ms':>8} {'%fw':>5} {'KERN ms':>8} {'GAP ms':>8} {'n':>5} {'avgFW us':>9} {'avgGAP us':>9} {'cores':>5}  OP"
)
for op, a in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print(
        f"{a[1]/1e6:8.1f} {a[1]/tot_fw*100:5.1f} {a[2]/1e6:8.1f} {a[3]/1e6:8.1f} {a[0]:5d} {a[1]/a[0]/1e3:9.1f} {a[3]/a[0]/1e3:9.1f} {a[4]/a[0]:5.0f}  {op}"
    )
