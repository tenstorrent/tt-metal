"""python pf_trace_summary.py <profiler dir> [session id]: traced-replay op device time vs op-to-op gaps from cpp_device_perf_report.csv.
Per device: sum of DEVICE KERNEL DURATION (kernel time), sum of OP TO OP LATENCY (gaps), op count; per OP NAME top table (mean over devices)."""
import csv
import sys
from collections import defaultdict

d = sys.argv[1]
K, G = "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"
rows = [r for r in csv.DictReader(open(d + "/.logs/cpp_device_perf_report.csv")) if r["METAL TRACE ID"]]
sess = sorted({(r["METAL TRACE ID"], r["METAL TRACE REPLAY SESSION ID"]) for r in rows})
print("trace sessions", sess)
sid = (
    sys.argv[2]
    if len(sys.argv) > 2
    else max(s[1] for s in sess if sum(1 for r in rows if r["METAL TRACE REPLAY SESSION ID"] == s[1]) > 10000)
)
rows = [r for r in rows if r["METAL TRACE REPLAY SESSION ID"] == sid]
dev = defaultdict(lambda: [0.0, 0.0, 0])
byname = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
for r in rows:
    k = float(r[K] or 0)
    g = float(r[G] or 0)
    x = dev[r["DEVICE ID"]]
    x[0] += k
    x[1] += g
    x[2] += 1
    b = byname[r["OP NAME"] or "?"][r["DEVICE ID"]]
    b[0] += k
    b[1] += g
    b[2] += 1
nd = len(dev)
mk = sum(v[0] for v in dev.values()) / nd / 1e6
mg = sum(v[1] for v in dev.values()) / nd / 1e6
mn = sum(v[2] for v in dev.values()) / nd
print(
    f"session {sid}: {nd} devices, ops/device {mn:.0f}, kernel sum {mk:.2f} ms, op-to-op gap sum {mg:.2f} ms (gap/(kernel+gap) = {mg / (mk + mg):.1%}); max-dev kernel {max(v[0] for v in dev.values()) / 1e6:.2f}"
)
tab = []
for n, v in byname.items():
    tab.append(
        (
            sum(x[0] for x in v.values()) / nd / 1e6,
            sum(x[1] for x in v.values()) / nd / 1e6,
            sum(x[2] for x in v.values()) / nd,
            n,
        )
    )
for k_, g_, c_, n_ in sorted(tab, reverse=True)[:40]:
    print(f"{n_:48s} kernel {k_:8.3f} ms  gap {g_:7.3f} ms  count {c_:7.1f}  avg {k_ * 1e3 / max(c_, 1):7.1f} us")
