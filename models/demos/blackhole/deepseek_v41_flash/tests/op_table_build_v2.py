"""Offline: raw.json + cpp_device_perf_report.csv -> per-program table with eager kernel time, trace kernel time and trace op-to-op gap.
GLOBAL CALL COUNT in the digest = (op_id << 10) | device_id.   python op_table_build_v2.py <dir> <csv> [out.json]"""
import csv
import json
import statistics
import sys
from collections import defaultdict

d, csvp = sys.argv[1], sys.argv[2]
raw = json.load(open(d + "/raw.json"))
K, G = "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"
op0, op1 = raw["op0"], raw["op1"]
n = op1 - op0
T0 = op0 - n
ek = defaultdict(list)  # prog -> [ns per device]
tk = defaultdict(lambda: defaultdict(list))
tg = defaultdict(lambda: defaultdict(list))  # prog -> session -> [..]
cores = {}
tr_ids = None
for r in csv.DictReader(open(csvp)):
    oid = int(float(r["GLOBAL CALL COUNT"])) >> 10
    if not r["METAL TRACE ID"]:
        if op0 <= oid < op1:
            ek[oid - op0].append(float(r[K]))
            cores[oid - op0] = max(cores.get(oid - op0, 0), int(float(r["CORE COUNT"] or 0)))
    else:
        if T0 <= oid < T0 + n:
            s = r["METAL TRACE REPLAY SESSION ID"]
            tk[oid - T0][s].append(float(r[K]))
            tg[oid - T0][s].append(float(r[G]))
# keep sessions complete on all 32 devices (>=n*32 rows)
cnt = defaultdict(int)
for p in tk:
    for s, v in tk[p].items():
        cnt[s] += len(v)
good = [s for s, c in cnt.items() if c == n * 32 and s not in ("1", "2", "3")]
print("good trace sessions", good)
sec = {}
st = 0
for name, _, nrows in raw["eager_marks"]:
    for i in range(st, nrows):
        sec[i] = name
    st = nrows
rows = []
for c in raw["eager_rows"]:
    for j, pid in enumerate(range(c["id0"], c["id1"])):
        p = pid - op0
        e = ek.get(p, [])
        tkv = [x for s in good for x in tk[p][s]]
        tgv = [x for s in good for x in tg[p][s]]
        rows.append(
            dict(
                p=p,
                call=c["idx"],
                sec=sec.get(c["idx"], "?"),
                op=c["op"].replace("ttnn.", ""),
                caller=c["caller"],
                kernels=c["kernels"],
                ins=c["ins"],
                outs=c["outs"],
                kw=c["kw"],
                cores=cores.get(p, 0),
                e_med=statistics.median(e) if e else None,
                e_max=max(e) if e else None,
                t_med=statistics.median(tkv) if tkv else None,
                t_max=max(tkv) if tkv else None,
                gap_med=statistics.median(tgv) if tgv else None,
                gap_max=max(tgv) if tgv else None,
            )
        )
json.dump(
    dict(traced_ms=raw["traced_ms"], B=raw["B"], layer=raw["layer"], rows=rows),
    open(sys.argv[3] if len(sys.argv) > 3 else d + "/table2.json", "w"),
    indent=0,
)
print(
    len(rows),
    "programs; sum trace kernel med us",
    sum(r["t_med"] or 0 for r in rows) / 1e3,
    "sum gap med us",
    sum(r["gap_med"] or 0 for r in rows[1:]) / 1e3,
    "traced ms",
    raw["traced_ms"],
)
