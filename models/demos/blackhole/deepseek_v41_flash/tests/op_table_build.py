"""Offline: merge <dir>/raw.json (recorded calls + id ranges) with <dir>/eager_cpp.csv / trace_cpp.csv (device profiler digest)
into a markdown per-op table.   python op_table_build.py <dir> [out.md]"""
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

d = sys.argv[1]
out_md = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "table.md")
raw = json.load(open(os.path.join(d, "raw.json")))
L = raw["layer"]


def load(path):
    rows = list(csv.DictReader(open(path)))
    return rows


def f(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return None


K = "DEVICE KERNEL DURATION [ns]"
eager = load(os.path.join(d, "eager_cpp.csv"))
# id -> device -> (kernel ns, cores)
per = defaultdict(dict)
for r in eager:
    if r.get("METAL TRACE ID"):
        continue
    gid = int(float(r["GLOBAL CALL COUNT"]))
    if raw["op0"] <= gid < raw["op1"] and f(r, K) is not None:
        per[gid][r["DEVICE ID"]] = (f(r, K), int(float(r["CORE COUNT"] or 0)))
devs = sorted({dv for v in per.values() for dv in v})
ref_dev = devs[0] if devs else None
print("devices", len(devs), "ids with data", len(per), "of", raw["op1"] - raw["op0"])


def stats(gid):
    v = per.get(gid)
    if not v:
        return None
    ks = [x[0] for x in v.values()]
    return dict(med=statistics.median(ks), mx=max(ks), mn=min(ks), cores=max(x[1] for x in v.values()))


sec_of = {}
marks = raw["eager_marks"]  # (name, id_at_end, nrows_at_end)
start = 0
for name, _, nrows in marks:
    for i in range(start, nrows):
        sec_of[i] = name
    start = nrows

calls = []
covered = set()
for r in raw["eager_rows"]:
    ids = list(range(r["id0"], r["id1"]))
    covered.update(ids)
    st = [stats(i) for i in ids]
    calls.append(dict(r, ids=ids, st=st, sec=sec_of.get(r["idx"], "?")))
untracked = [i for i in range(raw["op0"], raw["op1"]) if i not in covered]
tot_med = sum(s["med"] for c in calls for s in c["st"] if s)
tot_max = sum(s["mx"] for c in calls for s in c["st"] if s)
missing = sum(1 for c in calls for s in c["st"] if not s)

lines = []
P = lines.append
P(f"# Layer {L} per-op table (decode, batch {raw['B']}, 4x8 BH)\n")
P(
    f"Source: ttnn call recorder + device profiler digest (eager forward, kernel duration per program). {len(devs)} devices; "
    f"'med' = median over devices, 'max' = slowest device. Trace-replayed layer time: {raw['traced_ms']:.3f} ms"
    f" (profiler {'on' if raw['profiler'] else 'off'}).\n"
)
P(
    f"Calls with device programs: {len(calls)}; programs: {raw['op1'] - raw['op0']} (untracked ids {len(untracked)}, programs without profiler data {missing})."
)
P(
    f"Sum of per-program kernel time: median-device {tot_med / 1e3:.1f} us, slowest-device-per-op {tot_max / 1e3:.1f} us.\n"
)
P("| # | section | op | in -> out | kernel | progs | cores | med us | max us | % of layer |")
P("|---|---|---|---|---|---|---|---|---|---|")
sec_tot = defaultdict(lambda: [0.0, 0, 0])
for c in calls:
    m = sum(s["med"] for s in c["st"] if s)
    x = sum(s["mx"] for s in c["st"] if s)
    cores = max([s["cores"] for s in c["st"] if s] or [0])
    sec_tot[c["sec"]][0] += m
    sec_tot[c["sec"]][1] += 1
    sec_tot[c["sec"]][2] += len(c["ids"])
    ins = "; ".join(c["ins"][:4]) + ("..." if len(c["ins"]) > 4 else "")
    outs = "; ".join(c["outs"][:3])
    kw = " ".join(c["kw"][:5])
    name = c["op"].replace("ttnn.", "") + (f" ({kw})" if kw else "")
    P(
        f"| {c['idx']} | {c['sec']} | {name} | {ins} -> {outs} | {c['kernels'] or c['caller']} | {len(c['ids'])} | {cores} | {m / 1e3:.1f} | {x / 1e3:.1f} | {100 * m / tot_med:.1f}% |"
    )
P("\n## Section subtotals (median-device kernel time)\n")
P("| section | calls | programs | us | % |")
P("|---|---|---|---|---|")
for s, (t, n, p) in sec_tot.items():
    P(f"| {s} | {n} | {p} | {t / 1e3:.1f} | {100 * t / tot_med:.1f}% |")
P("\n## Calls that launch no device program (views/host-only), by caller\n")
for k, v in sorted(raw["eager_zero"].items(), key=lambda kv: -kv[1])[:40]:
    P(f"- {k}: {v}")
open(out_md, "w").write("\n".join(lines))
json.dump(
    dict(
        calls=[
            {k: v for k, v in c.items() if k != "st"}
            | dict(med=[s["med"] if s else None for s in c["st"]], mx=[s["mx"] if s else None for s in c["st"]])
            for c in calls
        ]
    ),
    open(os.path.join(d, "table.json"), "w"),
)
print("wrote", out_md)
