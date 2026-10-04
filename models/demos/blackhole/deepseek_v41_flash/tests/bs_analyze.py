"""Offline analysis of one test_batch_scaling.py output dir:  python bs_analyze.py <dir>  -> <dir>/analysis.json (+ printed summary)
 - routes.pt: distinct experts (global / per device e//12 / max per device / tokens-per-device) per MoE layer, mean over eager steps
 - layer_<L>/raw.json + cpp.csv: per-group trace kernel time (median over devices, slowest device) of that layer."""
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

import torch

d = sys.argv[1]
out = {}
EPD = 12  # experts per device (384 / 32), expert e lives on device e // 12
CSV = (
    os.path.join(d, "cpp.csv")
    if os.path.exists(os.path.join(d, "cpp.csv"))
    else os.path.join(d, "prof", ".logs", "cpp_device_perf_report.csv")
)


def routes():
    R = torch.load(os.path.join(d, "routes.pt"))  # [step][call] -> [B, 6]
    if not R:
        return None
    nl = min(len(s) for s in R)
    res = []
    for l in range(nl):
        dist, dmax, dmed, dmean, tmax, tmean = [], [], [], [], [], []
        for s in R:
            ids = s[l]
            dist.append(int(ids.unique().numel()))
            dev = ids // EPD
            per = []
            tk = []
            for dv in range(32):
                m = dev == dv
                per.append(int(ids[m].unique().numel()))
                tk.append(int(m.sum()))  # (token,expert) pairs handled by device dv
            dmax.append(max(per))
            dmed.append(statistics.median(per))
            dmean.append(sum(per) / 32)
            tmax.append(max(tk))
            tmean.append(sum(tk) / 32)
        f = lambda v: sum(v) / len(v)
        res.append(
            dict(
                layer_call=l,
                distinct=f(dist),
                dev_max=f(dmax),
                dev_med=f(dmed),
                dev_mean=f(dmean),
                pairs_max=f(tmax),
                pairs_mean=f(tmean),
            )
        )
    return res


def group(r):
    c, op, sec = r["caller"], r["op"], r["sec"]
    if c.startswith("mhc"):
        return "mhc"
    if c.startswith("router"):
        return "router"
    if c.startswith("shared_expert"):
        return "shared_expert"
    if "moe_compute" in op:
        return "moe_compute"
    if "dispatch" in op or "_format_dispatch_in" in c:
        return "dispatch"
    if sec == "moe":
        return "combine(tilize+fastred+reduce_scatter)"
    if sec == "moe_allgather":
        return "ccl_moe_allgather"
    if c.startswith("config.py"):
        return "ccl_attention"
    if sec == "attention":
        return "attention"
    return "other:" + sec


def layer_table(L):
    ld = os.path.join(d, f"layer_{L}")
    raw = json.load(open(os.path.join(ld, "raw.json")))
    n = raw["op1"] - raw["op0"]
    K, G = "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"
    best = None
    for T0 in (raw["op0"] - n, raw["cap0"]):
        sess = defaultdict(lambda: defaultdict(dict))  # prog -> session -> device -> ns
        gap = defaultdict(lambda: defaultdict(dict))
        cnt = defaultdict(int)
        for r in csv.DictReader(open(CSV)):
            if not r["METAL TRACE ID"]:
                continue
            oid = int(float(r["GLOBAL CALL COUNT"])) >> 10
            if T0 <= oid < T0 + n:
                s = r["METAL TRACE REPLAY SESSION ID"]
                sess[oid - T0][s][r["DEVICE ID"]] = float(r[K] or 0)
                gap[oid - T0][s][r["DEVICE ID"]] = float(r[G] or 0)
                cnt[s] += 1
        good = sorted([s for s, c in cnt.items() if c == n * 32], key=int)[1:]
        if good:
            best = (T0, sess, gap, good)
            break
    if not best:
        return None
    T0, sess, gap, good = best
    sec_of, st = {}, 0
    for name, _, nrows in raw["eager_marks"]:
        for i in range(st, nrows):
            sec_of[i] = name
        st = nrows
    rows = []
    for c in raw["eager_rows"]:
        for pid in range(c["id0"], c["id1"]):
            p = pid - raw["op0"]
            per_dev = defaultdict(list)
            for s in good:
                for dv, v in sess[p][s].items():
                    per_dev[dv].append(v)
            dv_med = [statistics.median(v) for v in per_dev.values()]
            gp = [statistics.median(list(gap[p][s].values())) for s in good]
            rows.append(
                dict(
                    call=c["idx"],
                    op=c["op"].replace("ttnn.", ""),
                    caller=c["caller"],
                    sec=sec_of.get(c["idx"], "?"),
                    med=statistics.median(dv_med),
                    mx=max(dv_med),
                    mn=min(dv_med),
                    gap=0.0 if p == 0 else statistics.median(gp),
                )
            )
    grp = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for r in rows:
        g = group(r)
        grp[g][0] += r["med"]
        grp[g][1] += r["mx"]
        grp[g][2] += r["gap"]
        grp[g][3] += 1
    return dict(
        layer=L,
        traced_ms=raw["traced_ms"],
        sessions=len(good),
        n_prog=len(rows),
        groups={g: dict(med_us=v[0] / 1e3, max_us=v[1] / 1e3, gap_us=v[2] / 1e3, n=v[3]) for g, v in grp.items()},
        sum_med_us=sum(r["med"] for r in rows) / 1e3,
        sum_gap_us=sum(r["gap"] for r in rows) / 1e3,
        rows=rows,
    )


if os.path.exists(os.path.join(d, "routes.pt")):
    out["routes"] = routes()
out["summary"] = (
    json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else None
)
out["layers"] = {}
for L in sorted(int(x.split("_")[1]) for x in os.listdir(d) if x.startswith("layer_")):
    try:
        out["layers"][L] = layer_table(L)
    except Exception as e:  # noqa
        out["layers"][L] = dict(error=repr(e))
json.dump(out, open(os.path.join(d, "analysis.json"), "w"), indent=1)
if out.get("routes"):
    r = out["routes"]
    print(
        "experts: distinct/layer %.1f  per-dev max %.2f med %.2f mean %.2f  pairs/dev max %.1f mean %.1f"
        % tuple(
            sum(x[k] for x in r) / len(r)
            for k in ("distinct", "dev_max", "dev_med", "dev_mean", "pairs_max", "pairs_mean")
        )
    )
for L, t in out["layers"].items():
    if "groups" in t:
        print(
            f"layer {L}: traced {t['traced_ms']*1e3:.0f} us, sum kernel med {t['sum_med_us']:.0f} us gaps {t['sum_gap_us']:.0f} us"
        )
        for g, v in sorted(t["groups"].items(), key=lambda kv: -kv[1]["max_us"]):
            print(f"   {g:42s} med {v['med_us']:7.1f}  max {v['max_us']:7.1f}  gap {v['gap_us']:6.1f}  n={v['n']}")
    else:
        print(L, t)
