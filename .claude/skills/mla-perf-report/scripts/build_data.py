import csv, json, os, glob

BASE = "generated/profiler"
SP = {"sparse": f"{BASE}/deepseek_v32_sparse_mla_perf", "dense": f"{BASE}/deepseek_v32_dense_mla_perf"}
SCEN = ["warm", "cold", "long"]


def read_summary(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "op": r["OP CODE"],
                    "count": int(float(r["count"])),
                    "total_ns": float(r["total_ns"]),
                    "avg_ns": float(r["avg_ns"]),
                    "pct": float(r["pct"]),
                }
            )
    return rows


data = {"modes": {}}
for mode, d in SP.items():
    data["modes"][mode] = {}
    for s in SCEN:
        p = f"{d}/deepseek_v32_{mode}_mla_perf_{s}.csv"
        rows = read_summary(p)
        entry = None
        if rows is not None:
            entry = {
                "ops": rows,
                "total_ns": sum(r["total_ns"] for r in rows),
                "total_calls": sum(r["count"] for r in rows),
                "csv_path": p,
            }
        data["modes"][mode][s] = entry


# cold by_iter: totals per iteration + per-op x iter matrix
def read_by_iter(path):
    if not os.path.exists(path):
        return None
    per_iter = {}  # iter -> {cache_depth, ops:[...]}
    with open(path) as f:
        for r in csv.DictReader(f):
            it = int(float(r["iteration"]))
            per_iter.setdefault(
                it, {"iteration": it, "cache_depth_tokens": int(float(r["cache_depth_tokens"])), "ops": []}
            )
            per_iter[it]["ops"].append(
                {
                    "op": r["OP CODE"],
                    "count": int(float(r["count"])),
                    "total_ns": float(r["total_ns"]),
                    "avg_ns": float(r["avg_ns"]),
                    "pct": float(r["pct"]),
                }
            )
    out = []
    for it in sorted(per_iter):
        e = per_iter[it]
        e["total_ns"] = sum(o["total_ns"] for o in e["ops"])
        e["op_count"] = sum(o["count"] for o in e["ops"])
        out.append(e)
    return out


data["cold_by_iter"] = {}
for mode, d in SP.items():
    data["cold_by_iter"][mode] = read_by_iter(f"{d}/deepseek_v32_{mode}_mla_perf_cold_by_iter.csv")

# raw report paths (latest per mode)
data["raw_reports"] = {}
for mode, d in SP.items():
    reps = sorted(glob.glob(f"{d}/reports/*/ops_perf_results_*.csv"))
    data["raw_reports"][mode] = [os.path.abspath(r) for r in reps]

# Baseline axis: totals of the merge-base "before" code, discovered via run-manifest (discover.py).
# data.baseline[mode][scenario] = total_ns; data.baseline_meta describes what the baseline is. The
# branch (data.modes) is the "after". Dense is the control (indexer-independent) — should be ~flat.
_SPdir = os.path.dirname(os.path.abspath(__file__))
# ADAPT per run: point at the baseline commit's totals produced by `python discover.py <baseline_commit>`
# (the merge-base "before", or a named baseline branch). Absent file -> no baseline axis in the report.
BASELINE_TOTALS = os.path.join(_SPdir, "totals_dc50ed81dc5.json")  # commit-keyed baseline totals
data["baseline"] = None
if os.path.exists(BASELINE_TOTALS):
    bt = json.load(open(BASELINE_TOTALS))  # keys "sparse/warm" -> {total_ns, calls, iters, dir}
    bl = {"sparse": {}, "dense": {}}
    for k, v in bt.items():
        m, s = k.split("/")
        bl[m][s] = {"total_ns": v["total_ns"], "total_calls": v["calls"]}
    data["baseline"] = bl
    data["baseline_meta"] = {
        "commit": "dc50ed8",
        "branch": "mvasilijevic/sparse_test_improvements",
        "label": "TP-head-sharded indexer (pre-SP×TP)",
        "desc": "merge-base 'before': the DSA indexer with the full-width TP logit all-reduce, "
        "identical harness/config to the branch. Dense is unchanged (indexer-independent) — a control.",
    }

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf_data.json")
with open(out, "w") as f:
    json.dump(data, f, indent=1)

# quick summary to stdout
for mode in SP:
    for s in SCEN:
        e = data["modes"][mode][s]
        if e is None:
            print(f"{mode:7}/{s:5}: MISSING")
        else:
            print(f"{mode:7}/{s:5}: {e['total_ns']/1e6:9.3f} ms  {e['total_calls']:4d} calls  {len(e['ops']):2d} ops")
for mode in SP:
    bi = data["cold_by_iter"][mode]
    if bi:
        print(
            f"cold_by_iter {mode}: {len(bi)} iters, depths {bi[0]['cache_depth_tokens']}..{bi[-1]['cache_depth_tokens']}"
        )
print("wrote", out)
