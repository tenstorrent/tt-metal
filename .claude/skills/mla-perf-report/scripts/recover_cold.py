import glob, os, sys
import pandas as pd

sys.path.insert(0, os.getcwd())
from models.tt_transformers.tests.test_utils import merge_device_rows

RPT = "generated/profiler/deepseek_v32_sparse_mla_perf/reports/2026_07_07_23_25_57"
f = glob.glob(RPT + "/ops_perf_results_*.csv")[0]
dur = "DEVICE KERNEL DURATION [ns]"
df0 = pd.read_csv(f)
mk = df0[df0["OP TYPE"] == "signpost"]["OP CODE"]
start = mk[mk == "start"].index[0]
stop = mk[mk == "stop"].index[0]
region = df0.iloc[start + 1 : stop].copy()
region[dur] = pd.to_numeric(region[dur], errors="coerce")

# aggregate (matched sparse cold)
df = merge_device_rows(region)
total = df[dur].sum()
by = (
    df.groupby("OP CODE")[dur]
    .agg(count="count", total_ns="sum", avg_ns="mean")
    .sort_values("total_ns", ascending=False)
)
by["pct"] = 100.0 * by["total_ns"] / total
out_dir = "/tmp/claude-1211414789/-localdev-mvasilijevic-tt-metal/0f0b040b-5dd5-4a93-b852-a5edd32a4cf9/scratchpad"
by.reset_index().to_csv(out_dir + "/sparse_cold_MATCHED.csv", index=False)

# by_iter
ri = region.reset_index(drop=True)
starts = list(ri.index[ri["OP CODE"] == "MLA_START"])
bounds = starts + [len(ri)]
CHUNK = 1280
rows = []
totals = []
for i in range(len(starts)):
    seg = merge_device_rows(ri.iloc[bounds[i] + 1 : bounds[i + 1]])
    tot = seg[dur].sum()
    g = (
        seg.groupby("OP CODE")[dur]
        .agg(count="count", total_ns="sum", avg_ns="mean")
        .sort_values("total_ns", ascending=False)
    )
    g["pct"] = 100.0 * g["total_ns"] / tot
    g = g.reset_index()
    g.insert(0, "cache_depth_tokens", i * CHUNK)
    g.insert(0, "iteration", i)
    rows.append(g)
    totals.append((i, i * CHUNK, tot, len(g)))
pd.concat(rows, ignore_index=True).to_csv(out_dir + "/sparse_cold_by_iter_MATCHED.csv", index=False)

print(
    f"MATCHED sparse cold: total={total/1e6:.3f} ms, calls={int(by['count'].sum())}, iters={len(starts)}, chunk step={CHUNK}"
)
print("per-iter totals (ms):", [round(t / 1e6, 3) for _, _, t, _ in totals])
print("\ntop ops:")
for op, r in by.head(8).iterrows():
    print(f"  {op:40} {int(r['count']):4d} {r['total_ns']/1e6:9.3f}ms {r['pct']:5.1f}%")
