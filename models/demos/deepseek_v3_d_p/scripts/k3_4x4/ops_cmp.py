"""Per-op device time per chip (ms, mean over devices) for several ops_perf CSVs.
usage: ops_cmp.py name=path.csv [name=path.csv ...] [--top N]"""
import sys

import pandas as pd

args = [a for a in sys.argv[1:] if "=" in a]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 40
agg = {}
for a in args:
    k, p = a.split("=", 1)
    df = pd.read_csv(p, low_memory=False, usecols=["OP CODE", "DEVICE ID", "DEVICE KERNEL DURATION [ns]"]).dropna(
        subset=["DEVICE KERNEL DURATION [ns]"]
    )
    nd = df["DEVICE ID"].nunique()
    g = df.groupby("OP CODE")["DEVICE KERNEL DURATION [ns]"].agg(["sum", "count"])
    agg[k] = pd.DataFrame({"ms": g["sum"] / nd / 1e6, "n": g["count"] / nd})
t = pd.concat(agg, axis=1).fillna(0)
names = list(agg)
for k in names[1:]:
    t[(f"{k}/{names[0]}", "")] = t[(k, "ms")] / t[(names[0], "ms")]
t = t.sort_values((names[-1], "ms"), ascending=False)
pd.set_option("display.width", 250)
print(t.round(2).head(top).to_string())
print("TOTAL ms", {k: round(agg[k]["ms"].sum(), 1) for k in names})
