# Summarize #43 ops_perf_results CSV: per-device device-kernel time by op class and top ops.
import re
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1])
# Keep only the profiled forward: weight prep and load ops run before the "start" signpost.
sp = df.index[df["OP TYPE"].astype(str) == "signpost"].tolist()
if len(sp) >= 2:
    df = df.loc[sp[0] + 1 : sp[-1] - 1]
dur = "DEVICE KERNEL DURATION [ns]" if "DEVICE KERNEL DURATION [ns]" in df else "DEVICE FW DURATION [ns]"
df = df[df[dur].notna()]
CLS = [
    ("conv3d", r"conv3d"),
    ("halo/neighbor-pad CCL", r"neighbor|halo|pad.*persistent"),
    ("other CCL", r"gather|scatter|all_reduce|allreduce|ccl|fabric"),
    ("norm (pixel/group/rms)", r"norm|moreh_sum|reduce"),
    ("upsample/depth-to-space", r"depth|space|upsample|fold|unpatch"),
    (
        "layout (tilize/untilize/permute/reshape/concat/slice/pad)",
        r"tilize|permute|transpose|reshape|view|concat|slice|pad|copy|clone|typecast|move|interleaved|shard",
    ),
    ("eltwise", r"binary|unary|silu|add|mul|sub|eltwise|where|ternary"),
]


def cls(op):
    s = op.lower()
    for name, rx in CLS:
        if re.search(rx, s):
            return name
    return "other"


df["cls"] = df["OP CODE"].map(cls)
dev = "DEVICE ID" if "DEVICE ID" in df else None
per_dev = df.groupby(dev)[dur].sum() / 1e6 if dev else None
print("per-device total ms:", per_dev.round(1).to_dict() if per_dev is not None else df[dur].sum() / 1e6)
d0 = df[df[dev] == per_dev.idxmax()] if dev else df
tot = d0[dur].sum()
print(f"\nslowest device {per_dev.idxmax() if dev else '-'}: {tot/1e6:.1f} ms, {len(d0)} ops")
c = d0.groupby("cls")[dur].agg(["sum", "count"]).sort_values("sum", ascending=False)
c["ms"] = c["sum"] / 1e6
c["%"] = 100 * c["sum"] / tot
print(c[["ms", "%", "count"]].round(2).to_string())
o = d0.groupby(["OP CODE", "cls"])[dur].agg(["sum", "count", "mean"]).sort_values("sum", ascending=False).head(25)
o["ms"] = o["sum"] / 1e6
o["%"] = 100 * o["sum"] / tot
o["mean_us"] = o["mean"] / 1e3
print("\n" + o[["ms", "%", "count", "mean_us"]].round(2).to_string())
