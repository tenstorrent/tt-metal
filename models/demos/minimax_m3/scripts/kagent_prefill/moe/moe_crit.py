import re
import sys

import pandas as pd

DUR = "DEVICE KERNEL DURATION [ns]"
df = pd.read_csv(
    sys.argv[1],
    low_memory=False,
    usecols=[
        "OP CODE",
        "OP TYPE",
        "GLOBAL CALL COUNT",
        "DEVICE ID",
        DUR,
        "DEVICE FW START CYCLE",
        "DEVICE FW END CYCLE",
    ],
)
df = df[["OP CODE", "OP TYPE", "GLOBAL CALL COUNT", "DEVICE ID", DUR, "DEVICE FW START CYCLE", "DEVICE FW END CYCLE"]]
in_chunk = False
layer = None
recs = []
lre = re.compile(r"M3_ZONE_(START|END) (layer\d+_(dense|sparse))")
for row in df.itertuples(index=False):
    code = str(row[0])
    if row[1] == "signpost":
        if "M3_ZONE_START profiled_chunk" in code:
            in_chunk = True
        elif "M3_ZONE_END profiled_chunk" in code:
            in_chunk = False
        m = lre.search(code)
        if m:
            layer = m.group(2) if m.group(1) == "START" else None
        continue
    if in_chunk and layer and "sparse" in layer:
        recs.append((layer, code, row[2] - row[3], row[3], row[4] / 1e3, row[5], row[6]))
r = pd.DataFrame(recs, columns=["layer", "op", "call", "dev", "us", "s", "e"])
MOE = [
    "DispatchDeviceOperation",
    "UnifiedRoutedExpertFfnDeviceOperation",
    "CombineDeviceOperation",
    "PostCombineReduceDeviceOperation",
]
tot = {}
for L, g in r.groupby("layer"):
    g = g.copy()
    g["inst"] = g.groupby("op")["call"].rank(method="dense").astype(int)
    p = g.pivot_table(index=["op", "inst"], columns="dev", values="us")
    rs2 = p.loc[("ReduceScatterMinimalAsyncDeviceOperation", 2)]
    d = p.loc[("DispatchDeviceOperation", 1)]
    x = p.loc[("UnifiedRoutedExpertFfnDeviceOperation", 1)]
    c = p.loc[("CombineDeviceOperation", 1)]
    pc = p.loc[("PostCombineReduceDeviceOperation", 1)]
    s = d + x + c + pc + rs2
    lb = d.min() + x.mean() + c.min() + pc.mean() + rs2.min()
    lb2 = d.min() + x.max() + c.min() + pc.mean() + rs2.min()
    print(
        f"{L}: per-chip sum(dispatch..RS) mean {s.mean():.0f} [min {s.min():.0f} max {s.max():.0f}] us; experts mean {x.mean():.0f} max {x.max():.0f} min {x.min():.0f} (max/mean {x.max()/x.mean():.2f}); "
        f"floor(all waits gone, experts=mean) {lb:.0f}; floor(experts=max) {lb2:.0f}; gap {s.mean()-lb:.0f} us"
    )
