import sys, pandas as pd
d = pd.read_csv(sys.argv[1], skiprows=1)
d.columns = [c.strip() for c in d.columns]
for c in ["zone name", "type", "RISC processor type"]:
    d[c] = d[c].astype(str).str.strip()
T = "time[cycles since reset]"
for grp in map(int, sys.argv[2:]):
    c = d[d["run host ID"] // 1024 == grp]
    st = c[c["type"] == "ZONE_START"]; en = c[c["type"] == "ZONE_END"]
    key = [c.columns[0], "core_x", "core_y", "RISC processor type", "zone name"]
    st = st.assign(n=st.groupby(key).cumcount()); en = en.assign(n=en.groupby(key).cumcount())
    z = st.merge(en, on=key + ["n"], suffixes=("_s", "_e"))
    z["dur"] = (z[T + "_e"] - z[T + "_s"]) / 1350
    z = z[z["zone name"].isin(["CMB-STEP", "CMB-FABRIC", "CMB-LOCAL", "CMB-GATE"])]
    # per reader core: total per zone; then mean over cores
    t = z.groupby(key[:-1] + ["zone name"])["dur"].sum().groupby("zone name").agg(["mean", "max"]).round(1)
    print("group", grp); print(t.to_string())
