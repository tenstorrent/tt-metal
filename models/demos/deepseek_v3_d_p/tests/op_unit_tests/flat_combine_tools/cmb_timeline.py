import sys, pandas as pd
d = pd.read_csv(sys.argv[1], skiprows=1)
d.columns = [c.strip() for c in d.columns]
for c in ["zone name", "type", "RISC processor type"]:
    d[c] = d[c].astype(str).str.strip()
dev = d.columns[0]
runs = sorted(d[d["zone name"].str.startswith("CMB-")]["run host ID"].unique())
rid = int(sys.argv[2]) if len(sys.argv) > 2 else runs[-1] // 1024
d = d[d["run host ID"] // 1024 == rid]
T = "time[cycles since reset]"
F = 1350.0
print("run", rid, "of", runs)
for chip, c in d.groupby(dev):
    t0 = c[T].min()
    st = c[c["type"] == "ZONE_START"]; en = c[c["type"] == "ZONE_END"]
    key = ["core_x", "core_y", "RISC processor type", "zone name"]
    st = st.assign(n=st.groupby(key).cumcount()); en = en.assign(n=en.groupby(key).cumcount())
    z = st.merge(en, on=key + ["n"], suffixes=("_s", "_e"))
    z["s"] = (z[T + "_s"] - t0) / F; z["e"] = (z[T + "_e"] - t0) / F
    rel = z[z["zone name"] == "CMB-RELEASE"].sort_values("n")
    steps = z[z["zone name"] == "CMB-STEP"]
    gate = z[z["zone name"] == "CMB-GATE"]
    fw = z[z["zone name"].str.endswith("-FW")]
    cmbcores = set(map(tuple, steps[["core_x", "core_y"]].drop_duplicates().values)) | set(map(tuple, rel[["core_x","core_y"]].drop_duplicates().values))
    flat_end = fw[[ (x, y) not in cmbcores for x, y in zip(fw.core_x, fw.core_y)]]["e"].max()
    end = fw["e"].max()
    se = steps.groupby("n")["e"].max(); ss = steps.groupby("n")["s"].min()
    gd = gate.groupby("n")["e"].max() if len(gate) else None
    print(f"chip {chip}: flat end {flat_end:.0f}  program end {end:.0f}")
    print("  release:", " ".join(f"{v:.0f}" for v in rel["e"].values))
    print("  step end:", " ".join(f"{v:.0f}" for v in se.values))
    print("  step dur:", " ".join(f"{a-b:.0f}" for a, b in zip(se.values, ss.values)))
