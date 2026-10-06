import pandas as pd, re, sys

d = pd.read_csv(sys.argv[1])
c = d[d["OP CODE"].str.contains("Conv3d")].copy()


def g(a, k):
    m = re.search(k + r"=?'?:?\s*'?([^;')]+)", a)
    return m.group(1) if m else ""


rows = []
for _, r in c.iterrows():
    a = r["ATTRIBUTES"]
    blk = "/".join(
        re.search(n + r"=(\d+)", a).group(1)
        for n in ["C_in_block", "C_out_block", "T_out_block", "H_out_block", "W_out_block"]
    )
    grid = re.search(r"compute_with_storage_grid_size=(\d+-\d+)", a).group(1)
    pm = re.search(r"'padding_mode': '(\w+)'", a)
    pad = re.search(r"'padding': '([^']+)'", a)
    shp = f"{r['INPUT_0_Z_PAD[LOGICAL]']}x{r['INPUT_0_Y_PAD[LOGICAL]']}x{r['INPUT_0_W_PAD[LOGICAL]']}x{r['INPUT_0_X_PAD[LOGICAL]']}"
    halo = str(r.get("INPUT_3_Y_PAD[LOGICAL]", "")) if str(r.get("INPUT_3_LAYOUT", "")) == "ROW_MAJOR" else "-"
    k = re.search(r"'kernel_size': '([^']+)'", a).group(1)
    rows.append(
        dict(
            dev=r["DEVICE ID"],
            shape=shp,
            k=k,
            cout=re.search(r"'output_channels': '(\d+)'", a).group(1),
            blk=blk,
            grid=grid,
            pad=pad.group(1) if pad else "",
            mode=pm.group(1) if pm else "",
            halo=halo,
            cores=r["CORE COUNT"],
            us=r["DEVICE KERNEL DURATION [ns]"] / 1e3,
        )
    )
t = pd.DataFrame(rows)
t["idx"] = t.groupby("dev").cumcount()
g = (
    t.groupby(["idx", "shape", "k", "cout", "blk", "grid", "pad", "mode", "halo", "cores"], sort=False)["us"]
    .agg(["max", "mean"])
    .reset_index()
)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print(g.to_string(index=False))
print("sum of per-call max (ms):", g["max"].sum() / 1e3)
s = (
    g.groupby(["shape", "k", "cout", "blk", "grid", "mode"], sort=False)["max"]
    .agg(["count", "mean", "sum"])
    .reset_index()
)
s["sum"] /= 1e3
print(s.sort_values("sum", ascending=False).to_string(index=False))
