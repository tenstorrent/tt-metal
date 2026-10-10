# t334: per-layer conv3d table from a tracy ops_perf_results CSV of one conv VAE decode.
# Usage: python conv_table.py <ops_perf.csv[.gz]> <out_prefix>
# Writes <out_prefix>_ops.csv (every op of the slowest chip, in order), <out_prefix>_conv3d.csv (42 convs,
# min/median/max over chips) and prints the class summary, the conv3d table and the halo totals.
import re
import sys

import pandas as pd

# Decoder conv order (reversed decoder_blocks of _LTX_PROD_DECODER_BLOCKS): 42 convs.
LAYERS = (
    ["conv_in"]
    + [f"res1024.{b}.c{c}" for b in range(2) for c in (1, 2)]
    + ["up_all/2"]
    + [f"res512a.{b}.c{c}" for b in range(2) for c in (1, 2)]
    + ["up_all_x1"]
    + [f"res512b.{b}.c{c}" for b in range(4) for c in (1, 2)]
    + ["up_time"]
    + [f"res256.{b}.c{c}" for b in range(6) for c in (1, 2)]
    + ["up_space"]
    + [f"res128.{b}.c{c}" for b in range(4) for c in (1, 2)]
    + ["conv_out"]
)
assert len(LAYERS) == 42

CLS = [
    ("conv3d", r"conv3d"),
    ("halo/neighbor-pad CCL", r"neighbor|halo|pad.*persistent"),
    ("other CCL", r"gather|scatter|all_reduce|allreduce|ccl|fabric"),
    ("norm", r"norm|moreh_sum|reduce"),
    ("layout", r"tilize|permute|transpose|reshape|view|concat|slice|pad|copy|clone|typecast|move|interleaved|shard"),
    ("eltwise", r"binary|unary|silu|add|mul|sub|eltwise|where|ternary"),
]
HIFI4_FLOP_PER_CYCLE_PER_CORE = 1024  # 4096 at LoFi; HiFi4 takes 4 math passes
FID_PASSES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}
DRAM_GBPS = 512  # Blackhole: 8 GDDR6 channels
DT_BYTES = {"BFLOAT16": 2, "BFLOAT8_B": 1088 / 1024, "BFLOAT4_B": 576 / 1024, "FLOAT32": 4}


def cls(op):
    s = op.lower()
    for name, rx in CLS:
        if re.search(rx, s):
            return name
    return "other"


def dim(v):
    # "19[19]" -> 19 (padded value)
    m = re.match(r"\s*(\d+)", str(v))
    return int(m.group(1)) if m else None


def attr(a, key):
    m = re.search(rf"{key}=([^;)]+)", a) or re.search(rf"'{key}': '([^']*)'", a)
    return m.group(1) if m else None


df = pd.read_csv(sys.argv[1])
out = sys.argv[2]
sp = df.index[df["OP TYPE"].astype(str) == "signpost"].tolist()
if len(sp) >= 2:
    df = df.loc[sp[0] + 1 : sp[-1] - 1]
dur = "DEVICE KERNEL DURATION [ns]"
df = df[df[dur].notna()].copy()
df["cls"] = df["OP CODE"].map(cls)
df["ms"] = df[dur] / 1e6
df["clk_ghz"] = (df["DEVICE FW END CYCLE"] - df["DEVICE FW START CYCLE"]) / df["DEVICE FW DURATION [ns]"]
df["idx"] = df.groupby("DEVICE ID").cumcount()

per_dev = df.groupby("DEVICE ID")["ms"].sum()
slow = per_dev.idxmax()
print(
    f"devices {len(per_dev)}; per-device op sum ms min/median/max {per_dev.min():.1f}/{per_dev.median():.1f}/{per_dev.max():.1f}"
)
print(f"clock GHz median {df['clk_ghz'].median():.3f} (min {df['clk_ghz'].min():.3f}, max {df['clk_ghz'].max():.3f})")
d0 = df[df["DEVICE ID"] == slow]
tot = d0["ms"].sum()
print(f"\nslowest device {slow}: {tot:.1f} ms, {len(d0)} ops")
c = d0.groupby("cls")["ms"].agg(["sum", "count"]).sort_values("sum", ascending=False)
c["%"] = 100 * c["sum"] / tot
print(c.round(2).to_string())
o = d0.groupby("OP CODE")["ms"].agg(["sum", "count", "mean"]).sort_values("sum", ascending=False)
print("\n" + o.round(3).to_string())

cols = ["idx", "OP CODE", "cls", "ms", "MATH FIDELITY", "CORE COUNT", "clk_ghz"]
cols += [c for c in df.columns if c.startswith(("INPUT_0_", "OUTPUT_0_")) and "PAD" in c]
cols += ["ATTRIBUTES"]
d0[cols].to_csv(f"{out}_ops.csv", index=False)

# Conv3d table: k-th conv3d on every chip.
cv = df[df["cls"] == "conv3d"].copy()
cv["k"] = cv.groupby("DEVICE ID").cumcount()
halo = df[df["cls"] == "halo/neighbor-pad CCL"].copy()
halo["k"] = halo.groupby("DEVICE ID").cumcount()
rows = []
for k, g in cv.groupby("k"):
    r = g.iloc[0]
    a = str(r["ATTRIBUTES"])
    T, H, W = (dim(r[f"OUTPUT_0_{x}_PAD[LOGICAL]"]) for x in "WZY")
    cin = dim(r["INPUT_0_X_PAD[LOGICAL]"])
    ti, hi, wi = (dim(r[f"INPUT_0_{x}_PAD[LOGICAL]"]) for x in "WZY")
    cout = int(attr(a, "output_channels"))
    # Upsampler convs write the depth-to-space result, so count FLOPs at the conv's own (input) grid.
    flop = 2 * 27 * cin * cout * ti * hi * wi
    cores = int(r["AVAILABLE WORKER CORE COUNT"])
    clk = g["clk_ghz"].median()
    mx = g["ms"].max()
    peak = HIFI4_FLOP_PER_CYCLE_PER_CORE * cores * clk * 1e9
    hk = halo[halo["k"] == k]["ms"]
    passes = next((v for f, v in FID_PASSES.items() if f.lower() in str(r["MATH FIDELITY"]).lower()), 4)
    ib = DT_BYTES.get(str(r.get("INPUT_0_DATATYPE", "BFLOAT16")).upper(), 2)
    ob = DT_BYTES.get(str(r.get("OUTPUT_0_DATATYPE", "BFLOAT16")).upper(), 2)
    in_b, out_b, w_b = ti * hi * wi * cin * ib, ti * hi * wi * cout * ob, 27 * cin * cout * 2
    # Lower bound: every byte once. Blocked estimate: the input is re-read once per C_out block and each
    # spatial block reads a 3x3x3 halo around itself; weights counted once (they stay in L1 per core).
    bt, bh, bw, bco = (int(attr(a, x) or 1) for x in ("T_out_block", "H_out_block", "W_out_block", "C_out_block"))
    halo_f = (bt + 2) * (bh + 2) * (bw + 2) / (bt * bh * bw)
    blk_b = in_b * -(-cout // bco) * halo_f + out_b + w_b
    rows.append(
        dict(
            k=k,
            layer=LAYERS[k] if len(cv["k"].unique()) == 42 else "",
            in_THWC=f"{ti}x{hi}x{wi}x{cin}",
            out_THW=f"{T}x{H}x{W}",
            Cin=cin,
            Cout=cout,
            blk_T=attr(a, "T_out_block"),
            blk_H=attr(a, "H_out_block"),
            blk_W=attr(a, "W_out_block"),
            blk_Cout=attr(a, "C_out_block"),
            blk_Cin=attr(a, "C_in_block"),
            grid=attr(a, "compute_with_storage_grid_size"),
            fidelity=r["MATH FIDELITY"],
            fp32_acc=attr(a, "fp32_dest_acc_en"),
            pad=attr(a, "padding_mode"),
            gflop_chip=flop / 1e9,
            ms_min=g["ms"].min(),
            ms_med=g["ms"].median(),
            ms_max=mx,
            pct_hifi4=100 * flop / (mx * 1e-3 * peak),
            pct_fid=100 * flop * passes / 4 / (mx * 1e-3 * peak),
            mb_min=(in_b + out_b + w_b) / 1e6,
            mb_blk=blk_b / 1e6,
            pct_dram_min=100 * (in_b + out_b + w_b) / (mx * 1e-3 * DRAM_GBPS * 1e9),
            pct_dram_blk=100 * blk_b / (mx * 1e-3 * DRAM_GBPS * 1e9),
            mem=str(r.get("INPUT_0_MEMORY", "")).replace("DEV_0_", "").replace("DEV_1_", ""),
            halo_ms_max=hk.max() if len(hk) else float("nan"),
        )
    )
t = pd.DataFrame(rows)
t.to_csv(f"{out}_conv3d.csv", index=False)
pd.set_option("display.width", 250)
print(
    f"\nconv3d per layer ({len(t)} ops; ms over chips; %HiFi4 = chip FLOP / (ms_max x 1024 FLOP/cyc/core x cores x clk); "
    f"pct_fid = same at the op's own fidelity; pct_dram_* = estimated bytes / (ms_max x {DRAM_GBPS} GB/s))"
)
print(t.drop(columns=["grid", "pad", "fp32_acc"]).round(2).to_string(index=False))
print(
    f"\nconv3d sum of per-layer max {t['ms_max'].sum():.1f} ms, {t['gflop_chip'].sum() / 1e3:.2f} TFLOP/chip, "
    f"weighted %HiFi4 {100 * t['gflop_chip'].sum() * 1e9 / (t['ms_max'].sum() * 1e-3 * peak):.1f}, "
    f"est DRAM GB/chip min {t['mb_min'].sum() / 1e3:.2f} blocked {t['mb_blk'].sum() / 1e3:.2f}"
)
hs = halo.groupby("DEVICE ID")["ms"].agg(["sum", "count"])
print(
    f"halo per chip: ops {hs['count'].iloc[0]}, ms min/median/max {hs['sum'].min():.2f}/{hs['sum'].median():.2f}/{hs['sum'].max():.2f}"
)
oc = df[df["cls"] == "other CCL"].groupby("DEVICE ID")["ms"].sum()
if len(oc):
    print(f"other CCL per chip ms min/median/max {oc.min():.2f}/{oc.median():.2f}/{oc.max():.2f}")
