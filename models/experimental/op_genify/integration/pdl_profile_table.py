# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stock vs genify_matmul per-matmul table for one Panoptic-DeepLab inference (#65150).

Inputs: two Tracy ops reports of
    TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=5000 scripts/run_safe_pytest.sh --profile \
        models/experimental/panoptic_deeplab/tests/pcc/test_tt_model.py -k panoptic_deeplab_110_cores -s
one stock, one with TT_GENIFY_MATMUL=1. Each report holds exactly one forward pass.

Matmuls: stock = MatmulDeviceOperation + the two 1x1 stride-2 Conv2dDeviceOperation (shortcuts);
genify = GenericOpDeviceOperation. They pair by order (55 per inference) and the shapes are checked.

Glue charged to a conv's matmul (the upsample passes carry none: their permute / tilize / typecast are
the same module ops in both modes):
  pre  = the run of conv-input ops right before the matmul: stock conv2d's own I2S / reshard + move,
         tilize of a row-major input, halo + move of the stride-2 sliding-window path; genify's own
         stride-2 subsample (rows slice + the column-selection GenericOp, "Select"), pool-input
         tilize, any reshard.
  post = the run of I2S / reshard / move right after the matmul that is not the next matmul's input
         glue: the next 3x3 conv sharding an interleaved output, or (charged to the block's shortcut)
         the identity I2S / reshard before the residual add.
Model-level data movement in both modes (moves to DRAM / L1, the upsample module's ops) stays uncharged
and shows in the per-op-code table and the totals.

    python models/experimental/op_genify/integration/pdl_profile_table.py --stock S.csv --genify G.csv \
        --out models/experimental/op_genify/integration/pdl_profile_table.md
"""

import argparse
import re

import pandas as pd

DUR = "DEVICE KERNEL DURATION [ns]"
GLUE = {
    "InterleavedToShardedDeviceOperation": "I2S",
    "ShardedToInterleavedDeviceOperation": "S2I",
    "ReshardDeviceOperation": "Reshard",
    "MoveDeviceOperation": "Move",
    "CopyDeviceOperation": "Copy",
    "SliceDeviceOperation": "Slice",
    "TilizeDeviceOperation": "Tilize",
    "TilizeWithValPaddingDeviceOperation": "TilizeVP",
    "UntilizeDeviceOperation": "Untilize",
    "UntilizeWithUnpaddingDeviceOperation": "UntilizeWU",
    "UntilizeCodegenDeviceOperation": "UntilizeCG",
    "TypecastDeviceOperation": "Typecast",
    "ReshapeDeviceOperation": "Reshape",
    "ReshapeViewDeviceOperation": "Reshape",
}


def pdl_matmul_names():
    """The 55 matmuls of one inference, in execution order (TtPanopticDeepLab.forward, 110-core path)."""
    names = []
    for stage, blocks in (("res2", 3), ("res3", 4), ("res4", 6), ("res5", 3)):
        for i in range(blocks):
            if i == 0:
                names.append(f"{stage}.0.shortcut" + (" (stride 2)" if stage in ("res3", "res4") else ""))
            names += [f"{stage}.{i}.conv1", f"{stage}.{i}.conv3"]
    for head in ("sem", "ins"):
        names += [
            f"{head} aspp.convs.0",
            f"{head} aspp.convs.4 (pool, M=1)",
            f"{head} aspp.project",
            f"{head} decoder.res3.project_conv",
            f"{head} decoder.res2.project_conv",
        ]
        if head == "sem":
            names += ["sem predictor (19/32)", "sem upsample W", "sem upsample H"]
        else:
            names += [
                "ins center_predictor (1/32)",
                "ins offset_predictor (2/32)",
                "ins center upsample W",
                "ins center upsample H",
                "ins offset upsample W",
                "ins offset upsample H",
            ]
    return names


def load(path):
    df = pd.read_csv(path, low_memory=False)
    df = df[~df["DEVICE ID"].isna()].reset_index(drop=True)  # drop signposts
    return df


def dims(r, t):
    out = []
    for a in "WZYX":
        v = r.get(f"{t}_{a}_PAD[LOGICAL]")
        if not isinstance(v, str):
            return None
        out.append(int(re.match(r"(\d+)\[(\d+)\]", v).group(2)))
    return out


def mem(s):
    s = re.sub(r"DEV_\d+_", "", s if isinstance(s, str) else "-")
    return {
        "DRAM_INTERLEAVED": "DRAM",
        "L1_INTERLEAVED": "L1",
        "L1_HEIGHT_SHARDED": "L1-HS",
        "L1_BLOCK_SHARDED": "L1-BS",
        "L1_WIDTH_SHARDED": "L1-WS",
    }.get(s, s)


def is_stock_mm(r):
    if r["OP CODE"] == "MatmulDeviceOperation":
        return True
    return r["OP CODE"] == "Conv2dDeviceOperation" and "window_hw=(1;1)" in str(r["ATTRIBUTES"])


def is_select(r):
    """genify's stride-2 column subsample: a selection matmul S' [1, groups, G/2, G] @ pixels [1, groups, G, C]
    (glue, not one of the model's matmuls)."""
    if r["OP CODE"] != "GenericOpDeviceOperation":
        return False
    a, b = dims(r, "INPUT_0"), dims(r, "INPUT_1")
    return a[1] > 1 and b[1] == a[1] and b[2] == a[3] and a[3] == 2 * a[2]


def is_genify_mm(r):
    return r["OP CODE"] == "GenericOpDeviceOperation" and not is_select(r)


def label(r):
    return "Select" if is_select(r) else LABEL.get(r["OP CODE"], r["OP CODE"])


def mkn(r):
    """(batch, M, K, N) of a matmul row."""
    a, b = dims(r, "INPUT_0"), dims(r, "INPUT_1")
    if r["OP CODE"] == "Conv2dDeviceOperation":
        at = str(r["ATTRIBUTES"])
        nhwc = [int(x) for x in re.search(r"'input_tensor_shape': '\{([^}]*)\}'", at).group(1).split(";")]
        sh, sw = (int(x) for x in re.search(r"stride_hw=\((\d+);(\d+)\)", at).groups())
        cout = int(re.search(r"'output_channels': '(\d+)'", at).group(1))
        return 1, nhwc[0] * -(-nhwc[1] // sh) * -(-nhwc[2] // sw), nhwc[3], cout
    batch = a[0] * a[1]
    return batch, a[2], a[3], b[3]


# Glue a conv brings with it (immediately before its matmul): the stock conv2d's own input sharding
# (I2S / reshard + move, tilize of a row-major input; halo + move on the stride-2 sliding-window path);
# genify's own input preparation (the stride-2 subsample: slice / untilize / slice / tilize; the tilize of
# the row-major pool output; a reshard of a rejected shard layout).
STOCK_PRE = {"I2S", "Reshard", "Move", "Halo", "Tilize", "TilizeVP"}
GENIFY_PRE = {"Slice", "Select", "Untilize", "Tilize", "TilizeVP", "Typecast", "I2S", "Reshard", "Move"}
# Glue an output placement causes downstream (immediately after the matmul, before the next compute op):
# e.g. the next 3x3 conv sharding an L1-interleaved genify output, the identity reshard before the add.
POST = {"I2S", "Reshard", "Move"}
LABEL = dict(GLUE, HaloDeviceOperation="Halo")  # a halo is glue only as a stride-2 1x1 conv's own op


def attribute_glue(df, is_mm, names, pre_set):
    """{matmul name: {"pre": [rows], "post": [rows]}} for the conv matmuls (the upsample passes carry
    no glue of their own: their permute / tilize / typecast are the same module ops in both modes).
    A block-0 conv3's post-glue that feeds the residual add is the identity's: charged to the shortcut."""
    codes = [label(df.iloc[i]) for i in range(len(df))]
    n = len(df)
    mm_rows = [i for i in range(n) if is_mm(df.iloc[i])]
    row_of = dict(zip(names, mm_rows))
    is_mm_row = set(mm_rows)
    out = {name: {"pre": [], "post": []} for name in names}
    taken = set()
    for name, j in row_of.items():
        if "upsample" in name:
            continue
        k = j - 1
        while k >= 0 and codes[k] in pre_set and k not in is_mm_row:
            out[name]["pre"].insert(0, k)
            taken.add(k)
            k -= 1
    for name, j in row_of.items():
        if "upsample" in name:
            continue
        k = j + 1
        run = []
        while k < n and codes[k] in POST and k not in taken:
            run.append(k)
            k += 1
        if not run or (k < n and k in is_mm_row):
            continue  # nothing, or the next matmul's own input glue
        target = name
        if name.endswith(".0.conv3") and k < n and df.iloc[k]["OP CODE"] == "BinaryNgDeviceOperation":
            stage = name.split(".")[0]
            target = next(x for x in names if x.startswith(f"{stage}.0.shortcut"))
        out[target]["post"] += run
        taken.update(run)
    return out


def glue_str(df, rows):
    if not rows:
        return "-", 0.0
    parts = [f"{label(df.iloc[i])} {df.iloc[i][DUR] / 1e3:.1f}" for i in rows]
    return " + ".join(parts), sum(df.iloc[i][DUR] for i in rows) / 1e3


def category_totals(df, is_mm):
    tot = {"matmul": 0.0, "glue (data movement)": 0.0, "other": 0.0}
    for _, r in df.iterrows():
        d = r[DUR] / 1e3
        if is_mm(r):
            tot["matmul"] += d
        elif r["OP CODE"] in GLUE or is_select(r):
            tot["glue (data movement)"] += d
        else:
            tot["other"] += d
    return tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stock", required=True)
    p.add_argument("--genify", required=True)
    p.add_argument(
        "--out", required=True, help="markdown output; a .csv with the per-matmul rows is written next to it"
    )
    a = p.parse_args()

    st, ge = load(a.stock), load(a.genify)
    st_mm = [i for i in range(len(st)) if is_stock_mm(st.iloc[i])]
    ge_mm = [i for i in range(len(ge)) if is_genify_mm(ge.iloc[i])]
    names = pdl_matmul_names()
    assert len(st_mm) == len(names), f"stock: {len(st_mm)} matmuls, expected {len(names)}"
    assert len(ge_mm) == len(names), f"genify: {len(ge_mm)} matmuls, expected {len(names)}"
    st_glue = attribute_glue(st, is_stock_mm, names, STOCK_PRE)
    ge_glue = attribute_glue(ge, is_genify_mm, names, GENIFY_PRE)

    rows = []
    for name, i, j in zip(names, st_mm, ge_mm):
        rs, rg = st.iloc[i], ge.iloc[j]
        s_shape, g_shape = mkn(rs), mkn(rg)
        assert s_shape == g_shape, f"{name}: stock {s_shape} != genify {g_shape}"
        b, m, k, nn = s_shape
        s_pre, s_pre_us = glue_str(st, st_glue[name]["pre"])
        s_post, s_post_us = glue_str(st, st_glue[name]["post"])
        g_pre, g_pre_us = glue_str(ge, ge_glue[name]["pre"])
        g_post, g_post_us = glue_str(ge, ge_glue[name]["post"])
        rows.append(
            {
                "matmul": name,
                "shape": (f"{b}x" if b > 1 else "") + f"{m}x{k}x{nn}",
                "A stock": f"{mem(rs['INPUT_0_MEMORY'])} {str(rs['INPUT_0_DATATYPE']).replace('BFLOAT8_B', 'bfp8').replace('BFLOAT16', 'bf16')}",
                "A genify": f"{mem(rg['INPUT_0_MEMORY'])} {str(rg['INPUT_0_DATATYPE']).replace('BFLOAT8_B', 'bfp8').replace('BFLOAT16', 'bf16')}",
                "out genify": mem(rg["OUTPUT_0_MEMORY"]),
                "stock op": "conv2d 1x1 s2" if rs["OP CODE"] == "Conv2dDeviceOperation" else "matmul",
                "stock us": rs[DUR] / 1e3,
                "genify us": rg[DUR] / 1e3,
                "stock cores": int(rs["CORE COUNT"]),
                "genify cores": int(rg["CORE COUNT"]),
                "stock pre-glue": s_pre,
                "stock post-glue": s_post,
                "stock glue us": s_pre_us + s_post_us,
                "genify pre-glue": g_pre,
                "genify pre us": g_pre_us,
                "genify post-glue": g_post,
                "genify glue us": g_pre_us + g_post_us,
            }
        )
    tab = pd.DataFrame(rows)
    tab["genify/stock"] = tab["genify us"] / tab["stock us"]
    tab["stock mm+glue us"] = tab["stock us"] + tab["stock glue us"]
    tab["genify mm+glue us"] = tab["genify us"] + tab["genify glue us"]
    csv_path = re.sub(r"\.md$", "", a.out) + ".csv"
    tab.to_csv(csv_path, index=False, float_format="%.2f")

    st_tot, ge_tot = category_totals(st, is_stock_mm), category_totals(ge, is_genify_mm)
    st_all, ge_all = st[DUR].sum() / 1e3, ge[DUR].sum() / 1e3

    # Per op code: count and time in each mode (genify's stride-2 selection matmuls split out).
    for df in (st, ge):
        df["op"] = [
            "GenericOpDeviceOperation (stride-2 select)" if is_select(df.iloc[i]) else df.iloc[i]["OP CODE"]
            for i in range(len(df))
        ]
    oc = (
        pd.concat(
            [
                st.groupby("op")[DUR].agg(["count", "sum"]).rename(columns={"count": "stock n", "sum": "stock us"}),
                ge.groupby("op")[DUR].agg(["count", "sum"]).rename(columns={"count": "genify n", "sum": "genify us"}),
            ],
            axis=1,
        )
        .fillna(0)
        .sort_values("stock us", ascending=False)
    )
    oc["stock us"] /= 1e3
    oc["genify us"] /= 1e3
    oc["delta us"] = oc["genify us"] - oc["stock us"]

    L = []
    L.append("# PDL stock vs genify_matmul, one inference (test_tt_model.py -k panoptic_deeplab_110_cores)\n")
    L.append(f"Stock CSV: `{a.stock}`  \nGenify CSV: `{a.genify}`\n")
    L.append("## Totals (device kernel time, µs)\n")
    L.append("| | stock | genify | genify - stock |")
    L.append("|---|---:|---:|---:|")
    L.append(
        f"| 55 matmuls (kernel only) | {tab['stock us'].sum():.1f} | {tab['genify us'].sum():.1f} | {tab['genify us'].sum() - tab['stock us'].sum():+.1f} |"
    )
    L.append(
        f"| glue charged to the matmuls (pre + post) | {tab['stock glue us'].sum():.1f} | {tab['genify glue us'].sum():.1f} | "
        f"{tab['genify glue us'].sum() - tab['stock glue us'].sum():+.1f} |"
    )
    L.append(
        f"| matmuls + their glue | {tab['stock mm+glue us'].sum():.1f} | {tab['genify mm+glue us'].sum():.1f} | "
        f"{tab['genify mm+glue us'].sum() - tab['stock mm+glue us'].sum():+.1f} |"
    )
    for cat in ("matmul", "glue (data movement)", "other"):
        L.append(f"| all ops: {cat} | {st_tot[cat]:.1f} | {ge_tot[cat]:.1f} | {ge_tot[cat] - st_tot[cat]:+.1f} |")
    L.append(
        f"| **whole inference (sum of all ops)** | **{st_all:.1f}** | **{ge_all:.1f}** | **{ge_all - st_all:+.1f}** |"
    )
    L.append(f"| device ops | {len(st)} | {len(ge)} | {len(ge) - len(st):+d} |")
    L.append("")
    sub = tab[tab["matmul"].str.contains("stride 2")]
    L.append(
        "Stride-2 subsample (genify pre-glue of res3.0 / res4.0 shortcut): "
        + "; ".join(f"{r['matmul']}: {r['genify pre-glue']} = {r['genify pre us']:.1f} µs" for _, r in sub.iterrows())
        + "\n"
    )
    L.append("## Per matmul (µs)\n")
    cols = [
        "matmul",
        "shape",
        "A stock",
        "A genify",
        "stock us",
        "genify us",
        "genify/stock",
        "stock pre-glue",
        "stock post-glue",
        "genify pre-glue",
        "genify post-glue",
    ]
    L.append("| " + " | ".join(cols) + " |")
    L.append("|" + "|".join("---:" if c.endswith("us") or c == "genify/stock" else "---" for c in cols) + "|")
    for _, r in tab.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f"{v:.2f}" if c == "genify/stock" else (f"{v:.1f}" if isinstance(v, float) else str(v)))
        L.append("| " + " | ".join(vals) + " |")
    L.append("")
    L.append("## Per op code (count, µs)\n")
    L.append("| op code | stock n | stock µs | genify n | genify µs | delta µs |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for code, r in oc.iterrows():
        L.append(
            f"| {code} | {int(r['stock n'])} | {r['stock us']:.1f} | {int(r['genify n'])} | {r['genify us']:.1f} | {r['delta us']:+.1f} |"
        )
    L.append("")
    with open(a.out, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L[:20]))
    print(f"wrote {a.out} and {csv_path}")


if __name__ == "__main__":
    main()
