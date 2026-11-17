# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stock vs genify_matmul in Qwen3.8-27B QB2 (#65150), per matmul and per model.

Inputs are two Tracy ops reports of matmul_shape_set/test_qwen38_profile.py on the same branch:
one with the stock ops, one with TT_GENIFY_MATMUL=1. Sections (signposts) are the prefill lengths
and one decode step; each has 12 matmuls in model order (8 projections of the profiled GDN and
full-attention layers, then 4 LM-head vocab chunks). genify_matmul shows up as
GenericOpDeviceOperation, one per stock matmul (gate_up + SwiGLU is one genify call; stock runs
gate_up, two slices and a SiLU-mul). Names, layer counts and the roofline come from
matmul_shape_set/build_table.py (#65148), so the numbers line up with that comment.

    python models/experimental/op_genify/integration/compare_qwen38.py STOCK.csv GENIFY.csv [--md out.md]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "matmul_shape_set"))
import build_table as bt  # noqa: E402

GENIFY_CODES = ("GenericOpDeviceOperation",)


def sections(path, codes):
    df = bt.load(path)
    d0 = df[df["DEVICE ID"] == 0]
    out = {}
    for sec, g in d0.groupby("section", sort=False):
        if sec is None:
            continue
        g = g.reset_index()
        out[sec] = (df, g, [i for i in g.index if g.loc[i, "OP CODE"] in codes])
    return out


def model_ms(g, idx):
    """Full-model device time of a section (device 0): pre + 48 x GDN layer + 16 x full layer + head."""

    def norm_before(i):
        while i > 0 and "Norm" not in g.loc[i, "OP CODE"]:
            i -= 1
        return i

    b0, b1, b2 = norm_before(idx[0]), norm_before(idx[4]), norm_before(idx[8])
    dur = g["DEVICE KERNEL DURATION [ns]"] / 1e3
    return (dur[:b0].sum() + 48 * dur[b0:b1].sum() + 16 * dur[b1:b2].sum() + dur[b2:].sum()) / 1e3


def compare(stock_csv, genify_csv):
    stock = sections(stock_csv, bt.MM_CODES)
    gen = sections(genify_csv, GENIFY_CODES)
    rows, totals = [], []
    for sec, (df_s, g_s, i_s) in stock.items():
        df_g, g_g, i_g = gen[sec]
        assert len(i_s) == len(i_g) == 12, (sec, len(i_s), len(i_g))
        decode = sec.startswith("DECODE")
        group = "decode token" if decode else f"prefill M={int(sec.split('_M')[1])}"
        mm_s = mm_g = roof = 0.0
        for j in range(12):
            if j < 8:
                name, kind, _ = bt.QWEN_ORDER[j]
                calls = bt.QWEN_LAYERS[kind]
            elif decode:
                name, calls = "LM head (vocab chunk)", 1
            else:
                continue  # the LM head runs decode-style once per prefill; #65148 lists it under decode
            m = None if decode else int(sec.split("_M")[1])
            s = bt.mm_from_row(df_s, g_s.loc[i_s[j]], model="Qwen3.8-27B", group=group, name=name, calls=calls, m=m)
            g_us = bt.mean_over_devices(df_g, g_g.loc[i_g[j]])
            rows.append(
                dict(
                    group=group,
                    name=name,
                    calls=calls,
                    shape=f"[{s.M} x {s.k}] . [{s.k} x {s.n}]",
                    stock_us=s.measured_us,
                    genify_us=g_us,
                    roofline_us=s.roofline_us,
                    bound=s.bound,
                    stock_cores=s.cores,
                    genify_cores=int(g_g.loc[i_g[j], "CORE COUNT"]),
                )
            )
            mm_s += calls * s.measured_us
            mm_g += calls * g_us
            roof += calls * s.roofline_us
        totals.append(
            dict(
                group=group,
                stock_mm_ms=mm_s / 1e3,
                genify_mm_ms=mm_g / 1e3,
                roofline_ms=roof / 1e3,
                stock_model_ms=model_ms(g_s, i_s),
                genify_model_ms=model_ms(g_g, i_g),
            )
        )
    return rows, totals


def speed(stock, new):
    r = stock / new
    if abs(r - 1) <= 0.03:
        return "even"
    return f"{r:.2f}x faster" if r > 1 else f"{1 / r:.2f}x slower"


def markdown(rows, totals):
    out = ["| Per | Calls | Stock matmul ms | Generated ms | Generated vs stock | Roofline ms | Generated / roofline |"]
    out.append("|---|---:|---:|---:|---|---:|---:|")
    for t in totals:
        out.append(
            f"| {t['group']} | {sum(r['calls'] for r in rows if r['group'] == t['group'])} | {t['stock_mm_ms']:.2f} | "
            f"{t['genify_mm_ms']:.2f} | {speed(t['stock_mm_ms'], t['genify_mm_ms'])} | {t['roofline_ms']:.2f} | "
            f"{t['genify_mm_ms'] / t['roofline_ms']:.2f}x |"
        )
    out += ["", "| Per | Stock model device ms | Generated model device ms | Change |", "|---|---:|---:|---|"]
    for t in totals:
        out.append(
            f"| {t['group']} | {t['stock_model_ms']:.2f} | {t['genify_model_ms']:.2f} | "
            f"{speed(t['stock_model_ms'], t['genify_model_ms'])} |"
        )
    out += [
        "",
        "| Per | Matmul | Calls | Shape per chip | Stock us | Generated us | Generated vs stock | Roofline us |",
        "|---|---|---:|---|---:|---:|---|---:|",
    ]
    for r in rows:
        out.append(
            f"| {r['group']} | {r['name']} | {r['calls']} | `{r['shape']}` | {r['stock_us']:.1f} | {r['genify_us']:.1f} | "
            f"{speed(r['stock_us'], r['genify_us'])} | {r['roofline_us']:.1f} |"
        )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stock_csv")
    ap.add_argument("genify_csv")
    ap.add_argument("--md")
    args = ap.parse_args()
    rows, totals = compare(args.stock_csv, args.genify_csv)
    text = markdown(rows, totals)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
