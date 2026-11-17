# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stock vs. genify_matmul per-matmul table for FLUX.2-dev on the QB2 (#65150).

Inputs are the Tracy ops reports of the two profiling tests run once per mode (TT_GENIFY_MATMUL unset / =1):
    models/experimental/op_genify/matmul_shape_set/test_flux2_profile.py              (--stock-dit / --genify-dit)
    models/experimental/op_genify/matmul_shape_set/test_flux2_text_encoder_profile.py (--stock-te / --genify-te)

Each stock matmul (Matmul / MinimalMatmul device op) is paired with its genify replacement (GenericOp) by order
within the section; both sequences are also aligned op by op (difflib on op codes, matmuls as one token) so the
ops that only the genify run has -- the slices that split a chunked output and the concat that joins a two-input
K -- are listed separately as glue. Times are DEVICE KERNEL DURATION averaged over the 4 chips. Scaling as in
matmul_shape_set/build_table.py: 1 double + 2 single blocks -> 8 double, 47 + 1 single per denoising step;
2 text-encoder layers -> 40 per image.

    python models/experimental/op_genify/integration/flux2_perf_table.py --stock-dit A.csv --genify-dit B.csv \
        --stock-te C.csv --genify-te D.csv --out models/experimental/op_genify/integration/flux2_perf_table
"""

import argparse
import difflib

import pandas as pd

from models.experimental.op_genify.matmul_shape_set.build_table import dims, load, mean_over_devices

STOCK_MM = ("MatmulDeviceOperation", "MinimalMatmulDeviceOperation")
GENIFY_MM = ("GenericOpDeviceOperation",)
DURATION = "DEVICE KERNEL DURATION [ns]"

# DiT step matmuls in execution order: (name, calls per denoising step, M key, stock op note).
DIT = [
    ("Timestep MLP linear_1", 1, "one", "ttnn.linear 1D"),
    ("Timestep MLP linear_2", 1, "one", "ttnn.linear 1D"),
    ("Guidance MLP linear_1", 1, "one", "ttnn.linear 1D"),
    ("Guidance MLP linear_2", 1, "one", "ttnn.linear 1D"),
    ("time_mod (+bias, split x17)", 1, "one", "minimal_matmul_split"),
    ("x_embedder", 1, "img", "minimal_matmul"),
    ("Double QKV img (split x3)", 8, "img", "minimal_matmul_split"),
    ("Double QKV txt (split x3)", 8, "txt", "minimal_matmul_split"),
    ("Double attn out img (+addcmul)", 8, "img", "dit_minimal_matmul_addcmul_fused"),
    ("Double attn out txt (+addcmul)", 8, "txt", "dit_minimal_matmul_addcmul_fused"),
    ("Double FF1 img (+SwiGLU)", 8, "img", "minimal_matmul fuse_swiglu"),
    ("Double FF2 img", 8, "img", "minimal_matmul"),
    ("Double FF1 txt (+SwiGLU)", 8, "txt", "minimal_matmul fuse_swiglu"),
    ("Double FF2 txt", 8, "txt", "minimal_matmul"),
    ("Single QKV (split x3)", 47, "joint", "minimal_matmul_split"),
    ("Single FF1 (+SwiGLU)", 47, "joint", "minimal_matmul fuse_swiglu"),
    ("Single out (concat-K)", 47, "joint", "minimal_matmul [x0, x1]"),
    ("Single QKV, final block (split x3)", 1, "joint", "minimal_matmul_split"),
    ("Single FF1, final block (+SwiGLU)", 1, "img", "minimal_matmul fuse_swiglu"),
    ("Single out, final block (concat-K)", 1, "img", "minimal_matmul [x0, x1]"),
    ("proj_out", 1, "img", "minimal_matmul"),
]
# Two profiled layers, each standing for 20 of the 40; merged into one row per projection below.
TE = [
    (name, 20, "minimal_matmul")
    for _ in range(2)
    for name in ("TE qkv_proj", "TE o_proj", "TE gate", "TE up (linear_in)", "TE down (linear_out)")
]


def section_ops(df, section):
    d0 = df[(df["DEVICE ID"] == 0) & (df.section == section)].reset_index(drop=True)
    d0["us"] = [mean_over_devices(df, r) for _, r in d0.iterrows()]
    return d0


def runs_ops(dfs, section):
    """One section's ops, `us` averaged op by op over repeated runs of the same mode (`us_runs` keeps each)."""
    runs = [section_ops(df, section) for df in dfs]
    base = runs[0]
    for r in runs[1:]:
        assert list(r["OP CODE"]) == list(base["OP CODE"]), f"{section}: runs differ in their op sequence"
    base["us_runs"] = list(zip(*[list(r["us"]) for r in runs]))
    base["us"] = [sum(v) / len(v) for v in base["us_runs"]]
    return base


def token(code):
    return "MM" if code in STOCK_MM + GENIFY_MM else code


def align(stock, genify):
    """Pairs (stock row index or None, genify row index or None) over the two op sequences."""
    a, b = [token(c) for c in stock["OP CODE"]], [token(c) for c in genify["OP CODE"]]
    pairs = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            pairs += list(zip(range(i1, i2), range(j1, j2)))
        else:
            pairs += [(i, None) for i in range(i1, i2)] + [(None, j) for j in range(j1, j2)]
    return pairs


def shape(r, t):
    d = dims(r, t)
    return "x".join(str(x) for x in d[2:]) if d else "-"


def compare(stock, genify, names, m_of=None):
    """Per-matmul rows + glue rows for one section."""
    s_mm = [i for i in stock.index if stock.loc[i, "OP CODE"] in STOCK_MM]
    g_mm = [i for i in genify.index if genify.loc[i, "OP CODE"] in GENIFY_MM]
    assert len(s_mm) == len(names), (len(s_mm), len(names))
    assert len(g_mm) == len(names), (len(g_mm), len(names), list(genify["OP CODE"]))
    rows = []
    for (name, calls, *rest), si, gi in zip(names, s_mm, g_mm):
        s, g = stock.loc[si], genify.loc[gi]
        rows.append(
            dict(
                matmul=name,
                calls=calls,
                M=m_of(rest[0]) if m_of else dims(g, "INPUT_0")[2],
                K=dims(g, "INPUT_1")[2],
                N=dims(g, "INPUT_1")[3],
                stock_op=rest[-1],
                stock_us=s["us"],
                genify_us=g["us"],
                genify_cores=int(g["CORE COUNT"]),
                stock_cores=int(s["CORE COUNT"]),
                genify_in0=shape(g, "INPUT_0"),
            )
        )
    # Glue: ops present only in the genify run, attached to the matmul they serve.
    glue = []
    pairs = align(stock, genify)
    extra_stock = [i for i, j in pairs if j is None]
    extra_genify = [j for i, j in pairs if i is None]
    mm_pos = {gi: k for k, gi in enumerate(g_mm)}
    for j in extra_genify:
        code = genify.loc[j, "OP CODE"]
        if code == "SliceDeviceOperation":  # splits the previous matmul's output
            owner = max(gi for gi in g_mm if gi < j)
        elif code == "ConcatDeviceOperation":  # joins the next matmul's K
            owner = min(gi for gi in g_mm if gi > j)
        else:
            owner = None
        glue.append(
            dict(
                op=code,
                for_matmul=names[mm_pos[owner]][0] if owner is not None else "?",
                calls=names[mm_pos[owner]][1] if owner is not None else 1,
                us=genify.loc[j, "us"],
                in0=shape(genify.loc[j], "INPUT_0"),
            )
        )
    return rows, glue, [stock.loc[i, "OP CODE"] for i in extra_stock]


def step_total(step, mm_codes):
    """Whole denoising step device time, scaled like build_table.flux2_rows (8 double, 47 + 1 single)."""
    mm = [i for i in step.index if step.loc[i, "OP CODE"] in mm_codes]
    dur = step["us"]
    tern = [i for i, c in enumerate(step["OP CODE"]) if c.startswith("Ternary")]
    d_end = [t for t in tern if t > mm[13]][0] + 1
    s1_end = [t for t in tern if t > mm[16]][0] + 1
    s2_end = [t for t in tern if t > mm[19]][0] + 1
    pre, dbl, s1, s2, post = (
        dur[: mm[5] + 1].sum(),
        dur[mm[5] + 1 : d_end].sum(),
        dur[d_end:s1_end].sum(),
        dur[s1_end:s2_end].sum(),
        dur[s2_end:].sum(),
    )
    return pre + 8 * dbl + 47 * s1 + s2 + post


def te_total(te, mm_codes):
    mm = [i for i in te.index if te.loc[i, "OP CODE"] in mm_codes]
    dur = te["us"]
    norm = [i for i in te.index if "Norm" in te.loc[i, "OP CODE"]]
    l0 = max(n for n in norm if n < mm[0])
    l1 = max(n for n in norm if n < mm[5])
    layer = dur[l0:l1].sum()
    return dur[:l0].sum() + 40 * layer + (dur[l1:].sum() - layer)


FORMATS = {"ratio": "{:.3f}", "ms_per_unit": "{:.2f}", "M": "{:,}", "K": "{:,}", "N": "{:,}"}


def fmt_table(df, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in FORMATS:
                cells.append(FORMATS[c].format(v))
            else:
                cells.append(f"{v:,.1f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stock-dit", required=True, help="CSV, or comma-separated CSVs of repeated runs (averaged)")
    p.add_argument("--genify-dit", required=True)
    p.add_argument("--stock-te", required=True)
    p.add_argument("--genify-te", required=True)
    p.add_argument("--out", required=True, help="output path prefix (.md and .csv are written)")
    a = p.parse_args()

    def loads(arg):
        return [load(path) for path in arg.split(",")]

    sd, gd = loads(a.stock_dit), loads(a.genify_dit)
    s_step, g_step = runs_ops(sd, "DIT_STEP"), runs_ops(gd, "DIT_STEP")
    s_ce, g_ce = runs_ops(sd, "CONTEXT_EMBEDDER"), runs_ops(gd, "CONTEXT_EMBEDDER")
    st, gt = loads(a.stock_te), loads(a.genify_te)
    s_te, g_te = runs_ops(st, "TEXT_ENCODER"), runs_ops(gt, "TEXT_ENCODER")
    n_runs = {k: len(v) for k, v in (("stock DiT", sd), ("genify DiT", gd), ("stock TE", st), ("genify TE", gt))}

    # True M from the model structure (Tracy logs a cached minimal_matmul's first-seen shape).
    txt = dims(g_ce[g_ce["OP CODE"].isin(GENIFY_MM)].iloc[0], "INPUT_0")[2]
    img = dims(g_step[g_step["OP CODE"].isin(GENIFY_MM)].iloc[5], "INPUT_0")[2]
    m_of = {"one": 1, "img": img, "txt": txt, "joint": img + txt}.get

    dit_rows, dit_glue, dit_extra_stock = compare(s_step, g_step, DIT, m_of)
    ce_rows, ce_glue, ce_extra_stock = compare(s_ce, g_ce, [("context_embedder", 1, "txt", "minimal_matmul")], m_of)
    te_rows, te_glue, te_extra_stock = compare(s_te, g_te, TE)
    te = pd.DataFrame(te_rows)
    te_rows = (
        te.groupby(["matmul", "M", "K", "N", "stock_op"], sort=False)
        .agg(
            calls=("calls", "sum"),
            stock_us=("stock_us", "mean"),
            genify_us=("genify_us", "mean"),
            genify_cores=("genify_cores", "first"),
            stock_cores=("stock_cores", "first"),
            genify_in0=("genify_in0", "first"),
        )
        .reset_index()
        .to_dict("records")
    )
    for r in dit_rows:
        r["group"] = "denoising step"
    for r in ce_rows + te_rows:
        r["group"] = "image (prompt encoding)"
    rows = pd.DataFrame(dit_rows + ce_rows + te_rows)
    rows["ratio"] = rows.genify_us / rows.stock_us
    glue = pd.DataFrame(dit_glue + ce_glue + te_glue)

    lines = ["# FLUX.2-dev: stock matmuls vs. genify_matmul (QB2, 2x2 mesh, per chip, mean of 4 chips)", ""]
    lines += [f"M: img {img}, txt {txt}, joint {img + txt} rows per chip. Profiled runs averaged: {n_runs}.", ""]
    lines += [
        f"Inputs: stock DiT {a.stock_dit}; genify DiT {a.genify_dit}; stock TE {a.stock_te}; genify TE {a.genify_te}",
        "",
    ]
    cols = [
        "matmul",
        "calls",
        "M",
        "K",
        "N",
        "stock_op",
        "stock_us",
        "genify_us",
        "ratio",
        "stock_cores",
        "genify_cores",
    ]
    for grp in ("denoising step", "image (prompt encoding)"):
        g = rows[rows.group == grp]
        lines += [f"## Per {grp}", "", fmt_table(g, cols), ""]

    summary = []
    for grp in ("denoising step", "image (prompt encoding)"):
        g = rows[rows.group == grp]
        s_tot = (g.calls * g.stock_us).sum() / 1e3
        g_tot = (g.calls * g.genify_us).sum() / 1e3
        gl = glue[glue.for_matmul.isin(g.matmul)] if len(glue) else glue
        glue_tot = (gl.calls * gl.us).sum() / 1e3 if len(gl) else 0.0
        summary.append((grp, int(g.calls.sum()), s_tot, g_tot, glue_tot))
    lines += ["## Totals (matmuls x calls)", ""]
    lines += ["| per | calls | stock ms | genify ms | genify / stock | glue ms (genify only) | genify + glue ms |"]
    lines += ["|---|---|---|---|---|---|---|"]
    for grp, calls, s_tot, g_tot, glue_tot in summary:
        lines.append(
            f"| {grp} | {calls} | {s_tot:.2f} | {g_tot:.2f} | {g_tot / s_tot:.3f} | {glue_tot:.2f} | {g_tot + glue_tot:.2f} |"
        )
    (_, _, sd_mm, gd_mm, gd_glue), (_, _, si_mm, gi_mm, gi_glue) = summary
    lines.append(
        f"| image incl. 50 steps | - | {si_mm + 50 * sd_mm:.1f} | {gi_mm + 50 * gd_mm:.1f} | "
        f"{(gi_mm + 50 * gd_mm) / (si_mm + 50 * sd_mm):.3f} | {gi_glue + 50 * gd_glue:.1f} | "
        f"{gi_mm + 50 * gd_mm + gi_glue + 50 * gd_glue:.1f} |"
    )
    lines.append("")

    # Whole-section device time (every op, scaled the same way), averaged and per run.
    def per_run(df, fn, codes):
        totals = []
        for k in range(len(df["us_runs"].iloc[0])):
            one = df.copy()
            one["us"] = [v[k] for v in df["us_runs"]]
            totals.append(fn(one, codes) / 1e3)
        return totals

    def without_ccl(df):
        """The same ops with the fabric-bound ones (all-gather, reduce-scatter, ring SDPA) zeroed: those vary
        by several percent run to run and are not touched by the switch."""
        out = df.copy()
        ccl = out["OP CODE"].str.contains("AllGather|ReduceScatter|RingJointSDPA")
        out["us_runs"] = [tuple(0.0 for _ in v) if c else v for v, c in zip(out["us_runs"], ccl)]
        out["us"] = [0.0 if c else v for v, c in zip(out["us"], ccl)]
        return out

    sections = [
        ("denoising step", step_total, s_step, g_step),
        ("denoising step without CCL ops", step_total, without_ccl(s_step), without_ccl(g_step)),
        ("text encoder (40 layers)", te_total, s_te, g_te),
        ("text encoder without CCL ops", te_total, without_ccl(s_te), without_ccl(g_te)),
    ]
    lines += ["## Whole-section device time (all ops, scaled)", ""]
    lines += [
        "| section | stock ms | genify ms | genify / stock | stock runs | genify runs |",
        "|---|---|---|---|---|---|",
    ]
    for name, fn, s_df, g_df in sections:
        s_tot, g_tot = fn(s_df, STOCK_MM) / 1e3, fn(g_df, GENIFY_MM) / 1e3
        s_runs = ", ".join(f"{t:.2f}" for t in per_run(s_df, fn, STOCK_MM))
        g_runs = ", ".join(f"{t:.2f}" for t in per_run(g_df, fn, GENIFY_MM))
        lines.append(f"| {name} | {s_tot:.2f} | {g_tot:.2f} | {g_tot / s_tot:.3f} | {s_runs} | {g_runs} |")
    lines.append(
        f"| context_embedder | {s_ce.us.sum() / 1e3:.3f} | {g_ce.us.sum() / 1e3:.3f} | "
        f"{g_ce.us.sum() / s_ce.us.sum():.3f} | | |"
    )
    lines.append("")

    lines += ["## Glue ops (genify run only), per call", ""]
    if len(glue):
        agg = (
            glue.groupby(["for_matmul", "op", "in0"], sort=False)
            .agg(count=("us", "size"), calls=("calls", "first"), us_each=("us", "mean"), us_sum=("us", "sum"))
            .reset_index()
        )
        agg["ms_per_unit"] = agg.calls * agg.us_sum / 1e3
        lines += [fmt_table(agg, ["for_matmul", "op", "in0", "count", "us_each", "us_sum", "calls", "ms_per_unit"]), ""]
    extra = dit_extra_stock + ce_extra_stock + te_extra_stock
    lines += [f"Ops only in the stock run: {extra or 'none'}", ""]

    md = "\n".join(lines)
    print(md)
    with open(a.out + ".md", "w") as f:
        f.write(md + "\n")
    rows.to_csv(a.out + ".csv", index=False)
    if len(glue):
        glue.to_csv(a.out + "_glue.csv", index=False)


if __name__ == "__main__":
    main()
