#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Analyze a tt-metal Tracy ops-perf CSV from the SP-prefill one-die engine
(sp_profile_one_die.py): totals, by-category breakdown, top op codes, and a
per-layer GDN/attention split, written to a markdown report.

Run (no device/env needed -- this only parses an existing CSV):
  python3 analyze_ops_csv.py <ops_perf_results.csv> [output.md]
"""
import re
import sys

import numpy as np
import pandas as pd

BH_CLOCK_GHZ = 1.35  # Blackhole AICLK assumption (cycles/ns); overridden if a
# direct ns start/end column is present (none in this schema).

KERNEL_COL = "DEVICE KERNEL DURATION [ns]"
FW_DUR_COL = "DEVICE FW DURATION [ns]"
FW_START_COL = "DEVICE FW START CYCLE"
FW_END_COL = "DEVICE FW END CYCLE"
OP2OP_COL = "OP TO OP LATENCY [ns]"


# ---------------------------------------------------------------- category ---


def category_of(op_code: str) -> str:
    c = op_code
    if re.search(r"^(Matmul|Linear)", c):
        return "matmul"
    if re.search(r"(ScaledDotProduct|SDPA)", c):
        return "sdpa"
    if re.search(r"(ChunkGdn|GatedDelta)", c):
        return "gdn_kernel"
    if re.search(r"(Conv|Halo)", c):
        return "conv1d"
    if re.search(r"(RMSNorm|LayerNorm|Norm)", c) and "GatedDelta" not in c:
        return "norm"
    if ("Heads" in c) and ("Create" in c or "Concat" in c):
        return "heads"
    if re.search(r"(Rotary|Rope)", c):
        return "rope"
    if re.search(r"^Embeddings", c):
        return "embedding"
    if re.search(r"(PagedFillCache|FillCache|UpdateCache|PagedUpdateCache)", c):
        return "kv_cache"
    if re.search(
        r"(Reshape|Permute|Transpose|Slice|Concat|Pad|ToLayout|Tilize|Untilize|"
        r"InterleavedToSharded|ShardedToInterleaved|ToMemoryConfig|Copy|Clone|"
        r"Typecast|Fill|Repeat|Move)",
        c,
    ):
        return "layout"
    if re.search(r"(Unary|Binary|Eltwise|Softplus|Sigmoid|Silu|Exp|Mul|Add|Sub|Div)", c):
        return "eltwise"
    return "reduce/other"


# ------------------------------------------------------------------ shapes ---

_DIM_RE = re.compile(r"^\s*(-?\d+)\[(-?\d+)\]\s*$")


def _parse_dim(val):
    if pd.isna(val):
        return None
    s = str(val)
    m = _DIM_RE.match(s)
    if m:
        return int(m.group(1)), int(m.group(2))
    try:
        v = int(float(s))
        return v, v
    except ValueError:
        return None


def shape_str(row, which: str, idx: int = 0) -> str:
    """Compact [W,Z,Y,X] logical shape string (padded/logical shown only if they differ)."""
    parts = []
    for d in ("W", "Z", "Y", "X"):
        col = f"{which}_{idx}_{d}_PAD[LOGICAL]"
        parsed = _parse_dim(row.get(col)) if col in row else None
        if parsed is None:
            parts.append("-")
        else:
            p, l = parsed
            parts.append(str(l) if p == l else f"{p}/{l}")
    return "[" + ",".join(parts) + "]"


def mem_pair(row) -> str:
    i = row.get("INPUT_0_MEMORY", "-")
    o = row.get("OUTPUT_0_MEMORY", "-")
    i = "-" if pd.isna(i) else i
    o = "-" if pd.isna(o) else o
    return f"{i} -> {o}"


def extract_binop(attrs) -> str:
    m = re.search(r"BinaryOpType::(\w+)", str(attrs))
    return m.group(1) if m else ""


# --------------------------------------------------------------- run split ---


def find_embedding_rows(df: pd.DataFrame):
    mask = df["OP CODE"].str.contains("Embedding", case=False, na=False) & ~df["OP CODE"].str.contains(
        "Rotary", case=False, na=False
    )
    return df.index[mask].tolist()


def split_runs(device_df: pd.DataFrame):
    emb_idx = find_embedding_rows(device_df)
    if len(emb_idx) < 2:
        # Fallback: no clean 2-run split found; treat whole thing as the timed run.
        return device_df.iloc[0:0], device_df, emb_idx
    second = emb_idx[1]
    run1 = device_df.loc[device_df.index < second]
    timed = device_df.loc[device_df.index >= second]
    return run1, timed, emb_idx


# ------------------------------------------------------------ layer split ---


def split_layers(timed: pd.DataFrame):
    """
    Returns a list of dicts, one per decoder layer:
      {type, start, mixer_end, end, rows(df slice)}
    plus the trailing tail rows (final norm + lm head) as a separate df slice.
    Falls back to marker-based (SDPA/gdn) segmentation if the residual-add
    pairing doesn't line up cleanly.
    """
    t = timed.copy()
    t["_binop"] = t["ATTRIBUTES"].apply(extract_binop)
    t["_prev_op"] = t["OP CODE"].shift(1)
    is_add_after_matmul = (
        (t["OP CODE"] == "BinaryNgDeviceOperation")
        & (t["_binop"] == "ADD")
        & (t["_prev_op"] == "MatmulDeviceOperation")
    )
    candidates = t.index[is_add_after_matmul].tolist()

    marker_mask = t["OP CODE"].apply(lambda c: category_of(c) in ("sdpa", "gdn_kernel"))
    # count distinct marker "events": consecutive gdn_kernel rows (prep+scan) collapse to 1
    marker_rows = t.index[marker_mask].tolist()

    layers = []
    used_fallback = False
    if candidates and len(candidates) % 2 == 0 and len(candidates) // 2 >= 1:
        mixer_ends = candidates[0::2]
        layer_ends = candidates[1::2]
        prev_end = timed.index[0] - 1  # row right before the timed run's first row (the embedding op)
        for i, end in enumerate(layer_ends):
            start = prev_end + 1
            mixer_end = mixer_ends[i]
            seg = t.loc[start:end]
            has_sdpa = (seg["OP CODE"] == "SDPAOperation").any()
            has_gdn = seg["OP CODE"].str.contains("ChunkGdn").any()
            ltype = "attention" if has_sdpa else ("gdn" if has_gdn else "unknown")
            layers.append(dict(type=ltype, start=start, mixer_end=mixer_end, end=end, rows=seg))
            prev_end = end
        tail = t.loc[layer_ends[-1] + 1 :]
        # sanity: need at least a couple of layers, else fall back to marker-based split
        if len(layers) < 2:
            used_fallback = True
    else:
        used_fallback = True

    if used_fallback:
        # Marker-based fallback: one layer per contiguous cluster of marker rows.
        layers = []
        clusters = []
        cur = []
        for r in marker_rows:
            if cur and r - cur[-1] > 40:
                clusters.append(cur)
                cur = []
            cur.append(r)
        if cur:
            clusters.append(cur)
        prev_end = timed.index[0] - 1
        for i, cl in enumerate(clusters):
            # end of this layer = row right before the next cluster starts, or end of timed df
            if i + 1 < len(clusters):
                end = clusters[i + 1][0] - 1
            else:
                end = timed.index[-1]
            start = prev_end + 1
            seg = t.loc[start:end]
            has_sdpa = (seg["OP CODE"] == "SDPAOperation").any()
            ltype = "attention" if has_sdpa else "gdn"
            layers.append(dict(type=ltype, start=start, mixer_end=None, end=end, rows=seg))
            prev_end = end
        tail = t.iloc[0:0]

    return layers, tail


# ------------------------------------------------------------------- main ---


def fmt_ms(ns) -> str:
    return f"{ns / 1e6:.3f}"


def main():
    if len(sys.argv) < 2:
        print("usage: analyze.py <csv> [out.md]", file=sys.stderr)
        sys.exit(1)
    csv_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "analysis.md"

    df = pd.read_csv(csv_path, low_memory=False)
    total_rows = len(df)

    device_mask = df[KERNEL_COL].notna()
    dropped = int((~device_mask).sum())
    device_df = df.loc[device_mask].copy()

    run1, timed, emb_idx = split_runs(device_df)

    lines = []
    A = []
    B = []
    C = []
    D = []
    E = []
    F = []
    G = []
    H = []

    # ---------------- A. run split ----------------
    A.append("## A. Run split\n")
    A.append(f"- Total CSV data rows: {total_rows}")
    A.append(f"- Device rows (have {KERNEL_COL}): {len(device_df)}; dropped host-only rows: {dropped}")
    A.append(f"- Embedding op rows found (exact 'Embeddings', excluding RotaryEmbedding): {emb_idx}")
    A.append(
        f"- Run 1 (warmup): {len(run1)} device op rows (rows {run1.index.min() if len(run1) else '-'}..{run1.index.max() if len(run1) else '-'})"
    )
    A.append(
        f"- Run 2 (TIMED, analyzed below): {len(timed)} device op rows (rows {timed.index.min()}..{timed.index.max()})"
    )
    A.append("")

    # ---------------- B. totals ----------------
    kernel_sum_ns = timed[KERNEL_COL].sum()
    fw_sum_ns = timed[FW_DUR_COL].sum()
    span_cycles = timed[FW_END_COL].iloc[-1] - timed[FW_START_COL].iloc[0]
    span_ns = span_cycles / BH_CLOCK_GHZ
    op2op_sum_ns = timed[OP2OP_COL].sum()
    op_count = len(timed)
    traced_estimate_ns = kernel_sum_ns + op_count * 1500

    B.append("## B. Totals (timed run only)\n")
    B.append(
        f"- Clock assumption: Blackhole AICLK = {BH_CLOCK_GHZ} GHz (no direct ns start/end column in this schema; span computed from FW cycle counters)."
    )
    B.append(f"- Sum of DEVICE KERNEL DURATION: **{fmt_ms(kernel_sum_ns)} ms**")
    B.append(
        f"- Sum of DEVICE FW DURATION: **{fmt_ms(fw_sum_ns)} ms** (overlaps across pipelined ops/RISCs; not additive wall time)"
    )
    B.append(f"- Span (last FW end cycle - first FW start cycle): **{fmt_ms(span_ns)} ms**")
    B.append(f"- Sum of OP TO OP LATENCY (idle/dispatch gap, untraced): **{fmt_ms(op2op_sum_ns)} ms**")
    B.append(f"- Op count: {op_count}")
    B.append(
        f"- Sanity: kernel_sum + op2op_sum = {fmt_ms(kernel_sum_ns + op2op_sum_ns)} ms vs span {fmt_ms(span_ns)} ms (should be close)"
    )
    B.append(
        f"- Best traced-replay approximation: kernel_sum + op_count*1.5us = **{fmt_ms(traced_estimate_ns)} ms** "
        f"(removes untraced dispatch overhead, replaces with a fixed ~1.5us/op trace-replay overhead)"
    )
    B.append("")

    # ---------------- C. by category ----------------
    timed = timed.copy()
    timed["_cat"] = timed["OP CODE"].apply(category_of)
    cat_group = timed.groupby("_cat")[KERNEL_COL].agg(["count", "sum"]).rename(columns={"count": "n", "sum": "ns"})
    cat_group["ms"] = cat_group["ns"] / 1e6
    cat_group["pct"] = 100 * cat_group["ns"] / kernel_sum_ns
    cat_group["avg_us"] = cat_group["ns"] / cat_group["n"] / 1e3
    cat_group = cat_group.sort_values("ns", ascending=False)

    C.append("## C. By category\n")
    C.append("| category | op count | total kernel ms | % of kernel sum | avg us/op |")
    C.append("|---|---:|---:|---:|---:|")
    for cat, r in cat_group.iterrows():
        C.append(f"| {cat} | {int(r['n'])} | {r['ms']:.3f} | {r['pct']:.1f}% | {r['avg_us']:.2f} |")
    unmapped = sorted(timed.loc[timed["_cat"] == "reduce/other", "OP CODE"].unique().tolist())
    C.append("")
    C.append(f"Unmapped op codes (fell into reduce/other): {unmapped if unmapped else '(none)'}")
    C.append("")

    # ---------------- D. top 30 op codes ----------------
    def most_common(s):
        s = s.dropna()
        return s.mode().iloc[0] if not s.empty else "-"

    timed["_mem_pair"] = timed.apply(mem_pair, axis=1)
    grp = timed.groupby("OP CODE")
    rows = []
    for code, g in grp:
        ns = g[KERNEL_COL]
        cat = category_of(code)
        fidelity = most_common(g["MATH FIDELITY"]) if cat == "matmul" else "-"
        rows.append(
            dict(
                op_code=code,
                n=len(g),
                total_ms=ns.sum() / 1e6,
                pct=100 * ns.sum() / kernel_sum_ns,
                avg_us=ns.mean() / 1e3,
                min_us=ns.min() / 1e3,
                max_us=ns.max() / 1e3,
                mem=most_common(g["_mem_pair"]),
                core=most_common(g["CORE COUNT"]),
                fidelity=fidelity,
            )
        )
    top_codes = sorted(rows, key=lambda r: -r["total_ms"])[:30]

    D.append("## D. Top 30 OP CODEs by total kernel time\n")
    D.append("| op code | count | total ms | % | avg us | min us | max us | mem (in->out) | core count | fidelity |")
    D.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---|")
    for r in top_codes:
        D.append(
            f"| {r['op_code']} | {r['n']} | {r['total_ms']:.3f} | {r['pct']:.1f}% | {r['avg_us']:.2f} | "
            f"{r['min_us']:.2f} | {r['max_us']:.2f} | {r['mem']} | {r['core']} | {r['fidelity']} |"
        )
    D.append("")

    # ---------------- E. top 20 individual instances ----------------
    top_inst = timed.sort_values(KERNEL_COL, ascending=False).head(20)
    E.append("## E. Top 20 individual op instances by kernel duration\n")
    E.append("| row | op code | us | input_0 shape | output_0 shape | mem (in->out) | core count |")
    E.append("|---:|---|---:|---|---|---|---:|")
    for idx, r in top_inst.iterrows():
        E.append(
            f"| {idx} | {r['OP CODE']} | {r[KERNEL_COL] / 1e3:.2f} | {shape_str(r, 'INPUT')} | "
            f"{shape_str(r, 'OUTPUT')} | {mem_pair(r)} | {r['CORE COUNT']} |"
        )
    E.append("")

    # ---------------- F. per-layer split ----------------
    layers, tail = split_layers(timed)
    for L in layers:
        L["kernel_ms"] = L["rows"][KERNEL_COL].sum() / 1e6
        L["n_ops"] = len(L["rows"])
        if L["mixer_end"] is not None:
            mixer_rows = timed.loc[L["start"] : L["mixer_end"]]
            mlp_rows = timed.loc[L["mixer_end"] + 1 : L["end"]]
            L["mixer_ms"] = mixer_rows[KERNEL_COL].sum() / 1e6
            L["mlp_ms"] = mlp_rows[KERNEL_COL].sum() / 1e6
        else:
            L["mixer_ms"] = None
            L["mlp_ms"] = None

    gdn_layers = [L for L in layers if L["type"] == "gdn"]
    attn_layers = [L for L in layers if L["type"] == "attention"]

    def stats(lst, key):
        vals = [L[key] for L in lst]
        return (np.mean(vals), np.min(vals), np.max(vals)) if vals else (float("nan"),) * 3

    F.append("## F. Per-layer split\n")
    F.append(
        f"- Layers found: {len(layers)} total = {len(gdn_layers)} GDN + {len(attn_layers)} attention "
        f"(+ {len(tail)}-row tail = final norm + LM head, excluded from per-layer stats)"
    )
    for name, lst in (("GDN", gdn_layers), ("attention", attn_layers)):
        if not lst:
            F.append(f"- {name}: none found")
            continue
        mean_k, min_k, max_k = stats(lst, "kernel_ms")
        mean_n, _, _ = stats(lst, "n_ops")
        F.append(
            f"- {name} layer kernel ms: mean {mean_k:.3f}, min {min_k:.3f}, max {max_k:.3f}; mean op count {mean_n:.1f}"
        )
        if lst[0]["mixer_ms"] is not None:
            mean_mix, _, _ = stats(lst, "mixer_ms")
            mean_mlp, _, _ = stats(lst, "mlp_ms")
            F.append(f"  - mean token-mixer part: {mean_mix:.3f} ms; mean MLP part: {mean_mlp:.3f} ms")
    F.append("")

    # ---------------- G. representative layers ----------------
    def pick_median(lst):
        if not lst:
            return None
        med = np.median([L["kernel_ms"] for L in lst])
        return min(lst, key=lambda L: abs(L["kernel_ms"] - med))

    def render_layer(L, title):
        out = [f"### {title} (rows {L['start']}..{L['end']}, {L['kernel_ms']:.3f} ms, {L['n_ops']} ops)\n"]
        for idx, r in L["rows"].iterrows():
            gap_us = r[OP2OP_COL] / 1e3
            out.append(
                f"- {r['OP CODE']}: {r[KERNEL_COL]/1e3:.2f} us, gap {gap_us:.2f} us, "
                f"in {shape_str(r,'INPUT')} out {shape_str(r,'OUTPUT')}, {mem_pair(r)}, cores={r['CORE COUNT']}"
            )
        out.append("")
        return out

    G.append("## G. Representative layers (median kernel-time of each type)\n")
    med_gdn = pick_median(gdn_layers)
    med_attn = pick_median(attn_layers)
    if med_gdn:
        G += render_layer(med_gdn, "Representative GDN layer")
    if med_attn:
        G += render_layer(med_attn, "Representative attention layer")

    # ---------------- H. summary ----------------
    top3 = cat_group.head(3)
    top3_str = ", ".join(f"{cat} {r['ms']:.3f} ms" for cat, r in top3.iterrows())
    # biggest avoidable-looking cost: layout churn inside GDN layers vs the conv1d math it wraps,
    # and how much of that layout time is a pure DRAM_INTERLEAVED -> DRAM_INTERLEAVED round trip.
    layout_ms = cat_group.loc["layout", "ms"] if "layout" in cat_group.index else 0.0
    layout_n = int(cat_group.loc["layout", "n"]) if "layout" in cat_group.index else 0
    conv_ms = cat_group.loc["conv1d", "ms"] if "conv1d" in cat_group.index else 0.0
    gdn_layout_ms = sum(
        L["rows"].loc[L["rows"]["_cat"] == "layout", KERNEL_COL].sum() / 1e6 for L in layers if L["type"] == "gdn"
    )
    gdn_total_ms = sum(L["kernel_ms"] for L in layers if L["type"] == "gdn")
    lay = timed[timed["_cat"] == "layout"]
    dram_rt_ms = (
        lay.loc[
            (lay["INPUT_0_MEMORY"] == "DEV_0_DRAM_INTERLEAVED") & (lay["OUTPUT_0_MEMORY"] == "DEV_0_DRAM_INTERLEAVED"),
            KERNEL_COL,
        ].sum()
        / 1e6
    )
    dram_rt_n = int(
        (
            (lay["INPUT_0_MEMORY"] == "DEV_0_DRAM_INTERLEAVED") & (lay["OUTPUT_0_MEMORY"] == "DEV_0_DRAM_INTERLEAVED")
        ).sum()
    )

    H.append("## H. Summary\n")
    H.append(
        f"1. kernel-sum {fmt_ms(kernel_sum_ns)} ms vs span {fmt_ms(span_ns)} ms vs op-to-op gap sum {fmt_ms(op2op_sum_ns)} ms "
        f"({100*op2op_sum_ns/span_ns:.1f}% of span is untraced dispatch gap) out of an external wall time of ~64.5-66.5 ms."
    )
    H.append(f"2. Top 3 categories by kernel time: {top3_str}.")
    H.append(
        f"3. {gdn_layout_ms:.3f} ms of the {layout_ms:.3f} ms total layout time ({100*gdn_layout_ms/layout_ms:.0f}%) sits "
        f"inside the 18 GDN layers ({100*gdn_layout_ms/gdn_total_ms:.0f}% of GDN-layer time), vs only {conv_ms:.3f} ms "
        f"of actual conv1d math there -- and {dram_rt_ms:.3f} ms of the {layout_ms:.3f} ms layout total ({dram_rt_n}/{layout_n} ops) "
        f"is a plain DRAM_INTERLEAVED -> DRAM_INTERLEAVED round trip with no sharding/compute in between."
    )
    H.append("")

    lines = ["# Qwen3.5-2B prefill (1024 tok, 1 die, untraced) ops profile\n"]
    lines += A + B + C + D + E + F + G + H

    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
