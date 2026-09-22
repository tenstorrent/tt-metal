#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E2E analyzer for the Qwen SP-prefill -> TP=4 decode handoff: two Tracy ops-perf CSVs,
one from the 4-die SP prefill process, one from the TP=4 inject+decode process.
Attributes each device op to the nearest Tracy signpost (layer/phase markers) for
per-layer/per-step tables; falls back to an embedding heuristic when none are present.

Usage: python3 analyze_e2e_ops.py <prefill_csv> <decode_csv> <out.md> [log_with_[e2e]_lines]

Capture (SP_PREFILL_HANDOFF.md sec 6a): plain pytest for the prefill test; `python -m
tracy -r -p -m pytest "<test>" -q -s` for the inject/decode tests. CSVs land under
generated/profiler/reports/<ts>/, .tracy captures under build/profiler/build_wasm/traces/.
"""
import re
import sys

import numpy as np
import pandas as pd

BH_CLOCK_GHZ = 1.35  # Blackhole AICLK assumption (cycles/ns); per-die/per-device
# self-relative only -- see analyze4.py's header comment on why
# DEVICE FW CYCLE counters are not cross-die/cross-process comparable.

KERNEL_COL = "DEVICE KERNEL DURATION [ns]"
FW_START_COL = "DEVICE FW START CYCLE"
FW_END_COL = "DEVICE FW END CYCLE"
OP2OP_COL = "OP TO OP LATENCY [ns]"

STEP_RE = re.compile(r"^decode step (\d+)$")
LAYER_RE = re.compile(r"^(prefill|decode) L(\d+) (gdn|attn)$")


# ---------------------------------------------------------------- category ---


def category_of(op_code: str) -> str:
    c = op_code
    if re.search(r"(AllGather|ReduceScatter|AllReduce|Gather|Scatter|Mcast)", c):
        return "ccl"
    if re.search(r"(Send|Recv|Socket)", c):
        return "comm"
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


def fmt_ms(ns) -> str:
    return f"{ns / 1e6:.3f}"


# --------------------------------------------------------------- signposts ---


def load_csv(path):
    """Read the full raw CSV (host + device rows). Tags every row with `_sp_occ` (int,
    0 = before any signpost, else the 1-based index of the most recent TT_SIGNPOST row
    at/before it in CSV row order) and `_sp_label` (that occurrence's header text, or
    None for occ 0). Returns (df, has_signposts, label_by_occ)."""
    df = pd.read_csv(path, low_memory=False).reset_index(drop=True)
    is_sp = df["OP TYPE"] == "signpost"
    has_signposts = bool(is_sp.any())
    occ = is_sp.cumsum()
    df["_sp_occ"] = occ
    label_by_occ = {0: None}
    for i in df.index[is_sp]:
        label_by_occ[int(occ.loc[i])] = df.loc[i, "OP CODE"]
    df["_sp_label"] = df["_sp_occ"].map(label_by_occ)
    return df, has_signposts, label_by_occ


def signpost_occurrences(rows: pd.DataFrame):
    """rows: a subset of device rows (has KERNEL_COL), in original order, carrying
    `_sp_occ`/`_sp_label`. Returns an ordered list of dicts, one per occurrence that has
    >=1 device op in `rows`: {occ, label, n, kernel_ms}."""
    if rows.empty:
        return []
    out = []
    for occ, g in rows.groupby("_sp_occ", sort=True):
        out.append(dict(occ=int(occ), label=g["_sp_label"].iloc[0], n=len(g), kernel_ms=g[KERNEL_COL].sum() / 1e6))
    return out


def signpost_table_md(occs, title):
    md = [f"{title}\n", "| occurrence | signpost | ops | kernel ms |", "|---:|---|---:|---:|"]
    for o in occs:
        md.append(
            f"| {o['occ']} | {o['label'] if o['label'] else '(before any signpost)'} | {o['n']} | {o['kernel_ms']:.3f} |"
        )
    md.append("")
    return md


# --------------------------------------------------------------- run split ---


def find_embedding_rows(df: pd.DataFrame):
    mask = df["OP CODE"].str.contains("Embedding", case=False, na=False) & ~df["OP CODE"].str.contains(
        "Rotary", case=False, na=False
    )
    return df.index[mask].tolist()


def split_last_run(device_df: pd.DataFrame):
    """Fallback (no signposts): last complete 'Embeddings ... -> end of rows' pass."""
    emb_idx = find_embedding_rows(device_df)
    if not emb_idx:
        return device_df, emb_idx
    last = emb_idx[-1]
    return device_df.loc[last:], emb_idx


# ------------------------------------------------------------ layer split ---


def split_layers(timed: pd.DataFrame):
    """Fallback (no signposts) heuristic layer split: residual-add-after-matmul pairing,
    else marker-based (SDPA/gdn) clustering. Returns (layers, tail)."""
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
    marker_rows = t.index[marker_mask].tolist()

    layers = []
    used_fallback = False
    if candidates and len(candidates) % 2 == 0 and len(candidates) // 2 >= 1:
        mixer_ends = candidates[0::2]
        layer_ends = candidates[1::2]
        prev_end = timed.index[0] - 1
        for i, end in enumerate(layer_ends):
            start = prev_end + 1
            seg = t.loc[start:end]
            has_sdpa = (seg["OP CODE"] == "SDPAOperation").any()
            has_gdn = seg["OP CODE"].str.contains("ChunkGdn").any()
            ltype = "attention" if has_sdpa else ("gdn" if has_gdn else "unknown")
            layers.append(dict(type=ltype, start=start, mixer_end=mixer_ends[i], end=end, rows=seg))
            prev_end = end
        tail = t.loc[layer_ends[-1] + 1 :]
        if len(layers) < 2:
            used_fallback = True
    else:
        used_fallback = True

    if used_fallback:
        layers = []
        clusters, cur = [], []
        for r in marker_rows:
            if cur and r - cur[-1] > 40:
                clusters.append(cur)
                cur = []
            cur.append(r)
        if cur:
            clusters.append(cur)
        prev_end = timed.index[0] - 1
        for i, cl in enumerate(clusters):
            end = clusters[i + 1][0] - 1 if i + 1 < len(clusters) else timed.index[-1]
            start = prev_end + 1
            seg = t.loc[start:end]
            has_sdpa = (seg["OP CODE"] == "SDPAOperation").any()
            ltype = "attention" if has_sdpa else "gdn"
            layers.append(dict(type=ltype, start=start, mixer_end=None, end=end, rows=seg))
            prev_end = end
        tail = t.iloc[0:0]
    return layers, tail


def signpost_layer_rows(rows: pd.DataFrame, prefix: str):
    """rows: device rows (subset, with _sp_occ/_sp_label) for one die/step. Returns an
    ordered list of {label, n, kernel_ms} for occurrences whose label matches
    '<prefix> L<idx> gdn|attn' or is the prefix's tail/head label."""
    occs = signpost_occurrences(rows)
    tail_label = "prefill tail" if prefix == "prefill" else "decode head"
    out = []
    for o in occs:
        lbl = o["label"]
        if not lbl:
            continue
        m = LAYER_RE.match(lbl)
        if (m and m.group(1) == prefix) or lbl == tail_label:
            out.append(o)
    return out


# ------------------------------------------------------------------- misc ---


def infer_die_order(per_die_stats):
    """Same recovery as analyze4.py: die 0 = send-only, die N = recv-only (+ tail),
    middle dies ordered by ascending span. Returns {device_id: die_index}."""
    devs = list(per_die_stats.keys())
    die0 = [d for d in devs if per_die_stats[d]["n_recv"] == 0]
    dieN = [d for d in devs if per_die_stats[d]["n_send"] == 0]
    if not (len(die0) == 1 and len(dieN) == 1 and die0[0] != dieN[0]):
        # No clean SP pipeline signature (e.g. a single die, or a TP/SPMD dump where
        # every die sends+receives symmetrically) -- fall back to raw DEVICE ID order.
        return {d: i for i, d in enumerate(sorted(devs))}
    middle = sorted([d for d in devs if d not in die0 and d not in dieN], key=lambda d: per_die_stats[d]["span_ms"])
    order = [die0[0]] + middle + [dieN[0]]
    return {dev: i for i, dev in enumerate(order)}


def top_ops_md(rows: pd.DataFrame, n: int, title: str):
    grp = rows.groupby("OP CODE")[KERNEL_COL].agg(["count", "sum", "mean", "min", "max"])
    grp = grp.sort_values("sum", ascending=False).head(n)
    md = [f"{title}\n", "| op code | count | total ms | avg us | min us | max us |", "|---|---:|---:|---:|---:|---:|"]
    for code, r in grp.iterrows():
        md.append(
            f"| {code} | {int(r['count'])} | {r['sum']/1e6:.3f} | {r['mean']/1e3:.2f} | {r['min']/1e3:.2f} | {r['max']/1e3:.2f} |"
        )
    md.append("")
    return md


def category_table_md(rows: pd.DataFrame, title: str):
    rows = rows.copy()
    rows["_cat"] = rows["OP CODE"].apply(category_of)
    kernel_sum_ns = rows[KERNEL_COL].sum()
    g = rows.groupby("_cat")[KERNEL_COL].agg(["count", "sum"]).rename(columns={"count": "n", "sum": "ns"})
    g["ms"] = g["ns"] / 1e6
    g["pct"] = 100 * g["ns"] / kernel_sum_ns if kernel_sum_ns else 0.0
    g["avg_us"] = g["ns"] / g["n"] / 1e3
    g = g.sort_values("ns", ascending=False)
    md = [
        f"{title}\n",
        "| category | op count | total kernel ms | % of kernel sum | avg us/op |",
        "|---|---:|---:|---:|---:|",
    ]
    for cat, r in g.iterrows():
        md.append(f"| {cat} | {int(r['n'])} | {r['ms']:.3f} | {r['pct']:.1f}% | {r['avg_us']:.2f} |")
    ccl_present = sorted(set(rows.loc[rows["_cat"] == "ccl", "OP CODE"]))
    md.append("")
    md.append(f"ccl op codes seen here: {ccl_present if ccl_present else '(none in this data)'}")
    md.append("")
    return md, g


def span_ms_of(rows: pd.DataFrame) -> float:
    if rows.empty:
        return 0.0
    return (rows[FW_END_COL].iloc[-1] - rows[FW_START_COL].iloc[0]) / BH_CLOCK_GHZ / 1e6


# =============================================================== PREFILL ===


def analyze_prefill(csv_path):
    df, has_signposts, label_by_occ = load_csv(csv_path)
    device_df = df.loc[df[KERNEL_COL].notna()].copy()
    dev_ids = sorted(device_df["DEVICE ID"].unique(), key=int)

    mode = (
        "signpost"
        if has_signposts and any(l == "prefill start" for l in label_by_occ.values())
        else "embedding-heuristic"
    )

    per_die = {}
    for dev in dev_ids:
        d = device_df[device_df["DEVICE ID"] == dev]
        if mode == "signpost":
            start_occs = [occ for occ, l in label_by_occ.items() if l == "prefill start"]
            timed = d[d["_sp_occ"] >= max(start_occs)]
            emb_idx = []
        else:
            timed, emb_idx = split_last_run(d)
        timed = timed.copy()
        kernel_sum_ns = timed[KERNEL_COL].sum()
        span_ns = (timed[FW_END_COL].iloc[-1] - timed[FW_START_COL].iloc[0]) if len(timed) else 0
        op2op_sum_ns = timed[OP2OP_COL].sum()
        timed["_cat"] = timed["OP CODE"].apply(category_of)
        comm_ns = timed.loc[timed["_cat"] == "comm", KERNEL_COL].sum()
        send_n = timed["OP CODE"].str.contains("Send", na=False).sum()
        recv_n = timed["OP CODE"].str.contains("Recv", na=False).sum()
        per_die[dev] = dict(
            timed=timed,
            emb_idx=emb_idx,
            kernel_sum_ms=kernel_sum_ns / 1e6,
            span_ms=span_ns / BH_CLOCK_GHZ / 1e6,
            op2op_sum_ms=op2op_sum_ns / 1e6,
            comm_ms=comm_ns / 1e6,
            n_send=send_n,
            n_recv=recv_n,
            n_ops=len(timed),
        )

    die_of = infer_die_order(per_die)
    die_indices = sorted(die_of.values())
    dev_of_die = {v: k for k, v in die_of.items()}
    last_die_dev = dev_of_die[die_indices[-1]]
    last = per_die[last_die_dev]

    # F4: bounded span "prefill start" -> end of "prefill tail" on the last die, as opposed to
    # `last["span_ms"]` above (which runs to the end of that die's remaining rows and can
    # include later, unrelated signposts such as "sp export start"). Also record which pass
    # this is: `timed` above is already sliced from max(start_occs) -- the LAST "prefill start"
    # occurrence in the CSV, i.e. the MEASURED pass when QWEN36_E2E_WARMUP=1 produced two.
    tail_bounded_span_ms = None
    pass_note = "no signposts (embedding-heuristic mode)"
    if mode == "signpost":
        start_occs = sorted(occ for occ, l in label_by_occ.items() if l == "prefill start")
        tail_occs = [occ for occ, l in label_by_occ.items() if l == "prefill tail"]
        pass_note = f"prefill start occurrence {len(start_occs)} of {len(start_occs)} in the CSV (the LAST one)"
        if tail_occs:
            bounded = last["timed"][last["timed"]["_sp_occ"] <= max(tail_occs)]
            tail_bounded_span_ms = span_ms_of(bounded)

    A = ["## A. Prefill\n"]
    A.append(f"- Source: `{csv_path}`; devices found: {dev_ids}; layer/phase-split mode: **{mode}**")
    A.append(f"- DEVICE ID -> die index: {dict(sorted(die_of.items(), key=lambda kv: kv[1]))}")
    if mode != "signpost":
        A.append(
            f"- Embedding passes per die (last used as timed run): {[len(per_die[d]['emb_idx']) for d in dev_ids]}"
        )
    A.append("")
    A.append("### A.1 Cross-die table\n")
    A.append("| die | ops | kernel sum ms | comm ms | gap ms | span ms |")
    A.append("|---:|---:|---:|---:|---:|---:|")
    for i in die_indices:
        st = per_die[dev_of_die[i]]
        A.append(
            f"| {i} | {st['n_ops']} | {st['kernel_sum_ms']:.3f} | {st['comm_ms']:.3f} | {st['op2op_sum_ms']:.3f} | {st['span_ms']:.3f} |"
        )
    A.append("")

    A.append(f"### A.2 Die {die_indices[-1]} category table\n")
    cat_md, cat_group = category_table_md(last["timed"], f"Die {die_indices[-1]} (DEVICE ID {last_die_dev})")
    A += cat_md

    A.append(f"### A.3 Die {die_indices[-1]} per-layer split (GDN / attention)\n")
    if mode == "signpost":
        occs = signpost_layer_rows(last["timed"], "prefill")
        if occs:
            A.append("| signpost | ops | kernel ms |")
            A.append("|---|---:|---:|")
            for o in occs:
                A.append(f"| {o['label']} | {o['n']} | {o['kernel_ms']:.3f} |")
        else:
            A.append("(no prefill L*/tail signposts found on this die)")
        A.append("")
    else:
        layers, tail = split_layers(last["timed"])
        for L in layers:
            L["kernel_ms"] = L["rows"][KERNEL_COL].sum() / 1e6
        A.append(f"- Layers found: {len(layers)} ({[L['type'] for L in layers]})")
        for L in layers:
            A.append(f"  - {L['type']}: {L['kernel_ms']:.3f} ms ({len(L['rows'])} ops, rows {L['start']}..{L['end']})")
        if len(tail):
            A.append(f"  - tail (final norm + LM head): {tail[KERNEL_COL].sum()/1e6:.3f} ms ({len(tail)} ops)")
        A.append("")

    A.append(f"### A.4 Die {die_indices[-1]} top 10 ops\n")
    A += top_ops_md(last["timed"], 10, "")

    A.append(f"### A.5 Die {die_indices[-1]} comm ops\n")
    comm_rows = last["timed"][last["timed"]["OP CODE"].apply(lambda c: category_of(c) == "comm")]
    if len(comm_rows):
        A.append("| op code | us | layer |")
        A.append("|---|---:|---|")
        for idx, r in comm_rows.iterrows():
            layer = r["_sp_label"] if mode == "signpost" and r.get("_sp_label") else "-"
            A.append(f"| {r['OP CODE']} | {r[KERNEL_COL]/1e3:.2f} | {layer} |")
    else:
        A.append("(no comm ops on this die's timed run)")
    A.append("")

    if has_signposts:
        A.append(f"### A.6 Die {die_indices[-1]} signposts (all occurrences)\n")
        A += signpost_table_md(signpost_occurrences(last["timed"]), "")

    A.append("")
    return A, dict(
        mode=mode,
        last_span_ms=last["span_ms"],
        die_indices=die_indices,
        tail_bounded_span_ms=tail_bounded_span_ms,
        pass_note=pass_note,
    )


# ================================================================ DECODE ===


def detect_decode_steps(dev_rows: pd.DataFrame, has_signposts: bool):
    """Returns (steps, mode, pre_step_rows). steps: list of {idx, rows}. pre_step_rows:
    rows before the first detected step (weight loads / cache injection)."""
    if has_signposts:
        uniq = dev_rows.drop_duplicates(subset="_sp_occ")[["_sp_occ", "_sp_label"]]
        step_idx_by_occ = {}
        for _, r in uniq.iterrows():
            m = STEP_RE.match(r["_sp_label"]) if r["_sp_label"] else None
            if m:
                step_idx_by_occ[int(r["_sp_occ"])] = int(m.group(1))
        occs = sorted(step_idx_by_occ)
        if occs:
            steps = []
            for k, occ in enumerate(occs):
                lo = occ
                hi = occs[k + 1] - 1 if k + 1 < len(occs) else dev_rows["_sp_occ"].max()
                seg = dev_rows[(dev_rows["_sp_occ"] >= lo) & (dev_rows["_sp_occ"] <= hi)]
                steps.append(dict(idx=step_idx_by_occ[occ], rows=seg))
            pre = dev_rows[dev_rows["_sp_occ"] < occs[0]]
            return steps, "signpost", pre
    emb_idx = find_embedding_rows(dev_rows)
    if not emb_idx:
        return [], "embedding-heuristic", dev_rows
    steps = []
    for k, e in enumerate(emb_idx):
        end = emb_idx[k + 1] - 1 if k + 1 < len(emb_idx) else dev_rows.index[-1]
        steps.append(dict(idx=k, rows=dev_rows.loc[e:end]))
    pre = dev_rows.loc[: emb_idx[0] - 1]
    return steps, "embedding-heuristic", pre


def analyze_decode(csv_path):
    df, has_signposts, label_by_occ = load_csv(csv_path)
    device_df = df.loc[df[KERNEL_COL].notna()].copy()
    dev_ids = sorted(device_df["DEVICE ID"].unique(), key=int)
    rep_dev = dev_ids[0]

    per_dev_steps = {}
    mode = None
    for dev in dev_ids:
        d = device_df[device_df["DEVICE ID"] == dev]
        steps, mode, pre = detect_decode_steps(d, has_signposts)
        for s in steps:
            s["kernel_ms"] = s["rows"][KERNEL_COL].sum() / 1e6
            s["span_ms"] = span_ms_of(s["rows"])
            s["gap_ms"] = s["rows"][OP2OP_COL].sum() / 1e6
            s["n_ops"] = len(s["rows"])
        per_dev_steps[dev] = dict(steps=steps, pre=pre)

    rep_steps = per_dev_steps[rep_dev]["steps"]
    n_steps = len(rep_steps)

    B = ["## B. Decode\n"]
    B.append(
        f"- Source: `{csv_path}`; devices found: {dev_ids}; representative device: {rep_dev}; step-split mode: **{mode}**"
    )
    B.append(f"- Steps detected (representative device): {n_steps}")
    B.append("")

    B.append("### B.1 Per-step table (device %d, min/max across devices)\n" % rep_dev)
    B.append(
        "| step | ops | kernel ms | kernel ms min | kernel ms max | span ms | span ms min | span ms max | gap ms |"
    )
    B.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in range(n_steps):
        rep = rep_steps[k]
        kmins, kmaxs, smins, smaxs = [], [], [], []
        for dev in dev_ids:
            steps_d = per_dev_steps[dev]["steps"]
            if k < len(steps_d):
                kmins.append(steps_d[k]["kernel_ms"])
                smins.append(steps_d[k]["span_ms"])
        kmax = max(kmins) if kmins else float("nan")
        kmin = min(kmins) if kmins else float("nan")
        smax = max(smins) if smins else float("nan")
        smin = min(smins) if smins else float("nan")
        B.append(
            f"| {rep['idx']} | {rep['n_ops']} | {rep['kernel_ms']:.3f} | {kmin:.3f} | {kmax:.3f} | "
            f"{rep['span_ms']:.3f} | {smin:.3f} | {smax:.3f} | {rep['gap_ms']:.3f} |"
        )
    B.append("")

    if n_steps:
        mid = n_steps // 2
        rep_rows = rep_steps[mid]["rows"]
        B.append(f"### B.2 Category table (step {rep_steps[mid]['idx']}, device {rep_dev})\n")
        cat_md, _ = category_table_md(rep_rows, "")
        B += cat_md

        B.append(f"### B.3 Per-layer split (step {rep_steps[mid]['idx']}, device {rep_dev})\n")
        if mode == "signpost":
            occs = signpost_layer_rows(rep_rows, "decode")
            if occs:
                B.append("| signpost | ops | kernel ms |")
                B.append("|---|---:|---:|")
                for o in occs:
                    B.append(f"| {o['label']} | {o['n']} | {o['kernel_ms']:.3f} |")
            else:
                B.append("(no decode L*/head signposts found in this step)")
            B.append("")
        else:
            layers, tail = split_layers(rep_rows)
            for L in layers:
                L["kernel_ms"] = L["rows"][KERNEL_COL].sum() / 1e6
            B.append(f"- Layers found: {len(layers)} ({[L['type'] for L in layers]})")
            for L in layers:
                B.append(f"  - {L['type']}: {L['kernel_ms']:.3f} ms ({len(L['rows'])} ops)")
            if len(tail):
                B.append(f"  - tail (norm + LM head): {tail[KERNEL_COL].sum()/1e6:.3f} ms ({len(tail)} ops)")
            B.append("")

        B.append(f"### B.4 Top 15 ops (step {rep_steps[mid]['idx']}, device {rep_dev})\n")
        B += top_ops_md(rep_rows, 15, "")

    B.append("### B.5 Op count per step (device %d)\n" % rep_dev)
    B.append("| step | op count |")
    B.append("|---:|---:|")
    for s in rep_steps:
        B.append(f"| {s['idx']} | {s['n_ops']} |")
    B.append("")

    if mode == "signpost":
        B.append("### B.6 Step x layer matrix (kernel ms, device %d)\n" % rep_dev)
        layer_cols = []
        matrix = []
        for s in rep_steps:
            occs = signpost_layer_rows(s["rows"], "decode")
            row = {}
            for o in occs:
                col = o["label"].replace("decode ", "")
                row[col] = o["kernel_ms"]
                if col not in layer_cols:
                    layer_cols.append(col)
            matrix.append((s["idx"], row))
        B.append("| step | " + " | ".join(layer_cols) + " |")
        B.append("|---:|" + "---:|" * len(layer_cols))
        for idx, row in matrix:
            cells = " | ".join(f"{row.get(c, float('nan')):.3f}" if c in row else "-" for c in layer_cols)
            B.append(f"| {idx} | {cells} |")
        B.append("")

    return B, dict(
        mode=mode,
        n_steps=n_steps,
        rep_steps=rep_steps,
        rep_dev=rep_dev,
        per_dev_steps=per_dev_steps,
        dev_ids=dev_ids,
        has_signposts=has_signposts,
        label_by_occ=label_by_occ,
    )


def analyze_pre_step(decode_info, rep_dev):
    pre = decode_info["per_dev_steps"][rep_dev]["pre"]
    C = ["## C. Non-step device ops before the first decode step\n"]
    if pre.empty:
        C.append("(none -- weight loads / any host->device cache injection produced no device ops here)")
        C.append("")
        return C
    if decode_info["mode"] == "signpost":
        inj_start = [o for o, l in decode_info["label_by_occ"].items() if l == "inject start"]
        inj_end = [o for o, l in decode_info["label_by_occ"].items() if l == "inject end"]
        if inj_start and inj_end:
            seg = pre[(pre["_sp_occ"] >= inj_start[0]) & (pre["_sp_occ"] <= inj_end[0])]
            C.append(f"- inject start..inject end window: {len(seg)} ops, {seg[KERNEL_COL].sum()/1e6:.3f} ms total")
    C.append(f"- Total: {len(pre)} ops, {pre[KERNEL_COL].sum()/1e6:.3f} ms")
    g = pre.groupby("OP CODE")[KERNEL_COL].agg(["count", "sum"]).sort_values("sum", ascending=False)
    C.append("")
    C.append("| op code | count | total ms |")
    C.append("|---|---:|---:|")
    for code, r in g.iterrows():
        C.append(f"| {code} | {int(r['count'])} | {r['sum']/1e6:.3f} |")
    C.append("")
    return C


def analyze_e2e_summary(prefill_info, decode_info, log_path):
    D = ["## D. E2E summary\n"]
    D.append("| metric | value |")
    D.append("|---|---:|")
    D.append(
        f"| prefill wavefront span (die {prefill_info['die_indices'][-1]}), ms | {prefill_info['last_span_ms']:.3f} |"
    )
    tbs = prefill_info.get("tail_bounded_span_ms")
    tbs_str = f"{tbs:.3f}" if tbs is not None else "n/a (no 'prefill tail' signpost found)"
    D.append(
        f"| prefill span die {prefill_info['die_indices'][-1]} "
        f"(from signposts: 'prefill start' -> 'prefill tail' end), ms | {tbs_str} |"
    )
    D.append("")
    D.append(f"- Prefill pass analysed: {prefill_info.get('pass_note', 'n/a')}.")
    D.append("")
    D.append("| metric | value |")
    D.append("|---|---:|")
    rep_steps = decode_info["rep_steps"]
    if rep_steps:
        mean_k = float(np.mean([s["kernel_ms"] for s in rep_steps]))
        mean_s = float(np.mean([s["span_ms"] for s in rep_steps]))
        sum_k = float(np.sum([s["kernel_ms"] for s in rep_steps]))
        sum_s = float(np.sum([s["span_ms"] for s in rep_steps]))
        D.append(f"| decode step mean kernel ms | {mean_k:.3f} |")
        D.append(f"| decode step mean span ms | {mean_s:.3f} |")
        D.append(f"| total device time, {len(rep_steps)} steps: sum kernel ms | {sum_k:.3f} |")
        D.append(f"| total device time, {len(rep_steps)} steps: sum span ms | {sum_s:.3f} |")
    D.append("")
    D.append(
        "Note: host-side handoff time (SP export -> host copy -> TP cache inject) is NOT in "
        "either CSV -- it happens between the two Tracy processes. Read it from the test's "
        "printed `[e2e] inject_ms=...` line."
    )
    D.append("")
    if log_path:
        try:
            with open(log_path) as f:
                # F4: the "[e2e] " lines carry a loguru/pytest prefix (timestamp | level |
                # module:func:line - [e2e] ...), so startswith("[e2e] ") misses every line.
                # Match "[e2e] " anywhere and print from the "[e2e]" marker onward. The space
                # after "]" excludes "[e2e-a]"/"[e2e-b]"/"[e2e-c]" lines, which use a different tag.
                e2e_lines = []
                for ln in f:
                    idx = ln.find("[e2e] ")
                    if idx != -1:
                        e2e_lines.append(ln[idx:].rstrip("\n"))
        except OSError as e:
            e2e_lines = [f"(could not read {log_path}: {e})"]
        if e2e_lines:
            D.append("`[e2e]` lines from the log:")
            D.append("```")
            D += e2e_lines
            D.append("```")
        else:
            D.append(f"(no `[e2e] ` lines found in {log_path})")
    else:
        D.append("(no log file given as argv[4] -- host-side handoff numbers must be filled in by hand)")
    D.append("")
    return D


# ------------------------------------------------------------------- main ---


def main():
    if len(sys.argv) < 4:
        print("usage: analyze_e2e_ops.py <prefill_csv> <decode_csv> <out.md> [log_with_[e2e]_lines]", file=sys.stderr)
        sys.exit(1)
    prefill_csv, decode_csv, out_path = sys.argv[1:4]
    log_path = sys.argv[4] if len(sys.argv) > 4 else None

    A, prefill_info = analyze_prefill(prefill_csv)
    B, decode_info = analyze_decode(decode_csv)
    C = analyze_pre_step(decode_info, decode_info["rep_dev"])
    D = analyze_e2e_summary(prefill_info, decode_info, log_path)

    text = "\n".join(["# SP-prefill -> TP-decode E2E ops profile\n"] + A + B + C + D)
    with open(out_path, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
