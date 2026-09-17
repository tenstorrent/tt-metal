#!/usr/bin/env python3
"""Summarise a tt-metal Tracy op-perf CSV for a Gated DeltaNet (GDN) prefill run.

Only the region between the GDN_PREFILL_START and GDN_PREFILL_END signposts is measured. If
those signposts are absent the whole file (minus signpost rows) is used and a warning is printed.

MULTI-DEVICE: at TP>1 the CSV carries one row per device per op. Summing them overstates the
cost by ~TP. By default the rows are merged to the CRITICAL PATH the way the rest of the repo
does it (models/tt_transformers/tests/test_utils.py::merge_device_rows): per-device rows are
matched in dispatch order, collectives are averaged across devices and every other op takes the
MAX. Pass --no-merge-devices to see the raw per-device sum instead.

Usage:
    python parse_gdn_perf.py <ops_perf_results_*.csv> --isl 4096 --chunks 2 [--layers 45]
                             [--top 25] [--json out.json] [--no-merge-devices]

Pure stdlib + pandas. No repo imports. Touches no device.
"""

import argparse
import json
import os
import re
import sys

import pandas as pd

OP_CODE_COL = "OP CODE"
OP_TYPE_COL = "OP TYPE"
DEVICE_ID_COL = "DEVICE ID"
KERNEL_SRC_COL = "COMPUTE KERNEL SOURCE"
SIGNPOST_TYPE = "signpost"
DEVICE_OP_TYPE = "tt_dnn_device"

START_MARKER = "GDN_PREFILL_START"
END_MARKER = "GDN_PREFILL_END"

# Duration column preference order.
DURATION_PREFERENCE = [
    "DEVICE KERNEL DURATION [ns]",
    "DEVICE FW DURATION [ns]",
]
DURATION_REGEX = re.compile(r"DEVICE.*DURATION.*ns", re.IGNORECASE)

# merge_device_rows averages these across devices and MAXes everything else.
COLLECTIVE_KEYS = ("AllGather", "ReduceScatter", "AllReduce", "Matmul_RS")

# Output-shape columns, in preference order, used to disambiguate same-named ops
# (e.g. the in-proj and out-proj both report as MatmulDeviceOperation).
SHAPE_COL_REGEX = re.compile(r"^OUTPUT_0_([WZYX])_PAD\[LOGICAL\]$")

STAGE_RULES = [
    ("matmul", ["matmul", "linear"]),
    ("conv1d", ["conv", "halo"]),
    ("delta_rule", ["gdn", "delta", "chunk_gated", "scan", "prep", "generic"]),
    ("norm", ["rms_norm", "layernorm"]),
    ("elementwise", ["silu", "sigmoid", "mul", "add", "binary", "unary", "typecast"]),
    (
        "reshape_slice",
        [
            "slice",
            "reshape",
            "concat",
            "permute",
            "transpose",
            "to_layout",
            "tilize",
            "untilize",
            "copy",
            "clone",
            "zeros",
            "nlp_concat_heads",
            "interleaved_to_sharded",
            "sharded_to_interleaved",
        ],
    ),
]
OTHER_STAGE = "other"
STAGE_ORDER = [name for name, _ in STAGE_RULES] + [OTHER_STAGE]


def _tokens(text):
    """Split an op code into lowercase word tokens ('NLPConcatHeads' -> nlp, concat, heads)."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text))
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
    return [t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def _matches(tokens, compact, key):
    key_compact = re.sub(r"[^a-z0-9]+", "", key.lower())
    if not key_compact:
        return False
    if len(key_compact) <= 4:
        return any(tok.startswith(key_compact) for tok in tokens)
    return key_compact in compact


def classify(op_code):
    tokens = _tokens(op_code)
    compact = "".join(tokens)
    for stage, keys in STAGE_RULES:
        for key in keys:
            if _matches(tokens, compact, key):
                return stage
    return OTHER_STAGE


def pick_duration_column(df):
    for col in DURATION_PREFERENCE:
        if col in df.columns:
            return col
    for col in df.columns:
        if DURATION_REGEX.search(str(col)):
            return col
    return None


def find_marker(markers, needle):
    """Index label of the first signpost row whose OP CODE contains needle."""
    for pos in range(len(markers)):
        if needle in str(markers.iloc[pos]):
            return markers.index[pos]
    return None


def select_region(df):
    """Keep rows strictly between the GDN start/end signposts."""
    if OP_TYPE_COL not in df.columns or OP_CODE_COL not in df.columns:
        print(f"WARNING: missing '{OP_TYPE_COL}' or '{OP_CODE_COL}' column; using all rows.")
        return df, False

    markers = df[df[OP_TYPE_COL] == SIGNPOST_TYPE][OP_CODE_COL]
    start_label = find_marker(markers, START_MARKER)
    stop_label = find_marker(markers, END_MARKER)

    if start_label is None or stop_label is None:
        missing = [m for m, lab in ((START_MARKER, start_label), (END_MARKER, stop_label)) if lab is None]
        found = sorted({str(v) for v in markers.tolist()})
        print("=" * 78)
        print(f"WARNING: signpost(s) not found: {', '.join(missing)}")
        print("WARNING: falling back to ALL non-signpost rows in the file.")
        print(f"         signposts present ({len(markers)} rows): {found if found else 'none'}")
        print("=" * 78)
        return df, False

    start = df.index.get_loc(start_label)
    stop = df.index.get_loc(stop_label)
    if stop <= start:
        print(f"WARNING: {END_MARKER} appears before {START_MARKER}; using all non-signpost rows.")
        return df, False
    return df.iloc[start + 1 : stop], True


def is_collective(op_code):
    return any(k.lower() in str(op_code).lower() for k in COLLECTIVE_KEYS)


def merge_device_rows(df, dur_col):
    """Collapse per-device rows to one critical-path row per op invocation.

    Faithful to models/tt_transformers/tests/test_utils.py::merge_device_rows: rows are bucketed
    per DEVICE ID in dispatch order, the i-th op of every device is one block, collectives take
    the MEAN across devices and everything else takes the MAX.
    """
    if DEVICE_ID_COL not in df.columns:
        print(f"WARNING: no '{DEVICE_ID_COL}' column; cannot merge devices.")
        return df, 1, []

    per_device = {dev: sub for dev, sub in df.groupby(DEVICE_ID_COL, sort=True)}
    n_dev = len(per_device)
    if n_dev <= 1:
        return df, n_dev, []

    lengths = {dev: len(sub) for dev, sub in per_device.items()}
    warnings = []
    if len(set(lengths.values())) != 1:
        warnings.append(f"per-device op counts differ: {lengths} -> merging over the shortest")
    n_ops = min(lengths.values())

    devs = sorted(per_device)
    seqs = {dev: per_device[dev].reset_index(drop=True) for dev in devs}

    # Sanity: op sequences must align position-by-position, else the merge is meaningless.
    ref_codes = seqs[devs[0]][OP_CODE_COL].astype(str).tolist()[:n_ops]
    for dev in devs[1:]:
        codes = seqs[dev][OP_CODE_COL].astype(str).tolist()[:n_ops]
        mismatch = [i for i in range(n_ops) if codes[i] != ref_codes[i]]
        if mismatch:
            warnings.append(
                f"device {dev} op sequence diverges from device {devs[0]} at "
                f"{len(mismatch)} of {n_ops} positions (first at index {mismatch[0]}: "
                f"{codes[mismatch[0]]} vs {ref_codes[mismatch[0]]})"
            )
            break

    merged = []
    for i in range(n_ops):
        rows = [seqs[dev].iloc[i] for dev in devs]
        durs = [float(pd.to_numeric(r[dur_col], errors="coerce") or 0.0) for r in rows]
        if is_collective(rows[0][OP_CODE_COL]):
            base = rows[0].copy()
            base[dur_col] = sum(durs) / len(durs)
            merged.append(base)
        else:
            merged.append(rows[int(max(range(len(durs)), key=lambda j: durs[j]))])
    return pd.DataFrame(merged).reset_index(drop=True), n_dev, warnings


PAD_LOGICAL_REGEX = re.compile(r"(-?\d+)\s*\[\s*(-?\d+)\s*\]")


def _shape_cell_value(v):
    """Extract the logical (bracketed) size from a 'PADDED[LOGICAL]' cell, e.g. '128[128]' -> '128'.

    Falls back to a plain numeric parse for cells that are not in that format. Never raises.
    """
    if v is None:
        return None
    match = PAD_LOGICAL_REGEX.search(str(v))
    if match:
        return match.group(2)
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return None


def _shape_from_outputs(out):
    """Best-effort compact shape (e.g. '1x1x2048x5152') parsed out of an OUTPUTS cell.

    Used only when the OUTPUT_0_*_PAD[LOGICAL] columns are absent or unusable. Never raises.
    """
    try:
        if not isinstance(out, str) or not out.strip():
            return None
        nums = re.findall(r"-?\d+", out)
        if len(nums) >= 2:
            return "x".join(nums[:4])
    except Exception:
        pass
    return None


def shape_tag(row, shape_cols):
    """Compact output-shape tag, e.g. '1x1x2048x5152', used to split same-named ops."""
    if shape_cols:
        vals = [_shape_cell_value(row.get(col)) for col in shape_cols]
        if any(v is not None for v in vals):
            return "x".join(v if v is not None else "?" for v in vals)
    shaped = _shape_from_outputs(row.get("OUTPUTS"))
    if shaped:
        return shaped
    out = row.get("OUTPUTS")
    if isinstance(out, str) and out:
        return out[:40]
    return "-"


def kernel_tag(row):
    """Compact kernel-source tag from a COMPUTE KERNEL SOURCE cell.

    The cell holds a Python-list-literal string, e.g. "['path/a.cpp']" or, with multiple
    kernels, "['path/a.cpp'; 'path/b.cpp']" (entries joined with '; ' rather than ', '). Pull
    out each quoted path, strip directories and the '.cpp' suffix, and join multiple kernel
    tags with '+'. Never raises on a malformed cell.
    """
    src = row.get(KERNEL_SRC_COL)
    if not isinstance(src, str) or not src.strip():
        return ""
    try:
        paths = re.findall(r"'([^']*)'", src)
        if not paths:
            stripped = src.strip().strip("[]").strip()
            paths = [stripped] if stripped else []
        tags = []
        for p in paths:
            p = p.strip()
            if not p:
                continue
            base = os.path.basename(p)
            if base.endswith(".cpp"):
                base = base[: -len(".cpp")]
            if base and base not in tags:
                tags.append(base)
        return "+".join(tags)
    except Exception:
        return ""


def fmt_row(cells, widths, aligns):
    out = []
    for cell, width, align in zip(cells, widths, aligns):
        text = str(cell)
        if len(text) > width:
            text = text[: width - 1] + "~"
        out.append(text.ljust(width) if align == "l" else text.rjust(width))
    return "  ".join(out)


def print_table(header, rows, widths, aligns):
    print(fmt_row(header, widths, ["l"] * len(header)))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print(fmt_row(row, widths, aligns))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="path to ops_perf_results_*.csv")
    ap.add_argument("--isl", type=int, required=True, help="input sequence length (tokens)")
    ap.add_argument("--chunks", type=int, required=True, help="number of prefill chunks in the region")
    ap.add_argument("--layers", type=int, default=45, help="GDN layer count for the projection (default 45)")
    ap.add_argument("--top", type=int, default=30, help="max rows in the op table (default 30)")
    ap.add_argument("--json", dest="json_path", default=None, help="also dump the tables + totals as JSON")
    ap.add_argument(
        "--no-merge-devices",
        dest="merge_devices",
        action="store_false",
        help="report the raw per-device row sum instead of the merged critical path",
    )
    args = ap.parse_args()

    if args.isl <= 0 or args.chunks <= 0 or args.layers <= 0:
        print("ERROR: --isl, --chunks and --layers must all be > 0")
        return 2

    df = pd.read_csv(args.csv, skip_blank_lines=True, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    total_rows = len(df)

    region, used_signposts = select_region(df)

    # Keep only real device ops (this also drops the signpost rows).
    if OP_TYPE_COL in region.columns:
        types = region[OP_TYPE_COL].astype(str)
        region = region[types == DEVICE_OP_TYPE] if (types == DEVICE_OP_TYPE).any() else region[types != SIGNPOST_TYPE]
    blank = region[OP_CODE_COL].isna() | (region[OP_CODE_COL].astype(str).str.strip() == "")
    skipped_blank = int(blank.sum())
    region = region[~blank]

    if len(region) == 0:
        print("NO ROWS IN REGION")
        return 1

    dur_col = pick_duration_column(df)
    if dur_col is None:
        print("ERROR: no device duration column found (looked for DEVICE.*DURATION.*ns)")
        print(f"       columns present: {list(df.columns)}")
        return 2

    raw_rows = len(region)
    raw_devices = region[DEVICE_ID_COL].nunique() if DEVICE_ID_COL in region.columns else 1
    merge_warnings = []
    if args.merge_devices:
        region, n_dev, merge_warnings = merge_device_rows(region, dur_col)
    else:
        n_dev = raw_devices

    dur_ns = pd.to_numeric(region[dur_col], errors="coerce").fillna(0.0)
    region = region.assign(_dur_ns=dur_ns)

    total_ns = float(region["_dur_ns"].sum())
    total_us = total_ns / 1e3
    total_ms = total_ns / 1e6

    print()
    print("=" * 78)
    print("GDN PREFILL OP-PERF SUMMARY")
    print("=" * 78)
    print(f"csv             : {args.csv}")
    print(f"duration column : {dur_col}")
    print(f"region          : {'signpost-bounded' if used_signposts else 'FALLBACK (all non-signpost rows)'}")
    print(f"rows in file    : {total_rows}")
    print(f"device rows     : {raw_rows} across {raw_devices} device(s)")
    print(
        f"accounting      : {'MERGED critical path (collectives averaged, others MAX)' if args.merge_devices and raw_devices > 1 else 'RAW per-device sum'}"
    )
    print(f"rows measured   : {len(region)}" + (f"  (skipped {skipped_blank} blank OP CODE)" if skipped_blank else ""))
    print(f"isl={args.isl}  chunks={args.chunks}  layers={args.layers}")
    for w in merge_warnings:
        print(f"WARNING: {w}")
    print()

    # Per-op table, split by output shape + compute kernel so same-named ops separate.
    shape_cols = [c for c in region.columns if SHAPE_COL_REGEX.match(str(c))]
    shape_cols.sort(key=lambda c: "WZYX".index(SHAPE_COL_REGEX.match(c).group(1)))
    region = region.assign(
        _variant=[
            f"{r[OP_CODE_COL]} [{shape_tag(r, shape_cols)}]" + (f" {kernel_tag(r)}" if kernel_tag(r) else "")
            for _, r in region.iterrows()
        ]
    )

    def table(group_col, title, width=56):
        grouped = region.groupby(group_col)["_dur_ns"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False)
        records = []
        for code, row in grouped.iterrows():
            t_us = float(row["sum"]) / 1e3
            records.append(
                {
                    "key": str(code),
                    "calls": int(row["count"]),
                    "total_us": t_us,
                    "mean_us": float(row["mean"]) / 1e3,
                    "per_chunk_us": t_us / args.chunks,
                    "pct_of_total": (100.0 * float(row["sum"]) / total_ns) if total_ns else 0.0,
                }
            )
        shown = records[: args.top] if args.top > 0 else records
        widths = [width, 7, 13, 11, 13, 7]
        aligns = ["l", "r", "r", "r", "r", "r"]
        print(f"{title} (top {len(shown)} of {len(records)})")
        print_table(
            ["KEY", "calls", "total_us", "mean_us", "per_chunk_us", "pct"],
            [
                [
                    r["key"],
                    r["calls"],
                    f"{r['total_us']:.2f}",
                    f"{r['mean_us']:.2f}",
                    f"{r['per_chunk_us']:.2f}",
                    f"{r['pct_of_total']:.2f}",
                ]
                for r in shown
            ],
            widths,
            aligns,
        )
        if len(shown) < len(records):
            rest = records[args.top :]
            print(
                fmt_row(
                    [
                        f"... {len(rest)} more",
                        sum(r["calls"] for r in rest),
                        f"{sum(r['total_us'] for r in rest):.2f}",
                        "-",
                        f"{sum(r['total_us'] for r in rest) / args.chunks:.2f}",
                        f"{sum(r['pct_of_total'] for r in rest):.2f}",
                    ],
                    widths,
                    aligns,
                )
            )
        print()
        return records

    op_records = table(OP_CODE_COL, "PER-OP BREAKDOWN (by OP CODE)", width=44)
    variant_records = table("_variant", "PER-OP BREAKDOWN (by OP CODE + output shape + kernel)", width=70)

    totals = {
        "duration_column": dur_col,
        "used_signposts": used_signposts,
        "merged_devices": bool(args.merge_devices and raw_devices > 1),
        "num_devices": int(n_dev),
        "raw_device_rows": int(raw_rows),
        "rows_measured": int(len(region)),
        "isl": args.isl,
        "chunks": args.chunks,
        "layers": args.layers,
        "total_ns": total_ns,
        "total_us": total_us,
        "total_ms": total_ms,
        "per_chunk_us": total_us / args.chunks,
        "ns_per_token": total_ns / args.isl,
        "one_layer_isl_total_ms": total_ms,
        "projection_ms": total_ms * args.layers,
    }

    print("TOTALS (one GDN layer)")
    print("-" * 78)
    print(f"{'total device time':<34}{total_us:>16.2f} us   ({total_ms:.3f} ms)")
    print(f"{'per-chunk total':<34}{totals['per_chunk_us']:>16.2f} us")
    print(f"{'ns per token (isl=' + str(args.isl) + ')':<34}{totals['ns_per_token']:>16.2f} ns")
    print(f"{'one-layer ISL total':<34}{total_ms:>16.3f} ms")
    print(f"{str(args.layers) + '-layer projection':<34}{totals['projection_ms']:>16.3f} ms")

    stages = region[OP_CODE_COL].astype(str).map(classify)
    region = region.assign(_stage=stages)
    st = region.groupby("_stage")["_dur_ns"].agg(["count", "sum"]).sort_values("sum", ascending=False)

    stage_records = []
    for stage, row in st.iterrows():
        t_us = float(row["sum"]) / 1e3
        stage_records.append(
            {
                "stage": str(stage),
                "calls": int(row["count"]),
                "total_us": t_us,
                "per_chunk_us": t_us / args.chunks,
                "pct": (100.0 * float(row["sum"]) / total_ns) if total_ns else 0.0,
            }
        )

    print()
    print("STAGE ROLLUP")
    print_table(
        ["stage", "calls", "total_us", "per_chunk_us", "pct"],
        [
            [r["stage"], r["calls"], f"{r['total_us']:.2f}", f"{r['per_chunk_us']:.2f}", f"{r['pct']:.2f}"]
            for r in stage_records
        ],
        [16, 7, 14, 14, 8],
        ["l", "r", "r", "r", "r"],
    )
    print()

    if args.json_path:
        payload = {
            "totals": totals,
            "ops": op_records,
            "variants": variant_records,
            "stages": stage_records,
            "csv": args.csv,
            "merge_warnings": merge_warnings,
        }
        with open(args.json_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote JSON -> {args.json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
