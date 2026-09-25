# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tabulate several zone-profile captures (chunk-size sweep) into one dense-vs-sparse comparison.

Usage:
    python3 summarize_chunk_sweep.py LABEL=path/to/ops_perf_results.csv [LABEL=... ...] [--md out.md]

Per capture: worst-device device-kernel ms for the whole profiled region, per dense / sparse layer,
and the attn / mlp split. For a PROFILE_ALL_CHUNKS capture the per-chunk rows are listed too.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd  # noqa: E402
from parse_zone_perf import ROOT_ZONE, ZoneAccumulator, io_byte_columns, layer_class, summarize  # noqa: E402


def load(csv):
    header = list(pd.read_csv(csv, nrows=0).columns)
    from parse_zone_perf import BASE_COLS, HOST_MOVEMENT_COLS, OPTIONAL_COLS

    byte_cols = io_byte_columns(header)
    usecols = BASE_COLS + [c for c in OPTIONAL_COLS if c in header]
    usecols += [c for c in HOST_MOVEMENT_COLS if c in header]
    usecols += sorted({c for cols in byte_cols.values() for c in cols.values()})
    acc = ZoneAccumulator()
    for chunk in pd.read_csv(csv, usecols=usecols, chunksize=200_000, low_memory=False):
        for row in chunk.to_dict("records"):
            acc.feed(row, byte_cols)
    return summarize(acc)


def ms(summary, zone):
    s = summary.get(zone)
    return s["ms_max"] if s else None


def fmt(v):
    return "-" if v is None else f"{v:.2f}"


def rows_for(label, summary):
    """One row per (chunk or whole), with dense/sparse layer totals and attn/mlp split."""
    # Zones directly under root that are not layers = chunk zones (PROFILE_ALL_CHUNKS).
    chunk_zones = sorted(
        z for z in summary if z.startswith(ROOT_ZONE + "/") and z.count("/") == 1 and layer_class(z) is None
    )
    prefixes = [ROOT_ZONE] if not chunk_zones else chunk_zones
    out = []
    for p in prefixes:
        layers = {}
        for z in summary:
            if not z.startswith(p + "/"):
                continue
            lc = layer_class(z)
            if lc is None:
                continue
            cls, rel, idx = lc
            if z.count("/") != p.count("/") + 1:
                continue  # only the layer zone itself here
            layers[cls] = {
                "idx": idx,
                "total": ms(summary, z),
                "attn": ms(summary, z + "/attn"),
                "mlp": ms(summary, z + "/mlp"),
            }
        name = label if p == ROOT_ZONE else f"{label} / {p.split('/')[-1]}"
        out.append((name, ms(summary, p), layers))
    if chunk_zones:
        # whole-sequence total across all chunks: sum the per-layer zones over chunks
        layers = {}
        for cls in ("dense", "sparse"):
            tot = [r[2][cls] for r in out if cls in r[2]]
            if tot:
                layers[cls] = {
                    "idx": tot[0]["idx"],
                    "total": sum(t["total"] or 0 for t in tot),
                    "attn": sum(t["attn"] or 0 for t in tot),
                    "mlp": sum(t["mlp"] or 0 for t in tot),
                }
        out.append((f"{label} / TOTAL ({len(chunk_zones)} chunks)", ms(summary, ROOT_ZONE), layers))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+", help="LABEL=ops_perf_results.csv")
    ap.add_argument("--md", help="write the table as markdown here")
    args = ap.parse_args()

    rows = []
    for cap in args.captures:
        label, _, csv = cap.partition("=")
        if not csv:
            sys.exit(f"expected LABEL=path, got {cap!r}")
        rows += rows_for(label, load(csv))

    hdr = [
        "config",
        "region ms",
        "dense L",
        "dense ms",
        "dense attn",
        "dense mlp",
        "sparse L",
        "sparse ms",
        "sparse attn",
        "sparse mlp",
    ]
    table = []
    for name, region, layers in rows:
        d = layers.get("dense", {})
        s = layers.get("sparse", {})
        table.append(
            [
                name,
                fmt(region),
                str(d.get("idx", "-")),
                fmt(d.get("total")),
                fmt(d.get("attn")),
                fmt(d.get("mlp")),
                str(s.get("idx", "-")),
                fmt(s.get("total")),
                fmt(s.get("attn")),
                fmt(s.get("mlp")),
            ]
        )

    widths = [max(len(h), *(len(r[i]) for r in table)) for i, h in enumerate(hdr)]
    line = lambda r: "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
    md = [line(hdr), "|" + "|".join("-" * (w + 2) for w in widths) + "|"] + [line(r) for r in table]
    print("\n".join(md))
    print("\nms = DEVICE KERNEL DURATION summed on the worst of 32 devices (not wall-clock).")
    if args.md:
        Path(args.md).write_text("\n".join(md) + "\n")
        print(f"[sweep] markdown -> {args.md}")


if __name__ == "__main__":
    main()
