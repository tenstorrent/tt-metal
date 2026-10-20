# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-module attribution of a single-chip ``test_perf_layer.py`` Tracy run (``GLM53_PERF_EAGER=1``).

    python models/demos/glm53_flash/tests/perf_layer_report.py <ops_perf_results_*.csv> --out <dir> [--top 12]

Runs ``tt-perf-report`` on the ``PERF_DECODE .. PERF_DECODE_END`` window (one traced replay), labels every
op with the innermost module signpost of the eager pass (same op order; ``tests/perf_mesh_report.py``),
and writes ``summary.md``: device time and op count per module (the two hyper-connection wraps
ATTN_HC / FFN_HC exclude their sublayers), plus the per-op table of the wraps' own ops.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_mesh_report import eager_labels, run_perf_report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tool", default=shutil.which("tt-perf-report") or "tt-perf-report")
    ap.add_argument("--title", default="")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv, low_memory=False)
    df["DEVICE ID"] = pd.to_numeric(df["DEVICE ID"], errors="coerce")
    present = sorted(int(d) for d in df[df["OP TYPE"] == "tt_dnn_device"]["DEVICE ID"].dropna().unique())
    assert len(present) == 1, present
    dev = present[0]
    txt, ops_csv = out / "decode_perf_report.txt", out / "decode_ops.csv"
    run_perf_report(args.tool, args.csv, txt, ops_csv, out / "decode_summary")
    rep = pd.read_csv(ops_csv)
    rep = rep[rep["OP Code"].astype(str) != "nan"]
    rep = rep[~rep["OP Code"].astype(str).str.startswith("PERF_")]
    rep["Device Time"] = pd.to_numeric(rep["Device Time"], errors="coerce").fillna(0.0)
    try:
        labels = eager_labels(df, dev)
        if len(labels) != len(rep):
            raise SystemExit(f"traced window has {len(rep)} ops, eager window {len(labels)}: cannot attribute")
    except SystemExit as e:
        # no usable eager window: attribute the composite TtMHCWrap path by its op signature. Each wrap's
        # pre-block runs from its single-core `32 x 16384 x 32` projection matmul to the fp32->bf16 typecast
        # that hands the hidden to the sublayer, the post-block from the bf16->fp32 typecast after the
        # sublayer to the `Concat` that packs the streams; the two LayerNorm ops are the decoder's norms.
        print(f"{e}; composite-signature attribution")
        codes = rep["OP Code"].astype(str).tolist()
        labels = [[] for _ in codes]
        wraps = ["ATTN_HC", "FFN_HC"]
        i, w = 0, 0
        while i < len(codes) and w < 2:
            if codes[i].startswith("MatmulDeviceOperation 32 x 16384 x 32"):
                j = i
                while not codes[j].startswith("TypecastDeviceOperation"):
                    j += 1
                for k in range(i, j + 1):
                    labels[k] = [wraps[w]]
                i = j + 1
                while not codes[i].startswith("TypecastDeviceOperation"):  # the sublayer's ops
                    labels[i] = [wraps[w], "SUBLAYER"]
                    i += 1
                j = i
                while not codes[j].startswith("ConcatDeviceOperation"):
                    j += 1
                for k in range(i, j + 1):
                    labels[k] = [wraps[w]]
                i = j + 1
                w += 1
            else:
                i += 1
        for k, c in enumerate(codes):
            if c.startswith("LayerNormDeviceOperation"):
                labels[k] = ["NORM"]
    rep["Stack"] = ["/".join(s) if s else "-" for s in labels]
    rep["Module"] = [s[-1] if s else "-" for s in labels]
    # hyper-connection ops: inside a *_HC wrap but outside its sublayer
    rep["HC"] = [s[-1] if s and s[-1].endswith("_HC") else "" for s in labels]
    tot = rep["Device Time"].sum()
    lines = [
        f"# {args.title or args.csv.name}",
        "",
        f"traced decode window: {len(rep)} ops, {tot / 1e3:.3f} ms device time",
        "",
    ]
    lines += ["| module (innermost label) | ops | device us | % |", "|---|---|---|---|"]
    for mod, grp in rep.groupby("Module", sort=False):
        lines.append(
            f"| {mod} | {len(grp)} | {grp['Device Time'].sum():.0f} | {100 * grp['Device Time'].sum() / tot:.1f} % |"
        )
    lines += ["", "## Hyper-connection wraps (their own ops, sublayers excluded)", ""]
    lines += ["| wrap | ops | device us |", "|---|---|---|"]
    for hc in ("ATTN_HC", "FFN_HC"):
        grp = rep[rep["HC"] == hc]
        lines.append(f"| {hc} | {len(grp)} | {grp['Device Time'].sum():.0f} |")
    both = rep[rep["HC"] != ""]
    lines.append(f"| both | {len(both)} | {both['Device Time'].sum():.0f} |")
    lines += ["", "### Wrap ops by kind (both wraps)", "", "| op | count | us total | us each |", "|---|---|---|---|"]
    kinds = both.groupby(both["OP Code"].astype(str).str.replace(r"\s+\d+ x .*$", "", regex=True))["Device Time"].agg(
        ["count", "sum"]
    )
    for op, r in kinds.sort_values("sum", ascending=False).iterrows():
        lines.append(f"| {op} | {int(r['count'])} | {r['sum']:.0f} | {r['sum'] / r['count']:.1f} |")
    lines += ["", "### ATTN_HC ops in order", "", "| # | op | cores | us |", "|---|---|---|---|"]
    for k, (_, r) in enumerate(rep[rep["HC"] == "ATTN_HC"].iterrows()):
        lines.append(f"| {k} | {r['OP Code']} | {r.get('Cores', '')} | {r['Device Time']:.0f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    rep.to_csv(out / "decode_ops_labelled.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
