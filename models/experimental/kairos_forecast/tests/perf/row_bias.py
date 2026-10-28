# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Names the outlier rows of the fixed record (D4 item 2) and measures the row-indexed bias between the precisions.

For every record whose head-space bias_row_max is above 0.05: the row (series, flip half, rollout forward), the
series' n_valid, its context std, the golden head row std and band half-width ((q0.9 - q0.1) / 2, mean over the
steps; normalized units), the signed performance and accuracy bias on that row, and the bias over the band
half-width. Then max_i |b_perf_i - b_acc_i| per record, the row drift that a whole precision change makes.

    python -m models.experimental.kairos_forecast.tests.perf.row_bias <suite.jsonl> <out.md>
"""
from __future__ import annotations

import sys

import torch

from models.experimental.kairos_forecast.tests.pcc import metrics as M
from models.experimental.kairos_forecast.tests.perf.check_gates import load_records
from models.experimental.kairos_forecast.tests.perf.suite import load

LIMIT = 0.05


def describe(rec_p, rec_a):
    name = rec_p["name"]
    g = load(name.removesuffix("_tf"))
    S = g["context"].shape[0]
    heads = g["head"] if name.endswith("_tf") else g["head"][:2] if g["gift"] and len(g["head"]) == 2 else g["head"][:1]
    gold = torch.cat(list(heads))
    row = rec_p["bias_row_argmax"]
    fwd, s = divmod(row, S)
    n = len(g["head"])
    half = int(g["gift"] and fwd >= n // 2) if name.endswith("_tf") else int(fwd >= 1)
    ctx = g["step_context"][fwd] if name.endswith("_tf") and "step_context" in g else g["context"]
    ctx = ctx[s] if ctx.shape[0] == S else ctx[s % ctx.shape[0]]
    valid = ctx[~torch.isnan(ctx)]
    h = M.head_view(gold[row : row + 1]).double().reshape(M.Q, -1)
    return {
        "record": name,
        "row": row,
        "series": s,
        "half": "-x" if half else "x",
        "forward": fwd,
        "n_valid": int(valid.numel()),
        "context_std": float(valid.std()),
        "golden_row_std": float(h.std()),
        "band_half_width": float(((h[-1] - h[0]) / 2).mean()),
        "bias_perf": rec_p["bias_rows"][row],
        "bias_acc": rec_a["bias_rows"][row],
    }


def drift_between(rp, ra, vec):
    d = [abs(x - y) for x, y in zip(rp[vec], ra[vec]) if x is not None and y is not None]
    return max(d) if d else 0.0


def main(path, out):
    recs = {(r["name"], r["precision"]): r for r in load_records(path)}
    lines = [
        "# Row bias of the fixed record (D4 items 1-2)",
        "",
        f"From `{path}`.",
        "",
        "## Outlier rows (head-space bias_row_max > 0.05)",
        "",
        "| record | precision | row | series | half | fwd | n_valid | ctx std | golden row std | band half-width "
        "| bias perf | bias acc | perf bias / half-width |",
        "|" + "---|" * 13,
    ]
    for (name, prec), r in sorted(recs.items()):
        if r["bias_row_max"] <= LIMIT:
            continue
        d = describe(r if prec == "performance" else recs[(name, "performance")], recs[(name, "accuracy")])
        d["row"] = r["bias_row_argmax"]
        lines.append(
            f"| {name} | {prec} | {d['row']} | {d['series']} | {d['half']} | {d['forward']} | {d['n_valid']} "
            f"| {d['context_std']:.4g} | {d['golden_row_std']:.4f} | {d['band_half_width']:.4f} "
            f"| {d['bias_perf']:+.4f} | {d['bias_acc']:+.4f} | {abs(d['bias_perf']) / max(d['band_half_width'], 1e-12):.2f} |"
        )
    lines += [
        "",
        "## Row drift between the precisions",
        "",
        "max_i |b_perf_i - b_acc_i|: the row-indexed move of a whole precision change (bound 0.02 per candidate).",
        "",
        "| record | head | final |",
        "|---|---|---|",
    ]
    for name in sorted({n for n, _ in recs}):
        rp, ra = recs[(name, "performance")], recs[(name, "accuracy")]
        lines.append(
            f"| {name} | {drift_between(rp, ra, 'bias_rows'):.4f} | "
            f"{drift_between(rp, ra, 'bias_rows_final'):.4f} |"
        )
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
