#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-chip MoE load of real routing, from M3_MOE_LOAD_STATS_FILE, one row per (timed point, MoE layer).

  load_skew.py --stats run.jsonl --log run.log --case single_prose --W 4096 --B 1 --input prose [--out load_skew.csv]

The stats file holds one JSON line per MoE layer per forward (raw per-expert counts). The run's log
(budget_sweep.py / budget_packed.py RESULT lines) gives the forward order: every fill / warmup / iter row is
one forward, so call k of a layer is the k-th such row. Each timed point is read at its last iter forward
(routing is deterministic, the repeats agree). h is the point's h (sweep) or the composition's segment depths
joined with '+' (packed).

Columns: per_chip_tokens = routed token-expert assignments landing on each chip (sum of its experts' counts),
chips in mesh (row, col) row-major order; max_over_mean over those 8; top_expert_share = hottest expert's
count / all assignments; experts_active_per_chip_max / _mean = experts with count > 0 per chip.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping  # noqa: E402

COLS = [
    "case",
    "layer",
    "W",
    "h",
    "B",
    "input",
    "per_chip_tokens",
    "max_over_mean",
    "top_expert_share",
    "experts_active_per_chip_max",
    "experts_active_per_chip_mean",
    "tokens_total",
    "worst_chip_active",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--case", required=True, help="case prefix; the point's h / compo name is appended")
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--input", required=True, help="label per point, or 'h=label,...' for a sweep")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "load_skew.csv"))
    args = ap.parse_args()

    fwds, points = [], []
    for line in open(args.log):
        if not line.startswith("RESULT "):
            continue
        r = json.loads(line[7:])
        if r["kind"] in ("fill", "warmup", "iter"):
            fwds.append(r)
        elif r["kind"] == "point":
            points.append((r, len(fwds) - 1))  # the point's last iter is the forward just before it
    calls = {}
    for line in open(args.stats):
        rec = json.loads(line)
        calls.setdefault(rec["layer"], []).append(rec)
    for layer, recs in calls.items():
        assert len(recs) == len(fwds), f"layer {layer}: {len(recs)} stats calls vs {len(fwds)} forwards in the log"

    labels = {}
    if "=" in args.input:
        labels = dict(kv.split("=") for kv in args.input.split(","))

    rows = []
    for pt, k in points:
        if "compo" in pt:
            name, h = pt["compo"], "+".join(str(s["h"]) for s in pt["segments"])
        else:
            name, h = str(pt["h"]), str(pt["h"])
        for layer in sorted(calls):
            rec = calls[layer][k]
            rows_, cols_ = rec["mesh"]
            epc = rec["experts_per_chip"]
            table = ExpertMapping.create_global_expert_idx_table(
                experts_per_chip=epc, dispatch_group_size=rows_, num_dispatch_groups=cols_
            ).to(
                torch.int64
            )  # (cols, rows, epc)
            counts = torch.tensor(rec["counts"], dtype=torch.int64)
            per = counts[table]  # (cols, rows, epc)
            chip_tok = [int(per[c, r].sum()) for r in range(rows_) for c in range(cols_)]
            chip_act = [int((per[c, r] > 0).sum()) for r in range(rows_) for c in range(cols_)]
            mean = sum(chip_tok) / len(chip_tok)
            worst = max(range(len(chip_tok)), key=chip_tok.__getitem__)
            rows.append(
                {
                    "case": f"{args.case}_{name}",
                    "layer": layer,
                    "W": args.W,
                    "h": h,
                    "B": args.B,
                    "input": labels.get(name, args.input),
                    "per_chip_tokens": ";".join(map(str, chip_tok)),
                    "max_over_mean": round(max(chip_tok) / mean, 4) if mean else "",
                    "top_expert_share": round(int(counts.max()) / int(counts.sum()), 4),
                    "experts_active_per_chip_max": max(chip_act),
                    "experts_active_per_chip_mean": round(sum(chip_act) / len(chip_act), 2),
                    "tokens_total": int(counts.sum()),
                    "worst_chip_active": chip_act[worst],
                }
            )
    out = Path(args.out)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['case']} L{r['layer']} h={r['h']} max/mean={r['max_over_mean']} per_chip={r['per_chip_tokens']}")
    print(f"[load_skew] {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
