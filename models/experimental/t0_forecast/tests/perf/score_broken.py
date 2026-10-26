# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Score the PLAN v1.9 broken outputs against the D3 / D4 floors (D4 item 2), on the bench goldens, CPU only.

    ttq run --model t0-gamma --part 3 --kind cpu -- python models/experimental/t0_forecast/tests/perf/score_broken.py \
        --out models/experimental/t0_forecast/doc/optimize/final/broken_outputs.json

Each broken output is the reference with its quantile band (``h - mean_q``) rescaled per row, in asinh space and in
data space alike: ``inactive_collapsed`` (every inactive row's band at 0), ``active_2pct_0.3x`` (2 % of the active rows
at 0.3x), ``band_0.88x`` and ``band_1.12x`` (every row). Each must miss at least one floor in each precision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    import torch

    from models.experimental.t0_forecast.tests.helpers import FLOORS, final_metrics, golden, misses, quantile_metrics

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    def scaled(t, rows, factor):
        """``t`` with the band of ``rows`` (over the leading flattened row axis) scaled by ``factor``."""
        flat = t.reshape(rows.numel(), -1, t.shape[-1]).clone()
        mean = flat.mean(dim=-1, keepdim=True)
        f = torch.ones(rows.numel(), 1, 1, dtype=flat.dtype)
        f[rows] = factor
        return (mean + f * (flat - mean)).reshape(t.shape)

    results = []
    # The bench goldens (realistic inputs) have no inactive rows; e2e (coverage floors) does.
    classes = {"accuracy": "accuracy", "performance": "performance"}
    for name in ("paper", "long_context", "multivariate", "e2e"):
        g = golden("e2e.pt" if name == "e2e" else f"bench_{name}.pt")
        if name == "e2e":
            classes = {"accuracy": "coverage", "performance": "coverage_performance"}
        head, raw = g["head_readout"], g["raw21"]
        n = head.shape[0]
        flat = head.double().reshape(n, -1)
        active = flat.std(dim=1) >= 0.05 * flat.std()
        idx = active.nonzero().flatten()
        pick = torch.zeros(n, dtype=torch.bool)
        pick[
            idx[
                torch.randperm(idx.numel(), generator=torch.Generator().manual_seed(2))[
                    : max(1, round(0.02 * idx.numel()))
                ]
            ]
        ] = True
        everything = torch.ones(n, dtype=torch.bool)
        broken = {
            "inactive_collapsed": (~active, 0.0),
            "active_2pct_0.3x": (pick, 0.3),
            "band_0.88x": (everything, 0.88),
            "band_1.12x": (everything, 1.12),
        }
        for kind, (rows, factor) in broken.items():
            m = {**quantile_metrics(head, scaled(head, rows, factor)), **final_metrics(raw, scaled(raw, rows, factor))}
            rec = {"golden": name, "broken": kind, "rows_changed": int(rows.sum()), "rows": n, "metrics": m}
            for precision in ("accuracy", "performance"):
                rec[f"misses_{precision}"] = misses(m, FLOORS[classes[precision]])
            results.append(rec)
            print(
                f"[broken] {name:13s} {kind:19s} rows {rec['rows_changed']:5d}/{n}"
                f" | acc {'FAIL' if rec['misses_accuracy'] else 'pass'} {rec['misses_accuracy'][:2]}"
                f" | perf {'FAIL' if rec['misses_performance'] else 'pass'} {rec['misses_performance'][:2]}",
                flush=True,
            )
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
