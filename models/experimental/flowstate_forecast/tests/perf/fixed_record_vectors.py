# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Add a vector the fixed record predates, from the device dumps of the fixed record's own jobs (CPU, D5 item 2).

    python models/experimental/flowstate_forecast/tests/perf/fixed_record_vectors.py --dumps <g1b/dump> --out <json>

Writes one record per case and precision of the fixed record: its name, precision, stamp and ``bias_quantiles``,
recomputed by ``metrics.gate_metrics`` on the same device outputs the fixed record was scored on (jobs 1123 and
1135). Also prints the recomputed scalar metrics that are in the fixed record, which must match it, so the dumps
are known to be those outputs. check_gates reads the file through a ``ref`` tolerance; the fixed record is unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from models.experimental.flowstate_forecast.tests.golden.test_dataflow import GOLDENS
from models.experimental.flowstate_forecast.tests.pcc import metrics as M
from models.experimental.flowstate_forecast.tests.perf.broken_outputs import to_data
from models.experimental.flowstate_forecast.tests.perf.check_gates import DEFAULT_BASELINE, load_records

CHECK = ("pcc_normalized", "abs_err_max", "inactive_max_err", "bias_row_max", "pcc_row_min")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dumps", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out, worst = [], 0.0
    for r in load_records(args.baseline):
        name, p = r["name"], r["precision"]
        case = name[6:] if name.startswith("suite:") else None
        g = torch.load(GOLDENS / f"{case or 'bench_' + name}.pt", weights_only=True)
        f = args.dumps / (f"suite_{case}-{p}.pt" if case else f"{name}-{p}.pt")
        h = torch.load(f, weights_only=True)["combined_normalized"]
        q, pt, _ = to_data(g, h)
        m = M.gate_metrics(h, g["combined_normalized"], q, g["quantile_outputs"], pt, g["prediction_outputs"])
        diffs = {k: abs(m[k] - r[k]) for k in CHECK if k in r and math.isfinite(r[k])}
        worst = max([worst, *diffs.values()])
        print(
            f"{name:20s} {p:11s} bias_quantiles {[round(v, 4) for v in m['bias_quantiles']]} "
            f"max |recomputed - record| {max(diffs.values()):.3g}"
        )
        stamp = {k: r[k] for k in ("git_head", "git_dirty", "started_at") if k in r}
        out.append({"name": name, "precision": p, "bias_quantiles": m["bias_quantiles"], "source": str(f), **stamp})
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(out)} records; worst |recomputed - fixed record| over {CHECK}: {worst:.3g}")


if __name__ == "__main__":
    main()
