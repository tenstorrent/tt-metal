# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Add the vectors the fixed record predates, from the saved heads of the fixed record's own jobs (CPU, D5 item 2, D6).

    python models/experimental/flowstate_forecast/tests/perf/fixed_record_vectors.py --out <json>

Writes one record per case and precision of the fixed record: its name, precision, stamp and VECTORS, recomputed
by ``metrics.gate_metrics`` on the same device outputs the fixed record was scored on (bench job 1123, suite job
1135, saved under FIXED_HEADS with the sha256 manifest iter_00/fixed_heads.sha256, v1.32). Every metric the fixed
record already has (scalars, bias_rows, spread_ratio_rows) is recomputed too and must match it within 1e-12, so the
heads are known to be those outputs; exit status 1 otherwise. check_gates reads the file through a ``ref``
tolerance; the fixed record is unchanged.

    ... fixed_record_vectors.py --baseline <record.json> --heads <dump dir> --flat --full --out <record.json>

backfills a later record (v1.32) from its own job's dumps (``<shape>-<precision>.pt``, ``suite_<case>-...``): the
same check, and the record is written whole with the new vectors and its original session stamp.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from models.experimental.flowstate_forecast.tests.golden.test_dataflow import GOLDENS
from models.experimental.flowstate_forecast.tests.pcc import metrics as M
from models.experimental.flowstate_forecast.tests.perf.broken_outputs import to_data
from models.experimental.flowstate_forecast.tests.perf.check_gates import DEFAULT_BASELINE, load_records

FIXED_HEADS = Path("/home/andy/bringup/goldens/flowstate/fixed_heads")
VECTORS = ("bias_quantiles", "inactive_bias_rows", "bias_rq")  # D6: active-row bias_q, inactive signed means, v1.33
ROW_VECTORS = ("bias_rows", "spread_ratio_rows")
TOL = 1e-12


def compare(m: dict, r: dict) -> dict:
    """|recomputed - record| for every finite scalar and row vector the record has (bias_quantiles is redefined)."""
    d = {}
    for k, v in r.items():
        if k in VECTORS or k not in m:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            d[k] = abs(m[k] - v)
        elif k in ROW_VECTORS:
            d[k] = max([abs(x - y) for x, y in zip(m[k], v) if y is not None] or [0.0])
            if [x is None for x in m[k]] != [y is None for y in v]:
                d[k] = math.inf
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heads", type=Path, default=FIXED_HEADS)
    ap.add_argument("--flat", action="store_true", help="one dump dir (a job's MODEL_PERF_DUMP / FLOWSTATE_SUITE_DUMP)")
    ap.add_argument("--full", action="store_true", help="write the whole record with the new vectors (backfill)")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out, worst = [], (0.0, "")
    for r in load_records(args.baseline):
        name, p = r["name"], r["precision"]
        case = name[6:] if name.startswith("suite:") else None
        g = torch.load(GOLDENS / f"{case or 'bench_' + name}.pt", weights_only=True)
        if args.flat:
            f = args.heads / (f"suite_{case}-{p}.pt" if case else f"{name}-{p}.pt")
        else:
            f = args.heads / (f"pcc_1135/suite_{case}-{p}.pt" if case else f"bench_1123/{name}-{p}.pt")
        h = torch.load(f, weights_only=True)["combined_normalized"]
        q, pt, _ = to_data(g, h)
        m = M.gate_metrics(h, g["combined_normalized"], q, g["quantile_outputs"], pt, g["prediction_outputs"])
        diffs = compare(m, r)
        k = max(diffs, key=diffs.get)
        worst = max(worst, (diffs[k], f"{name}/{p} {k}"))
        bq = [None if v is None else round(v, 4) for v in m["bias_quantiles"]]
        print(
            f"{name:20s} {p:11s} bias_quantiles {bq} {len(diffs)} metrics, max |recomputed - record| {diffs[k]:.3g} ({k})"
        )
        stamp = {k: r[k] for k in ("git_head", "git_dirty", "started_at") if k in r}
        new = {v: m[v] for v in VECTORS} | {"row_info": m["row_info"]}
        if args.full:
            out.append(r | new | {"backfilled_from": str(f)})
        else:
            out.append({"name": name, "precision": p, **{v: m[v] for v in VECTORS}, "source": str(f), **stamp})
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(out)} records; worst |recomputed - fixed record| {worst[0]:.3g} ({worst[1]}); tolerance {TOL}")
    sys.exit(0 if worst[0] <= TOL else 1)


if __name__ == "__main__":
    main()
