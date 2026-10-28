# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Score synthetic broken outputs against the gate floors, both precisions, every bench shape and suite case (CPU).

    python models/experimental/flowstate_forecast/tests/perf/broken_outputs.py [--gates gates.json] [--out out.json]

Copied from TiRex-2 651e8c4af93 and adapted: each broken output is made from the golden pre-denorm decoder output
``(rows, B, 9, fl, 1)`` per series, then goes through the reference's own host back end (de-normalization, CPM
combine, torch.quantile), so the head and data-space metrics stay consistent. ``collapsed`` (every quantile -> the
quantile mean), ``inactive_collapsed`` (only inactive series), ``active1pct_0.3x`` (the band of ceil(1 %) of the
active series, the widest, scaled 0.3x about the quantile mean), ``all_0.88x`` and ``all_1.12x``. Each must fail at
least one floor per precision (PLAN §9 v1.9, v1.12). Exit status 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from models.experimental.flowstate_forecast.reference import dataflow as dfl
from models.experimental.flowstate_forecast.tests.golden.test_dataflow import GOLDENS, TEST_CASES
from models.experimental.flowstate_forecast.tests.pcc import metrics as M
from models.experimental.flowstate_forecast.tests.perf.check_gates import DEFAULT_GATES, _floors, passes

BENCH = ("paper", "long_context", "long_horizon", "scale_q", "single", "single_long")


def scale_band(dec, series, factor):
    d = dec.clone()
    mean = d.mean(2, keepdim=True)
    d[:, series] = mean[:, series] + factor * (d[:, series] - mean[:, series])
    return d


def broken(g):
    dec, gh = g["decoder_pre_denorm"], M.head(g["combined_normalized"])
    act = M.active_rows(gh)
    idx = act.nonzero().flatten()
    k = math.ceil(0.01 * len(idx))
    width = M.spread(gh).double().reshape(gh.shape[0], -1).norm(dim=1)[idx]
    wide = idx[width.argsort(descending=True)[:k]]
    return {
        "collapsed": (scale_band(dec, slice(None), 0.0), "all series"),
        "inactive_collapsed": (scale_band(dec, ~act, 0.0), f"{int((~act).sum())} inactive series"),
        "active1pct_0.3x": (scale_band(dec, wide, 0.3), f"{k} of {len(idx)} active series"),
        "all_0.88x": (scale_band(dec, slice(None), 0.88), "all series"),
        "all_1.12x": (scale_band(dec, slice(None), 1.12), "all series"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    gates = json.loads(args.gates.read_text())
    results, bad = [], []
    for name in [f"bench_{s}" for s in BENCH] + list(TEST_CASES):
        g = torch.load(GOLDENS / f"{name}.pt", weights_only=True)
        mean, stdev = g["revin_mean"][..., :1], g["revin_stdev"][..., :1]
        key = name.removeprefix("bench_")
        for kind, (dec, what) in broken(g).items():
            q, pt = dfl.back_end(dec, mean, stdev, g["horizon"])
            m = M.gate_metrics(
                dfl.combine_cpm(dec, g["horizon"]),
                g["combined_normalized"],
                q,
                g["quantile_outputs"],
                pt,
                g["prediction_outputs"],
            )
            for precision in ("accuracy", "performance"):
                fl = _floors(gates, key, precision)
                fails = [k for k, f in fl.items() if k in m and not passes(k, m[k], f)]
                results.append(
                    {"case": key, "broken": kind, "rows": what, "precision": precision, "fails": fails, "metrics": m}
                )
                print(f"{key:13s} {precision:11s} {kind:19s} ({what}): fails {len(fails)} {fails}")
                if not fails:
                    bad.append(f"{key}/{precision}/{kind}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"{len(bad)} broken output(s) pass every floor" + (": " + ", ".join(bad) if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
