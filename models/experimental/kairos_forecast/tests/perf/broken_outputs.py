# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Score synthetic broken outputs against the gate floors, in both precisions, on every bench shape (CPU only).

    python models/experimental/tirex2_forecast/tests/perf/broken_outputs.py [--gates gates.json] [--out out.json]

Each broken output is made from the golden head output, and scored by the bench's metric code (``metrics.py``):
``collapsed`` (every quantile -> the quantile mean), ``inactive_collapsed`` (only inactive rows collapsed),
``active1pct_0.3x`` (the band of ceil(1 %) of the active rows, the widest ones, scaled 0.3x about the quantile
mean), ``all_0.88x`` and ``all_1.12x`` (every band scaled). Each must fail at least one floor per precision
(PLAN section 9 v1.9, D3 condition 1a). Exit status 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from models.experimental.tirex2_forecast.reference import dataflow as df
from models.experimental.tirex2_forecast.tests.pcc import metrics as M
from models.experimental.tirex2_forecast.tests.perf.check_gates import DEFAULT_GATES, _floors, passes
from models.experimental.tirex2_forecast.tests.perf.test_forward_perf import shape_inputs

SHAPES = ("paper", "long_context", "multivariate")


def scale_band(head, rows, factor):
    h = head.reshape(*head.shape[:-1], M.Q, M.P).clone()
    mean = h.mean(-2, keepdim=True)
    h[rows] = mean[rows] + factor * (h[rows] - mean[rows])
    return h.reshape(head.shape)


def broken(g_head):
    act = M.active_rows(g_head)
    idx = act.nonzero().flatten()
    k = math.ceil(0.01 * len(idx))
    width = M.spread(g_head).double().reshape(g_head.shape[0], -1).norm(dim=1)[idx]
    wide = idx[width.argsort(descending=True)[:k]]
    inact = M.collapsed(g_head).clone()
    inact[act] = g_head[act]
    return {
        "collapsed": (M.collapsed(g_head), "all rows"),
        "inactive_collapsed": (inact, f"{int((~act).sum())} inactive rows"),
        "active1pct_0.3x": (scale_band(g_head, wide, 0.3), f"{k} of {len(idx)} active rows"),
        "all_0.88x": (scale_band(g_head, slice(None), 0.88), "all rows"),
        "all_1.12x": (scale_band(g_head, slice(None), 1.12), "all rows"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    gates = json.loads(args.gates.read_text())
    results, bad = [], []
    for shape in SHAPES:
        g, ctx, horizon, gid, tm = shape_inputs(shape)
        gid = torch.arange(ctx.shape[0]) if gid is None else gid
        tm = torch.ones(ctx.shape[0], dtype=torch.bool) if tm is None else tm
        p = df.prepare(ctx, horizon, gid, tm)
        for name, (head, what) in broken(g["head_out"]).items():
            q = M.forecast_from_head(head, p.loc, p.scale, horizon, tm)
            m = M.gate_metrics(head, g["head_out"], q, g["q_one"], p.scale[tm])
            for precision in ("accuracy", "performance"):
                fails = [k for k, f in _floors(gates, shape, precision).items() if not passes(k, m[k], f)]
                results.append(
                    {"shape": shape, "broken": name, "rows": what, "precision": precision, "fails": fails, "metrics": m}
                )
                print(f"{shape:13s} {precision:11s} {name:19s} ({what}): fails {len(fails)} {fails}")
                if not fails:
                    bad.append(f"{shape}/{precision}/{name}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"{len(bad)} broken output(s) pass every floor" + (": " + ", ".join(bad) if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
