# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Score synthetic broken outputs against the gate floors, in both precisions, on every bench and suite case (CPU only).

    python models/experimental/tafsut_forecast/tests/perf/score_broken.py [--gates gates.json] [--out out.json]

Adapted from TiRex-2 ``651e8c4af93`` ``tests/perf/broken_outputs.py`` (PLAN v1.9, v1.20). Each broken output is
made from forward 1's golden head output and scored by the bench's metric code: ``collapsed`` (every quantile ->
the quantile mean), ``inactive_collapsed`` (only inactive rows collapsed), ``active1pct_0.3x`` (the band of ceil(1 %)
of the active rows, the widest ones, scaled 0.3x about the quantile mean), ``all_0.88x`` and ``all_1.12x`` (every
band scaled) and ``sign_flip`` (the head negated: the forecast mirrored about the series location, quantile order
reversed; Tafsut has no sign-flip path of its own, so this is the output-level analog of TiRex-2's D7 case). Each
must fail at least one floor per precision. Exit status 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


from models.experimental.tafsut_forecast.reference import dataflow as df
from models.experimental.tafsut_forecast.tests.golden.goldens import load_golden
from models.experimental.tafsut_forecast.tests.pcc import metrics as M
from models.experimental.tafsut_forecast.tests.perf.check_gates import DEFAULT_GATES, _floors, passes

BENCH = ("paper", "long_context", "long_horizon", "gift_ragged", "gift_ragged_odd")
SUITE = ("single", "single_long", "ragged4", "model16", "fallback", "horizon4")


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
        "sign_flip": (-g_head, "all rows"),
    }


def case_inputs(name):
    """Forward 1's golden head at its read-out tokens, its prep, its horizon slice and the golden quantiles there."""
    g = load_golden(name)
    take = min(g["horizon"], 1024)
    gh = g["forwards"][0]["head"][:, : -(-take // df.PATCH)]
    return gh, df.prepare_inputs(g["x"]), take, g["quantiles"][:, :take].transpose(1, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    gates = json.loads(args.gates.read_text())
    results, bad = [], []
    for shape in BENCH + SUITE:
        gh, prep, take, gq = case_inputs(shape)
        for name, (head, what) in broken(gh).items():
            q = df.postprocess(head.float(), prep, take).transpose(1, 2)
            m = M.gate_metrics(head, gh, q, gq, prep.scale)
            for precision in ("accuracy", "performance"):
                fl = _floors(gates, shape, precision)
                fails = [k for k, f in fl.items() if m.get(k) is not None and not passes(k, m[k], f)]
                results.append(
                    {"shape": shape, "broken": name, "rows": what, "precision": precision, "fails": fails, "metrics": m}
                )
                print(f"{shape:15s} {precision:11s} {name:19s} ({what}): fails {len(fails)} {fails}")
                if not fails:
                    bad.append(f"{shape}/{precision}/{name}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"{len(bad)} broken output(s) pass every floor" + (": " + ", ".join(bad) if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
