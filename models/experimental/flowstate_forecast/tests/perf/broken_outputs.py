# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Score synthetic broken outputs with the checker, both precisions, every bench shape and suite case (CPU).

    python models/experimental/flowstate_forecast/tests/perf/broken_outputs.py [--gates gates.json]
        [--baseline fixed_record.json] [--out out.json]

Each broken output is made in head space from the golden ``combined_normalized`` ``(B, 9, H, 1)`` and taken to
data space the reference's way (x stdev + mean per step, the per-step statistics combined like the head, then
torch.quantile and the weighted mean), so the head and data-space metrics stay consistent. Each is scored as a
candidate record by ``check_gates.check`` (floors and drift against the fixed record), with the same session stamp
as the scoring (v1.23). The golden itself is scored first, and only failures beyond the golden's own count
(v1.26: no vacuous catches).

    collapsed           every quantile -> the quantile mean (PLAN v1.9)
    inactive_collapsed  only the inactive series (not applicable without inactive series)
    active1pct_0.3x     the band of ceil(1 %) of the active series, the widest, scaled 0.3x
    all_0.88x, all_1.12x
    level_shift         +0.05 normalized on every row (v1.22); also given in data units against the row std
    level_shift_rows    +0.05 on 1 row at B <= 16, 2 at B <= 32, 4 at B <= 64, ceil(1 %) above (v1.25), the
                        active rows with the largest golden std
    lag1                the forecast one step late (v1.22)
    band_1.3x_row       the band of one active row (the widest) scaled 1.3x (v1.26)

Every one must fail on every applicable case and precision. Exit status 1 otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path

import torch

from models.experimental.flowstate_forecast.reference import dataflow as dfl
from models.experimental.flowstate_forecast.tests.golden.test_dataflow import GOLDENS, TEST_CASES
from models.experimental.flowstate_forecast.tests.pcc import metrics as M
from models.experimental.flowstate_forecast.tests.perf.check_gates import (
    DEFAULT_BASELINE,
    DEFAULT_GATES,
    check,
    load_records,
    session_stamp,
)

BENCH = ("paper", "long_context", "long_horizon", "scale_q", "single", "single_long")
SHIFT = 0.05


def scale_band(h, rows, factor):
    h = h.clone()
    mean = h.mean(1, keepdim=True)
    h[rows] = mean[rows] + factor * (h[rows] - mean[rows])
    return h


def n_shift_rows(B):
    return 1 if B <= 16 else 2 if B <= 32 else 4 if B <= 64 else math.ceil(0.01 * B)


def broken(gh):
    """gh (B, 9, H, 1) -> {name: (head, what)}."""
    h3 = M.head(gh)
    act = M.active_rows(h3)
    idx = act.nonzero().flatten()
    width = M.spread(h3).double().reshape(h3.shape[0], -1).norm(dim=1)
    wide = idx[width[idx].argsort(descending=True)]
    k = math.ceil(0.01 * len(idx))
    std = h3.double().reshape(h3.shape[0], -1).std(1)
    ks = n_shift_rows(gh.shape[0])
    big = idx[std[idx].argsort(descending=True)[:ks]]
    shifted = gh.clone()
    shifted[big] += SHIFT
    lag = gh.clone()
    lag[:, :, 1:] = gh[:, :, :-1]
    return {
        "collapsed": (scale_band(gh, slice(None), 0.0), "all series"),
        "inactive_collapsed": (scale_band(gh, ~act, 0.0), f"{int((~act).sum())} inactive series"),
        "active1pct_0.3x": (scale_band(gh, wide[:k], 0.3), f"{k} of {len(idx)} active series"),
        "all_0.88x": (scale_band(gh, slice(None), 0.88), "all series"),
        "all_1.12x": (scale_band(gh, slice(None), 1.12), "all series"),
        "level_shift": (gh + SHIFT, "all series"),
        "level_shift_rows": (shifted, f"rows {big.tolist()} of {gh.shape[0]}"),
        "lag1": (lag, "all series"),
        "band_1.3x_row": (scale_band(gh, wide[:1], 1.3), f"row {wide[:1].tolist()}"),
    }


def to_data(g, head):
    """The reference back end from head space: per-step stdev / mean combined like the head."""
    dec = g["decoder_pre_denorm"]
    rows = dec.shape[0]
    sd = g["revin_stdev"][..., :1][:, -rows:].transpose(0, 1).unsqueeze(2).unsqueeze(-2)
    mu = g["revin_mean"][..., :1][:, -rows:].transpose(0, 1).unsqueeze(2).unsqueeze(-2)
    one = torch.ones_like(dec)
    SD, MU = dfl.combine_cpm(one * sd, g["horizon"]), dfl.combine_cpm(one * mu, g["horizon"])
    pred = head * SD + MU
    q = torch.tensor(dfl.QUANTILES)
    quantiles = torch.quantile(pred, q, 1).transpose(0, 1)
    w = 0.5 - (0.5 - q).abs()
    w = (w / w.sum()).view(1, -1, 1, 1)
    return quantiles, (quantiles * w).sum(1), SD


def _metric(fail: str) -> str:
    """'(name, precision) metric ...' -> metric (the check that failed)."""
    rest = fail.split(")", 1)[1].split()
    return rest[0] if rest else fail


def score(rec, base, gates, gates_dir):
    with contextlib.redirect_stdout(io.StringIO()):
        return {_metric(f) for f in check([rec], base, gates, gates_dir, require_coverage=False)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    stamp = session_stamp()
    gates = json.loads(args.gates.read_text())
    base = load_records(args.baseline)
    results, bad = [], []
    for name in [f"bench_{s}" for s in BENCH] + list(TEST_CASES):
        g = torch.load(GOLDENS / f"{name}.pt", weights_only=True)
        gh = g["combined_normalized"]
        key = name.removeprefix("bench_") if name.startswith("bench_") else f"suite:{name}"
        rec = lambda h, prec: {  # noqa: E731
            "name": key,
            "precision": prec,
            **M.gate_metrics(h, gh, *to_data(g, h)[:1], g["quantile_outputs"], to_data(g, h)[1], g["prediction_outputs"]),
            **stamp,
        }
        own = {p: score(rec(gh, p), base, gates, args.gates.parent) for p in ("accuracy", "performance")}
        for p in own:
            print(f"{key:20s} {p:11s} golden itself fails {sorted(own[p])}")
        for kind, (h, what) in broken(gh).items():
            if kind == "inactive_collapsed" and not bool((~M.active_rows(M.head(gh))).any()):
                print(f"{key:20s} {kind:19s}: not applicable (no inactive series; identical to the golden)")
                continue
            extra = ""
            if kind == "level_shift":
                _, _, SD = to_data(g, gh)
                real = (SHIFT * SD[:, 0, :, 0]).mean(1) / M.head(g["quantile_outputs"]).double().reshape(gh.shape[0], -1).std(1)
                extra = f"; +0.05 normalized = {float(real.median()):.3g} x the data row std (median over rows)"
            for p in ("accuracy", "performance"):
                fails = sorted(score(rec(h, p), base, gates, args.gates.parent) - own[p])
                results.append(
                    {"case": key, "broken": kind, "rows": what + extra, "precision": p, "fails": fails, **stamp}
                )
                print(f"{key:20s} {p:11s} {kind:19s} ({what}{extra}): fails {len(fails)} {fails}")
                if not fails:
                    bad.append(f"{key}/{p}/{kind}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"{len(bad)} broken output(s) pass every floor and drift" + (": " + ", ".join(bad) if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
