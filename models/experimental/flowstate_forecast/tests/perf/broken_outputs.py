# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Score synthetic broken outputs with the checker, both precisions, every bench shape and suite case (CPU).

    python models/experimental/flowstate_forecast/tests/perf/broken_outputs.py [--gates gates.json]
        [--baseline fixed_record.json] [--out out.json]

Each broken output is made in head space from the golden ``combined_normalized`` ``(B, 9, H, 1)`` and taken to
data space the reference's way (x stdev + mean per step, the per-step statistics combined like the head, then
torch.quantile and the weighted mean), so the head and data-space metrics stay consistent. Each is scored as a
candidate record by ``check_gates.check`` (floors and drift against the fixed record), with the same session stamp
as the scoring (v1.23). Each base output (the golden, and the device output with ``--dumps``) is scored first, and
only failures beyond the base's own count, by (metric, row) for the row-indexed drift (v1.26, v1.27). The golden
must pass every drift (R1-0a). Shifts and single-row band scalings are probed in both signs.

    collapsed           every quantile -> the quantile mean (PLAN v1.9)
    inactive_collapsed  only the inactive series (not applicable without inactive series)
    active1pct_0.3x     the band of ceil(1 %) of the active series, the widest, scaled 0.3x
    all_0.88x, all_1.12x
    level_shift(_neg)   +-0.05 normalized on every row (v1.22); also given in data units against the row std
    level_shift_rows(_neg) +-0.05 on 1 row at B <= 16, 2 at B <= 32, 4 at B <= 64, ceil(1 %) above (v1.25), the
                        active rows with the largest golden std
    lag1                the forecast one step late (v1.22)
    band_1.3x_row       the band of one active row (the widest) scaled 1.3x (v1.26); band_0.7x_row 0.7x
    top_q_shift(_neg)   +-0.02 on the top quantile only, every row (v1.31, D5 item 2)
    inactive_shift(_neg) +-0.03 level shift of the inactive rows (v1.31, D5 item 3; n/a without them)

With ``--dumps`` (the bench's MODEL_PERF_DUMP and the suite's FLOWSTATE_SUITE_DUMP outputs of the fixed record's
jobs), each broken output is made from the device output of that case and precision instead of the golden, and
the device output itself is the "own" record: the perturbation a G2 candidate would bring. A row whose device bias
is already near +0.05 hides a +0.05 shift of the golden from the row-indexed drift, not one of the device output.

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
TOP_Q_SHIFT = 0.02  # D5 item 2
INACTIVE_SHIFT = 0.03  # D5 item 3


def scale_band(h, rows, factor):
    h = h.clone()
    mean = h.mean(1, keepdim=True)
    h[rows] = mean[rows] + factor * (h[rows] - mean[rows])
    return h


def n_shift_rows(B):
    return 1 if B <= 16 else 2 if B <= 32 else 4 if B <= 64 else math.ceil(0.01 * B)


def broken(base, gh):
    """base (B, 9, H, 1): the output to break (device or golden); gh: the golden, which picks the rows."""
    h3 = M.head(gh)
    act = M.active_rows(h3)
    idx = act.nonzero().flatten()
    width = M.spread(h3).double().reshape(h3.shape[0], -1).norm(dim=1)
    wide = idx[width[idx].argsort(descending=True)]
    k = math.ceil(0.01 * len(idx))
    std = h3.double().reshape(h3.shape[0], -1).std(1)
    ks = n_shift_rows(gh.shape[0])
    big = idx[std[idx].argsort(descending=True)[:ks]]
    shifted = base.clone()
    shifted[big] += SHIFT
    shifted_neg = base.clone()
    shifted_neg[big] -= SHIFT
    top, top_neg = base.clone(), base.clone()
    top[:, -1] += TOP_Q_SHIFT
    top_neg[:, -1] -= TOP_Q_SHIFT
    ina, ina_neg = base.clone(), base.clone()
    ina[~act] += INACTIVE_SHIFT
    ina_neg[~act] -= INACTIVE_SHIFT
    lag = base.clone()
    lag[:, :, 1:] = base[:, :, :-1]
    return {
        "collapsed": (scale_band(base, slice(None), 0.0), "all series"),
        "inactive_collapsed": (scale_band(base, ~act, 0.0), f"{int((~act).sum())} inactive series"),
        "active1pct_0.3x": (scale_band(base, wide[:k], 0.3), f"{k} of {len(idx)} active series"),
        "all_0.88x": (scale_band(base, slice(None), 0.88), "all series"),
        "all_1.12x": (scale_band(base, slice(None), 1.12), "all series"),
        "level_shift": (base + SHIFT, "all series"),
        "level_shift_neg": (base - SHIFT, "all series"),
        "level_shift_rows": (shifted, f"rows {big.tolist()} of {gh.shape[0]}"),
        "level_shift_rows_neg": (shifted_neg, f"rows {big.tolist()} of {gh.shape[0]}"),
        "lag1": (lag, "all series"),
        "band_1.3x_row": (scale_band(base, wide[:1], 1.3), f"row {wide[:1].tolist()}"),
        "band_0.7x_row": (scale_band(base, wide[:1], 0.7), f"row {wide[:1].tolist()}"),
        "top_q_shift": (top, "quantile 8, all series"),
        "top_q_shift_neg": (top_neg, "quantile 8, all series"),
        "inactive_shift": (ina, f"{int((~act).sum())} inactive series"),
        "inactive_shift_neg": (ina_neg, f"{int((~act).sum())} inactive series"),
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
    """'(name, precision) metric ...' -> metric, or 'metric row <i>' for a row-indexed failure (v1.27: the base's
    own failures are subtracted by (metric, row), so a probe failing another row of the same metric still counts)."""
    rest = fail.split(")", 1)[1].split()
    if len(rest) >= 3 and rest[1] in ("row", "quantile"):
        return f"{rest[0]} {rest[1]} {rest[2].rstrip(':')}"
    return rest[0] if rest else fail


def score(rec, base, gates, gates_dir):
    with contextlib.redirect_stdout(io.StringIO()):
        return {_metric(f) for f in check([rec], base, gates, gates_dir, require_coverage=False)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dumps", type=Path, help="device outputs: <shape>-<precision>.pt, suite_<case>-<precision>.pt")
    args = ap.parse_args()
    stamp = session_stamp()
    gates = json.loads(args.gates.read_text())
    base = load_records(args.baseline)
    results, bad = [], []
    for name in [f"bench_{s}" for s in BENCH] + list(TEST_CASES):
        g = torch.load(GOLDENS / f"{name}.pt", weights_only=True)
        gh = g["combined_normalized"]
        key = name.removeprefix("bench_") if name.startswith("bench_") else f"suite:{name}"

        def rec(h, prec):
            q, pt, _ = to_data(g, h)
            m = M.gate_metrics(h, gh, q, g["quantile_outputs"], pt, g["prediction_outputs"])
            return {"name": key, "precision": prec, **m, **stamp}

        for p in ("accuracy", "performance"):
            bases = [("golden", gh)]  # R1-0a: the golden is always scored, and must pass every drift
            if args.dumps:
                f = args.dumps / (f"{key}-{p}.pt" if not key.startswith("suite:") else f"suite_{key[6:]}-{p}.pt")
                if f.is_file():
                    bases.append(("device", torch.load(f, weights_only=True)["combined_normalized"]))
                else:
                    bad.append(f"{key}/{p}/no device dump")
            for src, base_h in bases:
                own = score(rec(base_h, p), base, gates, args.gates.parent)
                print(f"{key:20s} {p:11s} {src} output itself fails {sorted(own)}")
                if src == "golden" and (drift := sorted(x for x in own if "drift" in x or " row " in x)):
                    bad.append(f"{key}/{p}/golden fails drift {drift}")
                for kind, (h, what) in broken(base_h, gh).items():
                    if kind.startswith("inactive_") and not bool((~M.active_rows(M.head(gh))).any()):
                        print(f"{key:20s} {p:11s} {kind:20s}: not applicable (no inactive series)")
                        continue
                    extra = ""
                    if kind == "level_shift":
                        _, _, SD = to_data(g, gh)
                        gq = M.head(g["quantile_outputs"]).double().reshape(gh.shape[0], -1)
                        real = (SHIFT * SD[:, 0, :, 0]).mean(1) / gq.std(1)
                        extra = f"; +0.05 normalized = {float(real.median()):.3g} x the data row std (median over rows)"
                    fails = sorted(score(rec(h, p), base, gates, args.gates.parent) - own)
                    results.append(
                        {
                            "case": key,
                            "broken": kind,
                            "base": src,
                            "rows": what + extra,
                            "precision": p,
                            "fails": fails,
                            **stamp,
                        }
                    )
                    print(f"{key:20s} {p:11s} {kind:20s} ({src}; {what}{extra}): fails {len(fails)} {fails}")
                    if not fails:
                        bad.append(f"{key}/{p}/{src}/{kind}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(f"{len(bad)} broken output(s) pass every floor and drift" + (": " + ", ".join(bad) if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
