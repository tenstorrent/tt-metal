# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Broken-output scoring (skill §3; PLAN v1.9, v1.20, v1.22, v1.24; D3 item 1): wrong outputs built from the golden
head of every forward, put through the same post-processing as the device (inverse norm, flip average, clamp; a
rollout's steps each de-normalized with their own golden loc / scale), scored into the same records as the suite
(`<case>`, and `<case>_tf` for rollouts), and checked with check_gates.check against gates.json's floors and the
drift from a fixed record (suite.json and bench.json).

    collapsed           every quantile replaced by the quantile mean
    band_0.88/1.12      the band around the quantile mean scaled uniformly
    band_0.3_1pct       1 % of each forward's active rows (at least one) at a 0.3x band
    inactive_collapsed  the inactive rows (golden-defined) collapsed
    no_negate           the -x half carries the x half's head (flip cases)
    level_shift[_h0|_h1]  +0.05 normalized on both halves / the x half / the -x half (head space; a shift common to
                        both halves cancels in (a - flip(b)) / 2)
    lag1[_h0|_h1]       every quantile one step late along the 128-step token pair (head space)
    level_shift_final, lag1_final   the same on the final data-space forecast only (+0.05 x the target's scale)
    spread_shift_rows_<f>[_h0|_h1|_final]  one row's band x1.3 or x0.7, on the level_shift_rows row counts (D5 R1-0a)
    level_shift_rows[_h0|_h1|_final]  +0.05 on a few rows only (v1.25, D4 item 1): 1 row at B <= 16, 2 at 32, 4 at 64,
                        1 % of the rows on paper / ragged (B = series per forward; the first active rows)

The row-indexed drift (bias_row_drift_max[_final]) of a broken record is taken against the golden record itself
(row_base): the perturbation is the whole move.

A broken output is caught when a floor fails, or a drift check fails that the golden itself passes (the golden is
not a device output, so a drift check it fails, e.g. performance spread_ratio_p50 of 1.000 vs a fixed 1.012, says
nothing about the perturbation; those are listed as `golden_drift_misses`). A scoring identical to the golden (no
inactive rows, no flip) is `not applicable`. The golden must pass every floor.

    python -m models.experimental.kairos_forecast.tests.perf.broken_outputs <out.json> [case ...]
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
from pathlib import Path

import torch

from models.experimental.kairos_forecast.reference.dataflow import head_to_quantiles
from models.experimental.kairos_forecast.tests.pcc import metrics as M
from models.experimental.kairos_forecast.tests.perf.check_gates import DEFAULT_GATES, check, load_records, session_stamp
from models.experimental.kairos_forecast.tests.perf.suite import GIFT_CASES, _post, load, score

FIXED = {
    "suite": DEFAULT_GATES.parent / "iter_00" / "suite.jsonl",
    "bench": DEFAULT_GATES.parent / "iter_00" / "bench.json",
}
if os.environ.get("KAIROS_FIXED_SUITE"):  # development: score against another record (e.g. a pcc job's records)
    FIXED = {"suite": Path(os.environ["KAIROS_FIXED_SUITE"])}
SHIFT = 0.05
SPREAD_SHIFTS = (1.3, 0.7)  # D5 R1-0a: per-row band scaling probes
PERFECT = {"route_mismatch_rows": 0, "route_mismatch_patches": 0, "bias_max_abs_diff": 0.0, "min_top3_margin": 1.0}


def _seq(raw):
    """[rows, 2, 576] -> [rows, 9, 128] (both decoder tokens as one horizon) and back."""
    R = raw.shape[0]
    return raw.reshape(R, 2, M.Q, 64).permute(0, 2, 1, 3).reshape(R, M.Q, 128)


def _unseq(s):
    R = s.shape[0]
    return s.reshape(R, M.Q, 2, 64).permute(0, 2, 1, 3).reshape(R, 2, M.Q * 64)


def _band(raw, k, rows=None):
    h = raw.reshape(raw.shape[0], 2, M.Q, 64).clone()
    mu = h.mean(2, keepdim=True)
    sel = slice(None) if rows is None else rows
    h[sel] = (mu + k * (h - mu))[sel]
    return h.reshape(raw.shape)


def _lag(s):
    out = s.clone()
    out[..., 1:] = s[..., :-1]
    return out


def _active(raw):
    return M.active_rows(M.head_view(raw))


def shift_rows(name, n_series):
    """How many rows level_shift_rows moves (D4 item 1)."""
    if name.split("_tf")[0] in ("paper", "ragged"):
        return max(1, math.ceil(0.01 * n_series))
    return max(1, n_series // 16)


def _shift_rows(raw, k):
    """+SHIFT on the first k active rows of one forward's (one half's) head."""
    rows = _active(raw).nonzero().flatten()[:k]
    out = raw.clone()
    out[rows] += SHIFT
    return out


def head_transforms(k_rows=1):
    """label -> f(raw [rows, 2, 576], half) -> raw. half: 0 for the x half (or no flip), 1 for the -x half."""

    def one_pct(raw, _h):
        act = _active(raw).nonzero().flatten()
        k = max(1, math.ceil(0.01 * len(act)))
        return _band(raw, 0.3, act[:k])

    def inactive(raw, _h):
        return _band(raw, 0.0, ~_active(raw))

    t = {
        "collapsed": lambda r, h: M.collapsed_raw(r),
        "band_0.88": lambda r, h: _band(r, 0.88),
        "band_1.12": lambda r, h: _band(r, 1.12),
        "band_0.3_1pct": one_pct,
        "inactive_collapsed": inactive,
        "level_shift": lambda r, h: r + SHIFT,
        "lag1": lambda r, h: _unseq(_lag(_seq(r))),
    }
    for half in (0, 1):
        t[f"level_shift_h{half}"] = lambda r, h, half=half: r + SHIFT if h == half else r
        t[f"lag1_h{half}"] = lambda r, h, half=half: _unseq(_lag(_seq(r))) if h == half else r
        t[f"level_shift_rows_h{half}"] = lambda r, h, half=half: _shift_rows(r, k_rows) if h == half else r
    t["level_shift_rows"] = lambda r, h: _shift_rows(r, k_rows)
    for f in SPREAD_SHIFTS:  # D5 R1-0a: one row's band scaled, on the level_shift_rows row counts
        t[f"spread_shift_rows_{f}"] = lambda r, h, f=f: _band(r, f, _active(r).nonzero().flatten()[:k_rows])
        for half in (0, 1):
            t[f"spread_shift_rows_{f}_h{half}"] = (
                lambda r, h, f=f, half=half: _band(r, f, _active(r).nonzero().flatten()[:k_rows]) if h == half else r
            )
    return t


def case_records(name, transform=None, final=None):
    """The suite records of one case built from (transformed) golden heads. transform(raw, half) per forward;
    final(q, scale) on the data-space forecast; for no_negate, transform == "no_negate"."""
    rec = load(name)
    S, h, gift = rec["context"].shape[0], rec["horizon"], rec["gift"]
    heads = list(rec["head"])
    n = len(heads)
    halves = [int(gift and k >= n // 2) for k in range(n)]
    if transform == "no_negate":
        dev = heads[: n // 2] * 2
    elif transform is not None:
        dev = [transform(x, hf) for x, hf in zip(heads, halves)]
    else:
        dev = heads
    g_q, scale = rec["prediction_outputs"], rec["loc_scale"][0][1].reshape(-1)
    base = {"name": name, "rows": S * (2 if gift else 1), "forwards": n, **PERFECT}
    if n <= 2:  # one forward pair (H <= 128)
        d, g = torch.cat(dev), torch.cat(heads)
        q = _post(rec)(d)
        q = final(q, scale) if final else q
        return [{**base, **score(d, g, q, g_q, scale, 2 if gift else 1)}]
    steps = n // 2 if gift else n
    qs = [head_to_quantiles(dev[k], *rec["loc_scale"][k]) for k in range(n)]
    a = torch.cat(qs[:steps], -1)[..., :h]
    if gift:
        b = torch.cat(qs[steps:], -1)[..., :h]
        q = (a - torch.flip(b, dims=[1])) / 2
        pos = (~torch.any(rec["context"] < 0, dim=1)).view(-1, 1, 1)
        q = torch.where(pos, q.clamp_min(0.0), q)
    else:
        q = a
    q = final(q, scale) if final else q
    r = {**base, **score(dev[0], heads[0], q, g_q, scale), "free_step": 1}
    # teacher-forced record: every step's rows, data space per step with that step's golden loc / scale
    gq = torch.cat([head_to_quantiles(heads[k], *rec["loc_scale"][k]) for k in range(n)])
    dq = torch.cat(qs)
    sc = torch.cat([rec["loc_scale"][k][1].reshape(-1) for k in range(n)])
    # no final output in the teacher-forced record: a `final` scoring leaves it equal to the golden (not applicable)
    tf = score(torch.cat(dev), torch.cat(heads), dq, gq, sc, 2 if gift else 1)
    tf["pcc_normalized_step_min"] = min(M.pearson(M.head_view(heads[k]), M.head_view(dev[k])) for k in range(n))
    return [r, {**base, "name": f"{name}_tf", "rows": heads[0].shape[0] * n, **tf}]


def scorings(name):
    rec = load(name)
    gift = rec["gift"]
    out = {"golden": case_records(name)}
    k = shift_rows(name, rec["context"].shape[0])
    for label, f in head_transforms(k).items():
        if not gift and label[-3:] in ("_h0", "_h1"):
            continue
        out[label] = case_records(name, f)
    if gift:
        out["no_negate"] = case_records(name, "no_negate")
    out["level_shift_final"] = case_records(name, final=lambda q, s: q + SHIFT * s.view(-1, 1, 1))
    out["lag1_final"] = case_records(name, final=lambda q, s: _lag(q))

    def rows_final(q, s):
        q = q.clone()
        q[:k] += SHIFT * s[:k].view(-1, 1, 1)
        return q

    out["level_shift_rows_final"] = case_records(name, final=rows_final)

    def spread_final(f):
        def g(q, s):
            q = q.clone()
            mu = q[:k].mean(1, keepdim=True)
            q[:k] = mu + f * (q[:k] - mu)
            return q

        return g

    for f in SPREAD_SHIFTS:
        out[f"spread_shift_rows_{f}_final"] = case_records(name, final=spread_final(f))
    return out


def real_units(name):
    """+0.05 normalized in data units, against each series' context std (v1.22)."""
    rec = load(name)
    x = rec["context"]
    scale = rec["loc_scale"][0][1].reshape(-1)
    std = torch.stack([r[~torch.isnan(r)].std() for r in x])
    ratio = SHIFT * scale / std.clamp_min(1e-12)
    return {"shift_over_row_std_p50": float(ratio.median()), "min": float(ratio.min()), "max": float(ratio.max())}


def _failed_metrics(recs, base, gates, row_base=None):
    with contextlib.redirect_stdout(io.StringIO()):
        fails = check(recs, base, gates, require_coverage=False, allow_unstamped=True, row_base=row_base)
    out = []
    for f in fails:
        if ") " not in f:  # "(name, precision): dirty tree" and the other stamp failures
            out.append((f.split("): ", 1)[-1], "stamp"))
            continue
        rest = f.split(") ", 1)[1].split()
        if rest[0].endswith(":"):  # "<metric>: no floor" / "missing": a gates.json or record gap, never a catch
            out.append((rest[0][:-1], "stamp"))
            continue
        out.append((rest[0], "drift" if len(rest) > 1 and rest[1] == "drift" else "floor"))
    return sorted(set(out))


def verdicts(cases, gates, fixed):
    """{case/precision: {set: {label: {record name: caught-by list}}}}."""
    res = {}
    stamp = session_stamp()
    for name in cases:
        sc = scorings(name)
        for prec in ("accuracy", "performance"):
            row = {}
            for set_name, recs_fixed in fixed.items():
                base = [r for r in recs_fixed if r["precision"] == prec]
                mk = lambda recs: [{**r, "precision": prec, **stamp} for r in recs]  # noqa: E731
                golden = {}
                gold_recs = mk(sc["golden"])
                for r in gold_recs:
                    golden[r["name"]] = _failed_metrics([r], base, gates, gold_recs)
                per = {"golden": golden}
                for label, recs in sc.items():
                    if label == "golden":
                        continue
                    lab = {}
                    for r, g in zip(mk(recs), mk(sc["golden"])):
                        if all(r.get(k) == g.get(k) for k in r if isinstance(r[k], float)):
                            lab[r["name"]] = "not applicable (identical to the golden)"
                            continue
                        gd = {m for m, kind in golden[r["name"]] if kind == "drift"}
                        lab[r["name"]] = [
                            f"{kind}:{m}"
                            for m, kind in _failed_metrics([r], base, gates, gold_recs)
                            if kind == "floor" or (kind == "drift" and m not in gd)
                        ]
                    per[label] = lab
                row[set_name] = per
            res[f"{name}/{prec}"] = row
    return res


def problems(res):
    """Every way the scoring fails its purpose: a golden floor miss, or a broken record that nothing catches."""
    bad = []
    for cp, sets in res.items():
        for set_name, per in sets.items():
            for rname, fails in per["golden"].items():
                if any(kind in ("floor", "stamp") for _m, kind in fails):
                    bad.append(f"{cp} {set_name} golden {rname} fails a floor: {fails}")
            for label, lab in per.items():
                if label == "golden":
                    continue
                for rname, caught in lab.items():
                    if isinstance(caught, list) and not caught:
                        bad.append(f"{cp} {set_name} {label} {rname}: not caught")
    return bad


def main(out_path, cases):
    gates = json.loads(DEFAULT_GATES.read_text())
    fixed = {k: load_records(p) for k, p in FIXED.items()}
    res = verdicts(cases, gates, fixed)
    bad = problems(res)
    out = {
        **session_stamp(),
        "fixed": {k: str(p) for k, p in FIXED.items()},
        "level_shift_real_units": {c: real_units(c) for c in cases},
        "verdicts": res,
        "problems": bad,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    for p in bad:
        print("PROBLEM", p)
    print(f"{len(bad)} problem(s); wrote {out_path}")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1], sys.argv[2:] or GIFT_CASES) else 0)
