# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Broken-output scoring (skill §3, PLAN v1.9 / v1.20): wrong outputs built from the golden head, scored through the
same post-processing as the device (inverse norm, flip average on both halves, clamp), against gates.json floors.

    collapsed   every quantile replaced by the quantile mean
    band_0.88 / band_1.12   the band around the quantile mean scaled uniformly
    no_negate   the -x half's rows carry the x half's head (the flip's negation forgotten)
Every one must fail; the golden itself must pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from models.experimental.kairos_forecast.tests.pcc import metrics as M
from models.experimental.kairos_forecast.tests.perf.check_gates import DEFAULT_GATES, _floors, passes, report_only
from models.experimental.kairos_forecast.tests.perf.suite import _post, load, score


def _band(raw, k):
    h = raw.reshape(raw.shape[0], 2, M.Q, 64)
    mu = h.mean(2, keepdim=True)
    return (mu + k * (h - mu)).reshape(raw.shape)


def broken(raw, S):
    out = {"collapsed": M.collapsed_raw(raw), "band_0.88": _band(raw, 0.88), "band_1.12": _band(raw, 1.12)}
    nn = raw.clone()
    nn[S:] = raw[:S]
    out["no_negate"] = nn
    return out


def score_case(name):
    """{label: metrics} for the golden and each broken output of one H <= 128 GIFT case."""
    rec = load(name)
    S = rec["context"].shape[0]
    gold = torch.cat(rec["head"])
    post = _post(rec)
    g_q, scale = rec["prediction_outputs"], rec["loc_scale"][0][1].reshape(-1)
    perfect = {"route_mismatch_rows": 0, "bias_max_abs_diff": 0.0}
    res = {"golden": {**score(gold, gold, post(gold), g_q, scale), **perfect}}
    for label, raw in broken(gold, S).items():
        res[label] = {**score(raw, gold, post(raw), g_q, scale), **perfect}
    return res


def verdicts(name, precision, gates):
    floors = _floors(gates, name, precision)
    out = {}
    for label, m in score_case(name).items():
        fails = [k for k, f in floors.items() if k in m and not passes(k, m[k], f)
                 and not report_only({"name": name}, k)]
        out[label] = {"fails": fails, **{k: m[k] for k in floors if k in m}}
    return out


if __name__ == "__main__":
    gates = json.loads(DEFAULT_GATES.read_text())
    res = {f"{c}/{p}": verdicts(c, p, gates) for c in sys.argv[1:] or ["paper", "ragged"]
           for p in ("accuracy", "performance")}
    print(json.dumps(res, indent=1))
