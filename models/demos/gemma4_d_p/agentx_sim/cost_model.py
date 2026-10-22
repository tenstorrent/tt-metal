#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cost model for Gemma4 chunked prefill steps (BH Galaxy 8x4, CP8 / TP4).

    step(batch) = dense(M_total) + sum_r attn(C_r, P_r)

per layer type, times 10 global + 50 sliding layers. M = rows per device = sum(C_r) / CP. Data lives in a JSON file
(model_data_revc.json next to this script), with a silicon calibration per scheduler (Model.step_s).
"""

import bisect
import json
import math
import os

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data_revc.json")
CP = 8
N_LAYERS = {"global": 10, "sliding": 50}


def _per_lanes(value, lanes):
    """A calibration value: one number, or per lane count {"1": ..., "4": ...}; past the last count, its value."""
    if not isinstance(value, dict):
        return value
    return value[str(min(lanes, max(map(int, value))))]


class Model:
    @classmethod
    def load(cls, path=DEFAULT_DATA):
        with open(path) as f:
            return cls(json.load(f))

    def __init__(self, data):
        # dense(M) per layer type: piecewise linear through measured M points (no extrapolation past the ends).
        self.dense_pts = {L: sorted((int(m), v) for m, v in data["dense_us"][L].items()) for L in N_LAYERS}
        # attn_global(C, P) = a_C + b_C * P per layer (fitted over every measured prefix depth).
        self.g = {int(C): ab for C, ab in data["gattn_us"].items()}
        self.l = {int(C): v for C, v in data["lattn_us"].items()}
        self.cal = data["calibration"]
        self._cal_for = {}

    # ---- components (per layer, us) ----
    def dense(self, L, M):
        pts = self.dense_pts[L]
        ms = [m for m, _ in pts]
        if M <= ms[0]:
            # Below the smallest measured M: hold flat (rows are free when weights dominate).
            return pts[0][1]
        if M > ms[-1]:
            raise ValueError(f"M={M} past the last measured point {ms[-1]}; not extrapolating")
        i = bisect.bisect_left(ms, M)
        if ms[i] == M:
            return pts[i][1]
        (m0, v0), (m1, v1) = pts[i - 1], pts[i]
        return v0 + (v1 - v0) * (M - m0) / (m1 - m0)

    def gattn(self, C, P):
        a, b = self.g[C]
        return a + b * P

    # ---- one step ----
    def step(self, batch):
        """batch: list of (C, P). Returns dict of per-step us by component (all layers summed)."""
        M = sum(C for C, _ in batch) // CP
        return {
            "dense": sum(N_LAYERS[L] * self.dense(L, M) for L in N_LAYERS),
            "gattn": N_LAYERS["global"] * sum(self.gattn(C, P) for C, P in batch),
            "lattn": N_LAYERS["sliding"] * sum(self.l[C] for C, _ in batch),
        }

    def calibration(self, C, budget):
        """(calibration entry, None) for a scheduler fitted on silicon, else (the nearest entry, why it is not a fit).
        Unbatched units without their own entry take the nearest calibrated unit; lanes of any width take the
        'batched' entry, past its last lane count its last values."""
        if (C, budget) not in self._cal_for:
            if C == budget:
                u = min((int(k) for k in self.cal if k.isdigit()), key=lambda k: abs(math.log2(k / C)))
                why = None if u == C else f"unbatched {C} priced with the {u} calibration"
                self._cal_for[C, budget] = self.cal[str(u)], why
            else:
                cal = self.cal["batched"]
                lanes = max(map(int, cal["host_ms"]))
                fit = C == cal["unit"] and budget // C <= lanes
                why = (
                    None
                    if fit
                    else f"{C}-wide x{budget // C} lanes priced with the {cal['unit']}-wide x1-{lanes} calibration"
                )
                self._cal_for[C, budget] = cal, why
        return self._cal_for[C, budget]

    def step_s(self, batch, C, budget, calibrated=True):
        """(busy s, gap s) of one step of a scheduler with this unit and budget, calibrated on silicon: device time
        times a factor (one, or per component), plus host time; then the scheduler's gap before the next step, which
        is wall time but not busy time. The entry is from Model.calibration; any value may be per lane count.
        calibrated=False: the model's device time alone."""
        if not calibrated:
            return sum(self.step(batch).values()) / 1e6, 0.0
        cal = self.calibration(C, budget)[0]
        n, scale = len(batch), cal["scale"]
        us = sum(t * _per_lanes(scale[c] if isinstance(scale, dict) else scale, n) for c, t in self.step(batch).items())
        return us / 1e6 + _per_lanes(cal["host_ms"], n) / 1e3, cal.get("gap_ms", 0.0) / 1e3
