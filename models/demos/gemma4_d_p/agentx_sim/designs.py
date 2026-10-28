#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Step costs of Gemma4 prefill mesh designs from measured silicon step tables (SP4 x 2 vs SP8 study).

A design is one prefill server's mesh and sliding-attention path, e.g. sp8_lanes (the whole 8x4 Galaxy, one lanes
ring SDPA per sliding layer) or sp4_lanes (one CP4 x TP4 half; a Galaxy runs two of them). Its data are traced step
times of n 2048-token lanes at a few prefix depths (test_batched_step_perf / test_sp4x2_step_perf, trace replay
only), fitted per lane count as

    step(n lanes at prefixes P_1..P_n) = F[n] + slope[n] * sum(P_i)

which is how the calibrated cost model treats global attention (a + b * prefix per lane) once the dense and sliding
parts are folded into F. Host time per step and the scheduler's gap come from the batched calibration in
model_data_revc.json, the same for every design.
"""

import json
import os

DEFAULT_DESIGNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sp_designs_1028.json")


def fit(points, n):
    """(F ms, slope ms per prefix token) of F + slope * n * P through {prefix: ms}, least squares."""
    xs = [n * int(p) for p in points]
    ys = list(points.values())
    if len(xs) == 1:
        return ys[0] - 0.0, 0.0
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - slope * mx, slope


class DesignModel:
    """Model.step_s for one design: duck-types cost_model.Model for agentx_sim.run."""

    def __init__(self, name, steps, host_ms, gap_ms):
        self.name = name
        self.fits = {int(n): fit(points, int(n)) for n, points in steps.items()}
        self.max_lanes = max(self.fits)
        self.host_ms = {int(k): v for k, v in host_ms.items()}
        self.gap_s = gap_ms / 1e3

    @classmethod
    def load(cls, name, path=DEFAULT_DESIGNS):
        with open(path) as f:
            data = json.load(f)
        return cls(name, data["designs"][name]["steps"], data["host_ms"], data["gap_ms"])

    def device_ms(self, prefixes):
        f, slope = self.fits[len(prefixes)]
        return f + slope * sum(prefixes)

    def step_s(self, batch, C, budget, calibrated=True):
        n = len(batch)
        if n not in self.fits:
            raise ValueError(f"{self.name}: no measured step with {n} lanes (have {sorted(self.fits)})")
        ms = self.device_ms([p for _, p in batch])
        if calibrated:
            ms += self.host_ms[min(n, max(self.host_ms))]
        return ms / 1e3, self.gap_s if calibrated else 0.0

    def residuals(self, steps):
        """Measured minus fitted ms for every point of {lanes: {prefix: ms}}."""
        return {
            int(n): {int(p): round(ms - self.device_ms([int(p)] * int(n)), 2) for p, ms in points.items()}
            for n, points in steps.items()
        }


def designs(path=DEFAULT_DESIGNS):
    with open(path) as f:
        return json.load(f)["designs"]


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DESIGNS
    for name, d in designs(path).items():
        m = DesignModel.load(name, path)
        print(f"{name}: {d.get('note', '')}")
        for n, (f, s) in sorted(m.fits.items()):
            print(
                f"  {n} lanes: F {f:7.1f} ms, {s * 1e6:6.1f} ms per 1M prefix tokens per lane; residuals {m.residuals(d['steps'])[n]}"
            )
