# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Temporary instrument: measured accuracy of the WH LUT kernels. NOT FOR MERGE.

Feeds a dense ladder over each op's useful domain on Float32->Float32 dest_acc=Yes and
reports max/mean |error| against the exact function, per LUT segment. Run it before and
after a coefficient change to see what the retune actually bought on silicon.
"""

import math

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, FastMode, MathOperation
from helpers.stimuli_generator import StimuliSpec
from test_eltwise_unary_sfpu import eltwise_unary_sfpu

N = 250


def _ladder(hi):
    # 3/4 of the points inside [0, 2] where the knees are, 1/4 out to `hi`
    a = [2.0 * i / (3 * N // 4 - 1) for i in range(3 * N // 4)]
    b = [2.0 + (hi - 2.0) * (i + 1) / (N - 3 * N // 4) for i in range(N - 3 * N // 4)]
    return a + b


_PHI = lambda a: 0.5 * (1.0 + math.erf(a / math.sqrt(2.0)))

_CASES = [
    ("tanh (3-entry SFPLUT)", MathOperation.Tanh, ApproximationMode.Yes,
     _ladder(8.0), math.tanh, [1.0, 2.0]),
    ("sigmoid_appx (3-entry SFPLUT)", MathOperation.SigmoidAppx, ApproximationMode.No,
     _ladder(8.0), lambda x: 1.0 / (1.0 + math.exp(-x)), [1.0, 2.0]),
    ("gelu_appx (6-entry SFPLUTFP32)", MathOperation.GeluAppx, ApproximationMode.No,
     _ladder(8.0), lambda x: x * _PHI(x), [0.5, 1.0, 1.5, 2.0, 3.0]),
]

_RESULTS = {}


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    for label, rows, knees in _RESULTS.values():
        print(f"\n\n=== {label} ===")
        edges = [0.0] + knees + [float("inf")]
        overall = 0.0
        for lo, hi in zip(edges, edges[1:]):
            seg = [(x, y, t) for x, y, t in rows if lo <= x < hi]
            if not seg:
                continue
            errs = [abs(y - t) for _, y, t in seg]
            mx = max(errs)
            overall = max(overall, mx)
            worst = max(seg, key=lambda r: abs(r[1] - r[2]))
            print(f"  |x| in [{lo:g}, {hi:g}): n={len(seg):3d}  max|err|={mx:.6f} "
                  f"at x={worst[0]:.4f} (got {worst[1]:.6f}, want {worst[2]:.6f})  "
                  f"mean|err|={sum(errs)/len(errs):.6f}")
        print(f"  OVERALL max|err| = {overall:.6f}")


@pytest.mark.parametrize("case", list(range(len(_CASES))))
def test_lut_accuracy(monkeypatch, case):
    label, mathop, approx, probes, exact, knees = _CASES[case]

    import helpers.utils as utils

    captured = {}
    real = utils.passed_test

    def capture(golden, res, *a, **kw):
        captured["res"] = res.float().flatten()[: len(probes)].clone()
        return True

    monkeypatch.setattr(utils, "passed_test", capture)
    monkeypatch.setattr("test_eltwise_unary_sfpu.passed_test", capture)

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        DestAccumulation.Yes,
        approx,
        mathop,
        FastMode.No,
        [64, 64],
        spec_A=StimuliSpec.custom(values=probes, seed=0),
    )
    ys = captured["res"].tolist()
    _RESULTS[case] = (label, [(x, y, exact(x)) for x, y in zip(probes, ys)], knees)
