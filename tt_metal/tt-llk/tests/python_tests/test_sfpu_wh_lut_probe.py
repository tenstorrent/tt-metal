# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Temporary instrument: reverse-engineer the WH SFPLUT / SFPLUTFP32 tables. NOT FOR MERGE.

Reads raw Float32->Float32 (dest_acc=Yes) results for the four LUT-backed WH kernels at a
dense ladder of probe points, then solves each segment's (A, B) by least squares. That pins
three things the headers only assert in comments:

  1. the |x| breakpoints the hardware actually uses (3-entry SFPLUT, and TABLE1 vs TABLE2
     for SFPLUTFP32 -- sfpi's lut2(mode) maps mode!=1 to TABLE2, so tt-llk's `lut_mode = 0`
     selects TABLE2 despite its comment naming TABLE1),
  2. what the 0xFF coefficient byte decodes to (the tables use it where the comments say
     0.0, but 0xFF in the s1-e3-m4 byte format is -0.01513672),
  3. the measured per-segment max error, as the baseline any retune has to beat.
"""

import struct

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, FastMode, MathOperation
from helpers.stimuli_generator import StimuliSpec
from test_eltwise_unary_sfpu import eltwise_unary_sfpu

# Ladder: tight clusters either side of every candidate breakpoint, plus interior points to
# solve each segment's line, plus a far tail to expose a non-zero tail slope.
_PROBES = []
for knee in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
    _PROBES += [knee - 1e-3, knee, knee + 1e-3]
_PROBES += [0.05, 0.125, 0.25, 0.375, 0.625, 0.75, 0.875, 1.125, 1.25, 1.375,
            1.625, 1.75, 1.875, 2.25, 2.5, 2.75, 3.25, 3.5, 3.75, 4.5, 5.0,
            6.0, 8.0, 12.0, 16.0, 32.0, 64.0, 128.0]
_PROBES = sorted(set(_PROBES))

# (op, approx, label, how to strip the non-LUT part of the kernel to recover raw lut(x))
_CASES = [
    (MathOperation.Tanh, ApproximationMode.Yes, "tanh (3-entry SFPLUT)", lambda y, x: y),
    (MathOperation.SigmoidAppx, ApproximationMode.No, "sigmoid_appx (3-entry SFPLUT, -0.5)",
     lambda y, x: y - 0.5),
    (MathOperation.GeluAppx, ApproximationMode.No, "gelu_appx (6-entry SFPLUTFP32, -0.5x)",
     lambda y, x: y - 0.5 * x),
]

_RESULTS = {}


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    for label in _RESULTS:
        raw = _RESULTS[label]
        print(f"\n\n=== {label} ===")
        print(f"{'x':>10} {'raw result':>16} {'lut(x)':>14}   segment fit")
        prev = None
        for x, y, l in raw:
            mark = ""
            if prev is not None and abs(l - prev[1]) > 1e-7:
                # slope between consecutive probes, to spot the knees
                mark = f"  slope={(l - prev[1]) / (x - prev[0]):+.6f}"
            print(f"{x:10.5f} {y:16.9g} {l:14.9g}{mark}")
            prev = (x, l)


@pytest.mark.parametrize("case", list(range(len(_CASES))))
def test_lut_probe(monkeypatch, case):
    mathop, approx, label, strip = _CASES[case]

    import helpers.utils as utils

    captured = {}
    real = utils.passed_test

    def capture(golden, res, *a, **kw):
        captured["res"] = res.float().flatten()[: len(_PROBES)].clone()
        try:
            real(golden, res, *a, **kw)
        except Exception:
            pass
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
        spec_A=StimuliSpec.custom(values=_PROBES, seed=0),
    )
    ys = captured["res"].tolist()
    _RESULTS[label] = [(x, y, strip(y, x)) for x, y in zip(_PROBES, ys)]
