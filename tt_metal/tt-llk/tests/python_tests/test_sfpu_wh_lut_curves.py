# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Temporary instrument: dump dense LUT error curves to JSON. NOT FOR MERGE.

One pytest run per (op, LUT segment) so each segment gets a full 250-point face of
its own -- StimuliSpec.custom only reaches the first face, so a single run cannot
hold a fine ladder over the whole domain. Writes {x, y, exact} to $LUT_DUMP.

    LUT_DUMP=/tmp/base.json python -m pytest test_sfpu_wh_lut_curves.py -q
"""

import json
import math
import os

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, FastMode, MathOperation
from helpers.stimuli_generator import StimuliSpec
from test_eltwise_unary_sfpu import eltwise_unary_sfpu

N = 250
TAIL = 8.0
_PHI = lambda a: 0.5 * (1.0 + math.erf(a / math.sqrt(2.0)))

# op key -> (MathOperation, approx, exact fn, LUT segment edges)
_OPS = {
    "tanh": (MathOperation.Tanh, ApproximationMode.Yes, math.tanh, [0.0, 1.0, 2.0, TAIL]),
    "sigmoid_appx": (MathOperation.SigmoidAppx, ApproximationMode.No,
                     lambda x: 1.0 / (1.0 + math.exp(-x)), [0.0, 1.0, 2.0, TAIL]),
    "gelu_appx": (MathOperation.GeluAppx, ApproximationMode.No,
                  lambda x: x * _PHI(x), [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, TAIL]),
}

_CASES = [(k, i) for k, v in _OPS.items() for i in range(len(v[3]) - 1)]
_OUT = {}


@pytest.fixture(scope="module", autouse=True)
def _dump():
    yield
    path = os.environ.get("LUT_DUMP")
    if not path:
        pytest.fail("set LUT_DUMP to the output json path")
    merged = {}
    for (op, _), rows in sorted(_OUT.items()):
        merged.setdefault(op, []).extend(rows)
    for op in merged:
        merged[op].sort(key=lambda r: r[0])
    with open(path, "w") as f:
        json.dump({"segments": {k: v[3] for k, v in _OPS.items()}, "data": merged}, f)
    print(f"\nwrote {path}: " + ", ".join(f"{k}={len(v)} pts" for k, v in merged.items()))


@pytest.mark.parametrize("op,seg", _CASES, ids=[f"{o}-seg{s}" for o, s in _CASES])
def test_curve(monkeypatch, op, seg):
    mathop, approx, exact, edges = _OPS[op]
    lo, hi = edges[seg], edges[seg + 1]
    # Stay strictly inside the segment: the last sample sits just below the next knot
    # so a point never lands in the neighbour because of fp32 rounding.
    probes = [lo + (hi - lo) * i / N for i in range(N)]
    if seg == 0:
        probes[0] = 0.0

    import helpers.utils as utils

    captured = {}
    monkeypatch.setattr(utils, "passed_test", lambda *a, **k: True)

    def capture(golden, res, *a, **kw):
        captured["res"] = res.float().flatten()[: len(probes)].clone()
        return True

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
    _OUT[(op, seg)] = [[x, y, exact(x)] for x, y in zip(probes, captured["res"].tolist())]
