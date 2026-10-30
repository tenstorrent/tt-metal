# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Functional golden of the new format cells of perf_matmul_llkx: Float16 and Bfp8 (5 bit exponent), Bfp2_b operands and outputs, Int8 and UInt8 operands with a 32 bit integer DEST, run
with the body of test_matmul_grid2_tiny (math matmul test, golden on the operands as the unpackers write them) at one 1x2x2 block of 32x32 tiles per cell, fidelity drawn by the cell name.
Bfp8 (5 bit exponent) has no stimuli packer and the integer cells no stimuli generator in the harness: expected failures until they get one. Bfp2_b outputs are checked against the format's quantisation step."""

import pytest
import torch
from helpers.bfp_format_utils import bfp2b_to_float16b as _bfp2b_to_float16b
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.llk_params import MathFidelity
import perf_matmul_grid as _g
import perf_matmul_llkx as _x
import test_matmul_grid2_tiny as _t

_passed = _t.passed_test


def _passed_bfp2(golden, res, fmt, *a, **k):
    """The golden is float; a Bfp2_b result is one mantissa bit under a shared exponent per 16 datums, so the golden is quantised the same way before the format's tolerance applies."""
    if fmt == DataFormat.Bfp2_b:
        golden = _bfp2b_to_float16b(golden.reshape(-1).to(torch.bfloat16))
    return _passed(golden, res, fmt, *a, **k)


_t.passed_test = _passed_bfp2
_g._F.update({k: v for k, v in _x._F.items() if k not in _g._F})


def _golden_cases():
    out = []
    for a, b, o, acc in _x._new_cells():
        fi = _x._FIDS[_x._hash("g" + a + b + o + acc) % 4]
        out.append((a, b, o, acc, "H", getattr(MathFidelity, fi), 32, 32, 1, 2, 2))
    return out


def _marks(c):
    if "B8a" in c[:3]:
        return [pytest.mark.xfail(reason="no Bfp8 (5 bit exponent) stimuli packer in the harness", strict=False)]
    if c[0] in ("I8", "U8"):
        return [pytest.mark.xfail(reason="generate_face_matmul_data has no integer stimuli (torch.rand on int8)", strict=False)]
    return []


@pytest.mark.nightly
@skip_for_wormhole
@pytest.mark.parametrize(
    "case",
    [pytest.param(c, marks=_marks(c), id="in1fmt_%s_in0fmt_%s_out_%s_acc_%s_%s_1x2x2" % (c[0], c[1], c[2], c[3], c[5].name)) for c in _golden_cases()],
)
def test_matmul_llkx(case):
    _t.test_matmul_grid2_tiny(case)
