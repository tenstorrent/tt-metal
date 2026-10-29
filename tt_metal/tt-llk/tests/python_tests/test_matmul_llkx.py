# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Functional golden of the new format cells of perf_matmul_llkx: Float16 and Bfp8 (5 bit exponent), Bfp2_b operands and outputs, Int8 and UInt8 operands with a 32 bit integer DEST, run
with the body of test_matmul_grid2_tiny (math matmul test, golden on the operands as the unpackers write them) at one 1x2x2 block of 32x32 tiles per cell, fidelity drawn by the cell name.
Bfp8 (5 bit exponent) has no stimuli packer in the harness and stays an expected failure until it gets one."""

import pytest
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.llk_params import MathFidelity
import perf_matmul_grid as _g
import perf_matmul_llkx as _x
import test_matmul_grid2_tiny as _t

_g._F.update({k: v for k, v in _x._F.items() if k not in _g._F})


def _golden_cases():
    out = []
    for a, b, o, acc in _x._new_cells():
        fi = _x._FIDS[_x._hash("g" + a + b + o + acc) % 4]
        out.append((a, b, o, acc, "H", getattr(MathFidelity, fi), 32, 32, 1, 2, 2))
    return out


def _marks(c):
    return [pytest.mark.xfail(reason="no Bfp8 (5 bit exponent) stimuli packer in the harness", strict=False)] if "B8a" in c[:3] else []


@pytest.mark.nightly
@skip_for_wormhole
@pytest.mark.parametrize(
    "case",
    [pytest.param(c, marks=_marks(c), id="in1fmt_%s_in0fmt_%s_out_%s_acc_%s_%s_1x2x2" % (c[0], c[1], c[2], c[3], c[5].name)) for c in _golden_cases()],
)
def test_matmul_llkx(case):
    _t.test_matmul_grid2_tiny(case)
