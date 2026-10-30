# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Functional golden of the new format cells of perf_matmul_llkx: Float16 and Bfp8 (5 bit exponent), Bfp2_b operands and outputs, Int8 and UInt8 operands with a 32 bit integer DEST, run
with the body of test_matmul_grid2_tiny (math matmul test, golden on the operands as the unpackers write them) at one 1x2x2 block of 32x32 tiles per cell, fidelity drawn by the cell name.
Bfp8 (5 bit exponent) cells go through pack_bfp8, unpack_bfp8 and bfp8a_to_float16, the integer cells take small integer stimuli and the exact integer golden; Bfp2_b and Bfp8 outputs are checked against the golden quantised to the format."""

import pytest
import torch
from helpers.bfp_format_utils import bfp2b_to_float16b as _bfp2b_to_float16b, bfp8a_to_float16 as _bfp8a_to_float16
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.llk_params import MathFidelity
import perf_matmul_grid as _g
import perf_matmul_llkx as _x
import test_matmul_grid2_tiny as _t

_passed = _t.passed_test
_gg = _t.get_golden_generator


def _passed_bfp2(golden, res, fmt, *a, **k):
    """The golden is float; a Bfp2_b result is one mantissa bit under a shared exponent per 16 datums, so the golden is quantised the same way before the format's tolerance applies."""
    if fmt == DataFormat.Bfp2_b:
        golden = _bfp2b_to_float16b(golden.reshape(-1).to(torch.bfloat16))
    return _passed(golden, res, fmt, *a, **k)


def _passed_all(golden, res, fmt, *a, **k):
    if fmt == DataFormat.Bfp8:
        fmt = DataFormat.Bfp8_b  # the tolerance of the block float family; the golden is quantised to Bfp8 below
    return _passed_bfp2(golden, res, fmt, *a, **k)


_matmul_golden = _t.get_golden_generator(_t.MatmulGolden)


def _golden(op1, op2, out_fmt, fidelity, input_A_dimensions=None, input_B_dimensions=None, tilize=False, input_A_format=None, input_B_format=None, **k):
    """MatmulGolden with Bfp8 (5 bit exponent) operands quantised to what the unpacker writes and a Bfp8 output quantised as the packer writes it."""
    def operand(x, f, dims):
        # with tilize the operand arrives in tile order: quantise by 16 datum blocks (a face row) and untilize as the other block float formats do
        return (_bfp8a_to_float16(x, dims if tilize else None), DataFormat.Float16) if f == DataFormat.Bfp8 else (x, f)

    op1, fa = operand(op1, input_A_format, input_A_dimensions)
    op2, fb = operand(op2, input_B_format, input_B_dimensions)
    out = _matmul_golden(op1, op2, DataFormat.Float16 if out_fmt == DataFormat.Bfp8 else out_fmt, fidelity, input_A_dimensions=input_A_dimensions, input_B_dimensions=input_B_dimensions, tilize=tilize, input_A_format=fa, input_B_format=fb, **k)
    return _bfp8a_to_float16(out) if out_fmt == DataFormat.Bfp8 else out


_t.passed_test = _passed_all
_t.get_golden_generator = lambda cls: _golden if cls is _t.MatmulGolden else _gg(cls)
_g._F.update({k: v for k, v in _x._F.items() if k not in _g._F})


def _golden_cases():
    out = []
    for a, b, o, acc in _x._new_cells():
        fi = "HiFi4" if a in ("I8", "U8") else _x._FIDS[_x._hash("g" + a + b + o + acc) % 4]
        out.append((a, b, o, acc, "H", getattr(MathFidelity, fi), 32, 32, 1, 2, 2))
    return out


def _partial_int_cases():
    """Integer MVMUL multiplies 8 bit magnitudes in slices, one pair of slices per fidelity phase: below HiFi4 the product is partial (values 0 to 3 read back as 0 at LoFi to HiFi3)."""
    return [(x, x, "I32", "Y", "H", getattr(MathFidelity, fi), 32, 32, 1, 2, 2) for x in ("I8", "U8") for fi in ("LoFi", "HiFi2", "HiFi3")]


def _marks(c):
    """Bfp8 (5 bit exponent) output from a 32-bit DEST: the packer table has FP32 to FP32, FP16, FP16B and BFP8B only (knowledge/rtl/blackhole_tensix_core_architecture.md, PR:303-322), so the pair is undefined and the device writes zeros."""
    if c[2] == "B8a" and c[3] == "Y":
        return [pytest.mark.xfail(reason="packer: FP32 DEST to Bfp8 (5 bit exponent) is not in the packer conversion table; the device returns zeros", strict=True)]
    return []


@pytest.mark.nightly
@skip_for_wormhole
@pytest.mark.parametrize(
    "case",
    [pytest.param(c, marks=_marks(c), id="in1fmt_%s_in0fmt_%s_out_%s_acc_%s_%s_1x2x2" % (c[0], c[1], c[2], c[3], c[5].name)) for c in _golden_cases()],
)
def test_matmul_llkx(case):
    _t.test_matmul_grid2_tiny(case)


@pytest.mark.nightly
@skip_for_wormhole
@pytest.mark.xfail(reason="integer MVMUL accumulates one slice pair per fidelity phase: partial products below HiFi4", strict=True)
@pytest.mark.parametrize("case", [pytest.param(c, id="partial_in1fmt_%s_in0fmt_%s_out_%s_acc_%s_%s_1x2x2" % (c[0], c[1], c[2], c[3], c[5].name)) for c in _partial_int_cases()])
def test_matmul_llkx_int_partial(case):
    _t.test_matmul_grid2_tiny(case)
