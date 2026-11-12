# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Functional golden of the Bfp4 and Bfp2 (5 bit exponent) outputs of perf_matmul_llkx_a4: Float16 and Bfp8 (5 bit exponent) operands at half and full sync,
blocks of 2 and 4 tiles of one and three K tiles, on the body of test_matmul_llkx (test_matmul_grid2_tiny at 32x32 tiles), so the block pack forms of these
outputs are golden-checked. The golden is computed in Float16 and quantised as the packer writes the format.
"""

import perf_matmul_llkx_a4 as _a4
import pytest
import test_matmul_grid2_tiny as _t
import test_matmul_llkx as _l
from conftest import skip_for_wormhole
from helpers.bfp_format_utils import bfp2a_to_float16, bfp4a_to_float16
from helpers.format_config import DataFormat
from helpers.llk_params import MathFidelity

_l._g._F.update({k: v for k, v in _a4._F.items() if k not in _l._g._F})
_QUANT = {DataFormat.Bfp4: bfp4a_to_float16, DataFormat.Bfp2: bfp2a_to_float16}
_prev = _t.get_golden_generator


def _golden(op1, op2, out_fmt, fidelity, **k):
    if out_fmt not in _QUANT:
        return _prev(_t.MatmulGolden)(op1, op2, out_fmt, fidelity, **k)
    return _QUANT[out_fmt](_prev(_t.MatmulGolden)(op1, op2, DataFormat.Float16, fidelity, **k))


_t.get_golden_generator = lambda cls: _golden if cls is _t.MatmulGolden else _prev(cls)
_passed_prev = _t.passed_test


def _passed(golden, res, fmt, *a, **k):
    """Sums of three K tiles of [0, 1) stimuli sit in one binade a few Bfp4 steps wide, where two valid quantisations correlate at about 0.94:
    the block lattice compare (one step per 16 datums) carries the check, as the PCC floor of Bfp2 does."""
    if fmt == DataFormat.Bfp4:
        k.setdefault("custom_pcc_threshold", 0.9)
    return _passed_prev(golden, res, fmt, *a, **k)


_t.passed_test = _passed


def _cases():
    out = []
    for a in ("F16a", "B8a"):
        for b in ("F16a", "B8a"):
            for o in ("B4a", "B2a"):
                for acc in "NY":
                    for sy in "HF":
                        fi = _a4._FIDS[_a4._hash("g4" + a + b + o + acc + sy) % 4]
                        for rt, ct, kt in ((1, 2, 1), (2, 2, 3)):
                            out.append((a, b, o, acc, sy, getattr(MathFidelity, fi), 32, 32, rt, ct, kt))
    return out


def _marks(c):
    """A 32-bit DEST: the packer table converts FP32 to FP32, FP16, FP16B and BFP8B only (knowledge/rtl/blackhole_tensix_core_architecture.md, PR:303-322), as for Bfp8 in test_matmul_llkx."""
    if c[3] == "Y":
        return [
            pytest.mark.xfail(
                reason="packer: FP32 DEST to Bfp4 or Bfp2 (5 bit exponent) is not in the packer conversion table",
                strict=True,
            )
        ]
    return []


@pytest.mark.nightly
@skip_for_wormhole
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            c,
            marks=_marks(c),
            id="in1fmt_%s_in0fmt_%s_out_%s_acc_%s_sync_%s_%s_%dx%dx%d" % (c[0], c[1], c[2], c[3], c[4], c[5].name, c[8], c[9], c[10]),
        )
        for c in _cases()
    ],
)
def test_matmul_llkx_a4(case):
    _t.test_matmul_grid2_tiny(case)
