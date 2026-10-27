# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GELU derivative on the bit patterns a finite-value sweep never presents.

`calculate_gelu_derivative_simple` picks its region with a chain of ordered
float compares (x >= 3.1719 -> 1.0, x >= -3 -> polynomial, x > -13.375 -> tail,
else 0). SFPU compares order NaNs by sign and magnitude, so +NaN sorts above +inf
and -NaN below -inf: without an explicit case +NaN comes out as 1.0 and -NaN as 0
instead of propagating (tenstorrent/tt-llk#1701 item 12).

Wormhole and Blackhole propagate NaN with an fp32 destination. The
bfloat16-destination arm keeps 1.0 / 0 on both: it ends in convert<vFloat16b>
(SFPSTOCHRND), which turns any NaN into an infinity, so it could not return NaN
anyway.

Inputs are written as raw bit patterns through an integer view, so both NaN signs
reach L1 as asked -- torch's float -> bfloat16 conversion would collapse every NaN
onto one positive pattern. The two extreme payloads (0x7F800001 / 0xFFFFFFFF) also
survive for fp32; the bfloat16 rows (0x7F81 / 0xFFFF) are packed through float32
and ml_dtypes, which keep the sign but canonicalise the payload, so there they
repeat the plain +-NaN rows.

The probes and controls repeat across the whole tile, so every SFPU iteration --
every copy of the unrolled loop -- sees all of them, and every lane is checked.

Reference: GELU'(+inf) = 1, GELU'(-inf) = 0, GELU'(+-0) = 0.5, GELU'(NaN) = NaN.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import skip_for_quasar
from helpers.data_format_inference import unpacks_to_dest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    MathOperation,
    format_dict,
)
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.utils import classify_special
from test_eltwise_unary_sfpu import eltwise_unary_sfpu

# key -> (fp32 bits, bfloat16 bits)
SPECIALS = {
    "+inf": (0x7F800000, 0x7F80),
    "-inf": (0xFF800000, 0xFF80),
    "+NaN": (0x7FC00000, 0x7FC0),
    "-NaN": (0xFFC00000, 0xFFC0),
    "+NaN(min payload)": (0x7F800001, 0x7F81),
    "-NaN(max payload)": (0xFFFFFFFF, 0xFFFF),
    "+0.0": (0x00000000, 0x0000),
    "-0.0": (0x80000000, 0x8000),
}

# Region-boundary controls -> exact classify_special() label, or None: match the golden
# within CONTROL_ATOL, far below the |x| >= 3 a misfiring NaN case (returns x) gives.
CONTROLS = {
    3.1719: None,
    3.0: None,
    5.0: "1",
    1.0e30: "1",
    -3.0: None,
    -3.25: None,
    -13.0: None,
    -13.375: None,
    -20.0: "+0.0",
    -1.0e30: "+0.0",
}
CONTROL_ATOL = 0.02

PERIOD = len(SPECIALS) + len(CONTROLS)


def _run(in_fmt, out_fmt, dest_acc, approx):
    formats = InputOutputFormat(in_fmt, out_fmt)
    dims = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1]]
    src_A, tc_A, src_B, tc_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims,
        spec_A=StimuliSpec.constant(0.0),
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims,
    )
    flat = src_A.view(-1)
    lane = torch.arange(flat.numel()) % PERIOD

    is_control = lane >= len(SPECIALS)
    controls = torch.tensor(list(CONTROLS), dtype=torch.float32).to(flat.dtype)
    flat[is_control] = controls[lane[is_control] - len(SPECIALS)]

    is_special = ~is_control
    if in_fmt == DataFormat.Float32:
        bits = flat.view(torch.int32)
        patterns = np.array([fp32 for fp32, _ in SPECIALS.values()], dtype=np.uint32)
        patterns = torch.from_numpy(patterns.view(np.int32))
    else:
        bits = flat.view(torch.int16)
        patterns = np.array([bf16 for _, bf16 in SPECIALS.values()], dtype=np.uint16)
        patterns = torch.from_numpy(patterns.view(np.int16))
    bits[is_special] = patterns[lane[is_special]]

    res = eltwise_unary_sfpu(
        formats=formats,
        dest_acc=dest_acc,
        approx_mode=approx,
        mathop=MathOperation.GeluDerivative,
        fast_mode=FastMode.No,
        input_dimensions=dims,
        src_A=src_A,
        src_B=src_B,
        tile_cnt_A=tc_A,
        tile_cnt_B=tc_B,
        unpack_to_dest=unpacks_to_dest(in_fmt, dest_acc),
        clamp_negative=False,
        check_golden=False,
    )
    hw = torch.tensor(res, dtype=format_dict[formats.output_format])
    return (
        flat.to(torch.float32).numpy(),
        hw.to(torch.float32).numpy(),
        lane.numpy(),
    )


NON_FINITE = frozenset({"NaN", "+inf", "-inf"})


def _expected_specials(in_fmt, out_fmt, dest_acc):
    """key -> the set of acceptable classify_special() labels."""
    # Only the fp32-Dest arm has the NaN case; a Float32 input unpacks to it intact.
    if dest_acc == DestAccumulation.Yes and in_fmt == DataFormat.Float32:
        if out_fmt == DataFormat.Float32:
            nan_pos = nan_neg = {"NaN"}
        else:
            # The pack turns NaN into an infinity; the bug returned a finite 1 / 0.
            nan_pos = nan_neg = NON_FINITE
    else:
        nan_pos, nan_neg = {"1"}, {"+0.0"}
    return {
        "+inf": {"1"},
        "-inf": {"+0.0"},
        "+NaN": nan_pos,
        "-NaN": nan_neg,
        "+NaN(min payload)": nan_pos,
        "-NaN(max payload)": nan_neg,
        "+0.0": {"0.5"},
        "-0.0": {"0.5"},
    }


# Quasar's unary SFPU dispatch has no gelu_derivative arm (static_assert at compile time).
@skip_for_quasar
@pytest.mark.parametrize(
    "approx", [ApproximationMode.No, ApproximationMode.Yes], ids=["exact", "approx"]
)
@pytest.mark.parametrize(
    "in_fmt,out_fmt,dest_acc",
    [
        (DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes),
        # fp32 Dest packed to bfloat16: the NaN case runs, then the pack has to carry it.
        (DataFormat.Float32, DataFormat.Float16_b, DestAccumulation.Yes),
    ],
    ids=["bf16", "fp32", "fp32-to-bf16"],
)
def test_gelu_derivative_specials(in_fmt, out_fmt, dest_acc, approx):
    x, hw, lane = _run(in_fmt, out_fmt, dest_acc, approx)
    keys = list(SPECIALS)
    controls = list(CONTROLS.items())
    expected = _expected_specials(in_fmt, out_fmt, dest_acc)
    golden = get_golden_generator(UnarySFPUGolden)._gelu_derivative

    wrong = {}
    for xv, hv, li in zip(x, hw, lane):
        if li < len(SPECIALS):
            key = keys[li]
            got = classify_special(hv)
            ok = got in expected[key]
            want = sorted(expected[key])
        else:
            control, label = controls[li - len(SPECIALS)]
            key = "control %g" % control
            got = classify_special(hv)
            if label is not None:
                want = label
                ok = got == want
            else:
                ref = golden(float(xv))
                want = "%.9g +- %g" % (ref, CONTROL_ATOL)
                ok = bool(np.isfinite(hv)) and abs(float(hv) - ref) <= CONTROL_ATOL
        if not ok:
            entry = wrong.setdefault(key, {"want": want, "got": set(), "lanes": 0})
            entry["got"].add(got)
            entry["lanes"] += 1

    assert not wrong, "GELU' on special inputs and controls, by probe: %s" % wrong
