# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""What an exhaustive bfloat16 sweep of approximate tanh cannot reach.

Two disjoint blind spots, one test file.

**The 257 bit patterns.** `StimuliSpec.ulp_sweep` enumerates the 65,279 distinct
*finite* values and dedupes -0.0 against +0.0, so 2^16 - 65,279 = 257 bit patterns
never reach the kernel: 2 infinities, 254 NaNs, and negative zero.

**The gaps between bfloat16 values.** DEST holds fp32, so the kernel sees inputs a
bfloat16 sweep never presents. Segment 4 of the LUT once crossed 1.0 at |x| = 2.99532
and peaked at 1.000183 -- and the two bfloat16 neighbours either side of that window,
2.984375 and 3.0, both evaluate to <= 1.0, so an exhaustive bfloat16 sweep reported
exact saturation while fp32 inputs returned |tanh| > 1. `test_tanh_fp32_bound` walks
that window directly.

That matters for this kernel specifically. Segment 5 of the SFPLUT is the
constant pair (A=0, B=1), evaluated as `A*|x| + B` -- so an infinite input makes
it compute `0 * inf + 1`, which is NaN in IEEE arithmetic. Whether the hardware
actually does that is not something to reason about; a pre-scaled variant of this
same table was measured returning NaN for 120 large *finite* inputs for exactly
this reason.

Reference: tanh(+-inf) = +-1, tanh(NaN) = NaN, tanh(+-0.0) = +-0.0 (sign kept).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    FastMode,
    MathOperation,
    format_dict,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    generate_input_dim,
)

# exp field all ones -> 2 infinities + 254 NaNs, both signs
POS_SPECIALS = list(range(0x7F80, 0x8000))  # 128 bfloat16 bit patterns
NEG_SPECIALS = list(range(0xFF80, 0x10000))  # 128
ZEROS = [0x0000, 0x8000]  # +0.0, -0.0

FACE_SIZE = 256


def _write_bits(src, bits_by_face):
    """Write raw bfloat16 bit patterns into `src`, one face at a time.

    Not `StimuliSpec.custom(values=...)`, which would be the natural way to say
    this: that path ends in `torch.tensor(vals, dtype=torch.bfloat16)`, and torch's
    float -> bfloat16 conversion collapses every NaN onto one *positive* pattern.
    All 127 negative NaNs would arrive as +NaN and the two halves of this test
    would be the same half twice. Writing the bit patterns through a uint16 view
    converts nothing, so what reaches L1 is what is asked for. (The same hazard is
    in the shared strategy for any test that feeds it NaNs; fixing it there changes
    what several other suites are fed, so it is left alone here.)
    """
    u16 = src.view(-1).view(torch.uint16)
    for face, bits in bits_by_face.items():
        base = face * FACE_SIZE
        assert len(bits) <= FACE_SIZE, "face %d overflows" % face
        u16[base : base + len(bits)] = torch.tensor(bits, dtype=torch.uint16)


def _run(values_by_face, out_fmt, dest_acc, in_fmt=DataFormat.Float16_b, floats=False):
    """Run approximate tanh over `values_by_face`.

    `floats=True` takes ordinary float values per face instead of raw bit patterns;
    the bit-pattern path exists only for the NaN-sign hazard `_write_bits` documents,
    which ordinary finite values do not have.
    """
    formats = InputOutputFormat(in_fmt, out_fmt)
    dims = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1]]
    spec = (
        StimuliSpec.custom_faces(values_by_face)
        if floats
        else StimuliSpec.constant(0.0)
    )
    src_A, tc_A, src_B, tc_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims,
        spec_A=spec,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims,
    )
    if not floats:
        _write_bits(src_A, values_by_face)
    nb, ntb = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        dims,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )
    cfg = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(dims, dims),
            APPROX_MODE(ApproximationMode.Yes),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(False),
            MATH_OP(mathop=MathOperation.Tanh),
        ],
        runtimes=[TILE_COUNT(tc_A), NUM_BLOCKS(nb), NUM_TILES_IN_BLOCK(ntb)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tc_A,
            tile_count_B=tc_B,
            tile_count_res=tc_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=in_fmt.is_32_bit() and dest_acc == DestAccumulation.Yes,
    )
    hw = torch.tensor(cfg.run().result, dtype=format_dict[formats.output_format])
    return src_A.to(torch.float32).numpy(), hw.to(torch.float32).numpy()


@pytest.mark.parametrize(
    "out_fmt,dest_acc",
    [
        (DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float32, DestAccumulation.Yes),
    ],
    ids=["bf16-out", "fp32-out"],
)
def test_tanh_specials(out_fmt, dest_acc):
    x, hw = _run({0: POS_SPECIALS, 1: NEG_SPECIALS, 2: ZEROS}, out_fmt, dest_acc)

    def classify(v):
        if np.isnan(v):
            return "NaN"
        if np.isinf(v):
            return "+inf" if v > 0 else "-inf"
        if v == 0.0:
            return "+0.0" if np.signbit(v) == 0 else "-0.0"
        return "%.9g" % v

    groups = {}
    for xv, hv in zip(x, hw):
        if np.isnan(xv):
            key = "NaN" + ("(-)" if np.signbit(xv) else "(+)")
        elif np.isinf(xv):
            key = "+inf" if xv > 0 else "-inf"
        elif xv == 0.0:
            key = "-0.0" if np.signbit(xv) else "+0.0"
        else:
            continue
        groups.setdefault(key, {}).setdefault(classify(hv), 0)
        groups[key][classify(hv)] += 1

    # What the hardware actually does today, measured on a Wormhole n150 and
    # byte-for-byte identical on main's 3-entry table, so none of it is a property of
    # any particular coefficient table.
    #
    #   +-inf -> NaN on the fp32 path: segment 5 is the constant pair (A=0, B=1) and
    #            the hardware evaluates A*|x| + B, so an infinite input computes
    #            0 * inf + 1. IEEE tanh(+-inf) is +-1. This one *is* the LUT.
    #   NaN   -> the sign survives: +NaN and -NaN give distinct results on the bf16
    #            path (+inf and -inf), so the input path carries the sign bit.
    #   -0.0  -> +0.0, and this is NOT the LUT. SGN_RETAIN ends in copysign, and
    #            segment 0 computes A*0 + 0, so the LUT would return -0.0. The sign
    #            is already gone before the SFPU runs: the same stimuli through
    #            sources/eltwise_unary_datacopy_test.cpp -- unpack to DEST and pack
    #            back, no SFPU at all -- return +0.0 on both output formats, while
    #            helpers.pack writes 0x8000 to L1 and helpers.unpack reads -0.0 back,
    #            so neither host leg is responsible. A second, separate loss sits on
    #            the bf16 pack: SFPU negate turns +0.0 into -0.0 and that -0.0 reaches
    #            L1 intact with fp32 out but packs to +0.0 with bf16 out. Fixing
    #            either belongs to the unpack/pack path, not to this kernel.
    #   On the bf16 path a NaN packs out as an infinity of the same sign -- a NaN only
    #   survives to L1 on an fp32-end-to-end pipeline, so that arm says nothing about
    #   the kernel.
    #
    # Asserting the divergences on purpose: they predate this table, fixing them costs
    # instructions on every datum, and until that trade is made deliberately this test
    # is here to catch anyone changing them by accident.
    RECORDED = {
        DataFormat.Float32: {
            "+inf": "NaN",
            "-inf": "NaN",
            "NaN(+)": "NaN",
            "NaN(-)": "NaN",
            "+0.0": "+0.0",
            "-0.0": "+0.0",
        },
        DataFormat.Float16_b: {
            "+inf": "+inf",
            "-inf": "-inf",
            "NaN(+)": "+inf",
            "NaN(-)": "-inf",
            "+0.0": "+0.0",
            "-0.0": "+0.0",
        },
    }[out_fmt]

    # Each of these must both arrive in `groups` and carry a RECORDED entry. A key
    # that never arrives is a hole in the stimuli rather than a pass -- -0.0 would
    # disappear from `groups` entirely if its sign were lost before the kernel -- and
    # a key with nothing recorded against it can never fail, which is how the 127
    # negative-NaN patterns went unasserted.
    KEYS = ("+inf", "-inf", "NaN(+)", "NaN(-)", "+0.0", "-0.0")

    def shown(key):
        return ", ".join("%s x%d" % (k, v) for k, v in sorted(groups[key].items()))

    missing = [k for k in KEYS if k not in groups]
    unrecorded = [
        "%s (hardware returned %s)" % (k, shown(k))
        for k in KEYS
        if k in groups and k not in RECORDED
    ]
    failures = [
        "%s: recorded %s, got %s" % (k, RECORDED[k], shown(k))
        for k in KEYS
        if k in groups and k in RECORDED and list(groups[k]) != [RECORDED[k]]
    ]

    assert not missing, (
        "these input patterns never reached the kernel: "
        + ", ".join(missing)
        + "\nThe stimuli path lost them before the math did, so nothing below is "
        "asserting anything about them."
    )
    assert not unrecorded, (
        "no recorded behaviour for: "
        + ", ".join(unrecorded)
        + "\nA key with no RECORDED entry cannot fail. Add the measured value."
    )
    assert not failures, (
        "behaviour on the non-finite / signed-zero patterns changed:\n  "
        + "\n  ".join(failures)
        + "\nThese were identical on main. If the change is deliberate, update RECORDED."
    )


def test_tanh_fp32_bound():
    """|tanh| <= 1 for fp32 inputs between the last two bfloat16 values below 3.

    The LUT has no `min(result, 1.0f)` -- the polynomial path in the same header
    carries one -- so the bound is a property of the coefficients alone: every
    segment must stay at or below 1.0 across its whole range, not merely at the
    breakpoints a bfloat16 input can land on. With segment 4's slope at the
    unrounded minimax 0.039123535 this returned up to 1.000183 here while an
    exhaustive bfloat16 sweep still reported exact saturation.
    """
    # Up to the last fp32 value below 3.0, not 2.99999 -- roughly 40 representable
    # values sit above that, and they are the tightest part of the range.
    top = np.nextafter(np.float32(3.0), np.float32(0.0))
    xs = list(np.linspace(np.float32(2.9953), top, 256).astype(np.float32))
    _, hw = _run(
        {0: xs},
        DataFormat.Float32,
        DestAccumulation.Yes,
        in_fmt=DataFormat.Float32,
        floats=True,
    )
    got = np.abs(hw[: len(xs)])
    over = got > 1.0
    worst = got.max()

    # Pin that the probes actually arrived. |y| <= 1 is satisfied by anything small,
    # so a stimuli path that misplaced these or zeroed them would return tanh(0) = 0
    # and pass -- deterministically, so the bit-exact re-run would not catch it either.
    # Segment 4 gives 0.0390625 * 2.9953 + 0.8828125 = 0.99982 at the low end.
    assert worst > 0.9998, (
        "the fp32 probes did not reach the kernel: worst |y| = %.9f over %d inputs in "
        "(2.99532, 3.0), where every one of them should evaluate to >= 0.99982. "
        "Nothing below is asserting anything about segment 4." % (worst, len(xs))
    )
    assert not over.any(), (
        "approximate tanh returned |y| > 1 on %d of %d fp32 inputs in "
        "(2.99532, 3.0); worst |y| = %.9f. No LUT segment may exceed 1.0 inside "
        "its own range." % (over.sum(), len(xs), worst)
    )


def test_tanh_fp32_monotone_across_breakpoints():
    """No downward step where two LUT segments meet, on the fp32 path.

    The same blind spot as `test_tanh_fp32_bound`, one breakpoint over. fp16
    coefficients cannot make the segments exactly continuous for free, and a step
    of ~1.2e-4 is invisible to a bfloat16 sweep -- it is far below the ~2e-3 ulp
    of a bfloat16 near these values, so the sweep still reports a monotone result.
    `calculate_tanh<APPROXIMATION_MODE=true>` has no `convert<vFloat16b>` though,
    and also serves the fp32-dest path, where that same step is ~2048 fp32 ulp of
    non-monotonicity. The table holds the joins at |x| = 0.5, 1.0 and 2.0 exactly
    and steps *up* at |x| = 1.5; this walks fp32 values either side of each.
    """
    xs = []
    for bp in (0.5, 1.0, 1.5, 2.0, 3.0):
        b = np.float32(bp)
        below = b
        for _ in range(16):
            below = np.nextafter(below, np.float32(0.0))
        for _ in range(32):
            xs.append(below)
            below = np.nextafter(below, np.float32(8.0))
    xs = list(np.array(xs, dtype=np.float32))

    _, hw = _run(
        {0: xs},
        DataFormat.Float32,
        DestAccumulation.Yes,
        in_fmt=DataFormat.Float32,
        floats=True,
    )
    got = hw[: len(xs)].astype(np.float64)

    # The probes straddle 0.5 upward, so nothing here should be near zero.
    assert got.min() > 0.4, (
        "the fp32 probes did not reach the kernel: min y = %.9f over %d inputs "
        "straddling the breakpoints, where the smallest should be tanh(~0.5) ~ 0.46."
        % (got.min(), len(xs))
    )

    # Each breakpoint contributes one contiguous run of 32 ascending inputs.
    for start, bp in zip(range(0, len(xs), 32), (0.5, 1.0, 1.5, 2.0, 3.0)):
        block = got[start : start + 32]
        steps = np.diff(block)
        worst = steps.min()
        assert worst >= 0.0, (
            "approximate tanh steps down by %.6g across |x| = %s: y goes %.9f -> %.9f. "
            "No LUT segment may sit below its predecessor where they meet -- a bfloat16 "
            "sweep cannot see this, but the fp32-dest path returns it."
            % (
                -worst,
                bp,
                block[int(np.argmin(steps))],
                block[int(np.argmin(steps)) + 1],
            )
        )
