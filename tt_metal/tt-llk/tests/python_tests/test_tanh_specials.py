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
from conftest import skip_for_quasar, skip_for_wormhole
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
    SFPU_UNARY_ITERATIONS,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.utils import classify_special

# exp field all ones -> 2 infinities + 254 NaNs, both signs
POS_SPECIALS = list(range(0x7F80, 0x8000))  # 128 bfloat16 bit patterns
NEG_SPECIALS = list(range(0xFF80, 0x10000))  # 128
ZEROS = [0x0000, 0x8000]  # +0.0, -0.0

FACE_SIZE = 256
TILE_SIZE = 4 * FACE_SIZE
VECTOR = 32  # lanes per SFPU vector; 32 vectors walk a 32x32 tile


# Which tile datums SFPU vector v holds on a 16-bit Dest, measured on Blackhole (a tile
# of 2.0 through calculate_tanh with ITERATIONS = 1 changed exactly the even datums
# 0..62; ITERATIONS = 2 changed 0..63): vectors come in pairs over a block of four tile
# rows, the even vector holding the even columns and the odd vector the odd columns.
# So an x2 pair (2k, 2k+1) is one 4-row block split by column parity.
def _vector_datums(v):
    base = 64 * (v // 2)
    return np.arange(base + v % 2, base + 64, 2)


def _lanes(first_vector, count=1):
    """Tile datum indices covered by `count` SFPU vectors starting at `first_vector`."""
    return np.concatenate(
        [_vector_datums(v) for v in range(first_vector, first_vector + count)]
    )


# bfloat16 bit patterns for the mixed vectors: finite values next to the non-finite ones,
# so a guard that leaks or that is keyed on the wrong lane shows up. tanh(0) = 0 could not.
BF16 = {
    "+inf": 0x7F80,
    "-inf": 0xFF80,
    "NaN(+)": 0x7FC0,
    "NaN(-)": 0xFFC0,
    "sNaN(+)": 0x7F81,  # a non-canonical payload, both signs
    "sNaN(-)": 0xFF81,
    "0.5": 0x3F00,
    "-3.0": 0xC040,
    "max": 0x7F7F,  # largest finite bfloat16
    "-max": 0xFF7F,
}
# Finite and non-finite interleaved along one vector's 32 lanes...
MIXED_VECTOR = [
    BF16[k] for k in ("+inf", "0.5", "NaN(-)", "-3.0", "sNaN(+)", "max", "-inf", "-max")
] * 4
# ...and whole vectors of each, paired both ways across one x2 call.
FINITE_VECTOR = [BF16[k] for k in ("0.5", "-3.0", "max", "-max")] * 8
NONFINITE_VECTOR = [
    BF16[k]
    for k in (
        "+inf",
        "-inf",
        "NaN(+)",
        "NaN(-)",
        "sNaN(+)",
        "sNaN(-)",
        "+inf",
        "NaN(-)",
    )
] * 4

# fp32 bit patterns for the 32-bit Dest softcap arm.
FP32 = {
    "+inf": 0x7F800000,
    "-inf": 0xFF800000,
    "NaN(+)": 0x7FC00000,
    "NaN(-)": 0xFFC00000,
    "sNaN(+)": 0x7F800001,
    "sNaN(-)": 0xFF800001,
    "+0.0": 0x00000000,
    "-0.0": 0x80000000,
}

SOFTCAP_BETA = 5.0  # sfpu_operations.h passes beta = 5 and 1 / 5 to calculate_softcap


def _write_bits(src, bits_by_face):
    """Write raw bit patterns into `src`, one face at a time.

    Not `StimuliSpec.custom(values=...)`, which would be the natural way to say
    this: that path ends in `torch.tensor(vals, dtype=torch.bfloat16)`, and torch's
    float -> bfloat16 conversion collapses every NaN onto one *positive* pattern.
    All 127 negative NaNs would arrive as +NaN and the two halves of this test
    would be the same half twice. Writing the bit patterns through an integer view
    converts nothing, so what reaches L1 is what is asked for. (The same hazard is
    in the shared strategy for any test that feeds it NaNs; fixing it there changes
    what several other suites are fed, so it is left alone here.)

    bfloat16 tensors take 16-bit patterns, float32 tensors 32-bit ones.
    """
    if src.dtype == torch.bfloat16:
        view, width = src.view(-1).view(torch.uint16), 16
    else:
        assert src.dtype == torch.float32, src.dtype
        view, width = src.view(-1).view(torch.int32), 32
    for face, bits in bits_by_face.items():
        base = face * FACE_SIZE
        assert len(bits) <= FACE_SIZE, "face %d overflows" % face
        if width == 32:  # int32 view: patterns with the sign bit set wrap negative
            bits = [b - (1 << 32) if b >= (1 << 31) else b for b in bits]
        view[base : base + len(bits)] = torch.tensor(
            bits, dtype=torch.uint16 if width == 16 else torch.int32
        )


def _by_vector(patterns, base=None):
    """{face: bits} with each `patterns[v]` (32 bit patterns) on SFPU vector v's datums.

    `base` is a {face: bits} dict to start from (faces of contiguous patterns, as the
    257-pattern faces are); vectors are laid over it.
    """
    tile = [0] * TILE_SIZE
    for face, bits in (base or {}).items():
        tile[face * FACE_SIZE : face * FACE_SIZE + len(bits)] = bits
    for v, bits in patterns.items():
        assert len(bits) == VECTOR, "vector %d has %d patterns" % (v, len(bits))
        for d, b in zip(_vector_datums(v), bits):
            tile[d] = b
    return {f: tile[f * FACE_SIZE : (f + 1) * FACE_SIZE] for f in range(4)}


def _run(
    values_by_face,
    out_fmt,
    dest_acc,
    in_fmt=DataFormat.Float16_b,
    floats=False,
    approx=ApproximationMode.Yes,
    mathop=MathOperation.Tanh,
    iterations=None,
):
    """Run `mathop` over `values_by_face`; the LUT tanh unless `approx` says otherwise.

    `floats=True` takes ordinary float values per face instead of raw bit patterns;
    the bit-pattern path exists only for the NaN-sign hazard `_write_bits` documents,
    which ordinary finite values do not have. `iterations` narrows the SFPU loop
    from the 32 vectors that cover the tile (SFPU_UNARY_ITERATIONS).
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
    templates = [
        generate_input_dim(dims, dims),
        APPROX_MODE(approx),
        FAST_MODE(FastMode.No),
        CLAMP_NEGATIVE(False),
        MATH_OP(mathop=mathop),
    ]
    if iterations is not None:
        templates.append(SFPU_UNARY_ITERATIONS(iterations))
    cfg = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=templates,
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


def _classify(hv, nan_sign):
    """`classify_special`, optionally keeping the sign of a NaN result."""
    if nan_sign and np.isnan(hv):
        return "NaN(-)" if np.signbit(hv) else "NaN(+)"
    return classify_special(hv)


def _group_specials(x, hw, lanes=None, nan_sign=False):
    """{input class: {result class: count}} over the non-finite and signed-zero lanes.

    `lanes` restricts the walk to a slice of the tile (default: all of it).
    """
    if lanes is not None:
        x, hw = x[lanes], hw[lanes]
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
        label = _classify(hv, nan_sign)
        groups.setdefault(key, {}).setdefault(label, 0)
        groups[key][label] += 1
    return groups


def _shown(groups, key):
    return ", ".join("%s x%d" % (k, v) for k, v in sorted(groups[key].items()))


def _assert_specials(groups, expected, keys, changed):
    """Every key in `keys` must arrive in `groups` and match its `expected` class.

    A key that never arrives is a hole in the stimuli rather than a pass -- -0.0 would
    disappear from `groups` entirely if its sign were lost before the kernel -- and a
    key with nothing expected against it can never fail, which is how the 127
    negative-NaN patterns once went unasserted. `changed` is the sentence appended to
    a mismatch, saying what the expected values are a record of.
    """
    missing = [k for k in keys if k not in groups]
    unrecorded = [
        "%s (hardware returned %s)" % (k, _shown(groups, k))
        for k in keys
        if k in groups and k not in expected
    ]
    failures = [
        "%s: expected %s, got %s" % (k, expected[k], _shown(groups, k))
        for k in keys
        if k in groups and k in expected and list(groups[k]) != [expected[k]]
    ]

    assert not missing, (
        "these input patterns never reached the kernel: "
        + ", ".join(missing)
        + "\nThe stimuli path lost them before the math did, so nothing below is "
        "asserting anything about them."
    )
    assert not unrecorded, (
        "no expected behaviour for: "
        + ", ".join(unrecorded)
        + "\nA key with no expected entry cannot fail. Add the measured value."
    )
    assert not failures, (
        "behaviour on the non-finite / signed-zero patterns changed:\n  "
        + "\n  ".join(failures)
        + "\n"
        + changed
    )


def _assert_finite_lanes(x, hw, lanes, reference, what, tol=1e-2):
    """The finite inputs in `lanes` came back as `reference(x)`, within `tol` (abs + rel).

    The finite lanes of a mixed vector are what shows a non-finite neighbour's guard
    leaking into them; tanh's polynomial carries ~2.3e-3 relative error and the
    bfloat16 store another half ulp, both well inside the tolerance.
    """
    xs, hs = x[lanes], hw[lanes]
    finite = np.isfinite(xs)
    assert finite.any(), "no finite lanes in %s" % what
    ref = reference(xs[finite].astype(np.float64))
    err = np.abs(hs[finite].astype(np.float64) - ref)
    bad = err > tol * (1.0 + np.abs(ref))
    assert (
        not bad.any()
    ), "%s: %d finite lanes departed from the reference next to non-finite ones:\n  " % (
        what,
        bad.sum(),
    ) + "\n  ".join(
        "x=%g -> got %g, expected %g" % (xv, hv, rv)
        for xv, hv, rv in zip(xs[finite][bad], hs[finite][bad], ref[bad])
    )


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
    groups = _group_specials(x, hw)

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
    KEYS = ("+inf", "-inf", "NaN(+)", "NaN(-)", "+0.0", "-0.0")
    _assert_specials(
        groups,
        RECORDED,
        KEYS,
        "These were identical on main. If the change is deliberate, update RECORDED.",
    )


# What the polynomial kernel returns on a 16-bit Dest, as recorded on Blackhole and
# Wormhole. tanh(+-inf) = +-1 is the reference. The NaN lanes are not: the guard keeps
# the NaN, but the kernel's rounding to bfloat16 and the 16-bit Dest store each turn a
# NaN into an infinity of the same sign, whatever the output format (test_tanh_specials
# records the same on the LUT kernel; a Float32 pack of this kernel's Dest returns the
# same infinities; the same kernel with its rounding step removed on a 32-bit Dest stores
# NaN for NaN and +-1 for +-inf). So the NaN lanes are asserted as the sign-carrying
# non-finite result -- which the laundered +-1.0 is not -- and -0.0 -> +0.0 is the unpack
# path, not this kernel (see test_tanh_specials).
POLY_RECORDED = {
    "+inf": "1",
    "-inf": "-1",
    "NaN(+)": "+inf",
    "NaN(-)": "-inf",
    "+0.0": "+0.0",
    "-0.0": "+0.0",
}
POLY_CHANGED = (
    "These are the recorded 16-bit-Dest results of the polynomial kernel, not the "
    "reference; a NaN that comes back as +-1.0 is the min(result, 1.0f) clamp "
    "laundering it. If the change is deliberate, update POLY_RECORDED."
)


def test_tanh_polynomial_specials():
    """The paired-vector polynomial kernel on the 257 patterns, recorded 16-bit-Dest results.

    `calculate_tanh<false, false>` -- ApproximationMode.No on a 16-bit Dest, the default
    bfloat16 tanh -- walks the tile two vectors at a time through
    `_sfpu_tanh_polynomial_x2_` (the single-vector kernel that softcap and situ_glu call
    is `test_tanh_polynomial_single_vector_specials`). Each datum ends in a saturation
    guard: without one, SFPSWAP's min ranks a positive NaN above +inf (SFPSWAP.md) and
    SFPMAD emits a positive NaN (SFPMAD.md), so the Horner chain's NaN is clamped to 1.0
    and copysign returns +-1.0, a legal activation standing in for a NaN an upstream op
    produced. tanh(+-inf) = +-1 must survive the guard.

    Faces 0 and 1 are the 128 positive and 128 negative patterns, so there the two
    vectors of an x2 pair are both non-finite. Face 3 mixes, on the vector layout
    `_vector_datums` records: vector 24 interleaves finite values with +-inf and NaN
    along its lanes, and pairs (26, 27) and (28, 29) put a finite vector next to a
    non-finite one in both orders, so each vector's guard is checked on its own lanes
    and the finite lanes show whether the other vector's guard leaked.
    """
    x, hw = _run(
        _by_vector(
            {
                24: MIXED_VECTOR,
                25: FINITE_VECTOR,
                26: FINITE_VECTOR,
                27: NONFINITE_VECTOR,
                28: NONFINITE_VECTOR,
                29: FINITE_VECTOR,
            },
            base={0: POS_SPECIALS, 1: NEG_SPECIALS, 2: ZEROS},
        ),
        DataFormat.Float16_b,
        DestAccumulation.No,
        approx=ApproximationMode.No,
    )
    _assert_specials(
        _group_specials(x, hw), POLY_RECORDED, tuple(POLY_RECORDED), POLY_CHANGED
    )
    _assert_finite_lanes(x, hw, _lanes(24, 6), np.tanh, "face 3, vectors 24..29")


def test_tanh_polynomial_single_vector_specials():
    """The single-vector polynomial kernel on the same patterns, recorded 16-bit-Dest results.

    `_sfpu_tanh_polynomial_` is the odd-ITERATIONS tail of `calculate_tanh<false, false>`
    and the kernel under Blackhole's softcap and situ_glu. The driver's 32 iterations
    never reach it, so this variant runs ITERATIONS = 1: vector 0 alone goes through the
    kernel and vectors 1..31 stay as the datacopy wrote them. Vector 0 is the mixed
    vector; vector 1 -- the other column parity of the same four rows -- is the same
    pattern again, left untouched, which both pins the datum layout `_vector_datums`
    records (its +-inf lanes must come back as +-inf, vector 0's as +-1) and shows the
    finite lanes' bfloat16 round trip.
    """
    x, hw = _run(
        _by_vector({0: MIXED_VECTOR, 1: MIXED_VECTOR}),
        DataFormat.Float16_b,
        DestAccumulation.No,
        approx=ApproximationMode.No,
        iterations=1,
    )
    keys = ("+inf", "-inf", "NaN(+)", "NaN(-)")
    _assert_specials(
        _group_specials(x, hw, _lanes(0)), POLY_RECORDED, keys, POLY_CHANGED
    )
    _assert_finite_lanes(x, hw, _lanes(0), np.tanh, "vector 0")

    # The untouched copy: what a lane looks like when no SFPU op ran on it.
    untouched = _group_specials(x, hw, _lanes(1))
    _assert_specials(
        untouched,
        {"+inf": "+inf", "-inf": "-inf", "NaN(+)": "+inf", "NaN(-)": "-inf"},
        keys,
        "Vector 1 is outside the single ITERATIONS = 1 vector, so the kernel must not "
        "have touched it: +-inf stays +-inf (vector 0 turns them into +-1) and a NaN "
        "reaches L1 as an infinity of its sign through the 16-bit Dest. If the kernel "
        "reached these lanes, the datum layout _vector_datums records is wrong.",
    )
    _assert_finite_lanes(x, hw, _lanes(1), lambda v: v, "vector 1 (untouched)", tol=0.0)


@skip_for_wormhole  # softcap_tile exists on Blackhole (and Quasar) only
@skip_for_quasar  # Quasar's polynomial still clamps a NaN to +-beta; its record is separate
@pytest.mark.parametrize(
    "in_fmt,out_fmt,dest_acc",
    [
        (DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes),
    ],
    ids=["bf16", "fp32"],
)
def test_softcap_polynomial_specials(in_fmt, out_fmt, dest_acc):
    """softcap(x) = beta * tanh(x / beta) through `_sfpu_tanh_polynomial_` on the specials.

    softcap wraps the single-vector kernel in a multiply on each side, and Blackhole's
    SFPMAD re-emits the canonical positive NaN for a NaN operand (SFPMAD.md), so unlike
    bare tanh the sign of a NaN input is dropped: recorded here per Dest width. +-inf
    saturate to +-beta, which is what torch returns too.
    """
    bf16 = in_fmt == DataFormat.Float16_b
    specials = (
        _by_vector(
            {24: MIXED_VECTOR}, base={0: POS_SPECIALS, 1: NEG_SPECIALS, 2: ZEROS}
        )
        if bf16
        else {
            0: [FP32["+inf"], FP32["-inf"], FP32["NaN(+)"], FP32["NaN(-)"]] * 8
            + [FP32["sNaN(+)"], FP32["sNaN(-)"]] * 16,
            2: [FP32["+0.0"], FP32["-0.0"]],
        }
    )
    x, hw = _run(
        specials,
        out_fmt,
        dest_acc,
        in_fmt=in_fmt,
        approx=ApproximationMode.No,
        mathop=MathOperation.Softcap,
    )
    beta = "%.9g" % SOFTCAP_BETA
    RECORDED = {
        "+inf": beta,
        "-inf": "-" + beta,
        # measured on a Blackhole p100a; see the test docstring
        "NaN(+)": "+inf" if bf16 else "NaN(+)",
        "NaN(-)": "+inf" if bf16 else "NaN(+)",
        "+0.0": "+0.0",
        "-0.0": "+0.0",
    }
    _assert_specials(
        _group_specials(x, hw, nan_sign=not bf16),
        RECORDED,
        tuple(RECORDED),
        "These are the recorded Blackhole softcap results, not the reference: the "
        "sign of a NaN input does not survive softcap's multiply-adds, and a 16-bit "
        "Dest carries the NaN as an infinity. If the change is deliberate, update "
        "RECORDED.",
    )
    if bf16:
        _assert_finite_lanes(
            x,
            hw,
            _lanes(24),
            lambda v: SOFTCAP_BETA * np.tanh(v / SOFTCAP_BETA),
            "vector 24",
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
