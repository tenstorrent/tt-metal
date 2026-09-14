# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass
from itertools import chain, product

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    FastMode,
    MathOperation,
    format_dict,
)
from helpers.param_config import (
    build_param_id,
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.sfpu_domains import (
    _UNARY_OPS_NOT_SWEPT,
    BLOCK_SPREAD_DECADES,
    BLOCK_SPREAD_HIGH,
    EXTREMES_READY_OPS,
    SHIFT_EDGE_AMOUNTS,
    SPECIALS_READY_OPS,
    block_spread_spec,
    edge_spec,
    exclude_undefined,
    extreme_values,
    extremes_safe,
    for_op,
    for_op_pipeline,
    format_extremes,
    integer_specials,
    nan_sign_is_unspecified,
    nan_survives_to_l1,
    negative_zero_delivered,
    op_edge_points,
    op_threshold,
    sfpu_unary_ops,
    signed_zero_pole_cells,
    specials_after_nan_sign_gate,
    specials_safe,
)
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
    SFPU_RELU_MIN_INT_THRESHOLD,
    SFPU_SHIFT_AMOUNT,
    TILE_COUNT,
    DestSync,
    generate_input_dim,
)
from helpers.utils import passed_test

SUPPORTED_FAST_MODE_OPS = [
    MathOperation.Rsqrt,
    MathOperation.Sqrt,
]

# The unary op sweep: two coverage profiles, one test. Every op takes its stimuli from its
# registered domain in _OP_DOMAIN_REGISTRY; the profiles differ only in how much of the
# format/mode matrix each op is worth. BROAD_SWEEP_OPS gets the full format matrix, both
# approximation modes and both tile shapes; STANDARD_SWEEP_OPS is every other registered
# unary op, on Float16_b + Float32 with approximation off. To opt an op out of the sweep
# entirely, list it in sfpu_domains._UNARY_OPS_NOT_SWEPT.

BROAD_SWEEP_OPS = [
    MathOperation.Abs,
    MathOperation.Atanh,
    MathOperation.Asinh,
    MathOperation.Acosh,
    MathOperation.Cos,
    MathOperation.Log,
    MathOperation.Log1p,
    MathOperation.Reciprocal,
    MathOperation.Sin,
    MathOperation.Sqrt,
    MathOperation.Rsqrt,
    MathOperation.Square,
    MathOperation.Tanh,
    MathOperation.Celu,
    MathOperation.Silu,
    MathOperation.Tanhshrink,
    MathOperation.Floor,
    MathOperation.Ceil,
    MathOperation.Trunc,
    MathOperation.Frac,
    MathOperation.Gelu,
    MathOperation.GeluTanh,
    MathOperation.Neg,
    MathOperation.Fill,
    MathOperation.Elu,
    MathOperation.Exp,
    MathOperation.Exp2,
    MathOperation.Hardsigmoid,
    MathOperation.Threshold,
    MathOperation.ReluMax,
    MathOperation.ReluMin,
]

# Every registered unary SFPU op that the broad profile does not already cover, minus the
# ops sfpu_domains marks as deliberately unswept. Sorted so the parametrize ids are stable
# across runs.
STANDARD_SWEEP_OPS = sorted(
    sfpu_unary_ops() - set(BROAD_SWEEP_OPS) - set(_UNARY_OPS_NOT_SWEPT),
    key=lambda op: op.name,
)

# Per-op (atol, rtol) overrides for coarse LUT/polynomial ops; others use the
# per-format default in passed_test.
CUSTOM_TOLERANCES = {
    # Coarse 3-segment LUT: good PCC but abs error peaks ~0.12 near the knees.
    MathOperation.SigmoidAppx: (0.13, 0.05),
    MathOperation.GeluAppx: (0.13, 0.05),
}

BROAD_FORMATS = input_output_formats(
    [
        DataFormat.Float32,
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.Bfp8_b,
    ]
)

# The standard profile keeps the two formats that exercise the SFPU's own math without
# a block-float or 16-bit-exponent path in the way: bf16 for the 16-bit dst rounding
# and fp32 for full precision.
STANDARD_FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32])

BROAD_DIMENSIONS = [[64, 64], [128, 256]]
STANDARD_DIMENSIONS = [[64, 64]]

# Bfp4_b is only exercised as an input format, so the input is pinned to Bfp4_b here
# rather than building the full matrix and skipping the 12 non-Bfp4_b-input combos.
FORMATS_BFP4_B = [
    InputOutputFormat(DataFormat.Bfp4_b, output_format)
    for output_format in [
        DataFormat.Float16_b,
        DataFormat.Bfp8_b,
        DataFormat.Float16,
        DataFormat.Bfp4_b,
    ]
]

# Ops whose `#pragma GCC unroll X` loops miscompile to invalid assembly under coverage
# instrumentation, so they are skipped only when WITH_COVERAGE is set:
#   https://github.com/tenstorrent/tt-metal/issues/33268
#   https://github.com/tenstorrent/tt-llk/issues/883
# Covers ops from both sweep profiles.
COVERAGE_COMPILE_SKIP_OPS = [
    MathOperation.Acosh,
    MathOperation.Log,
    MathOperation.Log1p,
    MathOperation.Reciprocal,
    MathOperation.Sin,
    MathOperation.Sqrt,
    MathOperation.Rsqrt,
    MathOperation.Square,
    MathOperation.Celu,
    MathOperation.Silu,
    MathOperation.Neg,
    MathOperation.Exp2,
    MathOperation.Hardsigmoid,
    MathOperation.Threshold,
    MathOperation.ReluMax,
    MathOperation.ReluMin,
    MathOperation.Tanh,
    MathOperation.Gelu,
    MathOperation.GeluDerivative,
    MathOperation.LogWithBase,
    MathOperation.GeluAppx,
]


def _skip_coverage_unsupported(mathop):
    """Coverage-build exclusions, shared by every sweep that drives the unary ops.

    The exclusions are properties of the op under coverage instrumentation rather than of
    any one sweep's envelope, so every sweep that compiles these kernels needs this guard.
    """
    if not TestConfig.WITH_COVERAGE:
        return

    # Coverage runs skip the broad profile wholesale; only the standard profile runs.
    if mathop in BROAD_SWEEP_OPS:
        pytest.skip(
            reason="Broad-profile ops are not run under coverage: "
            "https://github.com/tenstorrent/tt-llk/issues/1435"
        )

    if mathop in COVERAGE_COMPILE_SKIP_OPS:
        pytest.skip(
            reason="`#pragma GCC unroll X` loops in these ops compile to invalid "
            "assembly under coverage instrumentation: "
            "https://github.com/tenstorrent/tt-metal/issues/33268 , "
            "https://github.com/tenstorrent/tt-llk/issues/883"
        )


def _sweep_params(formats, mathops, approx_modes, input_dimensions):
    """Build (formats, approx_mode, mathop, fast_mode, dest_acc, input_dimensions) tuples.

    Fast-mode-capable ops are swept with FastMode.No and FastMode.Yes; every other op
    runs with FastMode.No only. dest_acc always sweeps both values.
    """
    dest_accs = [DestAccumulation.No, DestAccumulation.Yes]
    fast_ops = [op for op in mathops if op in SUPPORTED_FAST_MODE_OPS]
    non_fast_ops = [op for op in mathops if op not in SUPPORTED_FAST_MODE_OPS]
    return list(
        chain(
            product(
                formats,
                approx_modes,
                fast_ops,
                [FastMode.No, FastMode.Yes],
                dest_accs,
                input_dimensions,
            ),
            product(
                formats,
                approx_modes,
                non_fast_ops,
                [FastMode.No],
                dest_accs,
                input_dimensions,
            ),
        )
    )


def _assert_broad_profile_valid():
    """Check the hand-maintained half of the sweep configuration.

    STANDARD_SWEEP_OPS is derived as the complement of BROAD_SWEEP_OPS, so only the
    hand-written lists can go wrong: duplicate entries, non-unary ops, and stale
    _UNARY_OPS_NOT_SWEPT exemptions.
    """
    duplicates = sorted(
        {op.name for op in BROAD_SWEEP_OPS if BROAD_SWEEP_OPS.count(op) > 1}
    )
    assert not duplicates, (
        "These ops are listed more than once in BROAD_SWEEP_OPS and would run their "
        f"whole matrix twice: {duplicates}"
    )
    not_unary = sorted(op.name for op in set(BROAD_SWEEP_OPS) - sfpu_unary_ops())
    assert not not_unary, (
        "These broad-profile ops are classified as having no unary SFPU kernel "
        f"(sfpu_domains._NON_SFPU_UNARY_OPS): {not_unary}"
    )
    stale_exemptions = sorted(
        op.name for op in set(_UNARY_OPS_NOT_SWEPT) - sfpu_unary_ops()
    )
    assert not stale_exemptions, (
        "These ops are exempted in sfpu_domains._UNARY_OPS_NOT_SWEPT but are not unary "
        f"SFPU ops, so the exemption does nothing: {stale_exemptions}"
    )


# The broad profile sweeps the full float matrix and the Bfp4_b input formats over the
# same op list — Bfp4_b is a second format axis, not a second op set. The standard
# profile is bf16/fp32 only, approx mode off, one tile shape.
UNARY_SWEEP_PARAMS = (
    _sweep_params(
        BROAD_FORMATS,
        BROAD_SWEEP_OPS,
        [ApproximationMode.No, ApproximationMode.Yes],
        BROAD_DIMENSIONS,
    )
    + _sweep_params(
        FORMATS_BFP4_B,
        BROAD_SWEEP_OPS,
        [ApproximationMode.No, ApproximationMode.Yes],
        BROAD_DIMENSIONS,
    )
    + _sweep_params(
        STANDARD_FORMATS,
        STANDARD_SWEEP_OPS,
        [ApproximationMode.No],
        STANDARD_DIMENSIONS,
    )
)


_assert_broad_profile_valid()


def _skip_bh_unsupported_float_combo(formats, dest_acc):
    """Blackhole with dest_acc=No supports neither Float16 input nor Float32->Float16."""
    if (
        dest_acc == DestAccumulation.No
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
        and (
            formats.input_format == DataFormat.Float16
            or formats == InputOutputFormat(DataFormat.Float32, DataFormat.Float16)
        )
    ):
        pytest.skip(reason="This combination is not supported on BH architecture")


def _gate_unspecified_nan_sign(mathop, formats, dest_acc, specials):
    """*specials*, minus the cells where the golden would assert an unspecified NaN sign.

    Blackhole is untouched -- its SFPMAD guarantees the canonical 0x7fc00000, so the golden's
    canonicalisation is sound there. The rule itself lives in sfpu_domains, shared with
    test_sfpu_binop_scalar's copy of this sweep.
    """
    return specials_after_nan_sign_gate(
        mathop,
        formats.input_format,
        formats.output_format,
        dest_acc,
        specials,
        TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE,
    )


def _skip_bh_unless_fp32(formats, dest_acc):
    """Blackhole with dest_acc=No only supports the Float32->Float32 combination."""
    if (
        dest_acc == DestAccumulation.No
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
        and formats != InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    ):
        pytest.skip(reason="This combination is not supported on BH architecture")


_UNARY_SWEEP_ARGNAMES = (
    "formats",
    "approx_mode",
    "mathop",
    "fast_mode",
    "dest_acc",
    "input_dimensions",
)


@pytest.mark.nightly
@pytest.mark.parametrize(
    ",".join(_UNARY_SWEEP_ARGNAMES),
    UNARY_SWEEP_PARAMS,
    ids=[build_param_id(_UNARY_SWEEP_ARGNAMES, p) for p in UNARY_SWEEP_PARAMS],
)
def test_eltwise_unary_sfpu(
    formats: list[InputOutputFormat],
    approx_mode: ApproximationMode,
    mathop: MathOperation,
    fast_mode: FastMode,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    """Every float unary SFPU op, over its registered domain.

    Stimuli come from the per-op registry for every op; the sweep envelope (which profile
    the op is in) is the only thing that varies. See BROAD_SWEEP_OPS.
    """
    broad = mathop in BROAD_SWEEP_OPS

    _skip_coverage_unsupported(mathop)

    # Each profile has its own Blackhole dest_acc=No guard, measured against its own
    # format set: the broad profile runs everything except a Float16 input or
    # Float32->Float16, while the standard profile allows only Float32->Float32.
    if broad:
        _skip_bh_unsupported_float_combo(formats, dest_acc)
    else:
        _skip_bh_unless_fp32(formats, dest_acc)

    custom_atol, custom_rtol = CUSTOM_TOLERANCES.get(mathop, (None, None))

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        approx_mode,
        mathop,
        fast_mode,
        input_dimensions,
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
    )


# Deliberate edge values. The random sweep above lands near knees and poles but never on
# them; this lands on them, using the shared metadata in sfpu_domains: domain singularities
# (cat A), op knees and rounding ties (cat D), and IEEE specials where the pipeline can carry
# them (cat B). It is one spec_A on the same driver, and edge_spec() is keyed off the op, so
# adding an op to the registry auto-enrols it here. The format axis is the standard profile;
# what varies is whether specials can be injected, which specials_safe() decides per
# (input, output, dest_acc).

_EDGE_SWEEP_OPS = sorted(
    sfpu_unary_ops() - set(_UNARY_OPS_NOT_SWEPT), key=lambda o: o.name
)

# What the cat-A/cat-D probes found on Wormhole, recorded as non-strict xfails so each case
# still executes and reports XPASS if the behaviour changes. Listed exhaustively per
# (input, output, dest_acc) so a combination drifting in or out shows up as a diff here.
# Sign(-0.0) and Heaviside(-0.0) diverge because SFPSETCC is specified only for inputs that
# are neither negative zero nor NaN, so they are outside the documented contract rather than
# hardware faults; they diverge on exactly the two unpack_to_dest combinations, the only ones
# where a real -0.0 reaches the LREG. See _assert_signed_zero_partition_valid below.
_EDGE_KNOWN_DIVERGENCES = {
    MathOperation.Sign: (
        (DataFormat.Float32, DataFormat.Float16_b, DestAccumulation.Yes),
        (DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes),
    ),
    MathOperation.Heaviside: (
        (DataFormat.Float32, DataFormat.Float16_b, DestAccumulation.Yes),
        (DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes),
    ),
}


# The cat-B divergences, derived rather than listed: each op diverges on exactly the
# combinations that deliver the probe it diverges on, so the sets stay right when the format
# axis grows or a delivery measurement is revised. Reciprocal and SqrtCustom diverge wherever
# specials are carried at all; Sqrt and Rsqrt only where a real -0.0 is also delivered.
def _cat_b_divergences(delivers):
    return tuple(
        (fmt.input_format, fmt.output_format, dest_acc)
        for fmt in input_output_formats([DataFormat.Float16_b, DataFormat.Float32])
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes)
        if specials_safe(fmt.input_format, fmt.output_format, dest_acc)
        and delivers(fmt.input_format, dest_acc)
    )


_EDGE_KNOWN_DIVERGENCES.update(
    {
        MathOperation.Reciprocal: _cat_b_divergences(lambda _fmt, _dest_acc: True),
        MathOperation.SqrtCustom: _cat_b_divergences(lambda _fmt, _dest_acc: True),
    }
)


# The cat-A twin of _cat_b_divergences, for the -0.0 that now arrives at a *registered zero
# pole* rather than through FLOAT_SPECIALS. The distinction matters: a cat-B probe is gated on
# specials_safe() as well, and an op reached only through its pole may not be in
# SPECIALS_READY_OPS at all -- its -0.0 comes from boundary_probes() and is gated on delivery
# alone.
def _signed_zero_pole_divergences():
    return signed_zero_pole_cells(
        input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
        (DestAccumulation.No, DestAccumulation.Yes),
    )


# Sqrt and Rsqrt are registered here rather than through _cat_b_divergences, and the move is a
# bug fix, not a tidy-up. Both diverge on a -0.0, and while cat B was that value's only source
# the two derivations picked the same cells and the marker could be gated on `specials`. Adding
# the cat-G probe took the -0.0 out of cat B's hands: boundary_probes() now emits it at their
# registered zero pole whenever delivery allows, gated on nothing else.
#
# The gap that opened is _gate_unspecified_nan_sign(). On Wormhole at Float32->Float16_b /
# dest_acc=Yes it withdraws cat B for Rsqrt -- its NaN result would reach L1 as an infinity
# whose sign SFPMAD leaves open -- so `specials` went False, the marker was withheld, and the
# -0.0 went in anyway and failed. Measured: rsqrt(-0.0) comes back as +inf there against the
# golden's -inf, the same kernel NaN the fp32-dest cell shows, substituted by the pack. Sqrt
# and Reciprocal survive the same gate on that cell, which is why Rsqrt was the only red.
#
# Derived from delivery alone now, so the registration matches where the probe comes from. The
# assert underneath pins that this did not change which cells are excused.
_EDGE_KNOWN_DIVERGENCES[MathOperation.Sqrt] = _signed_zero_pole_divergences()
_EDGE_KNOWN_DIVERGENCES[MathOperation.Rsqrt] = _signed_zero_pole_divergences()

assert _signed_zero_pole_divergences() == _cat_b_divergences(negative_zero_delivered), (
    "the signed-zero pole cells and the cat-B cells that deliver a -0.0 have come apart, so "
    "moving Sqrt and Rsqrt between the two derivations is no longer cell-for-cell: re-measure "
    "before trusting either set"
)

# RsqrtCompat is the one op the signed-zero pole probe found. Measured on a Blackhole p150 at
# Float32->Float32, dest_acc=Yes: rsqrt_compat(-0.0) returns +inf where IEEE and the golden
# give -inf. Recorded on its own rather than folded in with Rsqrt, because the two do *not*
# agree: Rsqrt(-0.0) returns NaN and this returns a wrongly-signed infinity, so one entry
# covering both would be a claim about the hardware that is false of one of them.
#
# The other five ops with a zero pole that the probe newly reaches all agree with their
# goldens: ReciprocalCompat(-0.0) = -inf, Log(-0.0) = -inf, LogWithBase, Rdiv and SqrtCustom
# likewise. That is the headline result of driving it -- the divergence is the exception.
_EDGE_KNOWN_DIVERGENCES[MathOperation.RsqrtCompat] = _signed_zero_pole_divergences()

# The two whose divergence needs the cat-B probe to be sent: 1/NaN and sqrt_custom(-inf),
# which only the specials set carries. Their xfails are conditional on specials surviving the
# NaN-sign gate; see where the marker is applied.
#
# Sqrt and Rsqrt were in here and are not any more. Their probe is the -0.0 boundary_probes()
# emits at a registered pole, so the gate can withdraw cat B without withdrawing the stimulus,
# and conditioning the marker on `specials` left one cell failing -- see the note above.
_CAT_B_DERIVED_DIVERGENCES = frozenset(
    {MathOperation.Reciprocal, MathOperation.SqrtCustom}
)

_EDGE_DIVERGENCE_REASON = {
    MathOperation.Sign: "sign(-0.0) returns -1; torch and IEEE give 0. Outside the "
    "documented SFPSETCC contract, which is specified only for inputs that are not "
    "negative zero. Scoped to the unpack-to-dest combinations, the only ones where a real "
    "-0.0 reaches the LREG.",
    MathOperation.Heaviside: "heaviside(-0.0) returns 0; -0.0 == 0 makes it 0.5. Same "
    "SFPSETCC negative-zero caveat as Sign, and the same unpack-to-dest scoping.",
    MathOperation.Reciprocal: "1/NaN returns +0; IEEE, torch and the golden all give NaN. "
    "Every other special agrees, so this is the NaN probe alone, and it diverges on every "
    "combination that delivers one. Not prescribed by the ISA.",
    MathOperation.SqrtCustom: "sqrt_custom(-inf) returns -inf; IEEE and the golden give "
    "NaN. The non-finite guard passes non-finite input straight through rather than "
    "synthesising a NaN, which is right for +inf and NaN and wrong for -inf -- a deliberate "
    "limit of the minimal fix, since a negative-to-NaN guard would regress erfinv on "
    "ordinary in-domain inputs. See https://github.com/tenstorrent/tt-metal/issues/52930.",
    MathOperation.Sqrt: "sqrt(-0) returns NaN; IEEE and the golden give -0. Scoped to the "
    "unpack-to-dest combinations, the only ones where a real -0.0 reaches the LREG. What "
    "reaches L1 depends on the pack: the NaN itself on an fp32 output, and a signed infinity "
    "where the pack narrows -- one kernel behaviour with two presentations, not two "
    "divergences.",
    MathOperation.Rsqrt: "rsqrt(-0) returns NaN; IEEE and the golden give -inf. Same cause "
    "and same unpack-to-dest scoping as Sqrt, including the pack's substitution: measured on "
    "a Wormhole n300, Float32->Float32 at dest_acc=Yes returns the NaN and "
    "Float32->Float16_b returns +inf against the golden's -inf.",
    MathOperation.RsqrtCompat: "rsqrt_compat(-0.0) returns +inf; IEEE and the golden give "
    "-inf. A wrongly-signed infinity, not the NaN Rsqrt returns for the same input, so the "
    "two are recorded separately. Reached through the cat-A zero pole rather than cat B -- "
    "this op is not in SPECIALS_READY_OPS -- so it is scoped by delivery alone, to the "
    "unpack-to-dest combinations where a real -0.0 arrives.",
}


def _unpack_to_dest(input_format: DataFormat, dest_acc: DestAccumulation) -> bool:
    """Mirror of the unpack_to_dest expression eltwise_unary_sfpu passes to TestConfig.

    Kept as one expression rather than two literals so the claim below is checked against
    the driver's actual routing, not against a copy of it that can drift.
    """
    return input_format.is_32_bit() and dest_acc == DestAccumulation.Yes


def _assert_signed_zero_partition_valid():
    """The signed-zero ops must partition on unpack_to_dest, exactly.

    The explanation recorded against those divergences is an inference from which combinations
    diverge, and a comment is prose that no run checks. Asserting the shape instead makes
    editing a table without revisiting the explanation fail at collection.
    """
    all_combos = [
        (fmt.input_format, fmt.output_format, dest_acc)
        for fmt in input_output_formats([DataFormat.Float16_b, DataFormat.Float32])
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes)
    ]

    expectations = {
        # SFPSETCC mishandles a -0.0 that does arrive, which is the unpack-to-dest path.
        MathOperation.Sign: True,
        MathOperation.Heaviside: True,
    }

    # Signbit used to hold the other side of this partition: six xfails recording that the -0.0
    # probe never arrived on the datacopy path. negative_zero_delivered() now keeps the probe
    # off those pipelines, so an entry here would be a non-strict xfail that can never fire.
    assert MathOperation.Signbit not in _EDGE_KNOWN_DIVERGENCES, (
        "Signbit's divergences were a stimulus limitation, not a kernel defect. An entry "
        "here means the delivery gate changed -- re-derive it rather than restoring it."
    )

    for op, diverges_when_unpack_to_dest in expectations.items():
        expected = {
            combo
            for combo in all_combos
            if _unpack_to_dest(combo[0], combo[2]) == diverges_when_unpack_to_dest
        }
        recorded = set(_EDGE_KNOWN_DIVERGENCES.get(op, ()))
        assert recorded == expected, (
            f"{op.name}'s recorded divergences no longer match the unpack_to_dest "
            f"partition (expected unpack_to_dest == "
            f"{diverges_when_unpack_to_dest}).\n"
            f"  missing: {sorted(str(c) for c in expected - recorded)}\n"
            f"  extra:   {sorted(str(c) for c in recorded - expected)}\n"
            "The comment above rests on this partition -- if the measurement really "
            "moved, re-derive the explanation rather than only editing the table."
        )

    assert set(_EDGE_KNOWN_DIVERGENCES[MathOperation.Sign]) == set(
        _EDGE_KNOWN_DIVERGENCES[MathOperation.Heaviside]
    ), "Sign and Heaviside share one SFPSETCC cause, so their sets must stay identical"


_assert_signed_zero_partition_valid()


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    mathop=_EDGE_SWEEP_OPS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_edges(
    request,
    formats: list[InputOutputFormat],
    mathop: MathOperation,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    # Same ops, same driver, same templates as test_eltwise_unary_sfpu, so the same
    # coverage-build exclusions apply — see _skip_coverage_unsupported.
    _skip_coverage_unsupported(mathop)

    _skip_bh_unless_fp32(formats, dest_acc)

    # Two independent gates, and both have to pass: _SPECIALS_READY_OPS says the *golden*
    # defines a result for non-finite inputs, specials_safe() says the *pipeline* delivers
    # them intact. Neither implies the other.
    specials = mathop in SPECIALS_READY_OPS and specials_safe(
        formats.input_format, formats.output_format, dest_acc
    )

    specials = _gate_unspecified_nan_sign(mathop, formats, dest_acc, specials)

    # Marked after the gate: where the gate has taken cat B away the probe is not sent, so the
    # entry would be a non-strict xfail that XPASSes every run. Sign and Heaviside are cat-A
    # signed zeros and are unaffected.
    diverges_here = (formats.input_format, formats.output_format, dest_acc) in (
        _EDGE_KNOWN_DIVERGENCES.get(mathop, ())
    )
    if diverges_here and (specials or mathop not in _CAT_B_DERIVED_DIVERGENCES):
        request.node.add_marker(
            pytest.mark.xfail(reason=_EDGE_DIVERGENCE_REASON[mathop], strict=False)
        )

    spec_A = edge_spec(
        mathop,
        formats.input_format,
        formats.output_format,
        specials=specials,
        dest_acc=dest_acc,
    )
    if spec_A is None:
        # Smooth everywhere: no singularity, no knee, and specials not carryable here, so the
        # random sweep above already covers everything an edge probe could add.
        pytest.skip(
            reason=f"{mathop.name} has no edge values for this pipeline "
            f"(no domain boundary, no op knee, specials not preserved)"
        )

    custom_atol, custom_rtol = CUSTOM_TOLERANCES.get(mathop, (None, None))

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        input_dimensions,
        spec_A=spec_A,
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Format extremes and the subnormal band (cat F)
#
# The registry's widest domain is +/-1000, so the sweeps above jump from ~10 straight to
# infinity with nothing in the thirty-odd decades between or in the band just above zero. That
# band is not decoration: the goldens model flush-to-zero carefully (_FTZ_THRESHOLD,
# _apply_ftz) and stimuli_generator's _format_elem_min_magnitude() exists to keep random draws
# away from denormals, yet until this nothing drove an input that reached either.
#
# Its own variant rather than a flag on test_eltwise_unary_sfpu_edges: one failure class per
# variant, a saturation failure at the ceiling and a signed-zero failure at a pole being
# unrelated. Two independent gates, as everywhere -- EXTREMES_READY_OPS says the op's *golden*
# defines an answer at a format extreme, extremes_safe() says the *pipeline* delivers one, and
# it is not specials_safe() because the breakers that stop a NaN reaching the SFPU say nothing
# about a finite datum with an extreme exponent.
# ─────────────────────────────────────────────────────────────────────────────

_EXTREME_SWEEP_OPS = sorted(
    set(EXTREMES_READY_OPS) & set(_EDGE_SWEEP_OPS), key=lambda op: op.name
)

assert set(EXTREMES_READY_OPS) <= set(_EDGE_SWEEP_OPS), (
    "these ops are enrolled for cat F but the unary edge sweep cannot drive them, so the "
    "enrolment reaches nothing: "
    f"{sorted(op.name for op in set(EXTREMES_READY_OPS) - set(_EDGE_SWEEP_OPS))}"
)

_EXTREME_CELLS = tuple(
    (formats, dest_acc)
    for formats in input_output_formats([DataFormat.Float16_b, DataFormat.Float32])
    for dest_acc in (DestAccumulation.No, DestAccumulation.Yes)
)


def _extremes_generate_nan(mathop, formats, dest_acc):
    """Does the golden answer NaN at any cat-F probe on this pipeline?

    The probes are finite, so the question is not whether one is a NaN but whether the op
    *makes* one out of them -- acos and asin at either ceiling, acosh and atanh, log1p at the
    negative one. No property of the format axis predicts it, so it is evaluated rather than
    tabulated. UnarySFPUGolden is instantiated directly rather than through
    get_golden_generator: the --compile-producer stub has no `ops` mapping.
    """
    golden = UnarySFPUGolden()
    golden.data_format = formats.output_format
    golden.dst_format = formats.output_format
    golden.dest_acc = dest_acc
    for value in extreme_values(formats.input_format, formats.output_format, dest_acc):
        result = golden.ops[mathop](float(value))
        if isinstance(result, float) and math.isnan(result):
            return True
    return False


# What keeps this sweep's assertions sound where the golden's NaN becomes an observable
# infinity. On six of the eight cells nan_survives_to_l1() is False, so a NaN the op invents is
# packed to an infinity and its *sign* becomes the result; the golden canonicalises that NaN
# positive, so the sweep asserts a sign bit -- and Wormhole's SFPMAD.md says of a generated NaN
# that the sign "might or might not be set". The edge sweep answers this with
# _gate_unspecified_nan_sign(); cat F needs the equivalent and cannot simply reuse
# GENERATED_NAN_SIGN_OPS, whose members are largely ops that do not produce a NaN from a finite
# extreme at all.
#
# The gate is in the test body. What makes the ops below safe today is not that gate -- none is
# in GENERATED_NAN_SIGN_OPS -- but the cat-B measurement that put them outside it: driving the
# specials set through every enrolled op on a Wormhole n300 found their generated NaN
# sign-clear. That is a real dependency of this sweep on another one's measurement and it was
# nowhere recorded, so it is asserted here.
_EXTREMES_NAN_OPS = frozenset(
    mathop
    for mathop in _EXTREME_SWEEP_OPS
    for formats, dest_acc in _EXTREME_CELLS
    if extremes_safe(formats.input_format, formats.output_format, dest_acc)
    and not nan_survives_to_l1(formats.input_format, formats.output_format, dest_acc)
    and _extremes_generate_nan(mathop, formats, dest_acc)
)

assert _EXTREMES_NAN_OPS <= set(SPECIALS_READY_OPS), (
    "these ops invent a NaN from a cat-F probe on a cell that packs it to an infinity, so "
    "this sweep asserts the sign Wormhole leaves unspecified -- and cat B never drove a NaN "
    "out of them, so GENERATED_NAN_SIGN_OPS makes no statement about that sign either: "
    f"{sorted(op.name for op in _EXTREMES_NAN_OPS - set(SPECIALS_READY_OPS))}. "
    "Measure the sign through cat B before enrolling the op for cat F."
)

# Not vacuous: if the probes or the goldens moved so that nothing invents a NaN any more, the
# assertion above would hold over an empty set and stop saying anything.
assert _EXTREMES_NAN_OPS, (
    "no cat-F op invents a NaN from a format extreme any more, so the sign dependency this "
    "records is gone -- delete it with the gate in the sweep body rather than leaving both"
)


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    mathop=_EXTREME_SWEEP_OPS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_extremes(
    formats: list[InputOutputFormat],
    mathop: MathOperation,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    """Drive the format's ceiling, its neighbour, its smallest normal and one subnormal."""
    _skip_coverage_unsupported(mathop)
    _skip_bh_unless_fp32(formats, dest_acc)

    if not extremes_safe(formats.input_format, formats.output_format, dest_acc):
        pytest.skip(
            reason="this pipeline cannot deliver a magnitude extreme intact "
            "(see sfpu_domains.extremes_safe)"
        )

    # See _EXTREMES_NAN_OPS. Inert while no NaN-inventing cat-F op is in
    # GENERATED_NAN_SIGN_OPS, which is the state the assertion there pins; it exists so that
    # adding one withdraws the variant instead of turning the sweep red against a sign the ISA
    # declines to promise.
    if (
        TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE
        and nan_sign_is_unspecified(
            mathop, formats.input_format, formats.output_format, dest_acc
        )
        and _extremes_generate_nan(mathop, formats, dest_acc)
    ):
        pytest.skip(
            reason=f"{mathop.name} invents a NaN at a format extreme and this pipeline packs "
            "it to an infinity, whose sign Wormhole's SFPMAD leaves unspecified "
            "(see tt-metal#52938)"
        )

    custom_atol, custom_rtol = CUSTOM_TOLERANCES.get(mathop, (None, None))

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        input_dimensions,
        # dest_acc decides whether the subnormal probe is sent at all: on the datacopy path
        # the LREG holds +0.0, and a probe there would blame the kernel for a datum it never
        # received. See sfpu_domains.subnormal_delivered().
        spec_A=StimuliSpec.custom(
            values=extreme_values(
                formats.input_format, formats.output_format, dest_acc
            ),
            seed=0,
            cycle=True,
        ),
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Overflow saturation
#
# The cat-F tranche above is deliberately the ops that *cannot* overflow, so its ceiling probe
# asks only whether the pipeline delivered the datum. This is the other half -- the ops whose
# *result* leaves the format -- and its own sweep because that failure is invisible to
# everything else here: the convert from the SFPU's fp32 to a narrower output must saturate to
# +/-inf, and one that wrapped would keep every cat-B probe green (a non-finite *input* still
# comes out right) while every large finite input silently returned a tiny wrong value.
#
# Every probe is exact in every format this runs on -- powers of two for Square, integers below
# 256 for the rest, which bfloat16's 8 mantissa bits hold exactly. A decimal near a threshold
# is the trap: 88.7 is 88.5 in bfloat16, so the test would pin a threshold other than the one
# it names. The overflowing probes also stay clear of the band between bfloat16's ceiling and
# fp32's, where a value is finite on one output format and infinite on the other.
#
# Underflow is absent: same convert, opposite end, but a result flushed to zero is the
# subnormal question cat F already covers, and one tensor would give one xfail two causes.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _SaturationProbe:
    """Magnitudes that straddle an op's overflow point, either side of it.

    *finite* are the controls -- large enough that a wrapped result would be obvious, small
    enough that the answer is still representable. *overflowing* must saturate. Both are
    needed: a list with no finite half asserts saturation with nothing to compare it to, and
    one with no overflowing half asserts ordinary arithmetic.

    *signed* emits the negation of every magnitude as well. Set it where the sign reaches the
    result -- Square and Cosh are even, Sinh is odd -- since a sign-handling defect at the
    ceiling would otherwise only be visible on half the domain.
    """

    finite: tuple
    overflowing: tuple
    signed: bool = False

    def values(self) -> list:
        magnitudes = self.finite + self.overflowing
        if not self.signed:
            return list(magnitudes)
        return [-m for m in magnitudes] + list(magnitudes)


# 2**63 squared is 2**126, the largest power-of-two square inside the bfloat16 exponent range;
# 2**64 squared is 2**128, the first one outside it. exp overflows just above 88, exp2 just
# above 127 (the exponent *is* the grid there), expwithbase is exp(x/2) so its threshold is
# twice exp's, and sinh/cosh are e**|x|/2 so theirs is just above 89.
_SATURATION_PROBES = {
    MathOperation.Square: _SaturationProbe(
        finite=(2.0**62, 2.0**63), overflowing=(2.0**64, 2.0**65), signed=True
    ),
    MathOperation.Exp: _SaturationProbe(finite=(80.0, 88.0), overflowing=(90.0, 100.0)),
    MathOperation.Exp2: _SaturationProbe(
        finite=(120.0, 127.0), overflowing=(128.0, 135.0)
    ),
    MathOperation.ExpWithBase: _SaturationProbe(
        finite=(160.0, 176.0), overflowing=(180.0, 200.0)
    ),
    MathOperation.Expm1: _SaturationProbe(
        finite=(80.0, 88.0), overflowing=(90.0, 100.0)
    ),
    MathOperation.Sinh: _SaturationProbe(
        finite=(80.0, 89.0), overflowing=(90.0, 100.0), signed=True
    ),
    MathOperation.Cosh: _SaturationProbe(
        finite=(80.0, 89.0), overflowing=(90.0, 100.0), signed=True
    ),
}

_SATURATION_FORMATS = [DataFormat.Float16_b, DataFormat.Float32]


def _assert_saturation_probes_straddle_the_ceiling():
    """Every op's finite probes must stay under the ceiling and its overflowing ones exceed it.

    Without this the probe list is literals that stay plausible while the thing they straddle
    moves: a wider ceiling makes every probe finite and the sweep asserts ordinary arithmetic,
    a narrower one makes every probe overflow and it asserts saturation with no control. Both
    still pass, which is what earns an assert.

    Classified by the *golden*, so nothing here restates what each op computes -- only where
    its overflow point is. `math.isfinite` alone will not do it, since the goldens evaluate in
    fp64 and Square(2**64) is finite there and above every ceiling this runs on.
    """
    golden = UnarySFPUGolden()
    for fmt in _SATURATION_FORMATS:
        golden.data_format = fmt
        golden.dst_format = fmt
        ceiling = max(abs(v) for v in format_extremes(fmt))
        for mathop, probe in _SATURATION_PROBES.items():
            for magnitude in probe.finite + probe.overflowing:
                result = abs(float(golden.ops[mathop](magnitude)))
                overflows = not math.isfinite(result) or result > ceiling
                expected = magnitude in probe.overflowing
                assert overflows == expected, (
                    f"{mathop.name} at {magnitude!r} on {fmt.name}: the golden gives "
                    f"{result!r} against a ceiling of {ceiling!r}, so it "
                    f"{'overflows' if overflows else 'does not overflow'} — but the table "
                    f"lists it as {'overflowing' if expected else 'finite'}. Re-choose the "
                    "magnitudes rather than moving the entry."
                )


_assert_saturation_probes_straddle_the_ceiling()


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats(_SATURATION_FORMATS),
    mathop=sorted(_SATURATION_PROBES, key=lambda op: op.name),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_saturation(
    formats: list[InputOutputFormat],
    mathop: MathOperation,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    """A result too large for the output format must saturate to ±inf, not wrap."""
    _skip_coverage_unsupported(mathop)
    _skip_bh_unless_fp32(formats, dest_acc)

    custom_atol, custom_rtol = CUSTOM_TOLERANCES.get(mathop, (None, None))

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        input_dimensions,
        # cycle=True for the reason edge_spec() gives: a zero tail would make the verdict a
        # statement about f(0), and 0 is the one input that cannot saturate.
        spec_A=StimuliSpec.custom(
            values=_SATURATION_PROBES[mathop].values(), seed=0, cycle=True
        ),
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mixed-magnitude block-float blocks
#
# The stimulus is sfpu_domains.block_spread_spec(), shared with the ternary suite so the two
# cannot come to model the same quantization differently; its shape and the decades it walks
# are argued for there. What the block does is measured here: in Bfp8_b, 0, 1792 and 2816 of
# 4096 elements flush to zero at 2**-4, 2**-12 and 2**-24 -- 2**-4 leaves the block intact and
# is the control -- while Bfp4_b takes 2048, 3328 and 3584 and Bfp2_b collapses fifteen of
# every sixteen at every spread. Those nine counts are pinned by test_sfpu_domains, so this
# paragraph cannot drift away from the stimulus. Measured against Abs on a Blackhole p150
# first, with zero mismatching lanes: the host quantizer models a mixed block the way the
# unpacker does, so a failure here is the op.
#
# Two variants, because the questions are independent -- whether an op survives a block with
# its small elements quantized away, and whether each format quantizes as modelled
# (op-independent, so one pass-through op is the instrument). On a Bfp4_b or Bfp2_b output this
# is also the only path that reaches `_bfp_block_aware_compare`'s lattice.
# ─────────────────────────────────────────────────────────────────────────────


def _block_spread_ops():
    """The broad-profile ops whose registered domain contains the whole spread.

    Derived rather than listed, so an op joins by having a domain wide enough to take it. The
    ones it leaves out -- Atanh, Acosh, Log, Reciprocal, Rsqrt -- are excluded because the
    spread would leave their domain, not because of anything about block floats: driving
    Reciprocal at an element the block flushed to zero would be a pole probe wearing a
    quantization probe's clothes, and the pole is cat A's job.
    """
    floor = BLOCK_SPREAD_HIGH * 2.0 ** -max(BLOCK_SPREAD_DECADES)
    selected = []
    for mathop in BROAD_SWEEP_OPS:
        try:
            spec = for_op(mathop, DataFormat.Bfp8_b).spec_A
        except KeyError:
            continue
        if spec.intervals or spec.low is None or spec.high is None:
            continue
        if spec.low <= floor and spec.high >= BLOCK_SPREAD_HIGH:
            selected.append(mathop)
    return selected


_BLOCK_SPREAD_OPS = _block_spread_ops()

assert _BLOCK_SPREAD_OPS, (
    "no broad-profile op has a domain wide enough for the block spread, so "
    "test_eltwise_unary_sfpu_block_spread would collect nothing"
)


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats([DataFormat.Bfp8_b]),
    mathop=_BLOCK_SPREAD_OPS,
    dest_acc=[DestAccumulation.Yes],
    # runtime(): the spread changes the tensor and nothing about the kernel, so the three
    # share one ELF per op.
    decades=runtime(list(BLOCK_SPREAD_DECADES)),
)
def test_eltwise_unary_sfpu_block_spread(
    formats: list[InputOutputFormat],
    mathop: MathOperation,
    dest_acc: DestAccumulation,
    decades: int,
):
    """Each op against a block whose small elements the shared exponent has quantized away."""
    _skip_coverage_unsupported(mathop)

    custom_atol, custom_rtol = CUSTOM_TOLERANCES.get(mathop, (None, None))

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        spec_A=block_spread_spec(decades),
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
    )


# Abs is the instrument, not the subject: it is a pass-through in magnitude, so a mismatch here
# is the block-float quantization model and cannot be the op. That is what lets this variant
# carry the format axis on its own instead of crossing it with the op sweep above.
_BLOCK_SPREAD_FORMAT_OPS = [MathOperation.Abs]


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats(
        [DataFormat.Bfp8_b, DataFormat.Bfp4_b, DataFormat.Bfp2_b], same=True
    ),
    mathop=_BLOCK_SPREAD_FORMAT_OPS,
    dest_acc=[DestAccumulation.Yes],
    decades=runtime(list(BLOCK_SPREAD_DECADES)),
)
def test_eltwise_unary_sfpu_block_spread_formats(
    formats: list[InputOutputFormat],
    mathop: MathOperation,
    dest_acc: DestAccumulation,
    decades: int,
):
    """Each block-float format's shared exponent, against a block that actually spans one."""
    # Abs is in BROAD_SWEEP_OPS, which the coverage build excludes wholesale, and this variant
    # is nightly -- the same job that runs the sweep above. Without the guard it fails the
    # coverage job at build time instead of skipping.
    _skip_coverage_unsupported(mathop)

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        spec_A=block_spread_spec(decades),
    )


# sqrt_custom(+inf): a strict regression assertion, deliberately outside the edge sweep.
#
# The edge sweep marks the whole SqrtCustom invocation non-strict XFAIL for the
# sqrt_custom(-inf) divergence, which would absorb a return to sqrt_custom(+inf) = NaN, so the
# repaired value is asserted here on its own. It runs on Float32 -> Float32 at dest_acc=Yes: a
# 16-bit output narrows NaN to inf on the way to L1, which is how the defect originally stayed
# hidden, so only a 32-bit output can show a regression.
@pytest.mark.nightly
def test_sqrt_custom_infinity_regression(request):
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    dest_acc = DestAccumulation.Yes
    input_dimensions = [32, 32]

    # Quasar still carries the pre-fix kernel (its ckernel_sfpu_sqrt_custom.h guards only
    # val != 0.0f), so it is expected to fail here rather than silently not being covered.
    # Non-strict: fixing Quasar should XPASS and prompt removing this, not error.
    if TestConfig.CHIP_ARCH == ChipArchitecture.QUASAR:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="Quasar's sfpu_sqrt_custom has not had the non-finite guard applied; "
                "sqrt_custom(+inf) is still NaN there. See tt-metal issue #52930.",
                strict=False,
            )
        )

    # If this ever goes False the pipeline stopped delivering +inf and the assertion below
    # would pass vacuously -- fail loudly instead of quietly testing nothing.
    assert specials_safe(formats.input_format, formats.output_format, dest_acc), (
        "Float32 -> Float32 at dest_acc=Yes no longer carries specials; re-derive the "
        "combination this regression test runs on before editing it."
    )

    num_elements = input_dimensions[0] * input_dimensions[1]
    # A finite control alongside the probe: if the guard is ever widened to pass everything
    # through, sqrt_custom(4.0) stops being 2.0 and this catches it in the same run.
    src_A = torch.full((num_elements,), 4.0, dtype=torch.float32)
    src_A[0] = float("inf")
    src_B = torch.zeros(num_elements, dtype=torch.float32)
    tile_cnt = (input_dimensions[0] // 32) * (input_dimensions[1] // 32)

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(ApproximationMode.No),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=MathOperation.SqrtCustom),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=tile_cnt,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=True,
    )

    res = torch.tensor(configuration.run().result, dtype=torch.float32)

    assert res[0] == float("inf"), (
        f"sqrt_custom(+inf) returned {res[0]!r}, expected +inf. The non-finite guard in "
        "ckernel_sfpu_sqrt_custom.h is what prevents this. See tt-metal issue #52930."
    )
    # Tolerance, not equality: sqrt_custom is an approximation, so the band only has to
    # separate a computed 2.0 from a passed-through 4.0.
    assert torch.allclose(res[1:], torch.tensor(2.0), rtol=1e-3, atol=0.0), (
        f"sqrt_custom(4.0) is no longer ~2.0 on the lanes around the probe "
        f"(max deviation {(res[1:] - 2.0).abs().max().item():.6g}); the non-finite guard's "
        "predicate has been widened to divert finite lanes."
    )


# reciprocal_compat(-0.0): the sign restore at the pole, deliberately outside the edge sweep.
#
# _reciprocal_compat_ returns |1/in|, so the signed wrapper's whole promise rests on the sign
# restore. The kernel takes the sign bit with SFPSETSGN rather than a comparison, which stays
# inside the documented SFPSETCC contract; this test is what holds that in place. It runs on
# Float32 -> Float32 at dest_acc=Yes, the only pipeline that both delivers a real -0.0 and
# keeps the two infinities distinguishable on the way back.
@pytest.mark.nightly
def test_reciprocal_compat_negative_zero_regression():
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    dest_acc = DestAccumulation.Yes
    input_dimensions = [32, 32]

    # If this ever goes False the pipeline stopped delivering -0.0 and the assertion below
    # would be testing +0.0 -- fail loudly rather than quietly testing nothing.
    assert negative_zero_delivered(formats.input_format, dest_acc), (
        "Float32 at dest_acc=Yes no longer delivers a real -0.0 to the LREG; re-derive the "
        "combination this regression test runs on before editing it."
    )

    num_elements = input_dimensions[0] * input_dimensions[1]
    # A positive control in the same tile: +0.0 must stay +inf. A restore that over-fires
    # (copying the wrong sign, or negating unconditionally) breaks this one, not the probe.
    src_A = torch.full((num_elements,), 1.0, dtype=torch.float32)
    src_A[0] = -0.0
    src_A[1] = 0.0
    src_B = torch.zeros(num_elements, dtype=torch.float32)
    tile_cnt = 1

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(ApproximationMode.No),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=MathOperation.ReciprocalCompat),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=tile_cnt,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=True,
    )

    res = torch.tensor(configuration.run().result, dtype=torch.float32)
    bits = res.view(torch.int32)

    assert bits[0].item() & 0xFFFFFFFF == 0xFF800000, (
        f"reciprocal_compat(-0.0) returned 0x{bits[0].item() & 0xFFFFFFFF:08X} "
        f"({res[0].item()!r}), expected -inf (0xFF800000). A 0x7F800000 means the sign "
        "restore did not fire on a delivered -0.0; a 0xFEFFFD9E means the pole guard did "
        "not fire either."
    )
    assert bits[1].item() & 0xFFFFFFFF == 0x7F800000, (
        f"reciprocal_compat(+0.0) returned 0x{bits[1].item() & 0xFFFFFFFF:08X} "
        f"({res[1].item()!r}), expected +inf. The restore is over-firing: it must move the "
        "input's sign bit, not set one."
    )
    # The rest of the tile is 1.0, catching a restore widened to every lane. Tolerance, not
    # equality: _reciprocal_compat_ is an approximation, so 1/1.0 lands near 1.0.
    assert torch.all(bits[2:] >= 0), (
        "reciprocal_compat(1.0) came back negative on some lane; the sign restore is "
        "firing outside the negative inputs."
    )
    assert torch.allclose(res[2:], torch.tensor(1.0), rtol=1e-3, atol=0.0), (
        f"reciprocal_compat(1.0) is no longer ~1.0 on the lanes around the probe "
        f"(max deviation {(res[2:] - 1.0).abs().max().item():.6g})."
    )


# Integer unary SFPU ops. Each has a dedicated integer kernel and runs through the
# shared driver with the input unpacked straight to DST (dest_acc=Yes is required for
# the 32-bit int path). Golden is exact (no PCC/tolerance).
_INT_UNARY_OPS = [
    MathOperation.LeftShift,
    MathOperation.RightShift,
    MathOperation.UnaryMaxInt32,
    MathOperation.UnaryMinInt32,
    MathOperation.UnaryMaxUint32,
    MathOperation.UnaryMinUint32,
    # relu_min is the only entry that is not an integer-only op. sfpu_operations.h selects
    # the vInt branch of _relu_min_ at runtime on math_format == Int32, and nothing else
    # drives that branch -- the float sweeps all take the vFloat one -- so without this the
    # integer half of the kernel, including its 2's-complement to sign+magnitude threshold
    # conversion, has no coverage at all.
    MathOperation.ReluMin,
]

# Ops whose kernel interprets DST as unsigned; run them under UInt32.
_UINT32_INT_UNARY_OPS = {
    MathOperation.UnaryMaxUint32,
    MathOperation.UnaryMinUint32,
}


def _int_unary_stimuli_spec(mathop):
    # Shifts use a fixed shift of 3, so keep inputs small-positive: x << 3 must stay
    # inside the positive int32 range (Dst is sign-magnitude, so hitting the sign bit
    # would diverge from the two's-complement golden).
    if mathop in (MathOperation.LeftShift, MathOperation.RightShift):
        return StimuliSpec.uniform(low=0.0, high=1_000_000.0)

    # Unary max/min compare against a fixed scalar, and a uniform draw reaches that scalar
    # with probability ~0, so the tie would never be driven. Take the exact value from
    # op_edge_points() and pair it with a deterministic spread either side.
    edges = [int(v) for v in op_edge_points(mathop)]
    if not edges:
        raise AssertionError(
            f"{mathop.name} has no op_edge_points() entry, so the int sweep cannot probe "
            "its comparison scalar — add one in sfpu_domains._OP_EDGE_POINTS"
        )
    straddle = [float(v + d) for v in edges for d in (-1, 0, 1)]
    # Positive-only keeps signed and unsigned interpretations identical (safe under
    # sign-magnitude Dst).
    spread = [float(v) for v in range(0, 2001, 125)]
    return StimuliSpec.custom(values=straddle + spread, seed=0)


@parametrize(
    mathop=_INT_UNARY_OPS,
    dest_acc=[DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_int(
    mathop: MathOperation,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    # ReluMin is in both BROAD_SWEEP_OPS and COVERAGE_COMPILE_SKIP_OPS, so this sweep needs
    # the same coverage guard the float ones use. It was unreachable before ReluMin joined
    # _INT_UNARY_OPS -- no integer-only op is in either list -- but without it the coverage
    # job compiles the relu_min kernel and fails at build time instead of skipping.
    _skip_coverage_unsupported(mathop)

    int_format = (
        DataFormat.UInt32 if mathop in _UINT32_INT_UNARY_OPS else DataFormat.Int32
    )
    formats = InputOutputFormat(int_format, int_format)

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        input_dimensions,
        spec_A=_int_unary_stimuli_spec(mathop),
    )


# Cat C for the unary integer ops. Its own sweep because test_eltwise_unary_sfpu_int above
# cannot reach these values: its shifts draw from [0, 1e6], eleven binades short of INT32_MAX,
# and its max/min straddle a scalar with a spread over [0, 2000], twenty short. Until this
# existed the coverage floor credited all six _INT_UNARY_OPS with cat C and no collected
# variant delivered it.
#
# INT32_MIN is out for every op and is not a gap: sign-magnitude Dst reads 0x80000000 as
# "negative zero" and cannot round-trip it, which the binary suite records the same way and
# covers with a dedicated xfail. INT32_MIN + 1 stands in. Every enrolment below is a
# measurement on a Wormhole n300, not a reading of the kernel.
_INT32_MIN = -(2**31)

_INT_UNARY_EXTREME_OPS = [
    MathOperation.RightShift,
    MathOperation.UnaryMaxInt32,
    MathOperation.UnaryMinInt32,
    MathOperation.UnaryMaxUint32,
    MathOperation.UnaryMinUint32,
]

# Driven at the *non-negative* extremes only, for the kernel's reason rather than the golden's.
# Measured: at the full signed set the right shift diverges on both negative values --
# `(INT32_MIN + 1) >> 3` comes back unshifted and `-1 >> 3` as 0x90000000, against -268435456
# and -1 from the two's-complement golden. That is the sign-magnitude Dst limitation
# SFPU_INT32_SHIFT.md documents and the binary suite already xfails, reached here through a
# magnitude rather than through INT32_MIN. Restricting the probe keeps the op covered at the
# extreme it can answer rather than recording a second copy of someone else's divergence.
_INT_UNARY_EXTREMES_NON_NEGATIVE = frozenset({MathOperation.RightShift})

# The one op with no answer at its extreme, recorded rather than driven or silently dropped.
# The exclusion is the *golden's*: with the fixed shift of 3 that sfpu_operations.h emits,
# `INT32_MAX << 3` does not fit in int32 and torch refuses the conversion, so there is no
# reference answer and the run errors before reaching the device. The largest input the op can
# be driven at is (2**31 - 1) >> 3, which is not a format extreme; even at shift amount 1 the
# ceiling is one binade below INT32_MAX.
_INT_UNARY_EXTREMES_NO_ANSWER = {
    MathOperation.LeftShift: "INT32_MAX << 3 overflows int32 and the golden cannot represent "
    "it, so the extreme has no reference answer; the op's reachable ceiling is INT32_MAX >> 3, "
    "which is not a format extreme",
}

# The ops already driven at their extremes by a sweep of their own, so an entry here is
# "covered elsewhere" rather than "not covered". ReluMin's is
# test_eltwise_unary_sfpu_relu_min_int_threshold, which drives both int32 extremes twice over:
# _RELU_MIN_INT_THRESHOLDS carries them as the compile-time threshold, and
# _relu_min_int_stimuli_spec() puts -/+INT32_MAX in the stimulus of every threshold variant.
_INT_UNARY_EXTREMES_ELSEWHERE = {
    MathOperation.ReluMin: "driven at both int32 extremes by "
    "test_eltwise_unary_sfpu_relu_min_int_threshold, as the threshold and as the stimulus",
}

# Totality: every op the int sweep drives is either enrolled at its extremes, covered by
# another sweep, or carries a recorded reason. Without this an op could join _INT_UNARY_OPS
# and be credited by none of the three.
_INT_UNARY_EXTREMES_VERDICTS = (
    set(_INT_UNARY_EXTREME_OPS)
    | set(_INT_UNARY_EXTREMES_NO_ANSWER)
    | set(_INT_UNARY_EXTREMES_ELSEWHERE)
)
_INT_UNARY_EXTREMES_UNDECIDED = set(_INT_UNARY_OPS) - _INT_UNARY_EXTREMES_VERDICTS
assert not _INT_UNARY_EXTREMES_UNDECIDED, (
    "these unary int ops are neither enrolled for cat C nor recorded as covered elsewhere or "
    f"as having no answer there: {sorted(op.name for op in _INT_UNARY_EXTREMES_UNDECIDED)}"
)
assert len(_INT_UNARY_EXTREMES_VERDICTS) == len(_INT_UNARY_EXTREME_OPS) + len(
    _INT_UNARY_EXTREMES_NO_ANSWER
) + len(
    _INT_UNARY_EXTREMES_ELSEWHERE
), "an op carries two cat-C verdicts, which cannot both be the reason"


def _int_unary_extreme_values(mathop):
    """The extremes *mathop* is driven at: its format's, less what it cannot answer."""
    int_format = (
        DataFormat.UInt32 if mathop in _UINT32_INT_UNARY_OPS else DataFormat.Int32
    )
    vals = [v for v in integer_specials(int_format) if v != _INT32_MIN]
    if mathop in _INT_UNARY_EXTREMES_NON_NEGATIVE:
        vals = [v for v in vals if v >= 0]
    return int_format, vals


@parametrize(
    mathop=_INT_UNARY_EXTREME_OPS,
    dest_acc=[DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_int_extremes(
    mathop: MathOperation,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    """The int32/uint32 extremes through the unary integer kernels (cat C).

    In the standard profile rather than nightly, matching test_eltwise_unary_sfpu_int, which
    drives these ops through the same driver at the same one cell: five variants, and splitting
    the profile would only make the class harder to see than the sweep it belongs to.

    cycle=True rather than custom()'s zero-fill: the list is three to five values long, so a
    zero-filled face would drive the probe on a handful of lanes and an ordinary zero on the
    other ~250 -- and for max/min a zero is a below-scalar value the base sweep already covers.
    """
    int_format, vals = _int_unary_extreme_values(mathop)
    assert vals, f"{mathop.name} is enrolled for cat C but has no extreme left to drive"

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(int_format, int_format),
        dest_acc,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        input_dimensions,
        spec_A=StimuliSpec.custom(values=[float(v) for v in vals], cycle=True, seed=0),
    )


_INT32_MAX = 2**31 - 1

# Both signs are swept. Every threshold reaches the vInt branch; what the negative half alone
# reaches is the overflow-safe compare it is split on, and Wormhole's hand-built threshold
# encoding. The negative extreme stops short of INT_MIN: CustomStrategy clamps stimuli at
# info.min + 1, so no input could straddle it.
_RELU_MIN_INT_THRESHOLDS = [-(_INT32_MAX - 1), -1000, -5, -1, 0, 5, 1000, _INT32_MAX]


def _relu_min_int_stimuli_spec(threshold: int) -> StimuliSpec:
    """Values straddling *threshold*, plus both ends of int32.

    Built around the threshold rather than a fixed span, so the clamp actually fires for a
    negative threshold. The range ends exercise the compare between far-apart operands.
    """
    # Straddling the boundary, then a decade either side of it. Offsets that leave the
    # stimuli range are dropped rather than folded onto its ends, which is what the thresholds
    # at the extremes would otherwise turn most of them into.
    offsets = (-1000, -100, -10, -2, -1, 0, 1, 2, 10, 100, 1000)
    candidates = [threshold + d for d in offsets] + [-_INT32_MAX, _INT32_MAX]
    values = sorted({v for v in candidates if -_INT32_MAX <= v <= _INT32_MAX})
    return StimuliSpec.custom(values=[float(v) for v in values], seed=0)


@parametrize(
    threshold=_RELU_MIN_INT_THRESHOLDS,
    dest_acc=[DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_relu_min_int_threshold(
    threshold: int,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    """relu_min on Int32 against both signs of threshold.

    The negative half is the point, and the golden is an exact integer max, so a wrong
    threshold shows up as a wrong clamp value rather than a tolerance miss.

    Int32 stimuli are two's complement, which is how ttnn feeds the device -- see
    use_int32_twos_complement in test_sfpu_reduce.py. Under this file's sign-magnitude
    default a kernel that reads Dst in the other encoding would pass instead.
    """
    formats = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        ApproximationMode.No,
        MathOperation.ReluMin,
        FastMode.No,
        input_dimensions,
        spec_A=_relu_min_int_stimuli_spec(threshold),
        relu_min_int_threshold=threshold,
        twos_complement=True,
    )


# Cat E: the shift amount itself, which SFPU_SHIFT_AMOUNT makes reachable. The amounts are
# shared with the binary shift sweep through sfpu_domains.SHIFT_EDGE_AMOUNTS.
_UNARY_SHIFT_OPS = [MathOperation.LeftShift, MathOperation.RightShift]

# Negatives collapse to one: the amount is emitted unsigned, so they all take the same
# out-of-range path. One is kept to pin that wrap.
_UNARY_SHIFT_AMOUNTS = [n for n in SHIFT_EDGE_AMOUNTS if n >= 0] + [-1]

# Interesting magnitudes only, since a shift is exact: powers of two, a few odd values, and
# zero. 2**30 keeps a large right shift working on a non-zero operand.
_SHIFT_STIMULUS_MAGNITUDES = [0, 1, 2, 3, 7, 255, 256, 1023, 65535, 65536, 2**30]


def _shift_stimulus_values(mathop, shift_amount):
    """Values that stay representable after *mathop* shifts them by *shift_amount*.

    A left shift is the only one that can leave int32, so the value set is chosen per variant.
    Positive-only, because Dst stores integers as sign-magnitude and a negative operand does
    not survive the round trip -- which also means the out-of-range half only covers the
    positive side, where the two kernels' rules coincide at 0.
    """
    magnitudes = _SHIFT_STIMULUS_MAGNITUDES
    if mathop == MathOperation.LeftShift and 0 <= shift_amount < 32:
        limit = _INT32_MAX >> shift_amount
        magnitudes = [m for m in magnitudes if m <= limit]
    return [float(m) for m in magnitudes]


@pytest.mark.nightly
@parametrize(
    mathop=_UNARY_SHIFT_OPS,
    shift_amount=_UNARY_SHIFT_AMOUNTS,
    dest_acc=[DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_int_shift(
    mathop: MathOperation,
    shift_amount: int,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    """Sweep the unary shift ops over the amounts worth driving, in range and out.

    Not the full axis -- see _UNARY_SHIFT_AMOUNTS for which amounts are kept and why.
    """
    values = _shift_stimulus_values(mathop, shift_amount)
    if not any(v for v in values):
        # Only 0 survives the representable-result filter, so the variant would assert
        # 0 << n == 0 and nothing else. Skipped rather than left as a green vacuous pass.
        pytest.skip(
            reason=f"every non-zero value overflows int32 at a left shift of {shift_amount}"
        )
    formats = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        input_dimensions,
        spec_A=StimuliSpec.custom(values=values, seed=0),
        shift_amount=shift_amount,
    )


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    approx_mode=[ApproximationMode.No],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_signbit(
    formats: list[InputOutputFormat],
    approx_mode: ApproximationMode,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    _skip_bh_unless_fp32(formats, dest_acc)

    # Sample both signs, avoiding 0 to sidestep -0.0 / rounding ambiguity.
    spec_A = StimuliSpec.uniform(intervals=[(-100.0, -0.5), (0.5, 100.0)])

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        approx_mode,
        MathOperation.Signbit,
        FastMode.No,
        input_dimensions,
        spec_A=spec_A,
    )


# Predicate ops (write 1.0/0.0). Finite-only stimuli give constant output (PCC
# undefined), so drive them with a spec interleaving +inf / -inf / nan and finite values.
ISINF_ISNAN_MATHOPS = [
    MathOperation.Isinf,
    MathOperation.Isposinf,
    MathOperation.Isneginf,
    MathOperation.Isnan,
    MathOperation.Isfinite,
]


# The predicates a bf16 input at dest_acc=Yes cannot answer. That unpack path delivers both
# NaN and -inf to the LREG as +inf, and which predicates that breaks follows from it rather
# than being a blanket property of the pipeline: is_nan reads 0 where the golden says 1,
# is_neg_inf reads 0 where it says 1, and is_inf reads 1 where it says 0. The other two
# survive precisely because +inf is what arrives -- is_pos_inf is untouched, and is_finite
# agrees by luck of the mapping, since isfinite(+inf) and isfinite(NaN) are both 0.
#
# Skipping the whole op list here withheld those two as well; they are swept now, so a
# regression in the +inf path is caught on a bf16 input instead of only on Float32.
_ISINF_ISNAN_BF16_DEST_UNSUPPORTED = [
    MathOperation.Isinf,
    MathOperation.Isneginf,
    MathOperation.Isnan,
]


def _isinf_isnan_stimuli_spec():
    def dist(size, dtype, generator):
        # Finite ramp in [-5, 5] with regular +inf / -inf / nan injected so every
        # face carries all special classes plus finite values.
        idx = torch.arange(size, dtype=torch.float32)
        x = (idx % 11) - 5.0
        x[0::7] = float("inf")
        x[1::7] = float("-inf")
        x[2::7] = float("nan")
        return x.to(dtype)

    return StimuliSpec(distribution=dist, seed=0)


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    approx_mode=[ApproximationMode.No],
    mathop=ISINF_ISNAN_MATHOPS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_isinf_isnan(
    formats: list[InputOutputFormat],
    approx_mode: ApproximationMode,
    mathop: MathOperation,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    _skip_bh_unless_fp32(formats, dest_acc)

    # bf16->fp32 dest unpack (non-32-bit input + dest_acc=Yes) delivers NaN and -inf as
    # +inf, which only the three predicates below can see; the rest are swept here.
    # See _ISINF_ISNAN_BF16_DEST_UNSUPPORTED.
    if (
        formats.input_format == DataFormat.Float16_b
        and dest_acc == DestAccumulation.Yes
        and mathop in _ISINF_ISNAN_BF16_DEST_UNSUPPORTED
    ):
        pytest.skip(
            reason="bf16->fp32 dest unpack delivers NaN and -inf as +inf, so this "
            "predicate cannot be evaluated on this pipeline"
        )

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        approx_mode,
        mathop,
        FastMode.No,
        input_dimensions,
        spec_A=_isinf_isnan_stimuli_spec(),
    )


# Ops whose behaviour turns on a comparison against a fixed scalar. A random float sweep
# reaches such a scalar with probability ~0, so the tie -- the one input where a `>` / `>=`
# slip is visible -- never gets driven. The 0/1 ops are here because their output would
# otherwise be constant; the clamps are here for the tie itself. Thresholds come from
# op_threshold().
_THRESHOLD_OPS = [
    MathOperation.LogicalNotUnary,
    MathOperation.UnaryEq,
    MathOperation.UnaryNe,
    MathOperation.ReluMin,
    MathOperation.ReluMax,
]


def _threshold_op_stimuli_spec(mathop):
    # Force a regular subset onto the op's threshold so the tie branch fires and, for the
    # 0/1 ops, the output is non-constant.
    #
    # The threshold comes from op_threshold() rather than a local literal, which could drift
    # from the dispatch constant the golden reads with no test noticing. Deliberately NOT
    # op_edge_points()[0]: that held only while every entry was exactly (threshold,), and the
    # clamp entries now straddle their cutoff, so index 0 is a probe beside the threshold.
    threshold = op_threshold(mathop)
    if threshold is None:
        raise AssertionError(
            f"{mathop.name} has no op_threshold() entry, so the threshold sweep cannot "
            "land on its comparison threshold — add one in sfpu_domains._OP_COMPARISON_THRESHOLD"
        )

    def dist(size, dtype, generator):
        idx = torch.arange(size, dtype=torch.float32)
        # Spread *relative to* the threshold: {t-2, t-1, t, t+1, t+2}. An absolute
        # {-2, -1, 0, 1, 2} spread works only for a threshold near zero -- against
        # relu_min's 5.0 every value sat on the clamped side and the golden went constant,
        # which is the same defect the widened domains fixed. Unchanged for logical_not,
        # whose threshold is 0.0.
        x = threshold + ((idx % 5) - 2.0)
        x[0::3] = threshold  # guaranteed threshold hits
        return x.to(dtype)

    return StimuliSpec(distribution=dist, seed=0)


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    approx_mode=[ApproximationMode.No],
    mathop=_THRESHOLD_OPS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_threshold(
    formats: list[InputOutputFormat],
    approx_mode: ApproximationMode,
    mathop: MathOperation,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    # ReluMin/ReluMax are COVERAGE_COMPILE_SKIP_OPS members, so this sweep needs the guard
    # too now that _THRESHOLD_OPS carries them.
    _skip_coverage_unsupported(mathop)
    _skip_bh_unless_fp32(formats, dest_acc)

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        approx_mode,
        mathop,
        FastMode.No,
        input_dimensions,
        spec_A=_threshold_op_stimuli_spec(mathop),
    )


def eltwise_unary_sfpu(
    test_name,
    formats: list[InputOutputFormat],
    dest_acc,
    approx_mode,
    mathop,
    fast_mode: FastMode,
    input_dimensions: list[int],
    spec_A=None,
    custom_atol=None,
    custom_rtol=None,
    shift_amount=None,
    relu_min_int_threshold=None,
    twos_complement=False,
):
    torch.manual_seed(0)
    torch.set_printoptions(precision=10)

    # The op's own signed domain, not generate_stimuli's positive-only format default, which
    # would leave the x<0 branch, the knees and the saturation tails unreached. A KeyError
    # means a new op arrived with no _OP_DOMAIN_REGISTRY entry -- register it. The domain has
    # to hold for the whole pipeline, so for_op_pipeline resolves against both formats and the
    # approximation mode and keeps the tightest result.
    if spec_A is None:
        spec_A = exclude_undefined(
            mathop,
            for_op_pipeline(
                mathop,
                formats.input_format,
                formats.output_format,
                approx_mode=approx_mode,
            ).spec_A,
        )

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_A,
    )

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
        **({} if shift_amount is None else {"shift_amount": shift_amount}),
        **(
            {}
            if relu_min_int_threshold is None
            else {"relu_min_int_threshold": relu_min_int_threshold}
        ),
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        test_name,
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(approx_mode),
            FAST_MODE(fast_mode),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=mathop),
            # Only emitted when swept: sfpu_operations.h keys off #ifdef, and every other
            # unary test has to keep compiling without the macro.
            *([] if shift_amount is None else [SFPU_SHIFT_AMOUNT(shift_amount)]),
            *(
                []
                if relu_min_int_threshold is None
                else [SFPU_RELU_MIN_INT_THRESHOLD(relu_min_int_threshold)]
            ),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            twos_complement=twos_complement,
        ),
        dest_acc=dest_acc,
        # dest_acc off: Float32 unpacks to 16-bit in src regs (later copied to dest for SFPU op)
        unpack_to_dest=(
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
    )

    res_from_L1 = configuration.run().result

    # res_from_L1 = res_from_L1[:1024]
    # golden_tensor = golden_tensor[:1024]
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
    ), "Assert against golden failed"


# Test exponential with APPROX_MODE=true, FAST_MODE=true, and CLAMP_NEGATIVE=true/false
@pytest.mark.parametrize("clamp_negative", [True, False])
def test_exponential_clamp_negative(clamp_negative: bool):
    torch.manual_seed(0)
    input_dimensions = [32, 32]
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    dest_acc = DestAccumulation.No

    # Generate custom stimuli with range [-5, 0.7]
    num_elements = input_dimensions[0] * input_dimensions[1]
    src_A = torch.rand(num_elements, dtype=torch.bfloat16) * 5.7 - 5.0
    # Set some values to be large and negative:
    src_A[0] = -10000
    src_A[1] = -1000
    src_A[2] = -200
    src_A[3] = -100
    src_A[4] = -88.5

    src_B = torch.zeros(num_elements, dtype=torch.bfloat16)
    tile_cnt_A = (input_dimensions[0] // 32) * (input_dimensions[1] // 32)
    tile_cnt_B = tile_cnt_A

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        MathOperation.Exp,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(ApproximationMode.Yes),
            FAST_MODE(FastMode.Yes),
            CLAMP_NEGATIVE(clamp_negative),
            MATH_OP(mathop=MathOperation.Exp),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # clamp_negative=False: require inputs < -88 to be negative (not necessarily
    # correct) and exclude them from the isclose check.
    if not clamp_negative:
        assert torch.all(
            res_tensor[:5] <= 0
        ), "Some of the first 5 elements are positive"
        res_tensor[:5] = golden_tensor[:5]

    # Use relaxed tolerance for this test
    atol, rtol = 0.02, 0.02
    is_close = torch.isclose(golden_tensor, res_tensor, rtol=rtol, atol=atol)
    is_nan = torch.isnan(golden_tensor) & torch.isnan(res_tensor)
    is_valid = is_close | is_nan

    assert torch.all(
        is_valid
    ), f"Test failed: {(~is_valid).sum()} elements outside tolerance (atol={atol}, rtol={rtol})"
