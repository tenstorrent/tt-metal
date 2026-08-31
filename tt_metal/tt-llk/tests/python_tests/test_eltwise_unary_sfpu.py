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
    EXTREMES_READY_OPS,
    SHIFT_EDGE_AMOUNTS,
    SPECIALS_READY_OPS,
    edge_spec,
    exclude_undefined,
    extreme_values,
    extremes_safe,
    for_op,
    for_op_pipeline,
    format_extremes,
    negative_zero_delivered,
    op_edge_points,
    sfpu_unary_ops,
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

# ─────────────────────────────────────────────────────────────────────────────
# The unary op sweep: two coverage profiles, one test
#
# Every op below runs through eltwise_unary_sfpu(), which takes its stimuli from the
# op's registered domain in _OP_DOMAIN_REGISTRY. The only thing that differs between
# the two lists is *how much of the format/mode matrix* each op is worth spending, so
# they are coverage profiles rather than different kinds of test:
#
#   BROAD_SWEEP_OPS    - the full format matrix including the block floats (and the
#                        Bfp4_b input formats), both approximation modes and both tile
#                        shapes. For ops whose kernels have format-specific or
#                        approx-mode-specific paths.
#   STANDARD_SWEEP_OPS - Float16_b + Float32, approximation mode off, one tile shape.
#                        Enough to validate the op's own math, and ~8x cheaper.
#
# Only the broad profile is listed by hand. The standard profile is every other unary
# SFPU op the registry knows about, so registering a domain is all it takes to get an op
# swept -- there is no second list to remember. Opting an op out of the sweep entirely is
# still a deliberate act: it goes in sfpu_domains._UNARY_OPS_NOT_SWEPT with a reason.
# ─────────────────────────────────────────────────────────────────────────────

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

    A helper rather than a copy per sweep, because the exclusions are properties of the
    *op* under coverage instrumentation, not of one sweep's envelope: any test that
    compiles these kernels hits the same invalid assembly. The scheduled llk-e2e job runs
    `not perf and not quasar` — nightly included — with coverage on, so a nightly sweep
    without this guard fails the coverage job at build time instead of being skipped.
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
    """Everything the derived standard profile cannot check for itself.

    Non-overlap, registration and exhaustiveness hold by construction now that
    STANDARD_SWEEP_OPS is the complement of BROAD_SWEEP_OPS within sfpu_unary_ops(). What
    is left is the hand-written half:

    - a repeated entry in BROAD_SWEEP_OPS runs that op's whole matrix twice,
    - a non-unary op in BROAD_SWEEP_OPS fails to compile once the sweep reaches it, and
      also silently drops out of the standard profile's complement,
    - an _UNARY_OPS_NOT_SWEPT entry that is not a unary op exempts nothing, so the reason
      recorded against it is misleading.
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


# Approximate exp overshoots the golden by a systematic ~5.7% (peak 6.75%) once its
# argument passes ~8 -- measured on Wormhole, the smallest output that breaches the
# default 5% rtol is exactly exp(8.00) = 2976, and 0.6% of elements in an affected
# tile breach it. That is a property of the approximation itself, not of the stimuli:
# it went unmeasured until this sweep stopped feeding exp uniform(0.1, 1.1), which
# never produced an argument above 1.1.
#
# Whether a given combination trips the 5% bar is marginal, and two things decide it:
# the domain its output format selects (high=16, or 10 when a Float16 output narrows
# it) and whether a 16-bit dst rounds golden and result back together -- dest_acc=Yes
# keeps an fp32 dst and exposes the full error. Hence Float32->Float16_b failing only
# at dest_acc=Yes. Listed exhaustively rather than by predicate so that a combination
# drifting in or out of tolerance shows up as a change here.
_APPROX_EXP_ACCURACY_XFAIL = {
    (DataFormat.Float16, DataFormat.Float16_b, DestAccumulation.No),
    (DataFormat.Float16, DataFormat.Float16_b, DestAccumulation.Yes),
    (DataFormat.Float32, DataFormat.Float16_b, DestAccumulation.Yes),
}

# ...and it is a **Wormhole** limit. Measured on a Blackhole p150b: both of the three
# combinations Blackhole can reach XPASSed, at both tile shapes, and no other unary variant
# XPASSed. So Blackhole's exp approximation holds the default 5% rtol where Wormhole's
# overshoots by ~5.7%, and Blackhole *asserts* the accuracy rather than tolerating it. Same
# shape of gate as _WORMHOLE_ONLY_EDGE_CLASSES in test_eltwise_binary_sfpu.py.
_APPROX_EXP_XFAIL_IS_WORMHOLE_ONLY = True


@pytest.mark.nightly
@pytest.mark.parametrize(
    ",".join(_UNARY_SWEEP_ARGNAMES),
    UNARY_SWEEP_PARAMS,
    ids=[build_param_id(_UNARY_SWEEP_ARGNAMES, p) for p in UNARY_SWEEP_PARAMS],
)
def test_eltwise_unary_sfpu(
    request,
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

    if (
        mathop == MathOperation.Exp
        and approx_mode == ApproximationMode.Yes
        and (formats.input_format, formats.output_format, dest_acc)
        in _APPROX_EXP_ACCURACY_XFAIL
        and not (
            _APPROX_EXP_XFAIL_IS_WORMHOLE_ONLY
            and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
        )
    ):
        # Marked dynamically rather than skipped so the case still executes: if the
        # approximation tightens, this reports XPASS instead of quietly staying green.
        request.node.add_marker(
            pytest.mark.xfail(
                reason="Approximate exp exceeds the default 5% rtol above an argument "
                "of ~8, peaking at 6.75%. See _APPROX_EXP_ACCURACY_XFAIL.",
                strict=False,
            )
        )

    if mathop == MathOperation.ReluMin:
        pytest.skip(reason="https://github.com/tenstorrent/tt-llk/issues/1120")

    if mathop == MathOperation.Tanh and approx_mode == ApproximationMode.Yes:
        pytest.skip(reason="Metal tanh does not support approximation mode")

    # Each profile has its own Blackhole dest_acc=No guard, measured against its own
    # format set: the broad profile runs everything except a Float16 input or
    # Float32->Float16, while the standard profile allows only Float32->Float32.
    if broad:
        _skip_bh_unsupported_float_combo(formats, dest_acc)
    else:
        _skip_bh_unless_fp32(formats, dest_acc)

    # Exp-family ops in approx mode can't run against bf8_b. Bfp4_b inputs are exempt:
    # that combination is validated by the Bfp4_b sweep, so only guard non-Bfp4_b inputs.
    if (
        approx_mode == ApproximationMode.Yes
        and mathop in [MathOperation.Exp, MathOperation.Exp2, MathOperation.Elu]
        and formats.input_format != DataFormat.Bfp4_b
        and (
            formats.input_format == DataFormat.Bfp8_b
            or formats.output_format == DataFormat.Bfp8_b
        )
    ):
        pytest.skip(
            reason="Exp-related operations are not supported for bf8_b format in approximation mode."
        )

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


# ─────────────────────────────────────────────────────────────────────────────
# Deliberate edge values (Phase 4)
#
# The sweep above widens the *random* domain, so it lands near knees and poles but never
# on them. This lands on them, using the shared metadata in sfpu_domains: domain
# singularities straddled by a format-relative epsilon (cat A), op knees and exact
# rounding ties (cat D), and IEEE specials where the pipeline can carry them (cat B).
#
# There is no new driver and no new C++ source — the whole thing is one spec_A. Because
# edge_spec() is keyed off the op, adding an op to the registry auto-enrols it here.
#
# The format axis is the standard profile rather than the broad one: an edge probe is a
# fixed value, so the block-float and approximation-mode axes vary nothing about it. What
# *does* vary is whether specials can be injected, which specials_safe() decides per
# (input, output, dest_acc) from the measured matrix — so the cat-A/cat-D probes run on
# all 8 combinations and the cat-B ones only where they mean something.
# ─────────────────────────────────────────────────────────────────────────────

_EDGE_SWEEP_OPS = sorted(
    sfpu_unary_ops() - set(_UNARY_OPS_NOT_SWEPT), key=lambda o: o.name
)

# What the cat-A/cat-D probes found on Wormhole, first time these points have been driven.
# Recorded as xfails rather than tolerated or probed-around, following Phase 0's precedent
# for approximate exp: the case still *executes* and reports XPASS if the behaviour
# changes. Listed exhaustively per (input, output, dest_acc) rather than by predicate so a
# combination drifting in or out shows up as a diff here.
#
# Cross-checked against tt-isa-documentation, which splits these into "documented" and
# "still open". Both groups stay xfailed — the test's job is to notice the divergence, not
# to judge it — but only the second group is worth a kernel-side look.
#
# DOCUMENTED, and the ISA is the authority:
#
#   -0.0 through a comparison, on the one path where -0.0 actually arrives. SFPSETCC is
#   specified only "provided that VC is neither negative zero nor any kind of NaN"
#   (WormholeB0/.../VectorUnit.md, and identically on Blackhole). So sign(-0.0) -> -1 and
#   heaviside(-0.0) -> 0 are *outside the documented contract* of the primitive those
#   kernels are built on, not hardware faults. The golden follows torch/IEEE-1985 and is
#   right about the mathematics; the hardware was never promised to agree.
#
# THE -0.0 PROBE REACHES THE SFPU ON ONLY TWO OF THE EIGHT COMBINATIONS:
#
#   The signed-zero divergences partition *exactly* on unpack_to_dest, which
#   eltwise_unary_sfpu sets to (input.is_32_bit() and dest_acc == Yes) — the only path on which
#   the datum skips SrcA and the datacopy. Sign and Heaviside diverge on those 2 combinations
#   and nowhere else; Signbit used to hold the complementary 6, which is what identified the
#   split. Asserted rather than observed — see _assert_signed_zero_partition_valid below.
#
#   One cause explains all three. Neither calculate_sign nor calculate_heaviside guards
#   |v| != 0 on its v_if(v < 0.0F), so a real -0.0 in the LREG would make them diverge on all
#   8; passing on 6 says the LREG holds +0.0 there. Signbit reads the sign bit directly, so it
#   returned 0 on those 6 and 1, correctly, on the 2 where the datum does arrive — a genuinely
#   broken sign-bit read would fail on all 8.
#
#   Signbit's 6 entries are therefore gone rather than kept: they recorded a *stimulus*
#   limitation that no kernel fix could clear. negative_zero_delivered() keeps the -0.0 probe
#   off the pipelines that flatten it, so Sign and Heaviside no longer pass vacuously there
#   either.
#
#   The host-side check on record establishes L1 only: -0.0 leaves the *host* correctly, which
#   says nothing about unpack -> SrcA -> DEST.
#
# STILL OPEN — not explained by the ISA:
#
#   Erfinv at ±1: golden ∓inf/±inf against a saturated result, on the two fp32-dest
#   combinations only, so tolerance-shaped rather than semantic.
_EDGE_KNOWN_DIVERGENCES = {
    MathOperation.Sign: (
        (DataFormat.Float32, DataFormat.Float16_b, DestAccumulation.Yes),
        (DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes),
    ),
    MathOperation.Heaviside: (
        (DataFormat.Float32, DataFormat.Float16_b, DestAccumulation.Yes),
        (DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes),
    ),
    MathOperation.Erfinv: (
        (DataFormat.Float16_b, DataFormat.Float32, DestAccumulation.Yes),
        (DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes),
    ),
}


# The cat-B divergences, derived rather than listed: each op diverges on exactly the
# combinations that *deliver* the probe it diverges on, so the sets stay right when the format
# axis grows or a delivery measurement is revised.
#
#   Reciprocal  every combination carrying specials at all -- 1/NaN is the probe.
#   Sqrt, Rsqrt every combination that also delivers a real -0.0, the strictly smaller
#               unpack-to-dest set. At dest_acc=No the kernel is handed +0.0 and agrees.
#
# Measured on a Blackhole p300a: Reciprocal on all 3 reachable combinations, Sqrt and Rsqrt on
# both of theirs. The rest of each set is Wormhole-only (_skip_bh_unless_fp32 takes dest_acc=No
# down to Float32->Float32 there) and follows from the same kernel path.
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
        MathOperation.Sqrt: _cat_b_divergences(negative_zero_delivered),
        MathOperation.Rsqrt: _cat_b_divergences(negative_zero_delivered),
    }
)


# The cat-A twin of _cat_b_divergences, for the -0.0 that now arrives at a *registered zero
# pole* rather than through FLOAT_SPECIALS. The distinction matters: a cat-B probe is gated on
# specials_safe() as well, and these ops are not in SPECIALS_READY_OPS at all -- their -0.0
# comes from boundary_probes() and is gated on delivery alone.
def _signed_zero_pole_divergences():
    return tuple(
        (fmt.input_format, fmt.output_format, dest_acc)
        for fmt in input_output_formats([DataFormat.Float16_b, DataFormat.Float32])
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes)
        if negative_zero_delivered(fmt.input_format, dest_acc)
    )


# RsqrtCompat is the one op the signed-zero pole probe found. Measured on a Blackhole p150 at
# Float32->Float32, dest_acc=Yes: rsqrt_compat(-0.0) returns +inf where IEEE and the golden give
# -inf. Recorded on its own rather than folded in with Rsqrt, because the two do *not* agree:
# Rsqrt(-0.0) returns NaN and this returns a wrongly-signed infinity, so one entry covering both
# would be a claim about the hardware that is false of one of them.
#
# The other five ops with a zero pole that the probe newly reaches all agree with their goldens:
# ReciprocalCompat(-0.0) = -inf, Log(-0.0) = -inf, LogWithBase, Rdiv and SqrtCustom likewise.
# That is the headline result of driving it -- the divergence is the exception.
_EDGE_KNOWN_DIVERGENCES[MathOperation.RsqrtCompat] = _signed_zero_pole_divergences()

# The three whose divergence needs the cat-B probe to be sent. Their xfails are conditional on
# specials surviving the NaN-sign gate; see where the marker is applied.
_CAT_B_DERIVED_DIVERGENCES = frozenset(
    {MathOperation.Reciprocal, MathOperation.Sqrt, MathOperation.Rsqrt}
)

_EDGE_DIVERGENCE_REASON = {
    MathOperation.Sign: "sign(-0.0) returns -1; torch and IEEE give 0. Outside the "
    "documented contract: SFPSETCC is specified only for inputs that are not negative "
    "zero (tt-isa-documentation WormholeB0/.../VectorUnit.md). These are the 2 "
    "unpack-to-dest combinations, the only ones where -0.0 reaches the LREG — the other 6 "
    "pass vacuously.",
    MathOperation.Heaviside: "heaviside(-0.0) returns 0; -0.0 == 0 makes it 0.5. Same "
    "SFPSETCC negative-zero caveat as Sign, and the same unpack-to-dest scoping.",
    MathOperation.Erfinv: "erfinv(±1) saturates instead of returning ±inf.",
    MathOperation.Reciprocal: "1/NaN returns +0: the kernel does not propagate NaN, where "
    "IEEE, torch and the golden all give NaN. Every other special agrees (1/±inf = ±0, "
    "1/±0 = ±inf), so this is the NaN probe alone and it diverges on every combination that "
    "delivers one. Not prescribed by the ISA, which says only that NaN inputs follow 'the "
    "usual IEEE754 rules'.",
    MathOperation.Sqrt: "sqrt(-0) returns NaN; IEEE and the golden give -0. Scoped to the "
    "unpack-to-dest combinations, the only ones where a real -0.0 reaches the LREG — at "
    "dest_acc=No the kernel is handed +0.0 and agrees, so the probe is not sent there.",
    MathOperation.Rsqrt: "rsqrt(-0) returns NaN; IEEE and the golden give -inf. Same cause "
    "and same unpack-to-dest scoping as Sqrt.",
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
    """The three signed-zero ops must partition on unpack_to_dest, exactly.

    This is the whole basis for reading Signbit's former six entries as "the probe is not
    delivered" rather than as a kernel-contract bug. It is an inference from *which*
    combinations diverge, so it stops holding the moment the sets stop lining up — and a reason
    string is prose, which no run checks. Assert the shape instead, so editing a table without
    revisiting the explanation fails at collection.

    Not asserting the *count*: what matters is that each op's divergent set is precisely one
    side of the unpack_to_dest split, which stays true if the format axis grows.
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
        "Signbit's divergences were a stimulus limitation, not a kernel defect. The -0.0 "
        "probe is no longer sent where it cannot be delivered, so re-adding entries here "
        "means the delivery gate changed -- re-derive it rather than restoring the table."
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
            "The signed-zero explanation above rests on this partition — if the "
            "measurement really moved, re-derive the explanation rather than only "
            "editing the table."
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

    if mathop == MathOperation.ReluMin:
        pytest.skip(reason="https://github.com/tenstorrent/tt-llk/issues/1120")

    # Two independent gates, and both have to pass: _SPECIALS_READY_OPS says the *golden*
    # defines a result for non-finite inputs, specials_safe() says the *pipeline* delivers
    # them intact. Neither implies the other.
    specials = mathop in SPECIALS_READY_OPS and specials_safe(
        formats.input_format, formats.output_format, dest_acc
    )

    specials = _gate_unspecified_nan_sign(mathop, formats, dest_acc, specials)

    # Marked after the gate, not before, because three of these divergences are cat-B's and the
    # gate can take cat B away: where it has, the probe is not sent, the divergence cannot
    # occur, and the entry would be a non-strict xfail that XPASSes every run. Sign, Heaviside,
    # RsqrtCompat and Erfinv are unaffected -- their divergences are cat-A poles and signed
    # zeros that edge_values() emits with or without specials.
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
        # Smooth everywhere: no singularity, no knee, and specials not carryable here.
        # 47 of the 97 unary ops are in this class, and for them the random sweep above
        # already covers everything an edge probe could add.
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
# The registry's widest domain is +/-1000, so the sweeps above jump from ~10 straight to infinity
# with nothing in the thirty-odd decades between or in the band just above zero. That band is not
# decoration: the goldens model flush-to-zero carefully (_FTZ_THRESHOLD, _apply_ftz) and
# stimuli_generator's _format_elem_min_magnitude() exists to keep random draws away from
# denormals, yet until this nothing drove an input that reached either.
#
# ITS OWN VARIANT, NOT A FLAG ON test_eltwise_unary_sfpu_edges: one failure class per variant. A
# saturation failure at the ceiling and a signed-zero failure at a pole are unrelated, and
# sharing a tensor lets one xfail hide the other. extreme_values() returns cat F alone for that
# reason.
#
# Two independent gates, as everywhere: EXTREMES_READY_OPS says the op's *golden* defines an
# answer at a format extreme, extremes_safe() says the *pipeline* delivers one -- and it is not
# specials_safe(), because the breakers that stop a NaN reaching the SFPU say nothing about a
# finite datum with an extreme exponent.
#
# THE FIRST TRANCHE is the ops whose behaviour at an extreme is uncontroversial and whose golden
# is plain arithmetic -- magnitude, sign, rounding, the pass-throughs, the comparisons. They
# settle a second question for free: above 2**mantissa every float is already an integer, so
# floor, ceil, round and trunc must be the identity there, and nothing else checks that.
# ─────────────────────────────────────────────────────────────────────────────

_EXTREME_SWEEP_OPS = sorted(
    set(EXTREMES_READY_OPS) & set(_EDGE_SWEEP_OPS), key=lambda op: op.name
)

assert set(EXTREMES_READY_OPS) <= set(_EDGE_SWEEP_OPS), (
    "these ops are enrolled for cat F but the unary edge sweep cannot drive them, so the "
    "enrolment reaches nothing: "
    f"{sorted(op.name for op in set(EXTREMES_READY_OPS) - set(_EDGE_SWEEP_OPS))}"
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
# *result* leaves the format -- and a different assertion with a different failure mode.
#
# Its own sweep because that failure is invisible to everything else here. The convert from the
# SFPU's fp32 to a narrower output must saturate to +/-inf; one that wrapped would keep every
# cat-B probe green (a non-finite *input* still comes out right) while every large finite input
# silently returned a tiny wrong value -- and no random sweep would reach it, the widest
# registered domain being +/-1000.
#
# Table-driven, and only now: Square was written alone first and the other six measured against
# it individually before this was derived, because a shared harness built before two or three ops
# exist fits the one it was written against. What the seven share turned out to be two lists of
# magnitudes and a sign flag.
#
# EVERY PROBE IS EXACT IN EVERY FORMAT THIS RUNS ON -- powers of two for Square, integers below
# 256 for the rest, which bfloat16's 8 mantissa bits hold exactly. A decimal near a threshold is
# the trap: 88.7 is 88.5 in bfloat16, so the test would pin a threshold other than the one it
# names. The overflowing probes also stay clear of the band between bfloat16's ceiling and
# fp32's, where a value is finite on one output format and infinite on the other and the variant
# would be measuring the format rather than the kernel.
#
# Underflow is absent: same convert, opposite end, but a result flushed to zero is the subnormal
# question cat F already covers through subnormal_delivered(), and one tensor would give one
# xfail two causes.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _SaturationProbe:
    """Magnitudes that straddle an op's overflow point, either side of it.

    *finite* are the controls -- large enough that a wrapped result would be obvious, small
    enough that the answer is still representable. *overflowing* must saturate. Both are
    needed: a list with no finite half asserts saturation with nothing to compare it to, and
    one with no overflowing half asserts ordinary arithmetic.

    *signed* emits the negation of every magnitude as well. Set it where the sign reaches the
    result -- Square and Cosh are even, Sinh is odd, and a sign-handling defect at the ceiling
    would otherwise only be visible on half the domain.
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
    moves: a wider ceiling makes every probe finite and the sweep asserts ordinary arithmetic, a
    narrower one makes every probe overflow and it asserts saturation with no control. Both still
    pass, which is what earns an assert.

    Classified by the *golden*, so nothing here restates what each op computes -- only where its
    overflow point is, which is the table. `math.isfinite` alone will not do it, since the
    goldens evaluate in fp64 and Square(2**64) and Cosh(90) are finite there and above every
    ceiling this runs on. UnarySFPUGolden is instantiated directly rather than through
    get_golden_generator for the reason _classify_edge_pair records: the harness swaps in a stub
    under --compile-producer, and this runs at import.
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
# Bfp8_b, Bfp4_b and Bfp2_b share one exponent per 16-element block, so the stimulus that
# exercises the format is a block of one large element and fifteen small ones -- everything else
# here is a narrow-range uniform or gaussian, and the shared exponent never bites. It does here:
# in Bfp8_b, 512 of 4096 elements flush to zero at 2**-8 and 2816 at 2**-24, and Bfp2_b collapses
# fifteen of every sixteen at every spread. Measured against Abs on a Blackhole p150 first, at
# all three formats and every spread, with zero mismatching lanes: the host quantizer models a
# mixed block the way the unpacker does, so a failure here is the op. Two variants, because the
# questions are independent -- whether an op survives a block with its small elements quantized
# away, and whether each format quantizes as modelled (op-independent, so one pass-through op is
# the instrument). On a Bfp4_b or Bfp2_b output this is also the only path that reaches
# `_bfp_block_aware_compare`'s lattice, passed_test having no tolerance pre-check there.
# ─────────────────────────────────────────────────────────────────────────────

_BLOCK_ELEMENTS = 16

# The spread, in binary decades below the block's largest element. 4 keeps every element inside
# Bfp8_b's 7 magnitude bits, 12 flushes the tail of the block, and 24 flushes most of it -- the
# walk from "all values keep full mantissa" to "the small ones are gone" that makes the lattice
# check earn its keep.
_BLOCK_SPREAD_DECADES = (4, 12, 24)

# The largest element of every block. 1.0 rather than the op's domain ceiling: it is exact in
# every format, and it keeps the whole spread inside the domain of every op selected below,
# which is what makes the selection a domain question rather than a magnitude one.
_BLOCK_SPREAD_HIGH = 1.0


def _block_spread_spec(decades):
    """One element at _BLOCK_SPREAD_HIGH per 16-element block, the rest log-spaced below it.

    Log-spaced rather than linear so the block spans the exponent range evenly: what the shared
    exponent does to an element depends on its distance from the block maximum in *binades*, so
    a linear ramp would put almost every element in the top binade and quantize none of them.
    """

    def face(size, dtype, generator):
        steps = torch.tensor(
            [0.0]
            + [
                -(decades * i) / (_BLOCK_ELEMENTS - 1)
                for i in range(1, _BLOCK_ELEMENTS)
            ],
            dtype=torch.float32,
        )
        block = _BLOCK_SPREAD_HIGH * torch.pow(2.0, steps)
        return block.repeat(-(-size // _BLOCK_ELEMENTS))[:size].to(dtype)

    return StimuliSpec(distribution=face, seed=0)


def _block_spread_ops():
    """The broad-profile ops whose registered domain contains the whole spread.

    Derived rather than listed, so an op joins by having a domain wide enough to take it. The
    five it leaves out -- Atanh, Acosh, Log, Reciprocal, Rsqrt -- are excluded because the
    spread would leave their domain, not because of anything about block floats: driving
    Reciprocal at an element the block flushed to zero would be a pole probe wearing a
    quantization probe's clothes, and the pole is cat A's job.
    """
    floor = _BLOCK_SPREAD_HIGH * 2.0 ** -max(_BLOCK_SPREAD_DECADES)
    selected = []
    for mathop in BROAD_SWEEP_OPS:
        try:
            spec = for_op(mathop, DataFormat.Bfp8_b).spec_A
        except KeyError:
            continue
        if spec.intervals or spec.low is None or spec.high is None:
            continue
        if spec.low <= floor and spec.high >= _BLOCK_SPREAD_HIGH:
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
    decades=runtime(list(_BLOCK_SPREAD_DECADES)),
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
        spec_A=_block_spread_spec(decades),
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
    decades=runtime(list(_BLOCK_SPREAD_DECADES)),
)
def test_eltwise_unary_sfpu_block_spread_formats(
    formats: list[InputOutputFormat],
    mathop: MathOperation,
    dest_acc: DestAccumulation,
    decades: int,
):
    """Each block-float format's shared exponent, against a block that actually spans one."""
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        spec_A=_block_spread_spec(decades),
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

    # Unary max/min compare against a fixed scalar; both branches plus the comparison
    # tie itself have to be exercised. A uniform draw straddles the scalar but reaches
    # it with probability ~0, so the tie — the one point where a `>` / `>=` slip is
    # visible — was never tested. Take the exact value from op_edge_points(), which is
    # where the golden's scalar is mirrored for exactly this purpose, and pair it with a
    # deterministic spread either side. custom() zero-fills the rest of each face, which
    # is itself a below-scalar probe.
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


# Cat E: the shift amount itself, which is the last gap in that category.
#
# The unary shift ops take their amount as a compile-time immediate, not as an operand, so
# until SFPU_SHIFT_AMOUNT existed the only amount ever tested was the fixed 3 that
# sfpu_operations.h hard-coded. (The *binary* shift ops take theirs as a second operand and
# have been swept over this same list for a while; that asymmetry is why this needed a C++
# change.)
#
# The amounts are shared with the binary shift sweep through sfpu_domains.SHIFT_EDGE_AMOUNTS
# rather than copied, so the two suites cannot drift on what counts as an interesting shift.
#
# The two unary shifts do not share an out-of-range rule -- see
# UnarySFPUGolden._shift_amount, which models each kernel separately.
_UNARY_SHIFT_OPS = [MathOperation.LeftShift, MathOperation.RightShift]

# The unary sweep takes the shared amounts but collapses the negatives to one.
#
# SFPU_SHIFT_AMOUNT emits the amount with a `u` suffix and both kernels branch on
# `shift_amt >= 32` as unsigned, so every negative amount arrives as a large unsigned and takes
# the same out-of-range path as 32, 33, 40, ... -- four amounts, one code path. The *binary*
# shift ops take theirs as a signed operand, where the four are genuinely distinct, which is
# why the shared list keeps them and only this consumer narrows.
#
# One is kept rather than none because the unsigned wrap is worth pinning: if SHIFT_AMOUNT ever
# became signed, -1 would compare as in-range and `v << -1` is undefined behaviour.
_UNARY_SHIFT_AMOUNTS = [n for n in SHIFT_EDGE_AMOUNTS if n >= 0] + [-1]

# A shift is exact, so the stimulus only has to reach the interesting magnitudes rather than
# straddle a boundary: powers of two around a byte and a half-word, a few odd values to catch a
# lost low bit, and zero. Non-negative only -- the docstring below says why.
#
# 2**30 is here for the *right* shift: without it the largest magnitude is 2**16, so every
# stimulus is already 0 by an amount of 17 and each of 17..30 asserts nothing but 0 >> n == 0,
# which cannot tell calculate_right_shift's `eff = 31` clamp from a clamp anywhere in that
# range. The limit filter below drops it from every LeftShift variant that would overflow.
_SHIFT_STIMULUS_MAGNITUDES = [0, 1, 2, 3, 7, 255, 256, 1023, 65535, 65536, 2**30]

_INT32_MAX = 2**31 - 1


def _shift_stimulus_values(mathop, shift_amount):
    """Values that stay representable after *mathop* shifts them by *shift_amount*.

    A left shift is the only one that can leave int32, and the amount is a compile-time
    immediate here, so the value set is chosen per variant. One bound suffices because the
    magnitudes are non-negative: keeping the result <= INT32_MAX also keeps it clear of
    INT32_MIN, which Dst stores as sign-magnitude and cannot represent. Out-of-range amounts
    need no filter -- left shift returns 0, right shift clamps to 31.

    Positive-only, for the reason _int_unary_stimuli_spec gives: Dst stores integers as
    sign-magnitude, so a negative operand does not survive the round trip.

    **That makes the out-of-range half weaker than it looks for RightShift.** An out-of-range
    right shift of a *negative* gives -1 rather than 0, which UnarySFPUGolden models, but no
    probe can reach it while negatives cannot be delivered -- so this only covers the positive
    half, where the two kernels' rules coincide at 0. Re-measure once a negative int32 operand
    can be delivered.
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

    Not the full axis: _UNARY_SHIFT_AMOUNTS carries 8 of the 32 in-range amounts, chosen the
    way SHIFT_EDGE_AMOUNTS is, and all six of its non-negative out-of-range ones. Only the
    *negative* amounts collapse to a single representative, for the reason recorded there --
    the `u` suffix makes all four arrive as the same large unsigned value.
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

    # bf16->fp32 dest unpack (non-32-bit input + dest_acc=Yes) doesn't preserve
    # -inf/nan, mangling is_neg/is_nan; skip — covered by the other input cases.
    if (
        formats.input_format == DataFormat.Float16_b
        and dest_acc == DestAccumulation.Yes
    ):
        pytest.skip(
            reason="bf16->fp32 dest unpack does not preserve -inf/nan special values"
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


# Threshold comparison ops: each maps every element to 0/1 by comparing against a
# fixed threshold, so a plain random float sweep never lands on the threshold and the
# output collapses to a constant (PCC undefined). Keyed by mathop:
#   logical_not(x) = (x == 0) ? 1 : 0   -> threshold 0.0
#   unary_eq / unary_ne(x)  compare vs 0.5 -> threshold 0.5
_THRESHOLD_OPS = [
    MathOperation.LogicalNotUnary,
    MathOperation.UnaryEq,
    MathOperation.UnaryNe,
]


def _threshold_op_stimuli_spec(mathop):
    # Force a regular subset onto the op's threshold so both the equal and not-equal
    # branches fire and the output is non-constant.
    #
    # The threshold comes from op_edge_points() rather than a local literal, which could drift
    # from UNARY_COMP_THRESHOLD -- the value the golden reads -- with no test noticing. These
    # three ops are outside _OP_DOMAIN_REGISTRY, so this is the only consumer of their
    # _OP_EDGE_POINTS entry, the same arrangement the int32 comparison ops have.
    edges = op_edge_points(mathop)
    if not edges:
        raise AssertionError(
            f"{mathop.name} has no op_edge_points() entry, so the threshold sweep cannot "
            "land on its comparison threshold — add one in sfpu_domains._OP_EDGE_POINTS"
        )
    # logical_not's entry is the signed-zero pair (+0.0, -0.0); both are the same
    # threshold, so the first element is the value to hit in every case.
    threshold = edges[0]

    def dist(size, dtype, generator):
        idx = torch.arange(size, dtype=torch.float32)
        x = (idx % 5) - 2.0  # {-2, -1, 0, 1, 2}; none equal 0.5
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
):
    torch.manual_seed(0)
    torch.set_printoptions(precision=10)

    # The op's own signed domain, not generate_stimuli's positive-only format default,
    # which would leave the x<0 branch, the piecewise knees and the saturation tails
    # unreached. A KeyError means a new op arrived with no _OP_DOMAIN_REGISTRY entry:
    # register it rather than falling back to the positive-only default.
    # The domain has to hold for the whole pipeline, so for_op_pipeline resolves against
    # both formats and keeps the tighter — see its docstring for why both matter.
    # approx_mode is passed because the exp family's positive side is bounded twice: by range
    # always, and by the approximation's accuracy only in ApproximationMode.Yes. The registry
    # carries the first; for_op applies the second from _APPROX_ACCURACY_MAX.
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
