# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


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
    PerfRunType,
    format_dict,
)
from helpers.param_config import (
    build_param_id,
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.perf.core import PerfConfig
from helpers.sfpu_domains import (
    _UNARY_OPS_NOT_SWEPT,
    SPECIALS_READY_OPS,
    edge_spec,
    exclude_undefined,
    for_op_pipeline,
    op_edge_points,
    sfpu_unary_ops,
    specials_safe,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    FRESH_CPP_IMPL,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    DestSync,
    TemplateParameter,
    generate_input_dim,
)
from helpers.utils import passed_test

SUPPORTED_FAST_MODE_OPS = [
    MathOperation.Rsqrt,
    MathOperation.Sqrt,
]


@dataclass
class ReciprocalImpl(TemplateParameter):
    """Select the production or fresh semantic-C++ reciprocal body.

    MUST stay a @dataclass with an annotated field: the variant hash keys the
    dataclass ``__repr__`` of every template, so a hand-written ``__init__``
    (empty inherited repr) makes every impl hash identically and lets
    ``.build_complete`` reuse the wrong impl's ELF (lane FO/FQ finding).
    """

    reciprocal_impl: int

    def convert_to_cpp(self) -> str:
        return f"constexpr int RECIPROCAL_IMPL = {self.reciprocal_impl};"


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
#   -0.0 through a comparison. SFPSETCC is specified only "provided that VC is neither
#   negative zero nor any kind of NaN" (WormholeB0/.../VectorUnit.md, and identically on
#   Blackhole). So sign(-0.0) -> -1 and heaviside(-0.0) -> 0 are *outside the documented
#   contract* of the primitive those kernels are built on, not hardware faults. The golden
#   follows torch/IEEE-1985 and is right about the mathematics; the hardware was never
#   promised to agree.
#
#   Note -0.0 *is* delivered correctly — verified host-side that the stimulus pipeline
#   preserves it into both Float32 and Float16_b — so the divergence is downstream of the
#   test, in the SFPU primitives.
#
# STILL OPEN — not explained by the ISA:
#
#   signbit(-0.0) returns 0 where the kernel's own docstring promises 1 ("logical-shift the
#   fp32 bit pattern right by 31 ... incl. -0.0"). Unlike sign/heaviside this op claims to
#   read the sign bit directly, so either the claim or the implementation is wrong. A
#   kernel-contract bug rather than a hardware one.
#
#   rsqrt at 0 saturates instead of returning inf. RsqrtCompat returns 1.7014118e38
#   (0x7F000000) where the golden gives inf, on all 8 combinations, while plain Rsqrt over
#   the same probe does *not* diverge — two implementations of one function disagreeing at
#   their shared pole. Nothing in the ISA prescribes either answer.
#
#   Erfinv at ±1: golden ∓inf/±inf against a saturated result, on the two fp32-dest
#   combinations only, so tolerance-shaped rather than semantic.
_EDGE_KNOWN_DIVERGENCES = {
    MathOperation.Signbit: (
        (DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.Yes),
        (DataFormat.Float16_b, DataFormat.Float32, DestAccumulation.No),
        (DataFormat.Float16_b, DataFormat.Float32, DestAccumulation.Yes),
        (DataFormat.Float32, DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float32, DataFormat.Float32, DestAccumulation.No),
    ),
    MathOperation.Sign: (
        (DataFormat.Float32, DataFormat.Float16_b, DestAccumulation.Yes),
        (DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes),
    ),
    MathOperation.Heaviside: (
        (DataFormat.Float32, DataFormat.Float16_b, DestAccumulation.Yes),
        (DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes),
    ),
    MathOperation.RsqrtCompat: (
        (DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.Yes),
        (DataFormat.Float16_b, DataFormat.Float32, DestAccumulation.No),
        (DataFormat.Float16_b, DataFormat.Float32, DestAccumulation.Yes),
        (DataFormat.Float32, DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float32, DataFormat.Float16_b, DestAccumulation.Yes),
        (DataFormat.Float32, DataFormat.Float32, DestAccumulation.No),
        (DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes),
    ),
    MathOperation.Erfinv: (
        (DataFormat.Float16_b, DataFormat.Float32, DestAccumulation.Yes),
        (DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes),
    ),
}

_EDGE_DIVERGENCE_REASON = {
    MathOperation.Signbit: "signbit(-0.0) returns 0; the kernel docstring promises 1 "
    "('incl. -0.0') and IEEE agrees. Not explained by the ISA — this op claims to read "
    "the sign bit directly, so the claim or the implementation is wrong.",
    MathOperation.Sign: "sign(-0.0) returns -1; torch and IEEE give 0. Outside the "
    "documented contract: SFPSETCC is specified only for inputs that are not negative "
    "zero (tt-isa-documentation WormholeB0/.../VectorUnit.md).",
    MathOperation.Heaviside: "heaviside(-0.0) returns 0; -0.0 == 0 makes it 0.5. Same "
    "SFPSETCC negative-zero caveat as Sign.",
    MathOperation.RsqrtCompat: "rsqrt(0) saturates to 1.7014118e38 (0x7F000000) instead "
    "of inf, while plain Rsqrt does not diverge at the same pole. Not prescribed by the "
    "ISA either way.",
    MathOperation.Erfinv: "erfinv(±1) saturates instead of returning ±inf.",
}


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

    if (formats.input_format, formats.output_format, dest_acc) in (
        _EDGE_KNOWN_DIVERGENCES.get(mathop, ())
    ):
        request.node.add_marker(
            pytest.mark.xfail(reason=_EDGE_DIVERGENCE_REASON[mathop], strict=False)
        )

    if mathop == MathOperation.ReluMin:
        pytest.skip(reason="https://github.com/tenstorrent/tt-llk/issues/1120")

    # Two independent gates, and both have to pass: _SPECIALS_READY_OPS says the *golden*
    # defines a result for non-finite inputs, specials_safe() says the *pipeline* delivers
    # them intact. Neither implies the other.
    specials = mathop in SPECIALS_READY_OPS and specials_safe(
        formats.input_format, formats.output_format, dest_acc
    )
    spec_A = edge_spec(
        mathop,
        formats.input_format,
        formats.output_format,
        specials=specials,
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


# Signed integer unary SFPU ops whose input or result crosses zero, so they need
# the two's-complement L1 stimuli plumbing (the test_sfpu_binary_rsub_int32
# convention) instead of the positive-only domain above. These were the last two
# perf-only int32 unary kernels with no functional golden anywhere (coverage-parity
# ledger class B-PERF-ONLY; Lane BK 2026-08-18 closes it): calculate_abs_int32
# (SFPABS on the I32 view, M32 magnitude store) and calculate_bitwise_not
# (~ on the raw I32 view). Golden is exact.
_INT32_SIGNED_UNARY_OPS = [
    MathOperation.AbsInt32,
    MathOperation.BitwiseNot,
]


@parametrize(
    mathop=_INT32_SIGNED_UNARY_OPS,
    dest_acc=[DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_int32_signed(
    mathop: MathOperation,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    formats = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)

    # Both signs, zero straddled deterministically; magnitudes stay far from
    # INT32_MIN (abs(INT32_MIN) is unrepresentable) and from INT32_MAX.
    spec_A = StimuliSpec.uniform(low=-1_000_000.0, high=1_000_000.0)

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        input_dimensions,
        spec_A=spec_A,
        twos_complement=True,
    )


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32]),
    approx_mode=[ApproximationMode.No],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
    fresh_cpp_impl=[0, 1],
)
def test_eltwise_unary_sfpu_signbit(
    formats: list[InputOutputFormat],
    approx_mode: ApproximationMode,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
    fresh_cpp_impl: int,
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
        fresh_cpp_impl=fresh_cpp_impl,
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
    threshold = 0.0 if mathop == MathOperation.LogicalNotUnary else 0.5

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
    fresh_cpp_impl=0,
    reciprocal_impl=0,
    twos_complement=False,
    golden_mathop=None,
):
    torch.manual_seed(0)
    torch.set_printoptions(precision=10)

    # The op's own signed domain, not generate_stimuli's positive-only format default,
    # which would leave the x<0 branch, the piecewise knees and the saturation tails
    # unreached. A KeyError means a new op arrived with no _OP_DOMAIN_REGISTRY entry:
    # register it rather than falling back to the positive-only default.
    # The domain has to hold for the whole pipeline, so for_op_pipeline resolves against
    # both formats and keeps the tighter — see its docstring for why both matter.
    if spec_A is None:
        spec_A = exclude_undefined(
            mathop,
            for_op_pipeline(mathop, formats.input_format, formats.output_format).spec_A,
        )

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_A,
    )

    # laneJO formal-equivalence witness-check hook (see test_sfpu_binary.py):
    # LANEJO_SRC_OVERRIDE holds a tensor replayed verbatim as src_A.
    import os as _lanejo_os

    _lanejo_src = _lanejo_os.environ.get("LANEJO_SRC_OVERRIDE")
    if _lanejo_src:
        _lanejo_t = torch.load(_lanejo_src).to(src_A.dtype).flatten()
        src_A = _lanejo_t.repeat(src_A.numel() // _lanejo_t.numel())

    generate_golden = get_golden_generator(UnarySFPUGolden)
    # golden_mathop: certification rows whose kernel selector rides a host
    # SfpuType (lane GW SFPARECIP probes on SfpuType::identity) key their
    # golden on their own MathOperation instead of the kernel-side mathop.
    golden_tensor = generate_golden(
        golden_mathop if golden_mathop is not None else mathop,
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
        test_name,
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(approx_mode),
            FAST_MODE(fast_mode),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=mathop),
            FRESH_CPP_IMPL(fresh_cpp_impl),
            ReciprocalImpl(reciprocal_impl),
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
            # Signed-integer ops whose inputs or results are negative need the
            # two's-complement L1 pack/unpack (the test_sfpu_binary_rsub_int32
            # convention); positive-only int ops keep the default.
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

    # laneJO witness-check hook (paired with LANEJO_SRC_OVERRIDE above).
    _lanejo_dump = _lanejo_os.environ.get("LANEJO_DUMP")
    if _lanejo_dump:
        torch.save({"src_A": src_A, "result": res_tensor}, _lanejo_dump)
    if _lanejo_os.environ.get("LANEJO_SKIP_ASSERT") == "1":
        return

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
    ), "Assert against golden failed"


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=["production", "fresh_cpp"])
@pytest.mark.parametrize("edge_values", [False, True], ids=["functional", "edges"])
def test_exp_fresh_cpp(fresh_cpp_impl, edge_values):
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    spec_A = None
    if edge_values:
        spec_A = edge_spec(
            MathOperation.Exp,
            formats.input_format,
            formats.output_format,
            specials=specials_safe(
                formats.input_format, formats.output_format, DestAccumulation.No
            ),
        )
        assert spec_A is not None

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        DestAccumulation.No,
        ApproximationMode.No,
        MathOperation.Exp,
        FastMode.No,
        [64, 64],
        spec_A=spec_A,
        fresh_cpp_impl=fresh_cpp_impl,
    )


@pytest.mark.parametrize(
    "fresh_cpp_impl",
    [0, 1, 2],
    ids=["production", "fresh_cpp", "fresh_cpp_tree"],
)
def test_sigmoid_appx_fresh_cpp(fresh_cpp_impl):
    """Three independently measurable selectors under one golden/tolerance
    contract: 0 = handwritten production kernel, 1 = fresh semantic 2-MAD
    cubic, 2 = fresh semantic 3-range magnitude dispatch tree (the
    LUT-eligible shape for -mtt-tensix-optimize-lut-select)."""
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        ApproximationMode.No,
        MathOperation.SigmoidAppx,
        FastMode.No,
        [64, 64],
        custom_atol=0.13,
        custom_rtol=0.05,
        fresh_cpp_impl=fresh_cpp_impl,
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=["production", "fresh_cpp"])
@pytest.mark.parametrize(
    "mathop",
    [MathOperation.UnaryMax, MathOperation.UnaryMin],
    ids=["unary_max", "unary_min"],
)
def test_unary_max_min_fresh_cpp(mathop, fresh_cpp_impl):
    """A/B the fresh semantic unary max/min against the production SFPLOADMACRO
    kernel with identical inputs (float contract: element tolerance + PCC)."""
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        fresh_cpp_impl=fresh_cpp_impl,
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=["production", "fresh_cpp"])
@pytest.mark.parametrize(
    "mathop",
    [
        MathOperation.UnaryMaxInt32,
        MathOperation.UnaryMinInt32,
        MathOperation.UnaryMaxUint32,
        MathOperation.UnaryMinUint32,
    ],
    ids=["unary_max_int32", "unary_min_int32", "unary_max_uint32", "unary_min_uint32"],
)
def test_unary_max_min_int_fresh_cpp(mathop, fresh_cpp_impl):
    """A/B the fresh semantic integer unary max/min against the production
    SFPLOADMACRO kernel with identical inputs (integer contract: exact)."""
    int_format = (
        DataFormat.UInt32 if mathop in _UINT32_INT_UNARY_OPS else DataFormat.Int32
    )
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(int_format, int_format),
        DestAccumulation.Yes,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        spec_A=_int_unary_stimuli_spec(mathop),
        fresh_cpp_impl=fresh_cpp_impl,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lane BR causal-tier lift: fresh semantic selectors paired with hand-shaped
# production bodies (raw-TTI streams, LUT l_reg idioms, hand pipelines,
# regression-flagged shapes).  Each test A/Bs impl 0 (production) against
# impl 1 (fresh typed C++) under the op's own golden and tolerance; formats
# and dest_acc mirror the op's measured sweep row exactly.


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=["production", "fresh_cpp"])
def test_ceil_fresh_cpp(fresh_cpp_impl):
    """A/B the fresh semantic ceil (typed 2^23 round + bump) against the
    production raw-TTI l_reg-pinned _ceil_body_ with identical inputs."""
    mathop = MathOperation.Ceil
    custom_atol, custom_rtol = CUSTOM_TOLERANCES.get(mathop, (None, None))
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
        fresh_cpp_impl=fresh_cpp_impl,
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=["production", "fresh_cpp"])
@pytest.mark.parametrize("edge_values", [False, True], ids=["functional", "edges"])
def test_eqz_fresh_cpp(fresh_cpp_impl, edge_values):
    """A/B the fresh semantic equal-zero (typed |v|==0 predicate) against the
    production all-raw-TTI calculate_comp float path with identical inputs.

    The edges leg puts exact +0.0 and -0.0 in the stimuli (op_edge_points for
    EqualZero) — the one pair of inputs the op is about, which the functional
    leg's uniform(-2, 2) draw reaches with probability ~0.  Both impls must
    return 1.0 for both zeros (ttnn golden: torch.eq(x, 0); production kernel
    comment: "handles ±0"), so the fresh body's sign handling is gated here,
    not just by review of its semantic statement.

    The edges leg runs dest_acc=Yes: -0.0 only survives to DEST on the
    32-bit unpack-to-dest path — on the 16-bit src-reg path the sign of
    zero is lost upstream and every zero reaches the SFPU as +0.0 (that is
    the registered Signbit _EDGE_KNOWN_DIVERGENCES xfail set), so a
    dest_acc=No edge probe cannot distinguish a sign-clearing body from a
    raw-bit one."""
    mathop = MathOperation.EqualZero
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    dest_acc = DestAccumulation.Yes if edge_values else DestAccumulation.No
    spec_A = None
    if edge_values:
        spec_A = edge_spec(
            mathop,
            formats.input_format,
            formats.output_format,
            specials=specials_safe(
                formats.input_format, formats.output_format, dest_acc
            ),
        )
        assert spec_A is not None
    custom_atol, custom_rtol = CUSTOM_TOLERANCES.get(mathop, (None, None))
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        spec_A=spec_A,
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
        fresh_cpp_impl=fresh_cpp_impl,
    )


_FLOAT_ZERO_COMP_MATHOPS = [
    MathOperation.NotEqualZero,
    MathOperation.LessThanZero,
    MathOperation.GreaterThanZero,
    MathOperation.LessThanEqualZero,
    MathOperation.GreaterThanEqualZero,
]


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=["production", "fresh_cpp"])
@pytest.mark.parametrize("edge_values", [False, True], ids=["functional", "edges"])
@pytest.mark.parametrize("mathop", _FLOAT_ZERO_COMP_MATHOPS, ids=lambda m: m.name)
def test_float_zero_comp_fresh_cpp(mathop, fresh_cpp_impl, edge_values):
    """A/B the fresh semantic float zero-comparisons against the production
    all-raw-TTI calculate_comp float path with identical inputs (laneED
    sem-only audit; the test_eqz_fresh_cpp / laneBR pattern extended to the
    five comparisons that hand kernel also implements).

    The production float comparison-to-zero body is one handwritten kernel
    for all six modes (metal ckernel_sfpu_comp.h calculate_comp: SFPSETSGN /
    SFPSETCC / SFPIADD-against-inf choreography, zero typed statements) — the
    corpus had it booked as "hand==semantic source".  These nodes give each
    remaining mode a semantic arm under the production golden and tolerance.

    Edge-leg rationale (dest_acc=Yes, exact +0.0/-0.0 stimuli) is inherited
    verbatim from test_eqz_fresh_cpp: -0.0 only survives to DEST on the
    32-bit unpack-to-dest path, and the zero-sign answer is exactly what
    separates a sign-blind body from the golden (torch treats -0.0 == 0, so
    ltz/gtz answer 0 and lez/gez answer 1 on both zeros)."""
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    dest_acc = DestAccumulation.Yes if edge_values else DestAccumulation.No
    spec_A = None
    if edge_values:
        spec_A = edge_spec(
            mathop,
            formats.input_format,
            formats.output_format,
            specials=specials_safe(
                formats.input_format, formats.output_format, dest_acc
            ),
        )
        assert spec_A is not None
    custom_atol, custom_rtol = CUSTOM_TOLERANCES.get(mathop, (None, None))
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        spec_A=spec_A,
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
        fresh_cpp_impl=fresh_cpp_impl,
    )


@pytest.mark.parametrize(
    "fresh_cpp_impl", [3, 1, 4], ids=["production_lut", "fresh_cpp", "licensed_cpp"]
)
def test_tanh_lut_fresh_cpp(fresh_cpp_impl):
    """A/B the byte-untouched production approximation-mode LUT tanh against
    the fresh semantic tanh body under one golden/tolerance contract (laneED
    sem-only audit).

    The hand arm is calculate_tanh<APPROXIMATION_MODE=true> — the raw
    3-region SFPLUT kernel (LReg0-2 coefficient choreography, imm16 table
    0x1DFF/0x481A/0xFF00), reached natively by building with approx mode ON
    (impl 3 has no selector branch for tanh, so it falls through to the
    production dispatch; the generic unary sweep skips Tanh at
    approx_mode:Yes, so before this node NO test ever raced that kernel).
    The sem arm is the fresh polynomial body at approx mode OFF — the approx
    flag is exactly the hand-vs-sem body axis; stimuli, golden (torch.tanh)
    and tolerance are identical.

    Tolerance derivation: the LUT is piecewise linear on |x| with breakpoints
    at 1.0 and 2.0 (the registered TanhDerivativeLut golden models the same
    table).  Its worst error against exact tanh is at |x| -> 1.0-:
    |0.90625 - tanh(1.0)| = |0.90625 - 0.76159| ~ 0.145, so atol = 0.16
    bounds the approximation class the production kernel actually ships
    (the SigmoidAppx/GeluAppx exact-golden + loose-tolerance precedent).

    impl 4 (lane GI, owner ratification 2026-08-24 item 2) is the LICENSED
    semantic arm: an independently fitted 3-piece minimax PWL on the same
    1.0/2.0 breakpoints (fresh_cpp/tanhlut_licensed.h), max |err| vs exact
    tanh 0.0411 vs the hand LUT's 0.1447 on the row's golden domain —
    equal-or-better proven exhaustively (laneGI accuracy oracle)."""
    approx = ApproximationMode.Yes if fresh_cpp_impl == 3 else ApproximationMode.No
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        approx,
        MathOperation.Tanh,
        FastMode.No,
        [64, 64],
        custom_atol=0.16,
        custom_rtol=0.05,
        fresh_cpp_impl=fresh_cpp_impl,
    )


@pytest.mark.parametrize(
    "fresh_cpp_impl", [3, 1, 4], ids=["production_lut6", "fresh_cpp", "licensed_cpp"]
)
def test_sigmoid_lut_fresh_cpp(fresh_cpp_impl):
    """A/B the byte-untouched LEGACY tt-llk 6-segment SFPLUTFP32 sigmoid hand
    kernel against the fresh semantic sigmoid body under one golden/tolerance
    contract (laneED sem-only audit).

    The hand arm (impl 3) is tt_llk_blackhole/common/inc/sfpu/
    ckernel_sfpu_sigmoid.h _calculate_sigmoid_ — lut2() over LReg0/1/2/4/5/6
    with the packed 6-segment coefficient table — a kernel the corpus
    manifest records as class D-ABSENT (zero test-source inclusion, zero
    dispatch anywhere under tests/): a genuinely distinct handwritten sigmoid
    that no node had ever raced.  It is NOT the 3-segment metal sigmoid_appx
    (raced by the sigmoidappx row) and NOT the typed metal calculate_sigmoid
    (raced by the sigmoid row).  The sem arm is the fresh sigmoid body.

    Tolerance derivation: golden is exact sigmoid (the SigmoidAppx
    precedent).  The 6-segment table's worst error against exact sigmoid is
    at the |x| = 4 knee: |0.9998 - sigmoid(4.0)| ~ 0.018, so
    atol = 0.05 bounds the table's approximation class with margin.

    Formats mirror the sigmoid family's other corr nodes (Float32->Float32,
    the standard-profile BH-legal dest_acc:No combination; the fp16-coded
    table itself dominates the error budget on any float pipeline).

    impl 4 (lane GI, owner ratification 2026-08-24 item 2) is the LICENSED
    semantic arm: an independently fitted 4-region poly-leaf magnitude tree
    (fresh_cpp/sigmoid_lut_licensed.h), max |err| vs exact sigmoid 0.0051
    vs the hand table's 0.0180 on the row's golden domain — equal-or-better
    proven exhaustively (laneGI accuracy oracle)."""
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        DestAccumulation.No,
        ApproximationMode.No,
        MathOperation.Sigmoid,
        FastMode.No,
        [64, 64],
        custom_atol=0.05,
        custom_rtol=0.05,
        fresh_cpp_impl=fresh_cpp_impl,
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 4], ids=["production", "licensed_cpp"])
def test_gelu_licensed_cpp(fresh_cpp_impl):
    """A/B the byte-untouched production ACCURATE bf16 gelu hand kernel
    against the gelu-255 LICENSED semantic arm under one golden/tolerance
    contract (lane GI, owner ratification 2026-08-24 item 2: "gelu at the
    hand kernel's 255-ulp contract").

    The hand arm (impl 0) is calculate_gelu_piecewise — the 4-region
    erf/erfc CDF kernel.  Its MEASURED accuracy contract on the row's
    golden domain (all finite bf16; the Gelu Gaussian stimulus is
    untruncated): max_abs 0.00777, max pure-bf16-ULP 253.19 vs the fp64
    golden — the same 253.19 the fitter board's ttnn_pure records (the
    255-ulp contract is real on this domain: torch-saturation flush +
    staircase vs an exact golden).

    The licensed arm (impl 4, fresh_cpp/gelu_255_licensed.h) keeps the hand
    kernel's region structure and spends the licensed slack on polynomial
    depth (deg-1 exp frac refine, linear H-correction, no grid snap, deg-11
    core).  Composite dominance (max-abs AND max-pure-ULP <= hand, all
    finite bf16) proven in laneGI accuracy-oracle/gelu255_verify.c — the
    licensed arm's extrema exactly tie the hand kernel's.

    Stimuli, golden (exact torch gelu) and tolerance (Float16_b default)
    are identical between arms and to the gelu-fresh family."""
    mathop = MathOperation.Gelu
    custom_atol, custom_rtol = CUSTOM_TOLERANCES.get(mathop, (None, None))
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
        fresh_cpp_impl=fresh_cpp_impl,
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 4], ids=["production", "licensed_cpp"])
def test_gelu_appx_licensed_cpp(fresh_cpp_impl):
    """A/B the byte-untouched production GeluAppx hand kernel against the
    LICENSED semantic arm under one golden/tolerance contract (lane GI,
    owner ratification 2026-08-24 item 2:
    review_records/OWNER-RATIFICATION-arm-preference-lut-license.md).

    The hand arm (impl 0) is calculate_gelu_appx — the 6-segment SFPLUTFP32
    FP16 TABLE1 kernel (lut2_sign over LReg0/1/2/4/5/6) computing the even
    part gelu(x) - 0.5x, plus 0.5x.  Its measured accuracy contract on the
    row's golden domain (all finite bf16 — the GeluAppx stimulus Gaussian is
    untruncated): max |err| vs exact gelu 0.0234 raw / 0.0239 bf16-stored,
    at x ~ 0.249 (the [0,0.5) segment is a chord through the origin).

    The licensed arm (impl 4, fresh_cpp/gelu_appx_licensed.h) is an
    independently fitted 4-region poly-leaf magnitude tree at max |err|
    0.0097 — equal-or-better than the hand kernel, proven exhaustively over
    the bf16 grid and the fp32 stimulus sweep (laneGI accuracy oracle).

    Stimuli, golden (exact torch gelu) and tolerance (the registered
    GeluAppx CUSTOM_TOLERANCES) are identical between arms and identical to
    the test_causal_lift_fresh_cpp[GeluAppx-*] family this row's sem arm
    previously pointed at (torch.manual_seed(0) + same registry spec =>
    byte-identical stimuli)."""
    mathop = MathOperation.GeluAppx
    custom_atol, custom_rtol = CUSTOM_TOLERANCES.get(mathop, (None, None))
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        DestAccumulation.No,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
        fresh_cpp_impl=fresh_cpp_impl,
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=["production", "fresh_cpp"])
def test_clamp_fresh_cpp(fresh_cpp_impl):
    """A/B the fresh semantic clamp (typed float bounds) against the production
    fp16-bit-punned _calculate_clamp_ with identical inputs and bounds."""
    mathop = MathOperation.Clamp
    custom_atol, custom_rtol = CUSTOM_TOLERANCES.get(mathop, (None, None))
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        DestAccumulation.No,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
        fresh_cpp_impl=fresh_cpp_impl,
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=["production", "fresh_cpp"])
def test_hardtanh_fresh_cpp(fresh_cpp_impl):
    """A/B the fresh semantic hardtanh (typed clamp) against the production
    chained add-then-zero-select _calculate_hardtanh_ with identical inputs."""
    mathop = MathOperation.Hardtanh
    custom_atol, custom_rtol = CUSTOM_TOLERANCES.get(mathop, (None, None))
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        DestAccumulation.No,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
        fresh_cpp_impl=fresh_cpp_impl,
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=["production", "fresh_cpp"])
def test_tanh_fresh_cpp(fresh_cpp_impl):
    """A/B the fresh semantic tanh (same Sollya polynomial, one datum per row,
    all-local coefficients) against the production two-datum hand software
    pipeline with programmed constant registers, identical inputs."""
    mathop = MathOperation.Tanh
    custom_atol, custom_rtol = CUSTOM_TOLERANCES.get(mathop, (None, None))
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
        fresh_cpp_impl=fresh_cpp_impl,
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=["production", "fresh_cpp"])
def test_tanh_derivative_lut_fresh_cpp(fresh_cpp_impl):
    """A/B the fresh semantic LUT-contract tanh' (typed 3-region piecewise —
    the golden's own model — then 1-t^2) against the production l_reg-pinned
    raw-SFPLUT _calculate_tanh_derivative_ with identical inputs."""
    mathop = MathOperation.TanhDerivativeLut
    custom_atol, custom_rtol = CUSTOM_TOLERANCES.get(mathop, (None, None))
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        DestAccumulation.No,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
        fresh_cpp_impl=fresh_cpp_impl,
    )


# Lane CM fitted-kernel placeholders (tt-polynomial-fitter frontier
# selections, PLACEHOLDER-PENDING-UPSTREAM-MERGE; provenance headers in
# fresh_cpp/*_fitted.h).  Dedicated family: all-new node ids, bf16 corr
# contract matching the fits' bf16 target; impl 2 = fitted body, impl 0 =
# production hand kernel (the row's hand arm).  TanhDerivative (not the
# -Lut op) carries the fitted tanh_bw: its golden is the true derivative,
# while the -Lut row's golden IS the production LUT and would fail a more
# accurate kernel.
_FITTED_CPP_OPS = [
    MathOperation.Tanh,
    MathOperation.Sigmoid,
    MathOperation.Gelu,
    MathOperation.TanhDerivative,
    # Lane CR wave 2 (same contract; provenance in fresh_cpp/*_fitted.h).
    MathOperation.Digamma,
    MathOperation.Lgamma,
    MathOperation.Polygamma,
    MathOperation.I0,
    MathOperation.I1,
    MathOperation.Mish,
    MathOperation.Log,
    MathOperation.Log1p,
    MathOperation.Exp,
    MathOperation.Rsqrt,
    MathOperation.Celu,
    MathOperation.Elu,
    MathOperation.Selu,
    # Lane CW wave 3 (rlibm rounding-interval refits of the CR wave-2
    # honest-outs: threshold at the contract params 5.0/10.0, expm1 over the
    # contract domain U[-5,5], acosh with the x=1 branch point exact;
    # provenance in fresh_cpp/*_fitted.h).
    MathOperation.Threshold,
    MathOperation.Expm1,
    MathOperation.Acosh,
]


@pytest.mark.parametrize("fresh_cpp_impl", [0, 2], ids=["production", "fitted_cpp"])
@pytest.mark.parametrize("mathop", _FITTED_CPP_OPS, ids=lambda m: m.name)
def test_fitted_cpp(mathop, fresh_cpp_impl):
    """A/B the tt-polynomial-fitter frontier-selected fitted bodies
    (fresh_cpp/*_fitted.h: silicon-measured coefficient sets evaluated in
    the measured kernel's arithmetic order, plain typed SFPI) against the
    production kernels with identical inputs, golden, and tolerance."""
    custom_atol, custom_rtol = CUSTOM_TOLERANCES.get(mathop, (None, None))
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
        fresh_cpp_impl=fresh_cpp_impl,
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=["production", "fresh_cpp"])
def test_silu_fresh_cpp(fresh_cpp_impl):
    """A/B the fresh semantic silu (identical piecewise sigmoid math, plain
    loop and locals) against the production POLYVAL5-macro _calculate_silu_
    with identical inputs."""
    mathop = MathOperation.Silu
    custom_atol, custom_rtol = CUSTOM_TOLERANCES.get(mathop, (None, None))
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
        fresh_cpp_impl=fresh_cpp_impl,
    )


# Batch 2 (Lane BR): one shared dedicated corr family for the remaining
# causal-tier lifts.  Formats/dest_acc mirror each op's measured sweep-row
# corr leg; golden and tolerance are the op's own (same machinery as the
# swept production node).
_CAUSAL_LIFT_B2_F32_OPS = [
    MathOperation.Fmod,
    MathOperation.Remainder,
    MathOperation.Xielu,
    MathOperation.UnaryPower,
    MathOperation.Expm1,
    # Batch 3 (F32 corr rows).
    MathOperation.Sigmoid,
    MathOperation.Cbrt,
    MathOperation.Softplus,
    MathOperation.Expm1Cw,
    MathOperation.I1,
    # Storm S4 (F32 corr rows; fresh bodies in fresh_cpp/<op>.h).
    MathOperation.Rdiv,
    MathOperation.Rpow,
    MathOperation.Selu,
    MathOperation.Sign,
    # Storm lane S1 (fresh_cpp/<op>.h bodies; formats mirror each op's swept
    # corr leg).
    MathOperation.Add1,
    # CastFp32ToFp16a moved to its own test_cast_fp32_to_fp16a_fresh_cpp (lane CX
    # golden re-spec): this family's Float32/dest_acc=No pipeline bf16-truncates
    # the operand, making the cast the identity and its rounding dead code.
    # Storm S2 (agent/storm-s2, fresh_cpp/<op>.h canonical bodies; F32 corr rows).
    MathOperation.Erf,
    MathOperation.Erfc,
    MathOperation.Erfinv,
    MathOperation.Digamma,
    MathOperation.Hardmish,
    MathOperation.Hardshrink,
    MathOperation.Heaviside,
    # Storm S5 (F32 corr rows).
    MathOperation.Softshrink,
    MathOperation.Softsign,
    MathOperation.UnaryGe,
    # laneED sem-only audit: GeluAppx's production body is the 6-segment
    # SFPLUTFP32 hand kernel calculate_gelu_appx (NOT a typed body) — this
    # node pair gives it the semantic arm it never had.  Golden = exact gelu,
    # tolerance = the registered GeluAppx CUSTOM_TOLERANCES entry; the fresh
    # exact-gelu body passes the loose contract trivially.
    MathOperation.GeluAppx,
]
_CAUSAL_LIFT_B2_F16B_OPS = [
    MathOperation.Log,
    MathOperation.Sqrt,
    MathOperation.Rsqrt,
    # Batch 3 (F16b corr rows).
    MathOperation.Hardsigmoid,
    MathOperation.Gelu,
    # Storm S4 (F16b corr rows; fresh bodies in fresh_cpp/<op>.h — ReluMax is
    # the relu op file's vehicle; Floor/Trunc/Frac are the rounding_ops file's
    # remaining raw-TTI variants, Ceil's fresh row predates the storm).
    MathOperation.ReluMax,
    MathOperation.Floor,
    MathOperation.Trunc,
    MathOperation.Frac,
    # Storm lane S3 (fresh_cpp/ per-op headers).
    MathOperation.Log1p,
    # Storm lane S1.
    MathOperation.Abs,
    MathOperation.Celu,
    # Storm S2 (agent/storm-s2; F16b corr rows, mirroring each op's sweep row).
    MathOperation.Elu,
    MathOperation.Exp2,
    MathOperation.Fill,
    # Storm S5 (F16b corr rows).
    MathOperation.Square,
    MathOperation.Tanhshrink,
    MathOperation.Threshold,
    MathOperation.Acosh,
]


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=["production", "fresh_cpp"])
@pytest.mark.parametrize(
    "mathop",
    _CAUSAL_LIFT_B2_F32_OPS + _CAUSAL_LIFT_B2_F16B_OPS,
    ids=lambda m: m.name,
)
def test_causal_lift_fresh_cpp(mathop, fresh_cpp_impl):
    """A/B the batch-2 fresh semantic bodies (fresh_cpp_operations.h: same
    golden-math algorithm as the production kernel, every constant a plain
    local, no programmed constant registers / builtin-MAD pins / exponent-
    shift strength reductions) against the hand-shaped production kernels
    with identical inputs, golden, and tolerance."""
    fmt = (
        DataFormat.Float32
        if mathop in _CAUSAL_LIFT_B2_F32_OPS
        else DataFormat.Float16_b
    )
    custom_atol, custom_rtol = CUSTOM_TOLERANCES.get(mathop, (None, None))
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(fmt, fmt),
        DestAccumulation.No,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        custom_atol=custom_atol,
        custom_rtol=custom_rtol,
        fresh_cpp_impl=fresh_cpp_impl,
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=["production", "fresh_cpp"])
@pytest.mark.parametrize("edge_values", [False, True], ids=["functional", "edges"])
def test_cast_fp32_to_fp16a_fresh_cpp(fresh_cpp_impl, edge_values):
    """A/B the fresh cast body against the production sfpi::convert kernel on
    the one pipeline where the cast is observable, under the hardware-semantics
    golden.

    Golden re-spec (lane CX, owner-signed 2026-08-20): the golden states the
    HARDWARE cast semantics -- SFP_STOCH_RND FP32_TO_FP16A rnd_mode=0 rounds
    half-AWAY on the 13 discarded mantissa bits, flushes exponent-0 inputs
    (denormals and both zeros) to +0.0, and collapses every NaN payload to
    signed infinity -- machine-checked by lane CT's exhaustive 2^32 proof
    against the pinned craq oracle (sfpi-gcc agent/cast-peephole-harvest,
    gcc/config/riscv/tt/proofs/cast-fp16a-rne/).

    Reachability (the eqz -0.0 / lane CL discipline): both legs run
    Float32->Float32 at dest_acc=Yes because that is exactly the driver's own
    unpack_to_dest condition -- the only pipeline that delivers the low 13
    mantissa bits (and denormals, -0.0, NaN payloads) to the SFPU.  At
    dest_acc=No the operand is bf16-truncated upstream on BOTH the golden and
    device paths, the cast is the identity on everything that arrives, and the
    rounding logic of either impl is dead code -- which is how the old
    software-RNE golden survived unfalsified (lane CT's harness audit).

    The edges leg lands exactly on the rounding-visible inputs: ties of both
    kept-LSB parities (the even tie is the RNE-vs-half-away discriminator),
    midpoint neighbours, the exponent-carry tie, denormals of both signs,
    +-0.0, and (via specials_safe) NaN and +-inf.

    Comparison is exact (custom_atol=custom_rtol=0.0, the harness's own strict
    form): the op is a pure bit-lattice function, losslessly representable in
    an fp32 dest, and the default Float32 tolerance is blind to the
    2^-11-relative rounding differences this row exists to price.  Known
    comparator limit: torch.isclose equates -0.0 with +0.0, so the single
    exact -0.0 input of CT's 4,097-case sign-loss class is enforced by the
    golden spec (and the archived legacy-body failing differential) rather
    than by this assert; the other 4,096 sign-loss inputs are negative
    denormals and DO fail loudly (nonzero vs +0)."""
    mathop = MathOperation.CastFp32ToFp16a
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    dest_acc = DestAccumulation.Yes
    spec_A = None
    if edge_values:
        spec_A = edge_spec(
            mathop,
            formats.input_format,
            formats.output_format,
            specials=specials_safe(
                formats.input_format, formats.output_format, dest_acc
            ),
        )
        assert spec_A is not None
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        spec_A=spec_A,
        custom_atol=0.0,
        custom_rtol=0.0,
        fresh_cpp_impl=fresh_cpp_impl,
    )


# Lane GW SFPARECIP-mode boundary stimuli: every branch edge of the doc
# functional model (ApproxExp: denormal->1.0 / <0.015625 / <0.6953125 / <2 /
# tail; ApproxRecip: denormal->Inf / normal / >=2**126 -> 0), both signs,
# +-0 (the custom spec zero-fills the remainder, so +0 dominates the tile).
_ARECIP_PROBE_BOUNDARIES = [
    float.fromhex("0x0.0p+0"),  # 0x00000000
    float.fromhex("-0x0.0p+0"),  # 0x80000000
    float.fromhex("0x1.0000000000000p-149"),  # 0x00000001
    float.fromhex("-0x1.0000000000000p-149"),  # 0x80000001
    float.fromhex("0x1.fffffc0000000p-127"),  # 0x007FFFFF
    float.fromhex("-0x1.fffffc0000000p-127"),  # 0x807FFFFF
    float.fromhex("0x1.0000000000000p-126"),  # 0x00800000
    float.fromhex("-0x1.0000000000000p-126"),  # 0x80800000
    float.fromhex("0x1.fffffe0000000p-7"),  # 0x3C7FFFFF
    float.fromhex("-0x1.fffffe0000000p-7"),  # 0xBC7FFFFF
    float.fromhex("0x1.0000000000000p-6"),  # 0x3C800000
    float.fromhex("-0x1.0000000000000p-6"),  # 0xBC800000
    float.fromhex("0x1.63fffe0000000p-1"),  # 0x3F31FFFF
    float.fromhex("-0x1.63fffe0000000p-1"),  # 0xBF31FFFF
    float.fromhex("0x1.6400000000000p-1"),  # 0x3F320000
    float.fromhex("-0x1.6400000000000p-1"),  # 0xBF320000
    float.fromhex("0x1.0000000000000p+0"),  # 0x3F800000
    float.fromhex("-0x1.0000000000000p+0"),  # 0xBF800000
    float.fromhex("0x1.fffffe0000000p+0"),  # 0x3FFFFFFF
    float.fromhex("-0x1.fffffe0000000p+0"),  # 0xBFFFFFFF
    float.fromhex("0x1.0000000000000p+1"),  # 0x40000000
    float.fromhex("-0x1.0000000000000p+1"),  # 0xC0000000
    float.fromhex("0x1.921fb60000000p+1"),  # 0x40490FDB
    float.fromhex("-0x1.921fb60000000p+1"),  # 0xC0490FDB
    float.fromhex("0x1.fffffe0000000p+125"),  # 0x7E7FFFFF
    float.fromhex("-0x1.fffffe0000000p+125"),  # 0xFE7FFFFF
    float.fromhex("0x1.0000000000000p+126"),  # 0x7E800000
    float.fromhex("-0x1.0000000000000p+126"),  # 0xFE800000
    float.fromhex("0x1.fffffe0000000p+127"),  # 0x7F7FFFFF
    float.fromhex("-0x1.fffffe0000000p+127"),  # 0xFF7FFFFF
]


@pytest.mark.parametrize(
    "stimuli_kind", ["core", "boundaries"], ids=["core", "boundaries"]
)
@pytest.mark.parametrize(
    "mathop",
    [MathOperation.ApproxExpProbe, MathOperation.ApproxCondRecipProbe],
    ids=lambda m: m.name,
)
def test_arecip_mode_probe_cpp(mathop, stimuli_kind):
    """Lane GW ISA-unlock certification rows for the two SFPARECIP modes the
    sfpi surface ships but nothing exercised (GS-3): Mod1=2 EXP and Mod1=1
    COND_RECIP.

    The kernel body is the bare mode (fresh_cpp/arecipprobe.h: approx_exp /
    approx_recip(..., RecipMode::IfNegative)); the golden IS the ISA
    functional model (tt-isa-documentation BlackholeA0 SFPARECIP.md),
    transcribed mechanically into golden_generators.py.  Comparison is EXACT
    (atol=rtol=0) on the Float32/dest_acc=Yes pipeline — the lane-CX
    reachability discipline: the only pipeline that delivers and returns
    full fp32 bit patterns (denormals and -0.0 included).

    Run against the lane-GW extended craq-sim, a pass certifies the sim
    extension's transcription; run on silicon, a pass adjudicates the
    doc-vs-sim SIM GAP (the where-adjudication precedent: silicon is the
    authority).  The 'boundaries' leg lands exactly on every branch edge of
    the doc model, both signs; the 'core' leg sweeps the useful ranges.
    COND_RECIP's compiler contract is VB == VC (recip where the SOURCE is
    negative, sign NOT rejoined) — the raw-word encoding the lane-GW
    sfpi-gcc change emits."""
    if TestConfig.CHIP_ARCH != ChipArchitecture.BLACKHOLE:
        pytest.skip(reason="SFPARECIP EXP/COND_RECIP probes are BH-only")
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    if stimuli_kind == "boundaries":
        spec_A = StimuliSpec.custom(values=_ARECIP_PROBE_BOUNDARIES)
    else:
        # Core sweep: the EXP mode's useful domain |x| < 2 plus the tail and
        # recip-relevant magnitudes; signs exercise the sign-preserve (EXP)
        # and negative-only (COND_RECIP) clauses.
        spec_A = StimuliSpec.uniform(low=-4.0, high=4.0)
    # Kernel selector: hosted on SfpuType::identity (generic init; the R7
    # LLK-pristine rule forbids extending the metal SfpuType enum), impl 5 =
    # EXP probe, impl 6 = COND_RECIP probe; the golden keys on the probe's
    # own MathOperation.
    impl = 5 if mathop is MathOperation.ApproxExpProbe else 6
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        DestAccumulation.Yes,
        ApproximationMode.No,
        MathOperation.Identity,
        FastMode.No,
        [64, 64],
        spec_A=spec_A,
        custom_atol=0.0,
        custom_rtol=0.0,
        fresh_cpp_impl=impl,
        golden_mathop=mathop,
    )


# Storm lane S1: int32 causal-tier lifts (fresh_cpp/absint32.h /
# fresh_cpp/bitwisenot.h).  Same stimuli, golden, and exact integer contract
# as the swept production node (test_eltwise_unary_sfpu_int32_signed).
@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=["production", "fresh_cpp"])
@pytest.mark.parametrize(
    "mathop",
    [MathOperation.AbsInt32, MathOperation.BitwiseNot],
    ids=lambda m: m.name,
)
def test_causal_lift_int32_fresh_cpp(mathop, fresh_cpp_impl):
    """A/B the storm-S1 fresh typed int32 bodies (value-level negation-select /
    two's-complement inversion, typed vInt Dst views) against the production
    kernels with identical inputs and the exact integer golden."""
    formats = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)
    # Both signs, zero straddled deterministically; magnitudes stay far from
    # INT32_MIN (abs(INT32_MIN) is unrepresentable) and from INT32_MAX —
    # identical to the swept production node's stimuli.
    spec_A = StimuliSpec.uniform(low=-1_000_000.0, high=1_000_000.0)
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        DestAccumulation.Yes,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        spec_A=spec_A,
        twos_complement=True,
        fresh_cpp_impl=fresh_cpp_impl,
    )


# Storm S5: dedicated corr family for the unaryshift row's fresh lift.  The
# existing int sweep (test_eltwise_unary_sfpu_int) cannot grow a fresh_cpp_impl
# axis without renaming its swept node ids (the retype-tripwire class), so the
# A/B gets its own nodes.  Same driver, formats, dest_acc, stimuli, and exact
# int golden as the swept LeftShift node.
@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=["production", "fresh_cpp"])
def test_unary_shift_fresh_cpp(fresh_cpp_impl):
    formats = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        DestAccumulation.Yes,
        ApproximationMode.No,
        MathOperation.LeftShift,
        FastMode.No,
        [64, 64],
        spec_A=_int_unary_stimuli_spec(MathOperation.LeftShift),
        fresh_cpp_impl=fresh_cpp_impl,
    )


@pytest.mark.parametrize(
    "reciprocal_impl", [0, 1, 2], ids=["production", "semantic", "semantic-ilv2"]
)
@pytest.mark.parametrize(
    "formats,dest_acc",
    [
        (
            InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
            DestAccumulation.No,
        ),
        (
            InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
            DestAccumulation.Yes,
        ),
    ],
    ids=["bf16-dst16", "fp32-dst32"],
)
@pytest.mark.parametrize("approx_mode", [ApproximationMode.No, ApproximationMode.Yes])
def test_reciprocal_semantic(formats, dest_acc, approx_mode, reciprocal_impl: int):
    """A/B the typed reciprocal across its BH format and accuracy paths."""
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        approx_mode,
        MathOperation.Reciprocal,
        FastMode.No,
        [32, 32],
        reciprocal_impl=reciprocal_impl,
    )


@pytest.mark.parametrize("reciprocal_impl", [0, 1], ids=["production", "semantic"])
def test_reciprocal_semantic_edges(reciprocal_impl: int):
    """Drive zero and the registered reciprocal domain boundaries."""
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        DestAccumulation.Yes,
        ApproximationMode.No,
        MathOperation.Reciprocal,
        FastMode.No,
        [32, 32],
        spec_A=edge_spec(
            MathOperation.Reciprocal,
            formats.input_format,
            formats.output_format,
            # Match the existing edge-suite contract.  Reciprocal is not in
            # SPECIALS_READY_OPS: its current kernel intentionally does not
            # preserve NaN, so forcing IEEE specials makes production fail its
            # own golden and cannot be an A/B correctness gate.
            specials=False,
        ),
        reciprocal_impl=reciprocal_impl,
    )


@pytest.mark.parametrize(
    "reciprocal_impl,label",
    [(0, "production"), (1, "semantic"), (2, "semantic-ilv2")],
)
def test_reciprocal_device_profile(perf_report, reciprocal_impl: int, label: str):
    """Profile one accurate BF16 reciprocal body, excluding datacopy."""
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    input_dimensions = [32, 32]
    spec_A = exclude_undefined(
        MathOperation.Reciprocal,
        for_op_pipeline(
            MathOperation.Reciprocal,
            formats.input_format,
            formats.output_format,
        ).spec_A,
    )
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_A,
    )

    configuration = PerfConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(ApproximationMode.No),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=MathOperation.Reciprocal),
            ReciprocalImpl(reciprocal_impl),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(1),
            NUM_TILES_IN_BLOCK(1),
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
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
    )
    configuration.run(perf_report, run_count=1)
    rows = perf_report.frame()
    rows = rows[rows["marker"] == "RECIPROCAL_BODY"]
    assert len(rows) >= 1, rows.to_string(index=False)
    cycles = float(rows.iloc[-1]["mean(MATH_ISOLATE)"])
    assert cycles > 0
    print(f"RECIPROCAL_DEVICE_PROFILE impl={label} body_cycles={cycles:.2f}")


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
