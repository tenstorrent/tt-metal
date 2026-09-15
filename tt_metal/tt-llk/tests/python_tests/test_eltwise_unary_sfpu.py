# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


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
)
from helpers.sfpu_domains import (
    _UNARY_OPS_NOT_SWEPT,
    SHIFT_EDGE_AMOUNTS,
    SPECIALS_READY_OPS,
    edge_spec,
    exclude_undefined,
    for_op_pipeline,
    negative_zero_delivered,
    op_edge_points,
    op_threshold,
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
        MathOperation.Sqrt: _cat_b_divergences(negative_zero_delivered),
        MathOperation.Rsqrt: _cat_b_divergences(negative_zero_delivered),
    }
)

# The four whose divergence needs the cat-B probe to be sent. Their xfails are conditional on
# specials surviving the NaN-sign gate; see where the marker is applied.
_CAT_B_DERIVED_DIVERGENCES = frozenset(
    {
        MathOperation.Reciprocal,
        MathOperation.SqrtCustom,
        MathOperation.Sqrt,
        MathOperation.Rsqrt,
    }
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
    "unpack-to-dest combinations, the only ones where a real -0.0 reaches the LREG.",
    MathOperation.Rsqrt: "rsqrt(-0) returns NaN; IEEE and the golden give -inf. Same cause "
    "and same unpack-to-dest scoping as Sqrt.",
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
