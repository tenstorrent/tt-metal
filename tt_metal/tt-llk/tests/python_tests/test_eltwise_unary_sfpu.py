# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import math
import struct
from itertools import product

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.constraints import distinct_dest_accumulation_modes
from helpers.data_format_inference import effective_dest_acc, unpacks_to_dest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    FusedSort,
    MathOperation,
    PerfRunType,
    StableSort,
    Transpose,
    format_dict,
)
from helpers.param_config import (
    build_param_id,
    input_output_formats,
    parametrize,
)
from helpers.perf.core import ALL_PERF_RUN_TYPES, create_test_or_perf_config
from helpers.sfpu_accuracy_budget import accuracy_contract
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
    swept_unary_ops,
    unpacks_to_dest,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import (
    DistributionKind,
    StimuliSpec,
    calculate_tile_and_face_counts,
    generate_stimuli,
)
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    DEST_SYNC,
    FAST_MODE,
    FUSED_SORT,
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    SFPU_POLYGAMMA_ORDER,
    SFPU_RELU_MAX_THRESHOLD,
    SFPU_RELU_MIN_INT_THRESHOLD,
    SFPU_SHIFT_AMOUNT,
    STABLE_SORT,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    DestSync,
)
from helpers.tile_constants import MAX_FACE_R_DIM, MAX_NUM_FACES
from helpers.ulp import FLUSH_SUBNORMAL_OUTPUTS
from helpers.utils import passed_test

SUPPORTED_FAST_MODE_OPS = [
    MathOperation.Rsqrt,
    MathOperation.Sqrt,
]

# The unary op sweep: two coverage profiles, one test. Every op takes its stimuli from its
# registered domain in _OP_DOMAIN_REGISTRY; the profiles differ only in how much of the
# format/mode matrix each op is worth. BROAD_SWEEP_OPS gets the full format matrix, both
# approximation modes and the multi-block tile shape; STANDARD_SWEEP_OPS is every other
# registered unary op, on Float16_b + Float32 with approximation off. To opt an op out of
# the sweep entirely, list it in sfpu_domains._UNARY_OPS_NOT_SWEPT.

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
    swept_unary_ops() - set(BROAD_SWEEP_OPS),
    key=lambda op: op.name,
)

BROAD_FORMATS = input_output_formats(
    [
        DataFormat.Float32,
        DataFormat.Float16_b,
        DataFormat.Bfp8_b,
    ]
)

# The standard profile keeps the two formats that exercise the SFPU's own math without
# a block-float or 16-bit-exponent path in the way: bf16 for the 16-bit dst rounding
# and fp32 for full precision.
STANDARD_FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32])

# 128x256 is 32 tiles, so it runs several Dest blocks and swaps Dest halves between them.
# The single-block path (64x64 fits in one) is left to the standard profile and the 64x64
# edge sweeps below.
BROAD_DIMENSIONS = [[128, 256]]
STANDARD_DIMENSIONS = [[64, 64]]

# Bfp4_b is only exercised as an input format, so the input is pinned to Bfp4_b here
# rather than building the full matrix and skipping the 6 non-Bfp4_b-input combos.
FORMATS_BFP4_B = [
    InputOutputFormat(DataFormat.Bfp4_b, output_format)
    for output_format in [
        DataFormat.Float16_b,
        DataFormat.Bfp8_b,
        DataFormat.Bfp4_b,
    ]
]


def _sweep_params(formats, mathops, approx_modes, input_dimensions):
    """Build (formats, approx_mode, mathop, fast_mode, dest_acc, input_dimensions) tuples.

    Fast-mode-capable ops are swept with FastMode.No and FastMode.Yes; every other op
    runs with FastMode.No only. dest_acc sweeps both values, except where TestConfig
    promotes No to Yes for the format combination; those run as Yes only.
    """
    fast_ops = [op for op in mathops if op in SUPPORTED_FAST_MODE_OPS]
    non_fast_ops = [op for op in mathops if op not in SUPPORTED_FAST_MODE_OPS]
    params = []
    for fmt in formats:
        dest_accs = distinct_dest_accumulation_modes(
            fmt, [DestAccumulation.No, DestAccumulation.Yes]
        )
        params += product(
            [fmt],
            approx_modes,
            fast_ops,
            [FastMode.No, FastMode.Yes],
            dest_accs,
            input_dimensions,
        )
        params += product(
            [fmt],
            approx_modes,
            non_fast_ops,
            [FastMode.No],
            dest_accs,
            input_dimensions,
        )
    return params


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


def _assert_dest_acc_ids_match_kernels():
    """Every sweep id must name the Dest width its kernel is built with.

    TestConfig promotes dest_acc to Yes for an outlier format combination, so a dest_acc:No id for
    one would run the dest_acc:Yes kernel while the golden and tolerances assume No.
    """
    promoted = sorted(
        {
            f"{formats.input_format.name}->{formats.output_format.name}"
            for formats, _, _, _, dest_acc, _ in UNARY_SWEEP_PARAMS
            if effective_dest_acc(formats.input_format, formats.output_format, dest_acc)
            != dest_acc
        }
    )
    assert not promoted, (
        "These format combinations are swept with a dest_acc the kernel does not run with "
        f"(TestConfig promotes it): {promoted}"
    )


_assert_dest_acc_ids_match_kernels()


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


def bh_unless_fp32_skip_reason(formats, dest_acc, arch):
    """Why a driver calling :func:`_skip_bh_unless_fp32` skips this variant on *arch*,
    or ``None``: Blackhole with dest_acc=No only supports Float32->Float32. The exact-op
    guard in test_sfpu_accuracy_budget.py asks it at the table's architecture."""
    if (
        dest_acc == DestAccumulation.No
        and arch == ChipArchitecture.BLACKHOLE
        and formats != InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    ):
        return "This combination is not supported on BH architecture"
    return None


def _skip_bh_unless_fp32(formats, dest_acc):
    reason = bh_unless_fp32_skip_reason(formats, dest_acc, TestConfig.CHIP_ARCH)
    if reason:
        pytest.skip(reason=reason)


_UNARY_SWEEP_ARGNAMES = (
    "formats",
    "approx_mode",
    "mathop",
    "fast_mode",
    "dest_acc",
    "input_dimensions",
)


@pytest.mark.nightly
@pytest.mark.parametrize("dest_sync", [DestSync.Half])
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
    dest_sync: DestSync,
):
    """Every float unary SFPU op, over its registered domain.

    Stimuli come from the per-op registry for every op; the sweep envelope (which profile
    the op is in) is the only thing that varies. See BROAD_SWEEP_OPS.
    """
    broad = mathop in BROAD_SWEEP_OPS

    # The broad profile's formats all run on Blackhole at dest_acc=No; the standard
    # profile allows only Float32->Float32 there.
    if not broad:
        _skip_bh_unless_fp32(formats, dest_acc)

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        approx_mode,
        mathop,
        fast_mode,
        input_dimensions,
        dest_sync=dest_sync,
    )


# Deliberate edge values. The random sweep above lands near knees and poles but never on
# them; this lands on them, using the shared metadata in sfpu_domains: domain singularities
# (cat A), op knees and rounding ties (cat D), and IEEE specials where the pipeline can carry
# them (cat B). It is one spec_A on the same driver, and edge_spec() is keyed off the op, so
# adding an op to the registry auto-enrols it here. The format axis is the standard profile;
# what varies is whether specials can be injected, which specials_safe() decides per
# (input, output, dest_acc).

_EDGE_SWEEP_OPS = sorted(swept_unary_ops(), key=lambda o: o.name)

# What the cat-A/cat-D probes found on Wormhole, recorded as non-strict xfails so each case
# still executes and reports XPASS if the behaviour changes.
#
# Five of the six recorded divergences are fixed in the kernels: Sign and Heaviside on -0.0 in
# #55306, and Sqrt, Rsqrt and SqrtCustom here. Only Reciprocal on 1/NaN is left, with its
# reason in _EDGE_DIVERGENCE_REASON. Wormhole and Blackhole only; Quasar carries its own
# kernels, and a failure there is a divergence to fix at the kernel rather than record here.
#
# This sweep does not settle the signed-zero results: passed_test() treats -0.0 and +0.0 as
# equal, and it runs ApproximationMode.No only. test_sqrt_family_negative_zero_regression
# reads the raw 32-bit result on both approximation modes instead.
#
# Reciprocal is derived rather than listed: it diverges on exactly the combinations that
# deliver the NaN probe, so the set stays right if the format axis changes.
_EDGE_KNOWN_DIVERGENCES = {
    MathOperation.Reciprocal: tuple(
        (fmt.input_format, fmt.output_format, dest_acc)
        for fmt in input_output_formats([DataFormat.Float16_b, DataFormat.Float32])
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes)
        if specials_safe(fmt.input_format, fmt.output_format, dest_acc)
    ),
}

_EDGE_DIVERGENCE_REASON = {
    MathOperation.Reciprocal: "1/NaN returns +0; IEEE, torch and the golden all give NaN. "
    "Every other special agrees, so this is the NaN probe alone, and it diverges on every "
    "combination that delivers one. Not prescribed by the ISA, and a guard costs the op "
    "1.44x, so it is left unfixed.",
}


def _assert_signed_zero_partition_valid():
    """None of the signed-zero ops may carry edge divergences any more.

    Sign and Heaviside held the two unpack_to_dest combinations and Signbit the complementary
    six, and the partition between them was the evidence that -0.0 reaches the LREG on exactly
    those two. Sqrt and Rsqrt held the same two. All of it is gone, for the different reasons
    recorded below. Asserting their absence at collection keeps a table edit from skipping the
    reasoning: an entry here would be a non-strict xfail that XPASSes every run.
    """
    fixed_or_not_delivered = {
        MathOperation.Sign: (
            "sign(-0.0) returns 0 now. calculate_sign takes its zero arm on sfpi::abs(v), "
            "which puts -0.0 inside the documented SFPSETCC contract, so this was fixed rather than "
            "reclassified -- an entry here means the kernel regressed."
        ),
        MathOperation.Heaviside: (
            "heaviside(-0.0) returns the scalar now. Its zero arm ignores the sign bit, as Sign's "
            "does (SFPLZ NOSGN on Wormhole, sfpi::abs(v) on Blackhole), for the same SFPSETCC "
            "reasoning."
        ),
        MathOperation.Signbit: (
            "Signbit's divergences were a stimulus limitation, not a kernel defect. An entry "
            "here means the delivery gate changed -- re-derive it rather than restoring it."
        ),
        MathOperation.Sqrt: (
            "sqrt(-0.0) returns -0.0 now, from the zero-magnitude arm in ckernel_sfpu_sqrt.h; "
            "an entry here means that arm regressed. test_sqrt_family_negative_zero_regression "
            "is what actually pins the sign, since this sweep cannot tell the two zeros apart."
        ),
        MathOperation.Rsqrt: (
            "rsqrt(-0.0) returns -inf now. Same arm and the same regression test as Sqrt."
        ),
    }

    for op, why in fixed_or_not_delivered.items():
        assert op not in _EDGE_KNOWN_DIVERGENCES, why


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
    _skip_bh_unless_fp32(formats, dest_acc)

    # Two independent gates, and both have to pass: _SPECIALS_READY_OPS says the *golden*
    # defines a result for non-finite inputs, specials_safe() says the *pipeline* delivers
    # them intact. Neither implies the other.
    specials = mathop in SPECIALS_READY_OPS and specials_safe(
        formats.input_format, formats.output_format, dest_acc
    )

    specials = _gate_unspecified_nan_sign(mathop, formats, dest_acc, specials)

    # Marked after the gate: the one recorded divergence left is Reciprocal's on the NaN
    # probe, so where the gate has taken cat B away the probe is not sent and the entry
    # would be a non-strict xfail that XPASSes every run.
    diverges_here = (formats.input_format, formats.output_format, dest_acc) in (
        _EDGE_KNOWN_DIVERGENCES.get(mathop, ())
    )
    if diverges_here and specials:
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

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        input_dimensions,
        spec_A=spec_A,
    )


# sqrt_custom(+/-inf): strict assertions on two named values, outside the edge sweep because
# the sweep cannot distinguish the two NaNs it would have to. Float32 -> Float32 at dest_acc=Yes,
# because a 16-bit output narrows NaN to inf and could not show a regression.
#
# Wormhole and Blackhole only, and no Quasar expectation is recorded here: this test cannot run
# there at all. The driver it builds pulls in llk_sfpu/ckernel_sfpu_mask.h, which has no Quasar
# copy, and nothing collects it either -- it sits outside python_tests/quasar/, carries no
# `quasar` marker, and is nightly, each of which the Quasar and ttsim runners exclude.
#
# The -inf half pins the NEGATIVE_INFINITY_SAFE instantiation, which calculate_sqrt_custom opts
# into and nothing in production does. erfinv, asin and acos take the default and still get
# -inf; that is deliberate and unreachable for them, and priced in ckernel_sfpu_sqrt_custom.h.
@pytest.mark.nightly
def test_sqrt_custom_infinity_regression(request):
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    dest_acc = DestAccumulation.Yes
    input_dimensions = [32, 32]

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
    src_A[1] = float("-inf")
    src_B = torch.zeros(num_elements, dtype=torch.float32)

    res = torch.tensor(
        eltwise_unary_sfpu(
            formats=formats,
            dest_acc=dest_acc,
            approx_mode=ApproximationMode.No,
            mathop=MathOperation.SqrtCustom,
            fast_mode=FastMode.No,
            input_dimensions=input_dimensions,
            src_A=src_A,
            src_B=src_B,
            unpack_to_dest=True,
            check_golden=False,
        ),
        dtype=torch.float32,
    )

    assert res[0] == float("inf"), (
        f"sqrt_custom(+inf) returned {res[0]!r}, expected +inf. The non-finite guard in "
        "ckernel_sfpu_sqrt_custom.h is what prevents this. See tt-metal issue #52930."
    )
    assert torch.isnan(res[1]), (
        f"sqrt_custom(-inf) returned {res[1].item()!r}, expected NaN. The NEGATIVE_INFINITY_SAFE "
        "arm in ckernel_sfpu_sqrt_custom.h is what produces this; a -inf means the arm did not "
        "fire, and a +inf means the qNaN it writes is not a positive one."
    )
    # Tolerance, not equality: sqrt_custom is an approximation, so the band only has to
    # separate a computed 2.0 from a passed-through 4.0.
    assert torch.allclose(res[2:], torch.tensor(2.0), rtol=1e-3, atol=0.0), (
        f"sqrt_custom(4.0) is no longer ~2.0 on the lanes around the probes "
        f"(max deviation {(res[2:] - 2.0).abs().max().item():.6g}); the non-finite guard's "
        "predicate has been widened to divert finite lanes."
    )


# sqrt(-0) and rsqrt(-0), read back as raw bit patterns. Outside the edge sweep because
# passed_test() treats -0.0 and +0.0 as equal, so only an integer comparison can tell them
# apart, and because the sweep runs ApproximationMode.No while _calculate_sqrt_body_ has a
# second copy of these guards under APPROXIMATE. Float32 -> Float32 at dest_acc=Yes is the only
# pipeline that delivers a real -0.0 and returns 32 bits intact.
#
# FastMode.No pins default-mode behavior, including the negative-input clamp. Fast sqrt
# does not preserve signed zero, and fast rsqrt skips the negative-input clamp; these
# modes are selected by sqrt_tile<true> and rsqrt_tile<RsqrtMode::Fast>, respectively.
@pytest.mark.nightly
@pytest.mark.parametrize(
    "approx_mode",
    [ApproximationMode.No, ApproximationMode.Yes],
    ids=lambda m: f"approx_{m.name}",
)
@pytest.mark.parametrize(
    "mathop, negative_zero_bits, positive_zero_bits",
    [
        # IEEE: sqrt(-0) = -0, sqrt(+0) = +0.
        (MathOperation.Sqrt, 0x80000000, 0x00000000),
        # IEEE: rsqrt(-0) = -inf, rsqrt(+0) = +inf.
        (MathOperation.Rsqrt, 0xFF800000, 0x7F800000),
    ],
    ids=lambda v: v.name if isinstance(v, MathOperation) else f"0x{v:08X}",
)
def test_sqrt_family_negative_zero_regression(
    mathop, negative_zero_bits, positive_zero_bits, approx_mode
):
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    dest_acc = DestAccumulation.Yes
    input_dimensions = [32, 32]

    # If this ever goes False the pipeline stopped delivering -0.0 and the -0.0 assertion
    # below would really be testing +0.0 -- fail loudly rather than quietly testing nothing.
    assert negative_zero_delivered(formats.input_format, dest_acc), (
        "Float32 at dest_acc=Yes no longer delivers a real -0.0 to the LREG; re-derive the "
        "combination this regression test runs on before editing it."
    )

    num_elements = input_dimensions[0] * input_dimensions[1]
    # -0.0 and +0.0 side by side, so a guard that sets a sign rather than carrying the input's
    # fails one of the two.
    src_A = torch.full((num_elements,), 4.0, dtype=torch.float32)
    src_A[0] = -0.0
    src_A[1] = 0.0
    # A negative that must stay NaN: catches the zero-magnitude arm being widened.
    src_A[2] = -1.0
    src_B = torch.zeros(num_elements, dtype=torch.float32)

    res = torch.tensor(
        eltwise_unary_sfpu(
            formats=formats,
            dest_acc=dest_acc,
            approx_mode=approx_mode,
            mathop=mathop,
            fast_mode=FastMode.No,
            input_dimensions=input_dimensions,
            src_A=src_A,
            src_B=src_B,
            unpack_to_dest=True,
            check_golden=False,
        ),
        dtype=torch.float32,
    )
    bits = res.view(torch.int32)
    op = mathop.name.lower()

    assert bits[0].item() & 0xFFFFFFFF == negative_zero_bits, (
        f"{op}(-0.0) returned 0x{bits[0].item() & 0xFFFFFFFF:08X} ({res[0].item()!r}), "
        f"expected 0x{negative_zero_bits:08X}. See the signed-zero arms in "
        "ckernel_sfpu_sqrt.h."
    )
    assert bits[1].item() & 0xFFFFFFFF == positive_zero_bits, (
        f"{op}(+0.0) returned 0x{bits[1].item() & 0xFFFFFFFF:08X} ({res[1].item()!r}), "
        f"expected 0x{positive_zero_bits:08X}."
    )
    assert torch.isnan(res[2]), (
        f"{op}(-1.0) returned {res[2].item()!r}, expected NaN; the zero-magnitude arm is "
        "swallowing ordinary negative inputs."
    )
    # The rest of the tile is 4.0, catching a guard that fires on every lane.
    expected = 2.0 if mathop == MathOperation.Sqrt else 0.5
    assert torch.allclose(res[3:], torch.tensor(expected), rtol=1e-2, atol=0.0), (
        f"{op}(4.0) is no longer ~{expected} on the lanes around the probes "
        f"(max deviation {(res[3:] - expected).abs().max().item():.6g})."
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
            "its comparison scalar — add one in sfpu_domains._OP_EDGES"
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


# Cat F: relu_max's threshold, which SFPU_RELU_MAX_THRESHOLD makes reachable, probed where the
# kernel's two SFPSWAP folds (max(min(x, t), 0)) meet the values a sign-magnitude total order
# ranks differently from IEEE: both NaN signs, both zeros, the infinities, the subnormal extremes,
# and the threshold itself with its two fp neighbours. The thresholds are the ones production
# passes -- relu6's 6.0 -- plus the two the sweep's fixed 5.0 cannot reach: zero, where the
# relu clamp alone decides, and a negative one, where every lane must come out +0.0.
#
# The gate is exact bits, not the op's tolerance: relu_max's result is always one of its input,
# the threshold or +0.0, so there is no rounding to allow for, and a tolerance would pass a -0.0
# for a +0.0 and a flushed subnormal for the subnormal.
_RELU_MAX_THRESHOLDS = [6.0, 0.0, -1.0, -0.0]

_FP32_MIN_SUBNORMAL = struct.unpack("<f", struct.pack("<I", 0x00000001))[0]
_FP32_MAX_SUBNORMAL = struct.unpack("<f", struct.pack("<I", 0x007FFFFF))[0]
_NEGATIVE_NAN = struct.unpack("<f", struct.pack("<I", 0xFFC00000))[0]


def _relu_max_probe_spec(threshold, formats, dest_acc):
    """The probe values for one (threshold, pipeline), as a per-face custom spec.

    Each group is added only where the pipeline delivers it intact, on the same rules the
    edge sweep uses: specials_safe() for the non-finites and negative_zero_delivered() for
    -0.0. The subnormals go in on the unpack-to-dest path (32-bit input, dest_acc=Yes) only;
    measured on Blackhole they read back as +0.0 there too (both signs, both extremes), and
    the golden's FTZ model (_flush_subnormals_of_dtype) says the same, so what the probe pins
    is that the flush is unchanged, not that a subnormal survives.
    """
    torch_format = format_dict[formats.input_format]
    t = torch.tensor([threshold], dtype=torch_format)
    above = torch.nextafter(t, torch.tensor([float("inf")], dtype=torch_format)).item()
    below = torch.nextafter(t, torch.tensor([float("-inf")], dtype=torch_format)).item()

    values = [
        threshold,
        above,
        below,
        -threshold,
        2.0 * threshold,
        0.0,
        1.0,
        -1.0,
        3.0,
        -3.0,
    ]
    if specials_safe(formats.input_format, formats.output_format, dest_acc):
        values += [float("inf"), float("-inf"), float("nan"), _NEGATIVE_NAN]
    if negative_zero_delivered(formats.input_format, dest_acc):
        values.append(-0.0)
    if unpacks_to_dest(formats.input_format, dest_acc):
        values += [
            _FP32_MIN_SUBNORMAL,
            -_FP32_MIN_SUBNORMAL,
            _FP32_MAX_SUBNORMAL,
            -_FP32_MAX_SUBNORMAL,
        ]
    return StimuliSpec.custom(values=values, seed=0)


@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    threshold=_RELU_MAX_THRESHOLDS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_relu_max_threshold(
    formats: list[InputOutputFormat],
    threshold: float,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    """relu_max against production's thresholds, checked bit for bit on the probe table.

    The golden is sfpu_relu_max: min under the SFPU's total order, then the relu clamp, so
    it pins the order of the two folds as well as the values -- a kernel that clamped first
    would return the threshold for a negative threshold and keep a -NaN, and fail here.
    """
    _skip_bh_unless_fp32(formats, dest_acc)

    res_tensor, golden_tensor = eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        ApproximationMode.No,
        MathOperation.ReluMax,
        FastMode.No,
        input_dimensions,
        spec_A=_relu_max_probe_spec(threshold, formats, dest_acc),
        relu_max_threshold=threshold,
        return_golden=True,
    )

    torch_format = format_dict[formats.output_format]
    int_format = torch.int32 if torch_format == torch.float32 else torch.int16
    res_bits = res_tensor.to(torch_format).contiguous().view(int_format)
    golden_bits = (
        golden_tensor.to(torch_format).contiguous().view(int_format)
        if isinstance(golden_tensor, torch.Tensor)
        else torch.tensor(golden_tensor, dtype=torch_format).view(int_format)
    )
    mismatch = torch.nonzero(res_bits != golden_bits).flatten()
    mask = 0xFFFFFFFF if int_format == torch.int32 else 0xFFFF
    assert mismatch.numel() == 0, (
        f"{mismatch.numel()} lane(s) differ from sfpu_relu_max bit for bit; first: "
        + ", ".join(
            f"[{i}] got {int(res_bits[i]) & mask:#x} want {int(golden_bits[i]) & mask:#x}"
            for i in mismatch[:8].tolist()
        )
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


# Negative two's-complement operands, which the positive-only sweep above never shifts: they pin that
# `vInt >> n` is the arithmetic SFPSHFT. Amounts 32 and 63 clamp to 31 (-1 for every negative lane).
_RIGHT_SHIFT_SIGNED_VALUES = [
    -1, -2, -3, -255, -256, -65536, -(2**30), -(2**31 - 2**7), 0, 1, 7, 2**30
]  # fmt: skip
_RIGHT_SHIFT_SIGNED_AMOUNTS = [0, 1, 7, 31, 32, 63]


@parametrize(
    mathop=[MathOperation.RightShift],
    shift_amount=_RIGHT_SHIFT_SIGNED_AMOUNTS,
)
def test_eltwise_unary_sfpu_int_right_shift_signed(
    mathop: MathOperation, shift_amount: int
):
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Int32, DataFormat.Int32),
        DestAccumulation.Yes,
        ApproximationMode.No,
        mathop,
        FastMode.No,
        [64, 64],
        spec_A=StimuliSpec.custom(
            values=[float(v) for v in _RIGHT_SHIFT_SIGNED_VALUES], seed=0
        ),
        shift_amount=shift_amount,
        twos_complement=True,
    )


# Every polygamma order ttnn accepts. psi^(n)(x) ~ (n-1)! / x^n for large x, so each order is
# swept from 0.5 up to where the result is 2^-125, just inside the normal range. The bottom of
# that range is where the unscaled partial sums sit n! below the result, and atol is 0 because
# any absolute tolerance would accept a zero there.
_POLYGAMMA_ORDERS = list(range(1, 12))


def _polygamma_stimuli_spec(order):
    high = (math.factorial(order - 1) * 2.0**125) ** (1.0 / order)
    return StimuliSpec(distribution=DistributionKind.LOG_UNIFORM, low=0.5, high=high)


@pytest.mark.nightly
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    polygamma_order=_POLYGAMMA_ORDERS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_dimensions=[[64, 64]],
)
def test_eltwise_unary_sfpu_polygamma_order(
    formats: InputOutputFormat,
    polygamma_order: int,
    dest_acc: DestAccumulation,
    input_dimensions: list[int],
):
    _skip_bh_unless_fp32(formats, dest_acc)
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip(
            reason="Float32 reaches a 16-bit Dest as Float16_b, which is swept already"
        )
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        ApproximationMode.No,
        MathOperation.Polygamma,
        FastMode.No,
        input_dimensions,
        spec_A=_polygamma_stimuli_spec(polygamma_order),
        custom_atol=0.0,
        # atol is 0 so a flushed lane fails outright at a relative error of 1. The rtol is loose
        # because the sweep reaches the last binades before underflow, where the Euler-Maclaurin
        # corrections themselves flush: measured worst case there is 2.0% for float32 at order 11
        # and 1.4% for bfloat16, against 32.6 float32 ULP and 0.74 bfloat16 ULP above 2^-100.
        custom_rtol=0.05 if formats.output_format == DataFormat.Float32 else 0.02,
        polygamma_order=polygamma_order,
        return_golden=True,
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
        gate_on_step_budget=True,
    )


# Both int32 extremes, both signs and zero. INT_MIN is deliverable because the words go in
# as two's complement; StimuliSpec would clamp it to INT_MIN + 1, so the input is built here.
_SIGNBIT_INT32_VALUES = [-(2**31), -100, -5, -1, 0, 1, 5, 100, 2**31 - 1]
UINT32_MASK = 0xFFFFFFFF
MAX_REPORTED_MISMATCHES = 8


def test_eltwise_unary_sfpu_signbit_int32():
    """signbit on Int32 returns bit 31 of every two's-complement input word as 0 or 1.

    The values repeat over every lane rather than heading each face, so neighbouring rows
    differ: a body run under another op's SFPLOADMACRO program stores a value derived from
    the wrong row, and the exact compare catches it.
    """
    formats = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)
    dest_acc = DestAccumulation.Yes
    input_dimensions = [64, 64]
    num_elements = input_dimensions[0] * input_dimensions[1]
    values = torch.tensor(_SIGNBIT_INT32_VALUES, dtype=torch.int32)
    src_A = values.repeat(num_elements // values.numel() + 1)[:num_elements]

    golden_tensor = get_golden_generator(UnarySFPUGolden)(
        MathOperation.Signbit,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    res_from_L1 = eltwise_unary_sfpu(
        formats=formats,
        dest_acc=dest_acc,
        approx_mode=ApproximationMode.No,
        mathop=MathOperation.Signbit,
        fast_mode=FastMode.No,
        input_dimensions=input_dimensions,
        src_A=src_A,
        src_B=src_A.clone(),
        twos_complement=True,
        unpack_to_dest=True,
        check_golden=False,
    )

    res_tensor = torch.tensor(res_from_L1, dtype=torch.int64)
    golden = golden_tensor.to(torch.int64)
    assert res_tensor.shape == golden.shape, "Result and golden differ in length"
    mismatch = torch.nonzero(res_tensor != golden).flatten()
    assert mismatch.numel() == 0, (
        f"{mismatch.numel()} of {golden.numel()} words differ from bit 31 of the input; first: "
        + ", ".join(
            f"[{i}] in {int(src_A[i])} got {int(res_tensor[i]) & UINT32_MASK:#x} want {int(golden[i])}"
            for i in mismatch[:MAX_REPORTED_MISMATCHES].tolist()
        )
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


# The predicates a bf16 input at dest_acc=Yes cannot answer. That unpack path hands the
# SFPU a NaN as the infinity of its own sign (measured on Wormhole with both NaN signs
# driven: Identity writes +inf on every +NaN lane and -inf on every -NaN lane, while +inf
# and -inf arrive intact), and which predicates that breaks follows from it rather than
# being a blanket property of the pipeline: is_nan reads 0 on every NaN lane, is_inf 1,
# is_pos_inf 1 on +NaN and is_neg_inf 1 on -NaN, where the golden says the opposite. Only
# is_finite survives, being 0 for an infinity and a NaN alike. (A stimulus of one NaN sign
# hides half of that: torch's bf16 cast gives 0xFFFF, and is_pos_inf then read clean.)
#
# Skipping the whole op list here withheld is_finite as well; it is swept now, so a
# regression in the inf path is caught on a bf16 input instead of only on Float32.
_ISINF_ISNAN_BF16_DEST_UNSUPPORTED = [
    MathOperation.Isinf,
    MathOperation.Isneginf,
    MathOperation.Isnan,
    MathOperation.Isposinf,
]


def isinf_isnan_skip_reason(formats, mathop, dest_acc):
    """Why the isinf/isnan sweep skips this variant, or ``None``. The driver and the
    exact-op guard in test_sfpu_accuracy_budget.py both ask it, so the guard's
    exclusions cannot drift from the skip."""
    if (
        formats.input_format == DataFormat.Float16_b
        and dest_acc == DestAccumulation.Yes
        and mathop in _ISINF_ISNAN_BF16_DEST_UNSUPPORTED
    ):
        return (
            "bf16->fp32 dest unpack delivers a NaN as the infinity of its sign, so "
            "this predicate cannot be evaluated on this pipeline"
        )
    return None


#: A positive quiet NaN's bit pattern in each stimulus dtype, as the signed integer view
#: of that width. The NaNs are written as bits because neither a cast nor a negation says
#: which sign the kernel is fed: torch's fp32 -> bfloat16 cast canonicalises every NaN to
#: 0xFFFF, sign set, and bfloat16 arithmetic drops a NaN's sign.
_QUIET_NAN_BITS = {
    torch.bfloat16: (torch.int16, 0x7FC0),
    torch.float32: (torch.int32, 0x7FC00000),
}


def _isinf_isnan_stimuli_spec():
    def dist(size, dtype, generator):
        # Finite ramp in [-5, 5] with regular +inf / -inf / NaN injected so every
        # face carries all special classes plus finite values. The NaN lanes alternate
        # +NaN and -NaN: the bf16 -> 32-bit Dest unpack turns a NaN into the infinity of
        # its sign, so one sign alone would hide what half the predicates see.
        idx = torch.arange(size, dtype=torch.float32)
        x = ((idx % 11) - 5.0).to(dtype)
        x[0::7] = float("inf")
        x[1::7] = float("-inf")
        int_dtype, positive_nan = _QUIET_NAN_BITS[dtype]
        sign_bit = 1 << (torch.iinfo(int_dtype).bits - 1)
        bits = x.view(int_dtype)
        bits[2::14] = positive_nan
        bits[9::14] = positive_nan - sign_bit  # the same pattern with the sign bit set
        return x

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

    # bf16->fp32 dest unpack (non-32-bit input + dest_acc=Yes) delivers a NaN as the
    # infinity of its sign, which every predicate but is_finite can see; it is swept here.
    # See _ISINF_ISNAN_BF16_DEST_UNSUPPORTED.
    reason = isinf_isnan_skip_reason(formats, mathop, dest_acc)
    if reason:
        pytest.skip(reason=reason)

    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        dest_acc,
        approx_mode,
        mathop,
        FastMode.No,
        input_dimensions,
        spec_A=_isinf_isnan_stimuli_spec(),
        gate_on_step_budget=True,
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
            "land on its comparison threshold — add one in sfpu_domains._OP_EDGES"
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
        gate_on_step_budget=True,
    )


def eltwise_unary_sfpu(
    test_name="sources/eltwise_unary_sfpu_test.cpp",
    formats: list[InputOutputFormat] | None = None,
    dest_acc=None,
    approx_mode=None,
    mathop=None,
    fast_mode: FastMode = FastMode.No,
    input_dimensions: list[int] | None = None,
    dest_sync: DestSync = DestSync.Half,
    spec_A=None,
    shift_amount=None,
    relu_min_int_threshold=None,
    relu_max_threshold=None,
    polygamma_order=None,
    twos_complement=False,
    gate_on_step_budget=False,
    *,
    clamp_negative=True,
    iterations=32,
    stable_sort=StableSort.No,
    fused_sort=FusedSort.No,
    loop_factor=1,
    src_A=None,
    src_B=None,
    tile_cnt_A=None,
    tile_cnt_B=None,
    unpack_to_dest=None,
    is_perf=False,
    perf_report=None,
    run_types=None,
    collect_result=False,
    check_golden=True,
    return_golden=False,
    custom_atol=None,
    custom_rtol=None,
):
    """Shared unary SFPU runner for functional, perf, and accuracy.

    Stimuli come from the op's own domain unless *src_A* / *src_B* (with their tile
    counts) are given. *unpack_to_dest* defaults to ``unpacks_to_dest(input, dest_acc)``.

    *is_perf* runs every *run_types* entry under *perf_report* and returns ``None``,
    unless *collect_result* also reads the L1_TO_L1 result back.

    *check_golden=False* returns the raw L1 result without a golden check.
    *return_golden* returns ``(res_tensor, golden_tensor)`` instead, for callers that
    want a stricter gate than the op's tolerance; it needs *check_golden*.

    *gate_on_step_budget* is for a caller whose hand-built stimulus the op's step
    budget was measured on (``MEASURED_ON_SWEEP`` in test_sfpu_accuracy_budget.py): it
    gates on the whole contract rather than only its tolerance arm.

    *custom_atol* / *custom_rtol* replace the contract's gate for a caller whose stimulus
    needs a tolerance the op's registry row does not describe (the polygamma order sweep).
    """
    torch.manual_seed(0)
    torch.set_printoptions(precision=10)

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")
    if return_golden and not check_golden:
        raise ValueError("return_golden needs check_golden=True")

    need_stimuli = src_A is not None or not is_perf or collect_result or check_golden
    if src_A is None and need_stimuli:
        # The op's own signed domain, not generate_stimuli's positive-only format default,
        # which would leave the x<0 branch, the knees and the saturation tails unreached.
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
    elif src_A is not None:
        computed_A, computed_B, _ = calculate_tile_and_face_counts(
            input_dimensions,
            input_dimensions,
            face_r_dim=MAX_FACE_R_DIM,
            num_faces=MAX_NUM_FACES,
        )
        if tile_cnt_A is None:
            tile_cnt_A = computed_A
        if tile_cnt_B is None:
            tile_cnt_B = computed_B
        if src_B is None:
            src_B = torch.zeros_like(src_A)
    else:
        tile_cnt_A, tile_cnt_B, _ = calculate_tile_and_face_counts(
            input_dimensions,
            input_dimensions,
            face_r_dim=MAX_FACE_R_DIM,
            num_faces=MAX_NUM_FACES,
        )

    golden_tensor = None
    if check_golden:
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
            **(
                {}
                if relu_max_threshold is None
                else {"relu_max_threshold": relu_max_threshold}
            ),
            **({} if polygamma_order is None else {"polygamma_order": polygamma_order}),
        )

    configuration = build_eltwise_unary_sfpu_config(
        formats,
        dest_acc,
        approx_mode,
        mathop,
        input_dimensions,
        src_A if need_stimuli else None,
        src_B if need_stimuli else None,
        tile_cnt_A,
        tile_cnt_B,
        test_name=test_name,
        fast_mode=fast_mode,
        dest_sync=dest_sync,
        shift_amount=shift_amount,
        relu_min_int_threshold=relu_min_int_threshold,
        relu_max_threshold=relu_max_threshold,
        polygamma_order=polygamma_order,
        twos_complement=twos_complement,
        clamp_negative=clamp_negative,
        iterations=iterations,
        stable_sort=stable_sort,
        fused_sort=fused_sort,
        loop_factor=loop_factor,
        unpack_to_dest=unpack_to_dest,
        is_perf=is_perf,
        run_types=run_types,
    )

    if is_perf and not collect_result:
        configuration.run(perf_report)
        return None

    if is_perf and collect_result:
        configuration.variant_stimuli.write(TestConfig.TENSIX_LOCATION)
        configuration.run(perf_report)
        res_from_L1 = configuration.variant_stimuli.collect_results(
            TestConfig.TENSIX_LOCATION
        )
    else:
        res_from_L1 = configuration.run().result

    if not check_golden:
        return res_from_L1

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Gate on the op's tolerance only: its step budgets were measured over the whole
    # format, so only callers whose stimuli they were measured on opt into them.
    contract = accuracy_contract(
        mathop,
        output_format=formats.output_format,
        input_format=formats.input_format,
        approx_mode=approx_mode,
        dest_acc=dest_acc,
        arch=TestConfig.CHIP_ARCH,
    )
    if custom_atol is not None or custom_rtol is not None:
        gate_kwargs = {"custom_atol": custom_atol, "custom_rtol": custom_rtol}
    elif gate_on_step_budget:
        gate_kwargs = contract.passed_test_kwargs(
            flush_subnormals=FLUSH_SUBNORMAL_OUTPUTS
        )
    else:
        gate_kwargs = contract.tolerance_kwargs()
    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        **gate_kwargs,
    ), "Assert against golden failed"
    if return_golden:
        return res_tensor, golden_tensor
    return res_from_L1


def build_eltwise_unary_sfpu_config(
    formats,
    dest_acc,
    approx_mode,
    mathop,
    input_dimensions,
    src_A,
    src_B,
    tile_cnt_A,
    tile_cnt_B,
    *,
    test_name="sources/eltwise_unary_sfpu_test.cpp",
    fast_mode=FastMode.No,
    dest_sync=DestSync.Half,
    shift_amount=None,
    relu_min_int_threshold=None,
    relu_max_threshold=None,
    polygamma_order=None,
    twos_complement=False,
    clamp_negative=True,
    iterations=32,
    stable_sort=StableSort.No,
    fused_sort=FusedSort.No,
    loop_factor=1,
    unpack_to_dest=None,
    is_perf=False,
    run_types=None,
):
    """The templates, runtimes and stimuli the shared unary kernel reads, as one config.

    For callers that need the configuration itself (build-only, or asserting on how it
    was built); everything else goes through eltwise_unary_sfpu. *src_A* / *src_B* may
    be None for a perf run that writes no stimuli."""
    if run_types is None:
        run_types = ALL_PERF_RUN_TYPES if is_perf else [PerfRunType.L1_TO_L1]
    _, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions,
        input_dimensions,
        face_r_dim=MAX_FACE_R_DIM,
        num_faces=MAX_NUM_FACES,
    )
    if unpack_to_dest is None:
        unpack_to_dest = unpacks_to_dest(formats.input_format, dest_acc)

    test_config_kwargs = {
        "test_name": test_name,
        "formats": formats,
        "templates": [
            MATH_OP(mathop=mathop),
            APPROX_MODE(approx_mode),
            ITERATIONS(iterations),
            FAST_MODE(fast_mode),
            STABLE_SORT(stable_sort),
            FUSED_SORT(fused_sort),
            CLAMP_NEGATIVE(clamp_negative),
            DEST_SYNC(dest_sync),
            *([] if shift_amount is None else [SFPU_SHIFT_AMOUNT(shift_amount)]),
            *(
                []
                if relu_min_int_threshold is None
                else [SFPU_RELU_MIN_INT_THRESHOLD(relu_min_int_threshold)]
            ),
            *(
                []
                if relu_max_threshold is None
                else [SFPU_RELU_MAX_THRESHOLD(relu_max_threshold)]
            ),
            *(
                []
                if polygamma_order is None
                else [SFPU_POLYGAMMA_ORDER(polygamma_order)]
            ),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt_A),
            LOOP_FACTOR(loop_factor),
            NUM_FACES(num_faces=faces_to_generate),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
        ],
        "variant_stimuli": StimuliConfig(
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
        "dest_acc": dest_acc,
        "unpack_to_dest": unpack_to_dest,
    }

    return create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )


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

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        MathOperation.Exp,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    res_from_L1 = eltwise_unary_sfpu(
        formats=formats,
        dest_acc=dest_acc,
        approx_mode=ApproximationMode.Yes,
        mathop=MathOperation.Exp,
        fast_mode=FastMode.Yes,
        input_dimensions=input_dimensions,
        src_A=src_A,
        src_B=src_B,
        clamp_negative=clamp_negative,
        check_golden=False,
    )

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


_NAN_BITS = (0x7FC00000, 0x7F800001, 0x7FFFFFFF, 0xFFC00000, 0xFF800001, 0xFFFFFFFF)


def _float_tile(values, num_elements=1024):
    row = torch.as_tensor(values, dtype=torch.float32).flatten()
    return row.repeat(num_elements // row.numel() + 1)[:num_elements].contiguous()


def _nan_values(bits=_NAN_BITS):
    return torch.tensor(
        [b - (1 << 32) if b >> 31 else b for b in bits], dtype=torch.int32
    ).view(torch.float32)


def _run_unary_edge(
    mathop, formats, src_A, approx_mode=ApproximationMode.No, fast_mode=FastMode.No
):
    dest_acc = (
        DestAccumulation.Yes
        if formats.input_format.is_32_bit()
        else DestAccumulation.No
    )
    res = eltwise_unary_sfpu(
        formats=formats,
        dest_acc=dest_acc,
        approx_mode=approx_mode,
        mathop=mathop,
        fast_mode=fast_mode,
        input_dimensions=[32, 32],
        src_A=src_A,
        src_B=torch.zeros_like(src_A),
        check_golden=False,
    )
    return torch.tensor(res, dtype=torch.float32)


_EDGE_FORMATS = [
    InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
]


@pytest.mark.parametrize("formats", _EDGE_FORMATS, ids=lambda f: f.input_format.name)
@pytest.mark.parametrize(
    "mathop, approx_mode",
    [
        (MathOperation.Erf, ApproximationMode.No),
        (MathOperation.Erf, ApproximationMode.Yes),
        (MathOperation.Erfc, ApproximationMode.No),
    ],
    ids=lambda v: v.name,
)
def test_erf_erfc_nan_input(formats, mathop, approx_mode):
    finite = torch.tensor(
        [0.0, 0.5, -0.5, 4.0, -4.0, 6.0, -6.0, 20.0, -20.0], dtype=torch.float32
    )
    src_A = _float_tile(torch.cat([_nan_values(), finite]))
    res = _run_unary_edge(mathop, formats, src_A, approx_mode)

    nan_in = torch.isnan(src_A)
    if formats.output_format.is_32_bit():
        bad = nan_in & ~torch.isnan(res)
    else:
        bad = nan_in & torch.isfinite(res)
    assert (
        not bad.any()
    ), f"{mathop.name}: {int(bad.sum())} NaN inputs came out as {res[bad][:4].tolist()}"
    assert torch.isfinite(
        res[~nan_in]
    ).all(), f"{mathop.name}: finite inputs came out non-finite"


def test_gelu_accurate_bf16_nonfinite_and_saturated():
    saturated = [-5.5625, -6.0, -13.375, -100.0, -1.0e30, -3.0e38]
    src_A = _float_tile([float("nan"), -float("nan"), -float("inf")] + saturated)
    res = _run_unary_edge(MathOperation.Gelu, _EDGE_FORMATS[1], src_A)

    delivered = src_A.to(torch.bfloat16).float()
    nonfinite_in = ~torch.isfinite(delivered)
    assert not torch.isfinite(
        res[nonfinite_in]
    ).any(), "gelu: a NaN or -inf input came out finite"
    sat = res[torch.isfinite(delivered)]
    assert (
        torch.equal(sat, torch.zeros_like(sat)) and not torch.signbit(sat).any()
    ), sat.unique().tolist()


@pytest.mark.parametrize("formats", _EDGE_FORMATS, ids=lambda f: f.input_format.name)
def test_xielu_below_underflow(formats):
    near = torch.arange(-200.0, -95.0, 0.5)
    far = -torch.arange(400000.0, 16000000.0, 31200.0)
    src_A = _float_tile(torch.cat([near, far]))
    res = _run_unary_edge(MathOperation.Xielu, formats, src_A)

    x = src_A.to(format_dict[formats.input_format]).double()
    alpha_n, beta = UnarySFPUGolden._XIELU_ALPHA_N, UnarySFPUGolden._XIELU_BETA
    expected = (alpha_n * (torch.expm1(x) - x) + beta * x).float()
    rtol = 6e-5 if formats.output_format.is_32_bit() else 1e-2
    assert torch.allclose(
        res, expected, rtol=rtol, atol=1e-6
    ), f"xielu max relative error {((res - expected).abs() / expected.abs()).max().item():.3g}"
    if formats.output_format.is_32_bit():
        far_lanes = x < -1e5
        off = far_lanes & (res != expected)
        assert (
            not off.any()
        ), f"xielu of {x[off][:4].tolist()} returned {res[off][:4].tolist()}, expected {expected[off][:4].tolist()}"


@pytest.mark.parametrize(
    "fast_mode", [FastMode.No, FastMode.Yes], ids=lambda m: f"fast_{m.name}"
)
@pytest.mark.parametrize("formats", _EDGE_FORMATS, ids=lambda f: f.input_format.name)
def test_log_zero_exponent_inputs(formats, fast_mode):
    if TestConfig.CHIP_ARCH != ChipArchitecture.BLACKHOLE:
        pytest.skip("Blackhole only: the Wormhole log kernel is unchanged here")
    sign = torch.tensor([0, 1], dtype=torch.int64).repeat_interleave(128)
    if formats.input_format.is_32_bit():
        mantissa = torch.randint(
            1,
            1 << 23,
            (128,),
            generator=torch.Generator().manual_seed(0),
            dtype=torch.int64,
        )
        mantissa[0], mantissa[1] = 0, (1 << 23) - 1
        bits = (sign << 31) | mantissa.repeat(2)
    else:
        bits = (sign << 31) | (torch.arange(128, dtype=torch.int64).repeat(2) << 16)
    src_A = _float_tile(bits.to(torch.int32).view(torch.float32))
    res = _run_unary_edge(MathOperation.Log, formats, src_A, fast_mode=fast_mode)

    bad = ~(torch.isinf(res) & (res < 0))
    assert (
        not bad.any()
    ), f"log of zero-exponent inputs {src_A[bad][:4].tolist()} returned {res[bad][:4].tolist()}"
