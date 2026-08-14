# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the sfpu_domains gates that decide what a sweep may inject.

No kernel, no device: these are pure-Python assertions about metadata. They are here
because specials_safe() is a *measured* matrix — 250 hardware variants reduced to a
handful of rules (see the section comment in sfpu_domains) — and until now nothing
executed it. Both production callers short-circuit on SPECIALS_READY_OPS, which is empty,
so the rules and the enum-normalisation trap underneath them could be rewritten without a
single test changing outcome. The measurement is expensive to redo and cheap to pin, so it
is pinned here.

The second half guards probe *spacing*, which has the same shape of problem: a probe that
is silently quantized back onto the boundary it was meant to straddle still reads as
coverage, and no hardware run reports it — the variant passes, having tested the boundary
twice.
"""

import struct

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, MathOperation
from helpers.sfpu_domains import (
    GENERATED_NAN_SIGN_OPS,
    Operand,
    edge_values,
    for_op,
    nan_sign_is_unspecified,
    nan_survives_to_l1,
    ops_with_singularity,
    probe_spacing_format,
    sfpu_unary_ops,
    specials_safe,
    specials_safe_formats,
)

# The formats the measurement covered: the 5x5 matrix driven over the isinf / isposinf /
# isneginf / isnan / isfinite predicates on Wormhole n150. Integer and Fp8/MX *output*
# formats are deliberately absent — the measurement never covered them and the gate makes
# no claim there, so pinning a verdict for them would invent one.
_MEASURED_FORMATS = [
    DataFormat.Float32,
    DataFormat.Float16,
    DataFormat.Float16_b,
    DataFormat.Bfp8_b,
    DataFormat.Bfp4_b,
]

# The whole accepted set, written out. Seven cells of fifty, and each one is a rule:
#
#   * A Float32 input carries specials at either dest_acc, into any non-block output —
#     Float16 included, but only with the 32-bit dest (breaker 1, output side).
#   * A Float16_b input carries them only at dest_acc=No; the bf16 -> fp32 dest unpack
#     loses -inf and NaN (breaker 2), and cannot reach a Float16 output at all.
#   * A Float16 input never carries them, at any dest_acc, into any output (breaker 1).
#   * A block-float input cannot carry them in the first place, and a block-float output
#     cannot express the result.
_ACCEPTED = frozenset(
    {
        (DataFormat.Float32, DataFormat.Float32, False),
        (DataFormat.Float32, DataFormat.Float16_b, False),
        (DataFormat.Float32, DataFormat.Float32, True),
        (DataFormat.Float32, DataFormat.Float16, True),
        (DataFormat.Float32, DataFormat.Float16_b, True),
        (DataFormat.Float16_b, DataFormat.Float32, False),
        (DataFormat.Float16_b, DataFormat.Float16_b, False),
    }
)

_MEASURED_MATRIX = [
    (inp, out, dest_acc)
    for inp in _MEASURED_FORMATS
    for out in _MEASURED_FORMATS
    for dest_acc in (False, True)
]


@pytest.mark.parametrize(
    "input_format,output_format,dest_acc",
    _MEASURED_MATRIX,
    ids=[
        f"{inp.name}-{out.name}-dest_acc_{'Yes' if d else 'No'}"
        for inp, out, d in _MEASURED_MATRIX
    ],
)
def test_specials_safe_matches_measured_matrix(input_format, output_format, dest_acc):
    """Every cell of the measured matrix, against the checked-in verdict."""
    expected = (input_format, output_format, dest_acc) in _ACCEPTED
    assert specials_safe(input_format, output_format, dest_acc) is expected, (
        f"specials_safe({input_format.name}, {output_format.name}, dest_acc={dest_acc}) "
        f"changed: expected {expected}. Either a rule moved or the measurement did — if "
        f"the latter, re-measure with the predicate sweep described in sfpu_domains and "
        f"update _ACCEPTED with the new verdict."
    )


def test_specials_safe_accepts_nothing_outside_the_measured_matrix():
    """The gate defaults to deny, so an unlisted input format is False rather than a
    wall of failures with one root cause."""
    for unlisted in (
        DataFormat.Tf32,
        DataFormat.Int32,
        DataFormat.UInt32,
        DataFormat.Fp8_e4m3,
        DataFormat.MxFp8P,
        DataFormat.Bfp2_b,
    ):
        for dest_acc in (False, True):
            assert not specials_safe(unlisted, DataFormat.Float32, dest_acc)


def test_specials_safe_rejects_mx_outputs():
    """MX outputs are excluded statically, not by measurement: an inf/NaN inside a block
    whose shared exponent is finite is not a value the format can express."""
    for mx_output in (DataFormat.MxFp8P, DataFormat.MxFp8R, DataFormat.MxFp4):
        assert not specials_safe(DataFormat.Float32, mx_output, True)


# The {Float16_b, Float32} matrix the two edge sweeps parametrize over, as (input, output).
_EDGE_SWEEP_PAIRS = [
    (DataFormat.Float16_b, DataFormat.Float16_b),
    (DataFormat.Float16_b, DataFormat.Float32),
    (DataFormat.Float32, DataFormat.Float16_b),
    (DataFormat.Float32, DataFormat.Float32),
]

_EDGE_SWEEP_CELLS = [
    (inp, out, dest_acc)
    for inp, out in _EDGE_SWEEP_PAIRS
    for dest_acc in (DestAccumulation.No, DestAccumulation.Yes)
]


def test_nan_survives_only_into_a_32_bit_dest_and_a_32_bit_pack():
    """Both legs have to stay 32-bit, and on this matrix exactly one cell manages it.

    Pinned because the gate's whole scope follows from it: five of the six triples
    specials_safe() accepts narrow a NaN somewhere, and those five are precisely where a
    generated NaN's sign becomes an observable +/-inf.
    """
    carrying = [c for c in _EDGE_SWEEP_CELLS if specials_safe(*c)]
    assert len(carrying) == 6, "specials_safe's verdict on this matrix moved"

    survives = [c for c in carrying if nan_survives_to_l1(*c)]
    assert survives == [
        (DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes)
    ], (
        "Float32->Float32 at dest_acc=Yes is the only cell on this matrix that carries a "
        f"NaN to L1 as a NaN; got {[(i.name, o.name, str(d)) for i, o, d in survives]}. "
        "If this moved, the Wormhole NaN-sign skip's scope moved with it."
    )


# The two suites do not share a format axis, so the gate's reach is counted per suite.
# ScalarRsub is the scalar suite's only member of the set; the other ten are unary.
_SCALAR_NAN_SIGN_OPS = frozenset({MathOperation.ScalarRsub})
_UNARY_NAN_SIGN_OPS = GENERATED_NAN_SIGN_OPS - _SCALAR_NAN_SIGN_OPS

# test_sfpu_binop_scalar's axis after _skip_unsupported: a Float32 tensor needs the 32-bit
# dest and a Float16_b tensor cannot use one, so two of its eight cells survive.
_SCALAR_SUITE_CELLS = [
    (DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.No),
    (DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes),
]


def test_nan_sign_gate_matches_the_measured_wormhole_failures():
    """Pinned against the Wormhole n300 run rather than against the rule that generated it.

    Unary: 50 (op, cell) pairs — the 49 recorded failures plus the one Rsqrt cell that was
    already xfailed for the unrelated -0.0 divergence, and which this gate now skips first.
    Scalar: 1, which is the run's single `ScalarRsub` failure.
    """
    unary_gated = [
        (op, cell)
        for op in _UNARY_NAN_SIGN_OPS
        for cell in _EDGE_SWEEP_CELLS
        if specials_safe(*cell) and nan_sign_is_unspecified(op, *cell)
    ]
    assert len(_UNARY_NAN_SIGN_OPS) == 10
    assert (
        len(unary_gated) == 50
    ), f"expected 50 gated unary (op, cell) pairs, got {len(unary_gated)}"

    scalar_gated = [
        (op, cell)
        for op in _SCALAR_NAN_SIGN_OPS
        for cell in _SCALAR_SUITE_CELLS
        if specials_safe(*cell) and nan_sign_is_unspecified(op, *cell)
    ]
    assert scalar_gated == [
        (MathOperation.ScalarRsub, _SCALAR_SUITE_CELLS[0])
    ], f"expected only ScalarRsub's Float16_b cell, got {scalar_gated}"

    # Every enrolled op keeps the one cell where the sign is readable, so none of them is
    # skipped out of its sweep entirely.
    for op in GENERATED_NAN_SIGN_OPS:
        assert not nan_sign_is_unspecified(
            op, DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes
        ), f"{op.name} lost its Float32->Float32 dest_acc=Yes assertion"


def test_nan_sign_gate_ignores_ops_that_forward_a_nan():
    """Neg, Abs and Identity move the sign bit rather than inventing one, so their NaN sign
    is a real datum and stays asserted on every cell. They are UnarySFPUGolden's
    _NAN_SIGN_TRANSPARENT_OPS, and this gate must never overlap that set."""
    for op in (MathOperation.Neg, MathOperation.Abs, MathOperation.Identity):
        assert op not in GENERATED_NAN_SIGN_OPS
        for cell in _EDGE_SWEEP_CELLS:
            assert not nan_sign_is_unspecified(op, *cell)


@pytest.mark.parametrize(
    "dest_acc,flag",
    [
        (True, True),
        (False, False),
        (DestAccumulation.Yes, True),
        (DestAccumulation.No, False),
    ],
    ids=["bool_True", "bool_False", "DestAccumulation_Yes", "DestAccumulation_No"],
)
def test_dest_acc_normalisation(dest_acc, flag):
    """A DestAccumulation member must read as its value, not as its truthiness.

    Both members are truthy objects, so a `bool(dest_acc)` implementation would evaluate
    DestAccumulation.No as the 32-bit-dest case and silently flip whole rows of the
    matrix. Float16_b -> Float16_b is the probe because its verdict inverts with the flag:
    accepted at dest_acc=No, rejected at dest_acc=Yes (breaker 2). So a member that
    normalised to the wrong bool would fail here rather than pass by coincidence.
    """
    assert specials_safe(DataFormat.Float16_b, DataFormat.Float16_b, dest_acc) is (
        not flag
    )


@pytest.mark.parametrize(
    "bad", [1, 0, "Yes", None, 1.0], ids=["int_1", "int_0", "str", "None", "float"]
)
def test_dest_acc_rejects_non_flags(bad):
    """Anything that is neither a bool nor a DestAccumulation raises rather than being
    guessed at. 1 and 0 included: they would work by accident and hide a caller bug."""
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        TypeError
    ):
        specials_safe(DataFormat.Float32, DataFormat.Float32, bad)


def test_specials_safe_formats_filters_to_the_accepted_rows():
    formats = [
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        InputOutputFormat(DataFormat.Float32, DataFormat.Float16),
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Float32),
    ]

    kept = specials_safe_formats(formats, DestAccumulation.No)
    assert [(f.input_format, f.output_format) for f in kept] == [
        (DataFormat.Float32, DataFormat.Float32),
        (DataFormat.Float16_b, DataFormat.Float16_b),
    ]

    kept = specials_safe_formats(formats, DestAccumulation.Yes)
    assert [(f.input_format, f.output_format) for f in kept] == [
        (DataFormat.Float32, DataFormat.Float32),
        (DataFormat.Float32, DataFormat.Float16),
    ]


def test_specials_safe_formats_validates_dest_acc_on_an_empty_list():
    """Normalisation happens once up front, so a bad flag raises even with nothing to
    filter — otherwise the error surfaces only for callers that happen to pass formats.
    """
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        TypeError
    ):
        specials_safe_formats([], "Yes")


# ─────────────────────────────────────────────────────────────────────────────
# Probe spacing: every edge probe has to still be distinct once the datapath has it
#
# The failure this pins is not a wrong answer, it is a probe that stops probing. With
# dest_acc=No the DEST holds 16 bits whatever the input format is, so an fp32 probe one fp32
# ULP above a pole of 1.0 (0x3F800002) is truncated straight back to 1.0 and the variant
# tests the pole twice while reading as "pole plus one above it". Acosh was the op it hit.
#
# The truncation is modelled here as an independent `& 0xFFFF0000` rather than by importing
# sfpu_domains' own helper, so these assert the *property* and not the implementation.
# ─────────────────────────────────────────────────────────────────────────────

# The format axis test_eltwise_unary_sfpu_edges actually collects.
_EDGE_SWEEP_FORMATS = [
    (inp, out)
    for inp in (DataFormat.Float16_b, DataFormat.Float32)
    for out in (DataFormat.Float16_b, DataFormat.Float32)
]

# Everything edge_values() can be asked about: the ops with a registry entry (which is what
# sfpu_unary_ops() gates the sweep on) plus the ops carrying a singularity on either operand.
_PROBED_OPS = sorted(
    set(sfpu_unary_ops()) | set(ops_with_singularity()), key=lambda op: op.name
)


def _as_16bit_dest(value: float) -> float:
    """*value* as a 16-bit DEST holds it: fp32 with the low mantissa half dropped."""
    if value != value or value in (float("inf"), float("-inf")):
        return value
    raw = struct.unpack("<I", struct.pack("<f", value))[0] & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", raw))[0]


def _distinct_key(value: float):
    """Identity of *value* as delivered, keeping the two zeros apart.

    -0.0 == +0.0 and they are zero ULPs apart, so a plain set() would collapse them — and
    for signbit / sign / heaviside the difference between them is the entire probe.
    """
    delivered = _as_16bit_dest(value)
    if delivered == 0.0:
        return ("zero", struct.pack("<f", delivered)[-1] & 0x80)
    return ("value", delivered)


@pytest.mark.parametrize(
    "input_format,output_format",
    _EDGE_SWEEP_FORMATS,
    ids=[f"{i.name}-{o.name}" for i, o in _EDGE_SWEEP_FORMATS],
)
@pytest.mark.parametrize(
    "operand", [Operand.A, Operand.B], ids=["operand_A", "operand_B"]
)
def test_edge_probes_stay_distinct_in_a_16bit_dest(
    input_format, output_format, operand
):
    """No op's probe list may contain two values a 16-bit DEST cannot tell apart.

    Parametrized over the formats and swept over every op rather than pinning Acosh, so an
    op that *gains* a nonzero singularity later is covered without touching this test —
    which is the failure mode that produced the original: the collapse was a property of
    (boundary magnitude, format pair, dest_acc), not of Acosh.
    """
    for op in _PROBED_OPS:
        probes = edge_values(
            op, input_format, output_format, operand, dest_acc=DestAccumulation.No
        )
        delivered = [_distinct_key(v) for v in probes]
        assert len(set(delivered)) == len(probes), (
            f"{op.name} probes {probes} collapse in a 16-bit DEST: they arrive as "
            f"{[_as_16bit_dest(v) for v in probes]}. A probe that quantizes onto its own "
            f"boundary tests the boundary twice while reading as coverage — widen the step "
            f"via probe_beside()/probe_spacing_format()."
        )


@pytest.mark.parametrize(
    "fmt,dest_acc,expected",
    [
        (DataFormat.Float32, DestAccumulation.No, DataFormat.Float16_b),
        (DataFormat.Float32, DestAccumulation.Yes, DataFormat.Float32),
        (DataFormat.Float16_b, DestAccumulation.No, DataFormat.Float16_b),
        (DataFormat.Float16_b, DestAccumulation.Yes, DataFormat.Float16_b),
        (DataFormat.Float16, DestAccumulation.No, DataFormat.Float16),
        (DataFormat.Bfp4_b, DestAccumulation.No, DataFormat.Bfp4_b),
        (DataFormat.Float32, None, DataFormat.Float32),
        # Integer formats are 32-bit but mantissa truncation is not what their DEST does.
        (DataFormat.Int32, DestAccumulation.No, DataFormat.Int32),
        (DataFormat.UInt32, DestAccumulation.No, DataFormat.UInt32),
    ],
    ids=[
        "fp32_destaccNo_coarsens",
        "fp32_destaccYes_keeps",
        "bf16_destaccNo_keeps",
        "bf16_destaccYes_keeps",
        "fp16_destaccNo_keeps",
        "bfp4_destaccNo_keeps",
        "no_dest_acc_keeps",
        "int32_destaccNo_keeps",
        "uint32_destaccNo_keeps",
    ],
)
def test_probe_spacing_format(fmt, dest_acc, expected):
    """Only a 32-bit format is coarsened, and only at dest_acc=No.

    A format that is already 16-bit or narrower keeps its own spacing: the DEST it lands in
    is no coarser than it is, so coarsening would loosen the probe for nothing. dest_acc=None
    keeps the format-only behaviour for callers with no flag to pass.
    """
    assert probe_spacing_format(fmt, dest_acc) is expected


def test_probe_spacing_format_rejects_non_flags():
    """Same normalisation trap as specials_safe(): a bare int would work by accident."""
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        TypeError
    ):
        probe_spacing_format(DataFormat.Float32, 1)


def test_acosh_above_pole_probe_is_widened_at_dest_acc_no():
    """The measured case, pinned with its actual values.

    Acosh is (1.0, ABOVE) and (Float32, Float16_b) ties on the bfloat16 ceiling and resolves
    to Float32, so the fp32 step is 2*2**-23 and the probe is 0x3F800002 — which arrives as
    1.0. The bfloat16 step is 2*2**-7, so 1.015625 arrives intact.
    """
    for output_format in (DataFormat.Float32, DataFormat.Float16_b):
        assert edge_values(
            MathOperation.Acosh,
            DataFormat.Float32,
            output_format,
            dest_acc=DestAccumulation.No,
        ) == [1.0, 1.015625]

        # dest_acc=Yes keeps the full fp32 dest, so the tight probe stands.
        assert edge_values(
            MathOperation.Acosh,
            DataFormat.Float32,
            output_format,
            dest_acc=DestAccumulation.Yes,
        ) == [1.0, 1.0000002384185791]


@pytest.mark.parametrize(
    "op",
    [MathOperation.Asin, MathOperation.Acos, MathOperation.Atanh, MathOperation.Erfinv],
    ids=lambda op: op.name,
)
def test_below_boundary_probes_are_not_loosened(op):
    """The other ±1 ops keep their tight below-1.0 probes at dest_acc=No.

    Stepping *down* from 1.0 crosses into the next binade (0x3F7FFFFF -> 0.99609375), so it
    survives the same truncation that destroys the step upward. Coarsening per boundary
    rather than per side would loosen these four for no gain in distinctness, which is why
    probe_beside() decides one side at a time.
    """
    probes = edge_values(
        op, DataFormat.Float32, DataFormat.Float32, dest_acc=DestAccumulation.No
    )
    # Two fp32 ULPs at 1.0, i.e. 0.9999997615814209 — not the two bfloat16 ULPs (0.984375)
    # a per-boundary coarsening would have produced.
    assert 1.0 - 2 * 2**-23 in probes
    assert -(1.0 - 2 * 2**-23) in probes


@pytest.mark.parametrize(
    "op",
    [MathOperation.Sqrt, MathOperation.Log, MathOperation.Reciprocal],
    ids=lambda op: op.name,
)
def test_zero_pole_probes_are_not_loosened(op):
    """Zero-poles keep the fp32 step too, and for a different reason than the ±1 ops.

    Truncation drops mantissa bits and keeps the exponent, and bfloat16 carries fp32's full
    8-bit exponent range, so 2**-23 is representable in a 16-bit DEST and stays a visible
    distance from zero. 17 ops sit on a zero pole; a blanket coarsening would loosen all of
    them.
    """
    probes = edge_values(
        op, DataFormat.Float32, DataFormat.Float32, dest_acc=DestAccumulation.No
    )
    assert min(abs(v) for v in probes if v != 0.0) == 2 * 2**-23


# ─────────────────────────────────────────────────────────────────────────────
# The exp family's two ceilings: range on the registry, accuracy behind approx mode
#
# One registry entry is consumed by both approximation modes, so an accuracy limit parked
# on it silently narrows the *accurate* path too. Exp lost (16, 80] that way — with it the
# exponent-overflow region and all large-exp saturation — and ExpWithBase lost 32..160 for a
# limit it never executes, since STANDARD_SWEEP_OPS runs ApproximationMode.No only.
#
# Nothing executed the split before these tests: the sweep asserts a tolerance, not a
# domain, so a bound could move in either direction without a single test changing outcome.
# ─────────────────────────────────────────────────────────────────────────────

# (op, range-bound high, approximation-accuracy high). The accuracy column is what the
# merged branch had on the shared entry; the range column is what the accurate path gets
# back. Wormhole-measured for the approximation (see _APPROX_EXP_ACCURACY_XFAIL).
_EXP_FAMILY_BOUNDS = [
    (MathOperation.Exp, 80.0, 16.0),
    (MathOperation.Exp2, 100.0, 23.0),
    (MathOperation.ExpWithBase, 160.0, 32.0),
]


@pytest.mark.parametrize(
    "op,range_high,approx_high", _EXP_FAMILY_BOUNDS, ids=lambda v: getattr(v, "name", v)
)
def test_exp_family_ceiling_depends_on_approximation_mode(op, range_high, approx_high):
    """The accurate path keeps the range bound; only the approximating path is narrowed."""
    assert (
        for_op(op, DataFormat.Float32, approx_mode=ApproximationMode.No).spec_A.high
        == range_high
    )
    assert (
        for_op(op, DataFormat.Float32, approx_mode=ApproximationMode.Yes).spec_A.high
        == approx_high
    )


@pytest.mark.parametrize(
    "op,range_high,approx_high", _EXP_FAMILY_BOUNDS, ids=lambda v: getattr(v, "name", v)
)
def test_exp_family_default_is_the_unnarrowed_domain(op, range_high, approx_high):
    """Omitting approx_mode applies no narrowing.

    That is the right default for a caller that measures error instead of asserting a
    tolerance (the accuracy harness), and it keeps the registry readable as "what the op and
    format can hold" rather than "what one mode happens to tolerate".
    """
    assert for_op(op, DataFormat.Float32).spec_A.high == range_high


def test_exp_with_base_argument_ceiling_matches_exp_in_both_modes():
    """exp_with_base computes exp(0.5*x), so its bound has to be exactly double exp's.

    This is the invariant its docstring claims, and the one that broke when the accuracy
    limit landed on the shared entry: exp went to 16 and exp_with_base to 32, but exp's
    *range* bound stayed 80 while exp_with_base's became 32 — an argument of 16 against
    exp's 80.
    """
    for mode in (ApproximationMode.No, ApproximationMode.Yes):
        exp_high = for_op(
            MathOperation.Exp, DataFormat.Float32, approx_mode=mode
        ).spec_A.high
        base_high = for_op(
            MathOperation.ExpWithBase, DataFormat.Float32, approx_mode=mode
        ).spec_A.high
        assert base_high == 2 * exp_high, (
            f"exp_with_base's argument ceiling is {0.5 * base_high} against exp's "
            f"{exp_high} in ApproximationMode.{mode.name}"
        )


@pytest.mark.parametrize(
    "fmt,high",
    [(DataFormat.MxFp8P, 5.0), (DataFormat.Fp8_e4m3, 5.0), (DataFormat.Float16, 10.0)],
    ids=lambda v: getattr(v, "name", v),
)
def test_narrow_format_exp_domains_are_not_widened_by_the_ceiling(fmt, high):
    """The ceiling only ever narrows, so a format branch already tighter is untouched."""
    for mode in (ApproximationMode.No, ApproximationMode.Yes, None):
        assert for_op(MathOperation.Exp, fmt, approx_mode=mode).spec_A.high == high


@pytest.mark.parametrize(
    "op",
    [
        MathOperation.Sqrt,
        MathOperation.Log,
        MathOperation.Reciprocal,
        MathOperation.Gelu,
    ],
    ids=lambda op: op.name,
)
def test_approx_mode_does_not_touch_ops_outside_the_table(op):
    """Only _APPROX_ACCURACY_MAX members react to the mode."""
    no = for_op(op, DataFormat.Float32, approx_mode=ApproximationMode.No).spec_A
    yes = for_op(op, DataFormat.Float32, approx_mode=ApproximationMode.Yes).spec_A
    assert (no.low, no.high, no.intervals) == (yes.low, yes.high, yes.intervals)


@pytest.mark.parametrize(
    "bad", [1, 0, "Yes", 1.0], ids=["int_1", "int_0", "str", "float"]
)
def test_for_op_rejects_non_flag_approx_mode(bad):
    """Same truthiness trap as dest_acc: ApproximationMode.No is a truthy object, so a
    bare int would work by accident on one branch and silently narrow on the other."""
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        TypeError
    ):
        for_op(MathOperation.Exp, DataFormat.Float32, approx_mode=bad)


def test_two_state_flag_rejects_the_other_two_state_enum():
    """A bool check alone cannot tell the two-state enums apart.

    DestAccumulation and ApproximationMode both wrap True/False, so passing one where the
    other is expected satisfies any `.value is a bool` test and selects a valid but
    unintended branch. That is the swap the guard exists to catch, and it is invisible in a
    result -- both arguments are legal, the answer is just quietly the wrong one.
    """
    for wrong, call in (
        (
            DestAccumulation.Yes,
            lambda v: for_op(MathOperation.Exp, DataFormat.Float32, approx_mode=v),
        ),
        (
            ApproximationMode.Yes,
            lambda v: specials_safe(DataFormat.Float32, DataFormat.Float32, v),
        ),
    ):
        with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
            TypeError
        ):
            call(wrong)

    # The right enum, and a bare bool, still pass.
    for_op(MathOperation.Exp, DataFormat.Float32, approx_mode=ApproximationMode.Yes)
    specials_safe(DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes)
    specials_safe(DataFormat.Float32, DataFormat.Float32, True)
