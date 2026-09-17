# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the SFPU accuracy budget registry.

No kernel, no device: this is a table and a resolution rule. Both need guarding for the
same reason ``sfpu_domains`` does — the rule reduces a four-dimensional lookup to "most
specific key wins", and a budget resolved from the wrong key is a silently wrong gate, not
an error. A test that reads a budget is worth more than one that reviews the table.

The resolution tests build their own small tables through :func:`resolve_contract` rather
than querying the live registry, so enrolling an op does not break them. Only the tests
that are *about* enrolment touch the real table.
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    MathOpType,
)
from helpers.sfpu_accuracy_budget import (
    _SFPU_ACCURACY_BUDGET,
    BFP8_B_EXACT_INTEGER_DOMAIN,
    DEFAULT,
    MEASURED_ARCH,
    TOLERANCE_CONTRACT,
    AccuracyContract,
    BudgetKey,
    Metric,
    accuracy_contract,
    enrolled_ops,
    resolve_contract,
    validate_registry,
)
from helpers.sfpu_domains import for_op
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from helpers.ulp import (
    _ULP_PROXY_DTYPES,
    INTEGER_FORMATS,
    MAX_MEANINGFUL_ULP,
    ULP_FORMATS,
    has_ulp_gate,
    ulp_dtype,
)
from helpers.utils import passed_test


def _integer_only_ops() -> set:
    """Every SFPU op whose operands and result are integers, from canonical sources.

    Three of them, because the harness has no single classification that covers all:

    * ``MathOpType.SFPU_BINARY_INT`` — the typed integer binaries (``SfpuGtInt`` and its
      siblings).
    * ``test_eltwise_unary_sfpu._INT_UNARY_OPS`` — the driver list for the integer unary
      kernels, which is the canonical statement of which unaries take the integer path.
      ``ReluMin`` is removed again: it is the one entry there that is not integer-only,
      and ``sfpu_operations.h`` picks its ``vInt`` branch only on ``math_format ==
      Int32``.
    * ``SfpuGcd`` — typed ``SFPU_BINARY`` rather than ``SFPU_BINARY_INT``, so neither
      source above catches it, and its operands are integers.
    """
    from test_eltwise_unary_sfpu import _INT_UNARY_OPS

    typed_binaries = {
        op
        for op in MathOperation
        if op.value.operation_type is MathOpType.SFPU_BINARY_INT
    }
    return (typed_binaries | set(_INT_UNARY_OPS) | {MathOperation.SfpuGcd}) - {
        MathOperation.ReluMin
    }


BLOCK_FORMATS_WITHOUT_ULP = [
    DataFormat.Bfp4_b,
    DataFormat.Bfp2_b,
    DataFormat.MxFp8P,
    DataFormat.MxFp4,
]


# ─────────────────────────────────────────────────────────────────────────────
# The contract's own coherence
# ─────────────────────────────────────────────────────────────────────────────


def test_a_ulp_contract_needs_a_budget():
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="needs max_ulp"
    ):
        AccuracyContract(metric=Metric.ULP)


def test_a_ulp_contract_rejects_a_tolerance():
    """Both cannot apply, and an entry carrying both is a half-finished conversion that
    would read as deliberate."""
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="silently ignored"
    ):
        AccuracyContract(max_ulp=1, atol=0.13)


def test_a_tolerance_contract_rejects_a_budget():
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="belong to the ulp metric"
    ):
        AccuracyContract(metric=Metric.TOLERANCE, max_ulp=1)


def test_a_negative_budget_is_rejected():
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="must not be negative"
    ):
        AccuracyContract(max_ulp=-1)


def test_the_metric_is_a_closed_set():
    """Replaces a test for a rejected unknown metric: ``Metric`` is an enum, so an
    unknown value is unrepresentable rather than caught by a hand-rolled check."""
    assert set(Metric) == {Metric.ULP, Metric.TOLERANCE}
    assert AccuracyContract(max_ulp=1).metric is Metric.ULP
    assert TOLERANCE_CONTRACT.metric is Metric.TOLERANCE


@pytest.mark.parametrize(
    "contract, expected",
    [
        (
            AccuracyContract(max_ulp=3),
            {"max_ulp": 3, "near_zero_atol": None},
        ),
        (
            AccuracyContract(max_ulp=3, near_zero_atol=1e-7),
            {"max_ulp": 3, "near_zero_atol": 1e-7},
        ),
        (
            AccuracyContract(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05),
            {"custom_atol": 0.13, "custom_rtol": 0.05},
        ),
        (TOLERANCE_CONTRACT, {"custom_atol": None, "custom_rtol": None}),
    ],
)
def test_a_contract_translates_to_passed_test_arguments(contract, expected):
    assert contract.passed_test_kwargs() == expected


def test_every_contract_is_accepted_by_passed_test():
    """The translation has to be callable, not merely shaped right — a renamed keyword
    would otherwise only surface on hardware."""
    golden = torch_ones()
    for contract in (
        AccuracyContract(max_ulp=1),
        AccuracyContract(max_ulp=1, near_zero_atol=1e-7),
        AccuracyContract(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05),
        TOLERANCE_CONTRACT,
    ):
        assert passed_test(
            golden,
            golden.clone(),
            DataFormat.Float16_b,
            **contract.passed_test_kwargs(),
        )


TILE_SIZE = DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM


def torch_ones():
    return torch.ones(TILE_SIZE, dtype=torch.bfloat16)


# ─────────────────────────────────────────────────────────────────────────────
# Key resolution
# ─────────────────────────────────────────────────────────────────────────────


def test_the_default_key_matches_every_variant():
    key = DEFAULT
    assert key.specificity == 0
    assert key.matches(
        approx_mode=ApproximationMode.Yes,
        output_format=DataFormat.Float32,
        dest_acc=DestAccumulation.No,
        arch=ChipArchitecture.WORMHOLE,
    )
    assert key.matches(approx_mode=None, output_format=None, dest_acc=None, arch=None)


def test_a_more_specific_key_wins_over_the_default():
    table = {
        DEFAULT: AccuracyContract(max_ulp=64),
        BudgetKey(output_format=DataFormat.Float32): AccuracyContract(max_ulp=4),
    }
    assert (
        resolve_contract(table, label="op", output_format=DataFormat.Float32).max_ulp
        == 4
    )
    assert (
        resolve_contract(table, label="op", output_format=DataFormat.Float16_b).max_ulp
        == 64
    )


def test_specificity_counts_every_set_dimension():
    table = {
        BudgetKey(output_format=DataFormat.Float32): AccuracyContract(max_ulp=4),
        BudgetKey(
            output_format=DataFormat.Float32, dest_acc=DestAccumulation.No
        ): AccuracyContract(max_ulp=8),
    }
    resolved = resolve_contract(
        table,
        label="op",
        output_format=DataFormat.Float32,
        dest_acc=DestAccumulation.No,
    )
    assert resolved.max_ulp == 8
    resolved = resolve_contract(
        table,
        label="op",
        output_format=DataFormat.Float32,
        dest_acc=DestAccumulation.Yes,
    )
    assert resolved.max_ulp == 4


def test_a_per_arch_override_beats_the_shared_entry():
    """Open question 4's answer: one value plus overrides, rather than per-arch from the
    start. WH and BH differ in available SFPU instructions and therefore in kernel, so the
    override has to exist; keying everything on arch from the start would double the table
    for the ops where it does not matter."""
    table = {
        DEFAULT: AccuracyContract(max_ulp=4),
        BudgetKey(arch=ChipArchitecture.BLACKHOLE): AccuracyContract(max_ulp=2),
    }
    for arch, expected in (
        (ChipArchitecture.BLACKHOLE, 2),
        (ChipArchitecture.WORMHOLE, 4),
        (ChipArchitecture.QUASAR, 4),
    ):
        resolved = resolve_contract(
            table, label="op", output_format=DataFormat.Float32, arch=arch
        )
        assert resolved.max_ulp == expected, arch


def test_an_unset_query_dimension_only_matches_a_wildcard():
    """A caller that does not know ``dest_acc`` must not be handed a budget measured for
    one setting of it."""
    table = {BudgetKey(dest_acc=DestAccumulation.Yes): AccuracyContract(max_ulp=1)}
    assert (
        resolve_contract(table, label="op", output_format=DataFormat.Float32)
        is TOLERANCE_CONTRACT
    )


def test_equally_specific_keys_are_an_error_not_a_tie_break():
    """Two keys, one dimension each, both matching: the table's iteration order must not
    decide a budget."""
    table = {
        BudgetKey(approx_mode=ApproximationMode.No): AccuracyContract(max_ulp=4),
        BudgetKey(output_format=DataFormat.Float32): AccuracyContract(max_ulp=64),
    }
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        ValueError, match="equally specific"
    ):
        resolve_contract(
            table,
            label="Ambiguous",
            output_format=DataFormat.Float32,
            approx_mode=ApproximationMode.No,
        )


def test_an_empty_table_falls_back_to_the_tolerance_metric():
    assert (
        resolve_contract({}, label="op", output_format=DataFormat.Float32)
        is TOLERANCE_CONTRACT
    )


def test_a_key_describes_itself_for_an_error_message():
    assert DEFAULT.describe() == "DEFAULT"
    described = BudgetKey(
        output_format=DataFormat.Float32, dest_acc=DestAccumulation.No
    ).describe()
    assert "output_format" in described and "dest_acc" in described


# ─────────────────────────────────────────────────────────────────────────────
# The live registry
# ─────────────────────────────────────────────────────────────────────────────


def test_the_registry_resolves_unambiguously_for_every_variant():
    """Exhaustive over the variant space, which is small. An ambiguity that only appears
    for one format is exactly what a reader of the table misses."""
    validate_registry()


def test_an_unenrolled_op_keeps_todays_gate():
    """Enrolment is incremental: nothing changes for an op until it is in the table."""
    assert MathOperation.Exp not in enrolled_ops()
    for fmt in ULP_FORMATS:
        assert (
            accuracy_contract(MathOperation.Exp, output_format=fmt, arch=MEASURED_ARCH)
            is TOLERANCE_CONTRACT
        )


@pytest.mark.parametrize("fmt", BLOCK_FORMATS_WITHOUT_ULP, ids=lambda f: f.name)
def test_an_enrolled_op_keeps_todays_gate_on_a_format_without_a_per_element_ulp(fmt):
    """Enrolment is per format as well as per op. Abs has a step budget, but Bfp4_b and
    the MX formats keep their block-aware lattice compares — and this must *fall back*
    rather than raise, or enrolling one op would break every block-format variant of it.
    """
    assert not has_ulp_gate(fmt)
    assert MathOperation.Abs in enrolled_ops()
    assert (
        accuracy_contract(MathOperation.Abs, output_format=fmt, arch=MEASURED_ARCH)
        is TOLERANCE_CONTRACT
    )


@pytest.mark.parametrize(
    "arch", [a for a in ChipArchitecture if a != MEASURED_ARCH], ids=lambda a: a.name
)
@pytest.mark.parametrize(
    "op", [MathOperation.SigmoidAppx, MathOperation.GeluAppx], ids=lambda o: o.name
)
def test_a_declared_tolerance_survives_an_unswept_architecture(op, arch):
    """The arch gate exists to stop *WH-measured step budgets* binding elsewhere. It must
    not take a declared tolerance with it.

    These two carry ``atol=0.13`` because a coarse 3-segment LUT peaks near its knees --
    the number they had as ``CUSTOM_TOLERANCES``, which applied on every architecture.
    Returning the tolerance contract before the registry lookup dropped them back to the
    default ``atol=0.05`` on Blackhole and Quasar, which is the gate that number exists to
    widen. Resolving first and downgrading only a ULP contract is what keeps both true.
    """
    contract = accuracy_contract(
        op,
        output_format=DataFormat.Float16_b,
        approx_mode=ApproximationMode.Yes,
        dest_acc=DestAccumulation.No,
        arch=arch,
    )
    assert contract.metric is Metric.TOLERANCE
    assert contract.atol == 0.13 and contract.rtol == 0.05


@pytest.mark.parametrize(
    "arch", [a for a in ChipArchitecture if a != MEASURED_ARCH], ids=lambda a: a.name
)
def test_a_step_budget_does_not_bind_on_an_unswept_architecture(arch):
    """The other half: every number in the table was measured on Wormhole with no
    headroom, so a ULP contract must not survive the trip."""
    on_wh = accuracy_contract(
        MathOperation.Abs, output_format=DataFormat.Float32, arch=MEASURED_ARCH
    )
    assert on_wh.metric is Metric.ULP and on_wh.max_ulp == 0
    assert (
        accuracy_contract(
            MathOperation.Abs, output_format=DataFormat.Float32, arch=arch
        )
        is TOLERANCE_CONTRACT
    )


def test_arch_must_be_passed_explicitly():
    """``arch`` is the one dimension where the numbers do not transfer, so unlike the
    other three it cannot be left unset and quietly resolved against the Wormhole table.
    A second enroller that forgets the keyword fails at the call."""
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        TypeError, match="arch"
    ):
        accuracy_contract(MathOperation.Abs, output_format=DataFormat.Float32)


def test_a_variant_specific_tolerance_needs_no_driver_override():
    """Why removing ``custom_atol``/``custom_rtol`` from the driver loses nothing: a
    tolerance narrower than the op's default is expressible as a more specific
    :class:`BudgetKey`, so it does not have to move the op's other variants.

    The registry is the single source of truth for a number; the alternative is the
    per-test magic number this whole mechanism removes.
    """
    table = {
        DEFAULT: AccuracyContract(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05),
        BudgetKey(output_format=DataFormat.Float32): AccuracyContract(
            metric=Metric.TOLERANCE, atol=0.001, rtol=0.001
        ),
    }
    narrow = resolve_contract(
        table, label="probe", output_format=DataFormat.Float32, arch=MEASURED_ARCH
    )
    assert narrow.atol == 0.001
    broad = resolve_contract(
        table, label="probe", output_format=DataFormat.Float16_b, arch=MEASURED_ARCH
    )
    assert broad.atol == 0.13


def test_every_enrolled_op_resolves_to_something_usable_on_a_float_format():
    for op in enrolled_ops():
        for fmt in ULP_FORMATS:
            contract = accuracy_contract(
                op,
                output_format=fmt,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
                arch=ChipArchitecture.WORMHOLE,
            )
            assert contract.metric in (Metric.ULP, Metric.TOLERANCE)
            if contract.metric == Metric.ULP:
                assert contract.max_ulp is not None and contract.max_ulp >= 0


def test_enrolled_ops_is_sorted_and_stable():
    ops = enrolled_ops()
    assert list(ops) == sorted(ops, key=lambda op: op.name)
    assert len(set(ops)) == len(ops)


# ─────────────────────────────────────────────────────────────────────────────
# Invariants the enrolled budgets rest on
#
# Not a copy of the table — a reader can see the numbers. These are the properties that
# are *not* visible from reading it, and each one guards against the specific way a
# budget gets quietly ruined: raising it until a failure goes away.
# ─────────────────────────────────────────────────────────────────────────────

# Derived the same way validate_registry() derives it, not hardcoded: add a second entry
# to _ULP_PROXY_DTYPES and has_ulp_gate() starts accepting that format and the validator
# picks it up, but a literal list here would not -- and then every guard in this file
# quietly stops covering it, which is where a too-wide budget would hide.
ULP_CAPABLE_FORMATS = list(ULP_FORMATS) + list(_ULP_PROXY_DTYPES)

#: Every op whose result is exact by construction — sign-bit manipulation, a copy, or an
#: integer-valued result. None of these can legitimately need a wide budget, so a large
#: one means the number was fitted to a failure rather than measured.
EXACT_BY_CONSTRUCTION = (
    MathOperation.Abs,
    MathOperation.Neg,
    MathOperation.Identity,
    MathOperation.Floor,
    MathOperation.Ceil,
    MathOperation.Trunc,
)


def _every_variant(op):
    """Every contract an op can resolve to, across the whole keyed variant space.

    Passing only ``output_format`` is not enough: by the ``matches()`` rule an unset
    caller dimension cannot match a key that sets one, so any ``BudgetKey(arch=...)`` or
    ``BudgetKey(dest_acc=...)`` entry is invisible to such a query — which is exactly the
    growth path this file advertises, and would hide a wide budget from the guards below.
    """
    for fmt in ULP_CAPABLE_FORMATS:
        for approx_mode in list(ApproximationMode) + [None]:
            for dest_acc in list(DestAccumulation) + [None]:
                for (
                    arch
                ) in ChipArchitecture:  # arch is required; None is unrepresentable
                    yield fmt, accuracy_contract(
                        op,
                        output_format=fmt,
                        approx_mode=approx_mode,
                        dest_acc=dest_acc,
                        arch=arch,
                    )


@pytest.mark.parametrize("op", EXACT_BY_CONSTRUCTION, ids=lambda op: op.name)
def test_an_exact_op_never_carries_a_wide_budget(op):
    """These ops clear a sign bit, copy, or land on an integer. One step of slack is the
    pack path; more than that is not the op, and a budget hiding it defeats the point of
    having these enrolled as the canaries."""
    for fmt, contract in _every_variant(op):
        if contract.metric == Metric.ULP:
            assert contract.max_ulp <= 1, (
                f"{op.name} on {fmt.name} carries max_ulp={contract.max_ulp}. These "
                "ops are exact by construction; a budget this wide means the number was "
                "fitted to a failure. Investigate the datapath or the golden instead."
            )


def test_no_budget_exceeds_its_formats_meaningful_ceiling():
    """ttnn's ``2**mantissa_bits`` line, applied to the table rather than to one call.

    Past it the two values differ by more than an order of magnitude and ULP has stopped
    being the right metric — the op belongs on the tolerance metric, as Square on Float32
    and the Bfp8_b entries are. This is the guard against "the sweep reported 15616, so
    the budget is 15616".
    """
    for op in enrolled_ops():
        for fmt, contract in _every_variant(op):
            if contract.metric != Metric.ULP:
                continue
            ceiling = MAX_MEANINGFUL_ULP[ulp_dtype(fmt)]
            assert contract.max_ulp <= ceiling, (
                f"{op.name} on {fmt.name} has max_ulp={contract.max_ulp}, past the "
                f"{ceiling}-step point where ULP stops meaning anything for that format. "
                "Put the op on the tolerance metric and record the measurement instead."
            )


def test_the_integer_valued_ops_are_the_only_ones_enrolled_on_bfp8_b():
    """Bfp8_b's bf16 step space is a real criterion only where the block exponent is the
    one bf16 would have used. Measured: Abs and Neg — which cannot be wrong — reach 15616
    steps there, because a small element in a wide block is quantized to zero by design.
    Floor/Ceil/Trunc escape it by producing integers, which a shared exponent represents
    exactly.

    If this test fails because an op was added, the question to answer is whether that op
    produces block-exponent-friendly values, not whether the budget can be raised.
    """
    enrolled_on_bfp8 = {
        op
        for op in enrolled_ops()
        for fmt, contract in _every_variant(op)
        if fmt is DataFormat.Bfp8_b and contract.metric == Metric.ULP
    }
    assert enrolled_on_bfp8 == {
        MathOperation.Floor,
        MathOperation.Ceil,
        MathOperation.Trunc,
    }


def test_the_bfp8_b_enrolment_depends_on_the_swept_domain_not_on_the_format():
    """A shared exponent does not represent integers exactly in general — the in-block
    step scales with the block maximum, so an integer is exact only while every block
    maximum stays under ``2**7``. Floor/Ceil/Trunc qualify because
    ``_OP_DOMAIN_REGISTRY`` bounds them to ``uniform(-10, 10)``, which is a property of
    the *stimulus*, not of the format — the same mechanism takes Abs and Neg to 15616
    steps. Widen the domain and the 0-step Bfp8_b budget stops being legitimate, so the
    dependency is asserted rather than left in a comment.
    """
    for op in (MathOperation.Floor, MathOperation.Ceil, MathOperation.Trunc):
        spec = for_op(op).spec_A
        assert spec.low is not None and spec.high is not None, op.name
        assert max(abs(spec.low), abs(spec.high)) < BFP8_B_EXACT_INTEGER_DOMAIN, (
            f"{op.name} is swept over [{spec.low}, {spec.high}], whose block maxima can "
            f"reach {BFP8_B_EXACT_INTEGER_DOMAIN}. Its 0-step Bfp8_b budget relied on "
            "every block maximum staying below that; re-measure before widening."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Integers never reach the ULP metric through the registry
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fmt", INTEGER_FORMATS, ids=lambda f: f.name)
def test_the_integer_short_circuit_holds_for_every_op(fmt):
    """``accuracy_contract`` returns the tolerance contract on ``not has_ulp_gate`` before
    it ever consults the table, so this pins the short-circuit rather than the table's
    contents — adding ``LeftShift: {DEFAULT: AccuracyContract(max_ulp=0)}`` would leave it
    green. ``test_no_table_entry_can_gate_an_integer_format`` is what guards the table.
    """
    for op in MathOperation:
        contract = accuracy_contract(op, output_format=fmt, arch=MEASURED_ARCH)
        assert contract.metric == Metric.TOLERANCE, (
            f"{op.name} resolves to a {contract.metric} contract on {fmt.name}. ULP is "
            "not a gate for an integer format; it wants bit equality."
        )


def test_no_table_entry_can_gate_an_integer_format():
    """Asserted against ``_SFPU_ACCURACY_BUDGET`` directly, because the short-circuit
    above means a bad entry is unreachable through ``accuracy_contract`` and so invisible
    to it. The alternative guard — a name-token list — misses any integer op not named
    ``Int32``/``Int16``/``Int8``/``Shift``/``Bitwise``."""
    for op, table in _SFPU_ACCURACY_BUDGET.items():
        for key, contract in table.items():
            if contract.metric != Metric.ULP:
                continue
            fmt = key.output_format
            assert fmt is None or not fmt.is_integer(), (
                f"{op.name} carries a step budget keyed on {fmt.name}, an integer "
                "format, which wants bit equality rather than ULP."
            )


def test_the_integer_ops_are_not_enrolled():
    """No integer-only SFPU op carries a contract. If one is added it belongs on the
    tolerance metric — or on an exact-equality gate, which this harness does not have
    yet — not on a step count.

    Derived from the canonical classification and driver sets rather than from name
    tokens. A token list over ``Int32``/``Int16``/``Int8``/``Shift``/``Bitwise`` misses
    ``UnaryMaxUint32``/``UnaryMinUint32`` (``Uint32`` does not contain ``Int32``), every
    ``SFPU_BINARY_INT`` member such as ``SfpuGtInt``, and ``SfpuGcd`` — so enrolling one
    of those left this green, and under a ``DEFAULT`` key
    ``test_no_table_entry_can_gate_an_integer_format`` cannot see it either
    (``key.output_format`` is ``None``) and neither can the short-circuit test.
    """
    integer_ops = _integer_only_ops()
    # Pin the ops a name-token derivation used to miss, so this set cannot silently
    # narrow back to one.
    for op in (
        MathOperation.UnaryMaxUint32,
        MathOperation.UnaryMinUint32,
        MathOperation.SfpuGtInt,
        MathOperation.SfpuGcd,
        MathOperation.LeftShift,
    ):
        assert op in integer_ops, f"{op.name} dropped out of the integer-op derivation"

    enrolled = set(enrolled_ops())
    assert not (enrolled & integer_ops), sorted(
        op.name for op in enrolled & integer_ops
    )
