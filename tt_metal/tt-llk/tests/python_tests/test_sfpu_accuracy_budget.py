# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the SFPU accuracy budget registry.

No kernel, no device: this is a table and a resolution rule. Both need guarding for the
same reason ``sfpu_domains`` does — the rule reduces a five-dimensional lookup to "most
specific key wins", and a budget resolved from the wrong key is a silently wrong gate, not
an error. A test that reads a budget is worth more than one that reviews the table.

The resolution tests build their own small tables through :func:`resolve_contract` rather
than querying the live registry, so enrolling an op does not break them. Only the tests
that are *about* enrolment touch the real table.
"""

import math
import textwrap

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
    _BUDGET_KEY_TYPES,
    _SFPU_ACCURACY_BUDGET,
    BFP8_B_EXACT_INTEGER_DOMAIN,
    DEFAULT,
    MEASURED_ARCH,
    TOLERANCE_CONTRACT,
    AccuracyContract,
    BudgetKey,
    Metric,
    _load_table,
    accuracy_contract,
    enrolled_ops,
    resolve_contract,
    usable_budget_ceiling,
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
    * ``test_eltwise_binary_sfpu._INT_DRIVEN_BINARY_OPS`` — the binary driver's own
      integer set, which is where ``SfpuDivInt32``, ``SfpuLcm``, the bitwise ops and the
      ``*Int32``/``*Uint32`` min/max/remainder family live. Typed ``SFPU_BINARY``, so
      ``MathOpType`` alone misses all of them.
    * ``SfpuGcd`` — in neither driver set nor typed ``SFPU_BINARY_INT``, and its operands
      are integers.
    """
    from test_eltwise_binary_sfpu import _INT_DRIVEN_BINARY_OPS
    from test_eltwise_unary_sfpu import _INT_UNARY_OPS

    typed_binaries = {
        op
        for op in MathOperation
        if op.value.operation_type is MathOpType.SFPU_BINARY_INT
    }
    return (
        typed_binaries
        | set(_INT_UNARY_OPS)
        | set(_INT_DRIVEN_BINARY_OPS)
        | {MathOperation.SfpuGcd}
    ) - {MathOperation.ReluMin}


# Every format that is neither ULP-gateable nor integer, so the audit arms cover the whole
# enum between them. The MX int formats matter here: DataFormat.is_integer() is False for
# MxInt8/MxInt4/MxInt2 -- they are classified by the separate is_mx_int_format() -- so
# INTEGER_FORMATS does not reach them, and without this row they would have sat in neither
# arm. Widening INTEGER_FORMATS instead would route them through
# _mxint_block_aware_compare in test_an_integer_format_still_works_on_the_default_gate.
BLOCK_FORMATS_WITHOUT_ULP = sorted(
    (f for f in DataFormat if not has_ulp_gate(f) and f not in INTEGER_FORMATS),
    key=lambda f: f.name,
)


# ── The contract's own coherence ──────────────────────────────────────────────


def test_a_ulp_contract_needs_a_budget():
    with _refuses("needs max_ulp"):
        AccuracyContract(metric=Metric.ULP)


def test_a_ulp_contract_rejects_a_tolerance():
    """Both cannot apply, and an entry carrying both is a half-finished conversion that
    would read as deliberate."""
    with _refuses("silently ignored"):
        AccuracyContract(max_ulp=1, atol=0.13)


def test_a_tolerance_contract_rejects_a_budget():
    with _refuses("belong to the ulp metric"):
        AccuracyContract(metric=Metric.TOLERANCE, max_ulp=1)


def test_a_negative_budget_is_rejected():
    with _refuses("must not be negative"):
        AccuracyContract(max_ulp=-1)


def test_the_metric_is_a_closed_set():
    """Two members and no third gate to fall through to.

    Not a substitute for ``test_a_metric_that_is_not_a_metric_member_is_refused`` further
    down: a bare ``Enum`` does not stop ``metric="ulp"`` from being *constructed*, only
    from being one of these. The closed set is what makes the ``__post_init__`` check a
    complete one."""
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


# ── Key resolution ────────────────────────────────────────────────────────────


def test_the_default_key_matches_every_variant():
    key = DEFAULT
    assert key.specificity == 0
    assert key.matches(
        approx_mode=ApproximationMode.Yes,
        input_format=DataFormat.Float32,
        output_format=DataFormat.Float32,
        dest_acc=DestAccumulation.No,
        arch=ChipArchitecture.WORMHOLE,
    )
    assert key.matches(
        approx_mode=None,
        input_format=None,
        output_format=None,
        dest_acc=None,
        arch=None,
    )


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
    with _refuses("equally specific"):
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


# ── The live registry ─────────────────────────────────────────────────────────


def test_the_registry_resolves_unambiguously_for_every_variant():
    """Exhaustive over the variant space, which is small. An ambiguity that only appears
    for one format is exactly what a reader of the table misses."""
    validate_registry()


def test_an_unenrolled_op_keeps_todays_gate():
    """Enrolment is incremental: nothing changes for an op until it is in the table.

    The op is picked from whatever is still unenrolled rather than named, so enrolling
    another one later does not turn this into a false failure — which is exactly what it
    did when the transcendentals landed and it still named ``Exp``.
    """
    unenrolled = sorted(
        set(MathOperation) - set(enrolled_ops()), key=lambda op: op.name
    )
    assert unenrolled, "every op is enrolled; this test has nothing left to check"
    for op in unenrolled[:5]:
        for fmt in ULP_FORMATS:
            assert (
                accuracy_contract(op, output_format=fmt, arch=MEASURED_ARCH)
                is TOLERANCE_CONTRACT
            ), f"{op.name} is unenrolled but resolves to something other than tolerance"


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
    with _refuses("arch", TypeError):
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


#: The 19 ops P3 enrols from the accuracy sweep. Their keys all pin ``input_format``.
_TRANSCENDENTALS_ENROLLED_WITH_AN_INPUT_FORMAT = frozenset(
    op
    for op, table in _SFPU_ACCURACY_BUDGET.items()
    if any(key.input_format is not None for key in table)
)


#: Every enrolled op's resolved budget on every gateable output format, at the standard
#: variant -- so this pins behaviour rather than restating what __post_init__ guarantees,
#: and pins it on the formats a Float32-only table leaves unbounded. ``None`` is the
#: tolerance metric. A retune changes the number here in the same diff that changes the
#: registry.
#:
#: Float32 alone was not enough: ``Square`` resolves to tolerance there, so the only
#: assertion it received was ``metric is Metric.TOLERANCE``, and it is deliberately
#: outside ``EXACT_BY_CONSTRUCTION``, so the ``<= 1`` canary skipped it too. Its
#: ``DEFAULT`` 4 and its ``Float16_b`` 1 were bounded only by ``MAX_MEANINGFUL_ULP`` --
#: 128 for bf16 and 1024 for fp16 -- and widening either to 100 passed the whole suite.
_EXPECTED_BUDGET = {
    # op: {output format: max_ulp, or None for the tolerance metric}
    MathOperation.Abs: {
        DataFormat.Float32: 0,
        DataFormat.Float16_b: 1,
        DataFormat.Float16: 1,
    },
    MathOperation.Neg: {
        DataFormat.Float32: 0,
        DataFormat.Float16_b: 1,
        DataFormat.Float16: 1,
    },
    MathOperation.Identity: {
        DataFormat.Float32: 0,
        DataFormat.Float16_b: 1,
        # Never measured: Identity was not in BROAD_SWEEP_OPS, so no sweep reached a
        # Float16 output for it. An unmeasured format falls back to tolerance.
        DataFormat.Float16: None,
    },
    MathOperation.Floor: {
        DataFormat.Float32: 0,
        DataFormat.Float16_b: 0,
        DataFormat.Float16: 0,
    },
    MathOperation.Ceil: {
        DataFormat.Float32: 0,
        DataFormat.Float16_b: 0,
        DataFormat.Float16: 0,
    },
    MathOperation.Trunc: {
        DataFormat.Float32: 0,
        DataFormat.Float16_b: 0,
        DataFormat.Float16: 0,
    },
    MathOperation.Square: {
        # 65536 steps measured on Float32, deliberately unenrolled.
        DataFormat.Float32: None,
        DataFormat.Float16_b: 1,
        DataFormat.Float16: 4,  # via DEFAULT
    },
    MathOperation.SigmoidAppx: {
        DataFormat.Float32: None,
        DataFormat.Float16_b: None,
        DataFormat.Float16: None,
    },
    MathOperation.GeluAppx: {
        DataFormat.Float32: None,
        DataFormat.Float16_b: None,
        DataFormat.Float16: None,
    },
    MathOperation.SfpuElwpow: {
        DataFormat.Float32: None,
        DataFormat.Float16_b: None,
        DataFormat.Float16: None,
    },
    MathOperation.SfpuXlogy: {
        DataFormat.Float32: None,
        DataFormat.Float16_b: None,
        DataFormat.Float16: None,
    },
}

#: The declared *tolerance* of every op that carries one, as ``(atol, rtol)``. The table
#: above says only "not a step budget", so without this a widened atol -- the same drift
#: a widened ``max_ulp`` would be -- passes every test in the file.
_EXPECTED_TOLERANCE = {
    MathOperation.SigmoidAppx: {fmt: (0.13, 0.05) for fmt in ULP_FORMATS},
    MathOperation.GeluAppx: {fmt: (0.13, 0.05) for fmt in ULP_FORMATS},
    MathOperation.SfpuElwpow: {fmt: (None, 0.15) for fmt in ULP_FORMATS},
    MathOperation.SfpuXlogy: {
        DataFormat.Float32: (0.14, None),
        DataFormat.Float16_b: (0.6, None),
        DataFormat.Float16: (0.12, None),
    },
}


def test_every_declared_tolerance_is_the_number_that_was_measured():
    """A tolerance contract is as widenable as a budget, and nothing else pins these:
    ``_EXPECTED_BUDGET`` only records that they are not step budgets."""
    assert set(_EXPECTED_TOLERANCE) == ONLY_EVER_TOLERANCE
    for op, per_format in sorted(
        _EXPECTED_TOLERANCE.items(), key=lambda kv: kv[0].name
    ):
        for fmt, (atol, rtol) in per_format.items():
            contract = accuracy_contract(op, output_format=fmt, arch=MEASURED_ARCH)
            assert contract.metric is Metric.TOLERANCE, f"{op.name} on {fmt.name}"
            assert (contract.atol, contract.rtol) == (
                atol,
                rtol,
            ), f"{op.name} {fmt.name}"


#: Every distinct budget each of the 19 sweep-derived transcendentals resolves to, per
#: output format, over the whole keyed variant space (input format x approximation mode x
#: Dest accumulation). ``None`` in a tuple means some variant resolves to the tolerance
#: metric.
#:
#: These ops need their own table because they cannot appear in the one above: every one
#: of their keys pins ``input_format``, and ``matches()`` rejects a pinned field against
#: an unset query, so the input-unset queries there resolve all 19 to the tolerance
#: contract -- which is the documented fallback, not a measurement, and pins none of the
#: ~230 emitted numbers. Nothing else did either: ``test_no_budget_exceeds_its_formats_
#: usable_ceiling`` only bounds them by the ceiling, so a regenerated table that widened
#: Tanh fp32->fp32 from 2 to 400000 (still under fp32's 419430) passed the whole file.
#:
#: Distinct values rather than one row per variant: it is the same information in 3 lines
#: per op instead of 24, and any change to any emitted number changes a tuple. Regenerate
#: with the emitter in the same diff that changes the registry.
_EXPECTED_TRANSCENDENTAL_BUDGETS = {
    MathOperation.Acosh: {
        DataFormat.Float32: (
            2,
            5117,
            40658,
            51200,
            81920,
            None,
        ),
        DataFormat.Float16_b: (
            1,
            2,
            None,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Asinh: {
        DataFormat.Float32: (
            2,
            5082,
            40879,
            51200,
            81920,
            None,
        ),
        DataFormat.Float16_b: (
            1,
            2,
            None,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Atanh: {
        DataFormat.Float32: (
            3,
            40842,
            163840,
            None,
        ),
        DataFormat.Float16_b: (
            1,
            2,
            3,
            None,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Celu: {
        DataFormat.Float32: (
            1,
            12,
            5088,
            40628,
            40960,
            None,
        ),
        DataFormat.Float16_b: (
            1,
            2,
            None,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Cos: {
        DataFormat.Float32: (
            1,
            2,
            5115,
            40925,
            40960,
        ),
        DataFormat.Float16_b: (
            1,
            2,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Elu: {
        DataFormat.Float32: (
            1,
            12,
            5088,
            40628,
            40960,
            None,
        ),
        DataFormat.Float16_b: (
            1,
            2,
            None,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Erfinv: {
        DataFormat.Float32: (
            30522,
            30720,
            31534,
            62735,
            81920,
            None,
        ),
        DataFormat.Float16_b: (
            2,
            None,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Exp: {
        DataFormat.Float32: (
            2,
            8813,
            40940,
            71680,
            81920,
            None,
        ),
        DataFormat.Float16_b: (
            1,
            2,
            None,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Exp2: {
        DataFormat.Float32: (
            2,
            5064,
            39759,
            71680,
            81920,
            None,
        ),
        DataFormat.Float16_b: (
            1,
            2,
            None,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Gelu: {
        DataFormat.Float32: (
            46656,
            71680,
            81920,
            None,
        ),
        DataFormat.Float16_b: (
            2,
            None,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Hardsigmoid: {
        DataFormat.Float32: (
            1024,
            5120,
            10240,
            40960,
            81920,
            None,
        ),
        DataFormat.Float16_b: (
            1,
            2,
            None,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Log: {
        DataFormat.Float32: (
            2,
            40867,
            None,
        ),
        DataFormat.Float16_b: (
            1,
            2,
            None,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Log1p: {
        DataFormat.Float32: (
            2,
            40893,
            60959,
            81920,
            None,
        ),
        DataFormat.Float16_b: (
            1,
            2,
            None,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Reciprocal: {
        DataFormat.Float32: (
            2,
            2060,
            5114,
            6649,
            10240,
            40799,
            40960,
            41598,
            81920,
        ),
        DataFormat.Float16_b: (
            1,
            2,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Rsqrt: {
        DataFormat.Float32: (
            1,
            3,
            5110,
            18409,
            21954,
            40904,
            40960,
            51200,
            54068,
            81920,
        ),
        DataFormat.Float16_b: (
            1,
            2,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Silu: {
        DataFormat.Float32: (
            3,
            5110,
            40955,
            71680,
            81920,
            None,
        ),
        DataFormat.Float16_b: (
            1,
            2,
            None,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Sin: {
        DataFormat.Float32: (
            1,
            2,
            5115,
            40885,
            40960,
        ),
        DataFormat.Float16_b: (
            1,
            2,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Sqrt: {
        DataFormat.Float32: (
            1,
            2,
            5115,
            18415,
            23392,
            40920,
            40960,
            51200,
            59334,
            81920,
        ),
        DataFormat.Float16_b: (
            1,
            2,
        ),
        DataFormat.Float16: (None,),
    },
    MathOperation.Tanh: {
        DataFormat.Float32: (
            2,
            5110,
            40675,
            61440,
            81920,
            None,
        ),
        DataFormat.Float16_b: (
            1,
            2,
            None,
        ),
        DataFormat.Float16: (None,),
    },
}

#: The Float32 column of ``_EXPECTED_BUDGET``, plus the transcendentals, which resolve
#: to tolerance for an input-unset query. Derived, not a second hand-written copy, so the
#: two cannot disagree about the same op.
_EXPECTED_FLOAT32_BUDGET = {
    **{
        op: per_format[DataFormat.Float32]
        for op, per_format in _EXPECTED_BUDGET.items()
    },
    **{op: None for op in _TRANSCENDENTALS_ENROLLED_WITH_AN_INPUT_FORMAT},
}


def test_every_enrolled_op_resolves_to_the_budget_it_declares_on_every_format():
    """The Float32 column below bounds only one format per op. This pins all three.

    Without it, ``Square``'s ``DEFAULT`` ``max_ulp=4`` and its ``Float16_b`` 1 -- the two
    numeric entries no other test reaches -- were bounded only by ``MAX_MEANINGFUL_ULP``,
    so widening either to 100 passed the whole suite. That is the "raise the budget until
    it stops failing" drift the registry's docstrings warn against, on exactly the
    entries nothing else held.
    """
    assert set(_EXPECTED_BUDGET) | set(_EXPECTED_TRANSCENDENTAL_BUDGETS) == set(
        enrolled_ops()
    ), "an op was enrolled or removed without updating the expected budgets"
    for op, per_format in sorted(_EXPECTED_BUDGET.items(), key=lambda kv: kv[0].name):
        assert set(per_format) == set(ULP_FORMATS), op.name
        for fmt, expected in per_format.items():
            contract = accuracy_contract(
                op,
                output_format=fmt,
                approx_mode=ApproximationMode.No,
                dest_acc=DestAccumulation.No,
                arch=MEASURED_ARCH,
            )
            where = f"{op.name} on {fmt.name}"
            if expected is None:
                assert contract.metric is Metric.TOLERANCE, where
            else:
                assert contract.metric is Metric.ULP, where
                assert contract.max_ulp == expected, where


def test_every_sweep_derived_budget_is_the_number_that_was_measured():
    """The value pin for the ~230 numbers the emitter produced.

    Nothing else holds them. ``test_every_enrolled_op_resolves_to_something_usable_on_a_
    float_format`` asserts properties ``AccuracyContract.__post_init__`` already
    guarantees, and ``test_no_budget_exceeds_its_formats_usable_ceiling`` only bounds
    them from above -- so widening Tanh fp32->fp32 from 2 to 400000 passed every test in
    this file. It fails here.
    """
    assert set(_EXPECTED_TRANSCENDENTAL_BUDGETS) == set(
        _TRANSCENDENTALS_ENROLLED_WITH_AN_INPUT_FORMAT
    ), "a sweep-derived op was enrolled or removed without updating the expected budgets"
    for op, per_format in sorted(
        _EXPECTED_TRANSCENDENTAL_BUDGETS.items(), key=lambda kv: kv[0].name
    ):
        assert set(per_format) == set(ULP_FORMATS), op.name
        for fmt, expected in per_format.items():
            seen = set()
            for input_format in ULP_FORMATS:
                for approx_mode in ApproximationMode:
                    for dest_acc in DestAccumulation:
                        contract = accuracy_contract(
                            op,
                            output_format=fmt,
                            input_format=input_format,
                            approx_mode=approx_mode,
                            dest_acc=dest_acc,
                            arch=MEASURED_ARCH,
                        )
                        seen.add(
                            contract.max_ulp if contract.metric is Metric.ULP else None
                        )
            ordered = tuple(sorted(v for v in seen if v is not None)) + (
                (None,) if None in seen else ()
            )
            assert ordered == expected, f"{op.name} on {fmt.name}"


def test_every_enrolled_op_resolves_to_the_budget_it_declares_on_float32():
    """``metric in (ULP, TOLERANCE)`` and ``max_ulp is not None and >= 0`` are both
    guaranteed the instant an ``AccuracyContract`` exists -- ``Metric`` is a closed
    two-member enum and ``__post_init__`` raises on a missing or negative ``max_ulp`` --
    so asserting them pinned nothing, and ``validate_registry()`` already covers a
    superset of "no exception raised".

    So this asserts the resolved number instead, which is the property worth holding."""
    assert set(_EXPECTED_FLOAT32_BUDGET) == set(
        enrolled_ops()
    ), "an op was enrolled or removed without updating the expected budgets"
    for op, expected in sorted(
        _EXPECTED_FLOAT32_BUDGET.items(), key=lambda kv: kv[0].name
    ):
        contract = accuracy_contract(
            op,
            output_format=DataFormat.Float32,
            approx_mode=ApproximationMode.No,
            dest_acc=DestAccumulation.No,
            arch=MEASURED_ARCH,
        )
        if expected is None:
            assert contract.metric is Metric.TOLERANCE, op.name
        else:
            assert contract.metric is Metric.ULP, op.name
            assert contract.max_ulp == expected, op.name


#: The enrolled ops that can never reach the ULP branch: every row they declare is a
#: tolerance. The coarse 3-segment LUT pair, and the two binary ops whose numbers moved
#: out of ``BINARY_CUSTOM_TOLERANCES`` -- all four keep the tolerance metric until there
#: is a measured step budget to replace it with.
ONLY_EVER_TOLERANCE = frozenset(
    {
        MathOperation.SigmoidAppx,
        MathOperation.GeluAppx,
        MathOperation.SfpuElwpow,
        MathOperation.SfpuXlogy,
    }
)


def test_every_enrolled_op_resolves_to_something_usable_on_a_float_format():
    """Through ``_every_variant``, not a hand-rolled loop with ``input_format`` unset.

    Every transcendental key pins ``input_format``, and ``matches()`` rejects a pinned
    field against an unset query, so a loop that left it out sent all 19 of them to
    ``TOLERANCE_CONTRACT`` and the ULP branch below never ran for any — a test named
    "every enrolled op" exercising only the nine that predate them.
    """
    saw_ulp = set()
    for op in enrolled_ops():
        for fmt, contract in _every_variant(op):
            assert contract.metric in (Metric.ULP, Metric.TOLERANCE)
            if contract.metric == Metric.ULP:
                assert contract.max_ulp is not None and contract.max_ulp >= 0
                saw_ulp.add(op)
    # The regression itself: the sweep has to reach the ULP branch for every enrolled op
    # except the two whose entire contract set is _COARSE_LUT_TOLERANCE, not silently
    # resolve all of them to tolerance. Named rather than written as a bare "- 2", so a
    # third op quietly slipping off the ULP branch fails instead of fitting the slack.
    assert set(enrolled_ops()) - saw_ulp == ONLY_EVER_TOLERANCE, sorted(
        op.name for op in set(enrolled_ops()) - saw_ulp
    )


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


def _refuses(match, kind=ValueError):
    """The suite's ``expect_error`` fixture needs a device; these are host-only tests."""
    return pytest.raises(kind, match=match)  # allow-pytest.raises: host-only test


def _every_variant(op):
    """Every contract an op can resolve to, across the whole keyed variant space.

    Passing only ``output_format`` is not enough: by the ``matches()`` rule an unset
    caller dimension cannot match a key that sets one, so any ``BudgetKey(arch=...)``,
    ``BudgetKey(dest_acc=...)`` or ``BudgetKey(input_format=...)`` entry is invisible to
    such a query. The last one is not hypothetical — every one of the enrolled
    transcendental entries pins ``input_format``, so a query without it dropped all 19 of
    those ops out of the guards below and left
    ``test_no_budget_exceeds_its_formats_meaningful_ceiling`` covering 9 ops instead of 28.
    """
    for fmt in ULP_CAPABLE_FORMATS:
        for input_format in list(ULP_CAPABLE_FORMATS) + [None]:
            for approx_mode in list(ApproximationMode) + [None]:
                for dest_acc in list(DestAccumulation) + [None]:
                    # arch is required; None is unrepresentable.
                    for arch in ChipArchitecture:
                        yield fmt, accuracy_contract(
                            op,
                            output_format=fmt,
                            input_format=input_format,
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


def test_no_budget_exceeds_its_formats_usable_ceiling():
    """The line past which a budget stops being *stronger* than the gate it replaces,
    applied to the table rather than to one call.

    ``min(rtol * 2**mantissa_bits, MAX_MEANINGFUL_ULP)``, the same bound
    ``usable_budget_ceiling`` refuses to declare past — not
    ``MAX_MEANINGFUL_ULP`` alone, which is roughly 100% relative error and about 20x
    looser: 128 for bf16 against 6. Because the ULP arm of ``passed_test`` returns before
    both ``isclose`` and PCC, a budget past this line *is* the whole gate, and
    ``passed_test`` only warns — so a hand-edited or regenerated bf16 entry anywhere in
    7..127 steps (``max_ulp=64`` is 50% relative error) used to pass this guard and every
    other host test. It is the invariant that closed the Tanh and Gelu budgets in review,
    and the table side could not see it.

    Ops past the line belong on the tolerance metric, as Square on Float32 and the Bfp8_b
    entries are. This is the guard against "the sweep reported 15616, so the budget is
    15616".
    """
    for op in enrolled_ops():
        for fmt, contract in _every_variant(op):
            if contract.metric != Metric.ULP:
                continue
            ceiling = usable_budget_ceiling(fmt)
            assert contract.max_ulp <= ceiling, (
                f"{op.name} on {fmt.name} has max_ulp={contract.max_ulp}, past the "
                f"{ceiling:.0f}-step point where a budget stops being tighter than the "
                "tolerance it replaces. Put the op on the tolerance metric and record "
                "the measurement instead."
            )


def test_the_usable_ceiling_is_tighter_than_the_meaningful_one():
    """Why the guard above moved off ``MAX_MEANINGFUL_ULP``: the two are not close, and
    the looser one admits budgets that gate nothing."""
    for fmt in ULP_FORMATS:
        meaningful = MAX_MEANINGFUL_ULP[ulp_dtype(fmt)]
        assert usable_budget_ceiling(fmt) < meaningful, fmt.name
    assert usable_budget_ceiling(DataFormat.Float16_b) == 6.4


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
        # Resolved at Bfp8_b, the format the invariant is about: identical to the default
        # Float16_b for these three today, but wrong the moment a format-sensitive spec is
        # enrolled here.
        spec = for_op(op, DataFormat.Bfp8_b).spec_A
        assert spec.low is not None and spec.high is not None, op.name
        # ceil of the bound, because the invariant is about *result* block maxima and
        # floor/ceil/trunc round outward by up to one integer. An input domain inside
        # (127, 128) -- uniform(-127.5, 127.5), say -- produces block maxima of exactly
        # 128, where the in-block step becomes 2**(7-6) = 2, odd integers stop being
        # representable and the 0-step budget is no longer a valid criterion.
        reachable = math.ceil(max(abs(spec.low), abs(spec.high)))
        assert reachable < BFP8_B_EXACT_INTEGER_DOMAIN, (
            f"{op.name} is swept over [{spec.low}, {spec.high}], whose results reach "
            f"{reachable} and so whose block maxima can reach "
            f"{BFP8_B_EXACT_INTEGER_DOMAIN}. Its 0-step Bfp8_b budget relied on every "
            "block maximum staying below that; re-measure before widening."
        )


# ── Integers never reach the ULP metric through the registry ──────────────────


@pytest.mark.parametrize("fmt", INTEGER_FORMATS, ids=lambda f: f.name)
def test_the_integer_short_circuit_holds_for_every_op(fmt):
    """``accuracy_contract`` consults the table first and *then* downgrades a ULP
    contract on ``not has_ulp_gate`` -- the ordering changed when the off-Wormhole
    tolerance fix landed -- so this still pins the downgrade rather than the table's
    contents: adding ``LeftShift: {DEFAULT: AccuracyContract(max_ulp=0)}`` would leave it
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
            # has_ulp_gate, not is_integer: accuracy_contract downgrades a ULP contract on
            # *any* format without a per-element ULP -- Bfp4_b, Bfp2_b, the MX formats and
            # Tf32 as well as the integers -- and validate_registry sweeps only the
            # gateable ones plus the proxies, so a BudgetKey(output_format=Bfp4_b) carrying
            # a measured max_ulp would have passed every guard here and gated nothing.
            assert fmt is None or has_ulp_gate(fmt), (
                f"{op.name} carries a step budget keyed on {fmt.name}, which has no "
                "per-element ULP, so accuracy_contract will silently downgrade it to the "
                "tolerance metric and the number will gate nothing."
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
        MathOperation.SfpuDivInt32,
        MathOperation.SfpuLcm,
        MathOperation.SfpuBitwiseAnd,
        MathOperation.SfpuRsubInt32,
    ):
        assert op in integer_ops, f"{op.name} dropped out of the integer-op derivation"

    enrolled = set(enrolled_ops())
    assert not (enrolled & integer_ops), sorted(
        op.name for op in enrolled & integer_ops
    )


# ── The contract refuses what its annotations cannot ──────────────────────────


@pytest.mark.parametrize("bogus", ["ulp", "tolerance", "pcc", 0, None], ids=repr)
def test_a_metric_that_is_not_a_metric_member_is_refused(bogus):
    """A type annotation is not a check. ``metric="ulp"`` is not ``Metric.ULP`` -- the
    enum is bare, so the string compares unequal -- and it used to fall through to the
    *tolerance* arm of ``__post_init__``, silently switching off the gate the entry meant
    to declare. Exactly the typo a registry edit makes."""
    with _refuses("must be a Metric member"):
        AccuracyContract(metric=bogus, max_ulp=1)


@pytest.mark.parametrize(
    "field, bogus",
    [
        ("approx_mode", True),
        ("output_format", "Float32"),
        ("dest_acc", True),
        ("dest_acc", False),
        ("arch", "wormhole"),
    ],
    ids=lambda v: str(v),
)
def test_a_budget_key_dimension_that_is_not_an_enum_member_is_refused(field, bogus):
    """The same rule as ``AccuracyContract.metric``, on the four dimensions that had no
    check. All of them are bare ``Enum``s, so ``DestAccumulation.No.value is False`` and
    ``ChipArchitecture.WORMHOLE.value == "wormhole"`` never compare equal to their
    members -- and the failure mode is silence, not an exception: such a key is counted
    as set by ``specificity``, matched by nothing in ``matches()``, seen as no duplicate
    by ``budget_table()`` and as no tie by ``validate_registry()``, and rendered
    identically to the correct key by ``describe()``, since ``ChipArchitecture.__str__``
    returns ``.value``. The budget it declares would gate nothing at all.
    """
    with _refuses(f"BudgetKey.{field} must be a"):
        BudgetKey(**{field: bogus})


def test_every_budget_key_field_is_guarded():
    """``_BUDGET_KEY_TYPES`` is hand-maintained beside the dataclass, so a dimension
    added to one and not the other would be unguarded and silently inert."""
    from dataclasses import fields

    assert {f.name for f in fields(BudgetKey)} == set(_BUDGET_KEY_TYPES)
    # ...and the declared member of each really is accepted.
    assert BudgetKey(
        approx_mode=ApproximationMode.No,
        input_format=DataFormat.Float16_b,
        output_format=DataFormat.Float32,
        dest_acc=DestAccumulation.Yes,
        arch=ChipArchitecture.WORMHOLE,
    ).specificity == len(_BUDGET_KEY_TYPES)


def _table(tmp_path, text):
    """*text* as a budget table on disk, loaded the way the real one is."""
    path = tmp_path / "budget.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return _load_table(path)


def test_a_repeated_op_in_the_table_is_refused(tmp_path):
    """YAML keeps only the last of two identical mapping keys, so the earlier op's whole
    budget would vanish with nothing downstream able to see it."""
    with _refuses("duplicate entry for 'Abs'"):
        _table(
            tmp_path,
            """\
            Abs:
              - {max_ulp: 0}
            Neg:
              - {max_ulp: 1}
            Abs:
              - {max_ulp: 99}
            """,
        )


def test_a_duplicate_row_is_refused_rather_than_deduplicated(tmp_path):
    """Two rows with the same key are two list items, not one -- so nothing collapses
    them, and a copy-pasted row replacing a measured budget would otherwise take effect
    silently as the later of the two."""
    with _refuses("repeats BudgetKey"):
        _table(
            tmp_path,
            """\
            Abs:
              - {out: Float16_b, max_ulp: 1}
              - {out: Float16_b, max_ulp: 4}
            """,
        )
    both = _table(
        tmp_path,
        """\
        Abs:
          - {max_ulp: 4}
          - {out: Float16_b, max_ulp: 1}
        """,
    )
    assert len(both[MathOperation.Abs]) == 2


def test_the_loader_refuses_what_it_cannot_turn_into_a_contract(tmp_path):
    """Every failure here is the author's, so each one names the op it came from."""
    with _refuses("'Nope' is not a MathOperation"):
        _table(tmp_path, "Nope:\n  - {max_ulp: 1}\n")
    with _refuses("unknown field"):
        _table(tmp_path, "Abs:\n  - {max_ulp: 1, budget: 2}\n")
    with _refuses("not a DataFormat"):
        _table(tmp_path, "Abs:\n  - {out: Float17, max_ulp: 1}\n")
    with _refuses("has no rows"):
        _table(tmp_path, "Abs:\n")
    # ...and the contract invariants still come from AccuracyContract itself.
    with _refuses("a ulp contract replaces the tolerance gate"):
        _table(tmp_path, "Abs:\n  - {max_ulp: 1, atol: 0.5}\n")


def test_a_quoted_and_an_unquoted_no_mean_the_same_thing(tmp_path):
    """YAML 1.1 reads a bare ``No`` as ``False``, and ``ApproximationMode.No`` is spelled
    ``False`` too, so the two spellings must not disagree. The table quotes them; the
    loader takes either."""
    quoted = _table(tmp_path, 'Abs:\n  - {approx: "No", dest: "Yes", max_ulp: 1}\n')
    bare = _table(tmp_path, "Abs:\n  - {approx: No, dest: Yes, max_ulp: 1}\n")
    assert quoted == bare
    key = next(iter(quoted[MathOperation.Abs]))
    assert key.approx_mode is ApproximationMode.No
    assert key.dest_acc is DestAccumulation.Yes


@pytest.mark.parametrize("field", ["atol", "rtol", "near_zero_atol"], ids=str)
def test_a_negative_tolerance_field_is_refused(field):
    """``passed_test`` applies an override only ``if custom_atol is not None and
    custom_atol >= 0``, so a negative ``atol``/``rtol`` read in the table as a declared
    tolerance and then silently ran against the per-format default. A negative
    ``near_zero_atol`` is a silently inert floor. The ULP arm already rejected a negative
    ``max_ulp`` -- the one case that would have failed loudly anyway."""
    metric = Metric.ULP if field == "near_zero_atol" else Metric.TOLERANCE
    kwargs = {"metric": metric, field: -0.001}
    if metric is Metric.ULP:
        kwargs["max_ulp"] = 1
    with _refuses("must not be negative"):
        AccuracyContract(**kwargs)


@pytest.mark.parametrize(
    "arch", [a for a in ChipArchitecture if a != MEASURED_ARCH], ids=lambda a: a.name
)
def test_a_key_that_names_an_architecture_binds_on_it(arch):
    """The arch gate exists to stop *unkeyed* WH measurements binding elsewhere. A key
    that names the architecture explicitly is a measurement someone took there, and
    downgrading it made the advertised arch dimension impossible to use for enrolling
    Blackhole or Quasar -- which the previous round's test missed, because it called
    ``resolve_contract`` directly and never went through the public API."""
    op = MathOperation.Abs
    table = _SFPU_ACCURACY_BUDGET[op]
    pinned = BudgetKey(output_format=DataFormat.Float16, arch=arch)
    table[pinned] = AccuracyContract(max_ulp=7)
    try:
        contract = accuracy_contract(
            op,
            output_format=DataFormat.Float16,
            approx_mode=ApproximationMode.No,
            dest_acc=DestAccumulation.No,
            arch=arch,
        )
        assert contract.metric is Metric.ULP, "an arch-keyed budget must survive"
        assert contract.max_ulp == 7
        # A shared WH-measured entry on the same arch still does not bind.
        shared = accuracy_contract(
            op,
            output_format=DataFormat.Float32,
            approx_mode=ApproximationMode.No,
            dest_acc=DestAccumulation.No,
            arch=arch,
        )
        assert shared is TOLERANCE_CONTRACT
    finally:
        del table[pinned]


def test_no_enrolled_op_is_driven_by_a_sweep_that_was_never_measured():
    """The enrolment rule names five stimulus sources; three of them are hand-built specs
    that no recorded measurement covers.

    ``_signbit``, ``_isinf_isnan`` and ``_threshold`` each run ULP-gateable
    ``Float16_b``/``Float32`` through the same driver, so a budget binds there too, with
    the ULP arm returning before both the tolerance gate and PCC. None of the nine
    enrolled ops reaches them today -- but ``ReluMin``/``ReluMax`` are parked in the
    registry "for a later pass" and the threshold sweep drives exactly those two, with
    every third lane on the tie. This fails at that enrolment rather than in a device run.
    """
    from test_eltwise_unary_sfpu import _THRESHOLD_OPS, ISINF_ISNAN_MATHOPS

    unmeasured = {
        # The signbit sweep is not parametrised over a list; it drives this one op.
        "signbit": {MathOperation.Signbit},
        "isinf_isnan": set(ISINF_ISNAN_MATHOPS),
        "threshold": set(_THRESHOLD_OPS),
    }
    assert all(unmeasured.values()), "a sweep set went empty; the derivation has broken"

    enrolled = set(enrolled_ops())
    for sweep, ops in sorted(unmeasured.items()):
        overlap = sorted(op.name for op in enrolled & ops)
        assert not overlap, (
            f"{', '.join(overlap)} carries a budget but is driven by the {sweep} sweep, "
            "whose hand-built stimulus no recorded measurement covers. Measure it there "
            "before enrolling, or key the budget away from the formats it reaches."
        )
