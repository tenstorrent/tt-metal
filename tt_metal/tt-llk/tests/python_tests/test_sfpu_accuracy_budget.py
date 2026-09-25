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
import re
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
    _TABLE_PATH,
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
from helpers.sfpu_domains import exclude_undefined, for_op_pipeline
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from helpers.ulp import (
    _ULP_PROXY_DTYPES,
    MAX_MEANINGFUL_ULP,
    ULP_FORMATS,
    has_ulp_gate,
    ulp_dtype,
)
from helpers.utils import passed_test


def _refuses(match, kind=ValueError):
    """The suite's ``expect_error`` fixture needs a device; these are host-only tests."""
    return pytest.raises(kind, match=match)  # allow-pytest.raises: host-only test


def _integer_only_ops() -> set:
    """Every SFPU op whose operands and result are integers, from canonical sources.

    Four of them, because the harness has no single classification that covers all:

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
# is_integer() does not reach them, and without this row they would have sat in neither
# arm. Widening is_integer() instead would route them through
# _mxint_block_aware_compare in test_an_integer_format_still_works_on_the_default_gate.
BLOCK_FORMATS_WITHOUT_ULP = sorted(
    (f for f in DataFormat if not has_ulp_gate(f) and not f.is_integer()),
    key=lambda f: f.name,
)


# ── The contract's own coherence ──────────────────────────────────────────────


# A half-finished conversion is the case these cover: an entry carrying both metrics,
# or neither's required field, reads as deliberate and would gate on whichever arm
# `passed_test` happens to take.
@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"metric": Metric.ULP}, "needs max_ulp"),
        ({"max_ulp": 1, "atol": 0.13}, "silently ignored"),
        ({"metric": Metric.TOLERANCE, "max_ulp": 1}, "belong to the ulp metric"),
        ({"max_ulp": -1}, "must not be negative"),
        # YAML 1.1 reads `true` as a bool, and bool is an int; `1.0e+1` lands as a float.
        ({"max_ulp": True}, "must be an int step count"),
        ({"max_ulp": 1.0}, "must be an int step count"),
        # `atol: yes` would apply as 1.0; `.nan` is silently ignored by passed_test and
        # `.inf` makes its gate unconditional.
        ({"metric": Metric.TOLERANCE, "atol": True}, "must be a number"),
        ({"metric": Metric.TOLERANCE, "atol": float("nan")}, "must be finite"),
        ({"metric": Metric.TOLERANCE, "rtol": float("inf")}, "must be finite"),
    ],
    ids=[
        "no-budget",
        "both-metrics",
        "budget-on-tolerance",
        "negative-budget",
        "bool-budget",
        "float-budget",
        "bool-atol",
        "nan-atol",
        "inf-rtol",
    ],
)
def test_an_incoherent_contract_is_refused(kwargs, match):
    with _refuses(match):
        AccuracyContract(**kwargs)


def test_the_metric_is_a_closed_set():
    """Two members and no third gate to fall through to.

    Not a substitute for ``test_a_metric_that_is_not_a_metric_member_is_refused`` further
    down: a bare ``Enum`` does not stop ``metric="ulp"`` from being *constructed*, only
    from being one of these. The closed set is what makes the ``__post_init__`` check a
    complete one."""
    assert set(Metric) == {Metric.ULP, Metric.TOLERANCE}
    assert AccuracyContract(max_ulp=1).metric is Metric.ULP
    assert TOLERANCE_CONTRACT.metric is Metric.TOLERANCE


#: Both translations, because only `tolerance_kwargs` has a production caller and only
#: `passed_test_kwargs` is otherwise pinned. `passed_test` takes `max_ulp` and
#: `near_zero_atol` as optional kwargs, so a `tolerance_kwargs` that regressed from
#: `return {}` to the `passed_test_kwargs` shape would not raise -- it would swap every
#: enrolled op from tolerance+PCC to a whole-format ULP budget, silently.
_KWARGS_METHODS = ("passed_test_kwargs", "tolerance_kwargs")

#: The ULP arm differs between them: `tolerance_kwargs` deliberately declines to gate.
_TRANSLATIONS = [
    (
        AccuracyContract(max_ulp=3),
        {"max_ulp": 3, "near_zero_atol": None},
        {},
    ),
    (
        AccuracyContract(max_ulp=3, near_zero_atol=1e-7),
        {"max_ulp": 3, "near_zero_atol": 1e-7},
        {},
    ),
    (
        AccuracyContract(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05),
        {"custom_atol": 0.13, "custom_rtol": 0.05},
        {"custom_atol": 0.13, "custom_rtol": 0.05},
    ),
    (
        TOLERANCE_CONTRACT,
        {"custom_atol": None, "custom_rtol": None},
        {"custom_atol": None, "custom_rtol": None},
    ),
]


@pytest.mark.parametrize("contract, by_ulp, by_tolerance", _TRANSLATIONS)
def test_a_contract_translates_to_passed_test_arguments(contract, by_ulp, by_tolerance):
    assert contract.passed_test_kwargs() == by_ulp
    assert contract.tolerance_kwargs() == by_tolerance


@pytest.mark.parametrize("method", _KWARGS_METHODS)
def test_every_contract_is_accepted_by_passed_test(method):
    """The translation has to be callable, not merely shaped right — a renamed keyword
    would otherwise only surface on hardware."""
    golden = torch_ones()
    for contract, _, _ in _TRANSLATIONS:
        assert passed_test(
            golden,
            golden.clone(),
            DataFormat.Float16_b,
            **getattr(contract, method)(),
        )


TILE_SIZE = DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM


def torch_ones():
    return torch.ones(TILE_SIZE, dtype=torch.bfloat16)


# ── Key resolution ────────────────────────────────────────────────────────────


def test_the_default_key_matches_every_variant():
    key = DEFAULT
    assert key.specificity == 0
    assert key.matches(
        BudgetKey(
            approx_mode=ApproximationMode.Yes,
            input_format=DataFormat.Float32,
            output_format=DataFormat.Float32,
            dest_acc=DestAccumulation.No,
            arch=ChipArchitecture.WORMHOLE,
        )
    )
    assert key.matches(
        BudgetKey(
            approx_mode=None,
            input_format=None,
            output_format=None,
            dest_acc=None,
            arch=None,
        )
    )


def test_a_more_specific_key_wins_over_the_default():
    table = {
        DEFAULT: AccuracyContract(max_ulp=64),
        BudgetKey(output_format=DataFormat.Float32): AccuracyContract(max_ulp=4),
    }
    assert (
        resolve_contract(
            table, BudgetKey(output_format=DataFormat.Float32), label="op"
        ).max_ulp
        == 4
    )
    assert (
        resolve_contract(
            table, BudgetKey(output_format=DataFormat.Float16_b), label="op"
        ).max_ulp
        == 64
    )


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
            table, BudgetKey(output_format=DataFormat.Float32, arch=arch), label="op"
        )
        assert resolved.max_ulp == expected, arch


def test_an_unset_query_dimension_only_matches_a_wildcard():
    """A caller that does not know ``dest_acc`` must not be handed a budget measured for
    one setting of it."""
    table = {BudgetKey(dest_acc=DestAccumulation.Yes): AccuracyContract(max_ulp=1)}
    assert (
        resolve_contract(table, BudgetKey(output_format=DataFormat.Float32), label="op")
        is TOLERANCE_CONTRACT
    )


def test_specificity_counts_every_set_dimension():
    """The only comparison of two *non-DEFAULT* keys: a 2-field key beating a 1-field
    one, and the 1-field one winning where the 2-field key does not match.

    Load-bearing here as it was not before: the ``Fill`` block now stacks 1-, 2- and
    4-field keys on one op, and ``resolve_contract`` raises only on an equal-specificity
    tie -- so a miscount would silently repoint budgets while every bounds-only guard in
    this file still passed.
    """
    table = {
        BudgetKey(output_format=DataFormat.Float32): AccuracyContract(max_ulp=4),
        BudgetKey(
            output_format=DataFormat.Float32, dest_acc=DestAccumulation.No
        ): AccuracyContract(max_ulp=8),
    }
    resolved = resolve_contract(
        table,
        BudgetKey(output_format=DataFormat.Float32, dest_acc=DestAccumulation.No),
        label="op",
    )
    assert resolved.max_ulp == 8
    resolved = resolve_contract(
        table,
        BudgetKey(output_format=DataFormat.Float32, dest_acc=DestAccumulation.Yes),
        label="op",
    )
    assert resolved.max_ulp == 4


def test_a_key_describes_itself_for_an_error_message():
    assert DEFAULT.describe() == "DEFAULT"
    described = BudgetKey(
        output_format=DataFormat.Float32, dest_acc=DestAccumulation.No
    ).describe()
    assert "output_format" in described and "dest_acc" in described


def test_a_query_left_over_from_another_test_is_replaced_not_flagged(monkeypatch):
    """`--ulp-measure` associates a reading with the variant `accuracy_contract` was
    last asked about, and refuses to record when two lookups race one comparison. That
    has to mean two lookups *in one test*.

    The exhaustive sweep resolves a contract and then skips the cell when it is on the
    tolerance metric, leaving a query nobody consumed. Treating that as ambiguity threw
    away the next test's reading: measured, it dropped all 40 readings that followed a
    skip in a 130-test run.
    """
    import helpers.sfpu_accuracy_budget as budget

    def resolve(test_id):
        monkeypatch.setenv("PYTEST_CURRENT_TEST", f"{test_id} (call)")
        accuracy_contract(
            MathOperation.Abs,
            output_format=DataFormat.Float16_b,
            arch=MEASURED_ARCH,
        )

    budget.LAST_QUERY = None
    budget.PENDING_AMBIGUOUS = False

    # A cell that resolved and then skipped, followed by a different test: ordinary.
    resolve("t_one")
    resolve("t_two")
    assert not budget.PENDING_AMBIGUOUS
    assert budget.LAST_QUERY[0] == "t_two"

    # Two lookups inside one test with nothing consumed between them: not ordinary.
    resolve("t_three")
    resolve("t_three")
    assert budget.PENDING_AMBIGUOUS

    budget.LAST_QUERY = None
    budget.PENDING_AMBIGUOUS = False


def test_enrolled_ops_is_sorted_and_stable():
    ops = enrolled_ops()
    assert list(ops) == sorted(ops, key=lambda op: op.name)
    assert len(set(ops)) == len(ops)


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
            BudgetKey(
                output_format=DataFormat.Float32, approx_mode=ApproximationMode.No
            ),
            label="Ambiguous",
        )


def test_an_empty_table_falls_back_to_the_tolerance_metric():
    assert (
        resolve_contract({}, BudgetKey(output_format=DataFormat.Float32), label="op")
        is TOLERANCE_CONTRACT
    )


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


@pytest.mark.parametrize("arch", list(ChipArchitecture), ids=lambda a: a.name)
@pytest.mark.parametrize(
    "output_format",
    [DataFormat.Float16_b, DataFormat.Bfp4_b],
    ids=lambda f: f.name,
)
def test_a_downgrade_lands_on_the_ops_own_tolerance_row(
    arch, output_format, monkeypatch
):
    """Both downgrades re-resolve against the tolerance rows rather than returning the
    global default: a ULP row winning on specificity must not shadow a *broader*
    tolerance row the same op declares, or an op carrying both shapes falls back past
    its own atol to the per-format default.

    Built here rather than taken from the registry -- no live op carries both a ULP row
    and a wider declared atol yet, and the shadow is only observable through
    ``passed_test_kwargs()``.
    """
    op = MathOperation.Abs
    monkeypatch.setitem(
        _SFPU_ACCURACY_BUDGET,
        op,
        {
            DEFAULT: AccuracyContract(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05),
            BudgetKey(output_format=DataFormat.Float16_b): AccuracyContract(max_ulp=3),
        },
    )
    contract = accuracy_contract(op, output_format=output_format, arch=arch)
    if output_format is DataFormat.Float16_b and arch is MEASURED_ARCH:
        assert contract.metric is Metric.ULP and contract.max_ulp == 3
    else:
        # Bfp4_b has no per-element ULP, and off Wormhole the unkeyed budget does not
        # bind -- either way the op keeps the 0.13 it declared, not the global default.
        assert contract.metric is Metric.TOLERANCE
        assert contract.atol == 0.13 and contract.rtol == 0.05


@pytest.mark.parametrize(
    "arch", [a for a in ChipArchitecture if a != MEASURED_ARCH], ids=lambda a: a.name
)
def test_a_ulp_row_that_names_its_arch_binds_there(arch, monkeypatch):
    """The arch gate exists because unkeyed numbers are Wormhole measurements. A row
    whose key names another arch *is* a measurement taken there, and must bind."""
    op = MathOperation.Abs
    monkeypatch.setitem(
        _SFPU_ACCURACY_BUDGET,
        op,
        {
            BudgetKey(output_format=DataFormat.Float16_b): AccuracyContract(max_ulp=3),
            BudgetKey(output_format=DataFormat.Float16_b, arch=arch): AccuracyContract(
                max_ulp=5
            ),
        },
    )
    contract = accuracy_contract(op, output_format=DataFormat.Float16_b, arch=arch)
    assert contract.metric is Metric.ULP and contract.max_ulp == 5
    # ...while the unkeyed row is still downgraded on any other unswept arch.
    for other in ChipArchitecture:
        if other not in (arch, MEASURED_ARCH):
            assert (
                accuracy_contract(op, output_format=DataFormat.Float16_b, arch=other)
                is TOLERANCE_CONTRACT
            )


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


@pytest.mark.parametrize("enrolled", [False, True], ids=["unenrolled", "enrolled"])
def test_a_bad_query_is_refused_whether_or_not_the_op_is_enrolled(enrolled):
    """Validation must not depend on the table's contents, or a miswired driver passes
    until the day its op is enrolled. The op is picked from the table, not named, so
    enrolling more ops later does not change what this checks."""
    op = next(
        op
        for op in sorted(MathOperation, key=lambda op: op.name)
        if (op in _SFPU_ACCURACY_BUDGET) == enrolled
    )
    with _refuses("BudgetKey.arch must be a"):
        accuracy_contract(op, output_format=DataFormat.Float32, arch="wormhole")


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
        table,
        BudgetKey(output_format=DataFormat.Float32, arch=MEASURED_ARCH),
        label="probe",
    )
    assert narrow.atol == 0.001
    broad = resolve_contract(
        table,
        BudgetKey(output_format=DataFormat.Float16_b, arch=MEASURED_ARCH),
        label="probe",
    )
    assert broad.atol == 0.13


#: The ops enrolled from the accuracy sweep. Their keys all pin ``input_format``, which
#: is what makes them the witnesses for the resolution regression below.
_TRANSCENDENTALS_ENROLLED_WITH_AN_INPUT_FORMAT = frozenset(
    op
    for op, table in _SFPU_ACCURACY_BUDGET.items()
    if any(key.input_format is not None for key in table)
)


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
# Sign and Heaviside are not here, despite the -0.0 divergence: WH's bit-pattern compare
# reads -0.0 as negative, so a 0-ULP budget on a cell where that lane is in play would
# fail a kernel behaving as specified. The whole-format sweep measures each cell
# separately, and both carry a budget on the cells where that lane is not in play.
# GeluTanh, Tanhshrink and SfpuElwmul used to sit here, on a per-op-per-format maximum
# that was past the ceiling everywhere. The full sweep measures each variant separately,
# and some of their cells are well inside it -- GeluTanh's Float32 worst lane is 8.7e8
# steps, but not in every cell. They now carry a budget where one is meaningful and fall
# through to tolerance elsewhere, which is what per-variant keying is for.


def test_every_enrolled_op_resolves_to_something_usable_on_a_float_format():
    """Through ``_every_variant``, not a hand-rolled loop with ``input_format`` unset.

    Every transcendental key pins ``input_format``, and ``matches()`` rejects a pinned
    field against an unset query, so a loop that left it out sent all 19 of them to
    ``TOLERANCE_CONTRACT`` and the ULP branch below never ran for any — a test named
    "every enrolled op" exercising only the nine that predate them.
    """
    assert len(_TRANSCENDENTALS_ENROLLED_WITH_AN_INPUT_FORMAT) == 70, sorted(
        op.name for op in _TRANSCENDENTALS_ENROLLED_WITH_AN_INPUT_FORMAT
    )
    saw_ulp = set()
    for op in enrolled_ops():
        for _, fmt, contract in _every_variant(op):
            assert contract.metric in (Metric.ULP, Metric.TOLERANCE)
            if contract.metric == Metric.ULP:
                assert contract.max_ulp is not None and contract.max_ulp >= 0
                saw_ulp.add(op)
    # The regression itself: the sweep has to reach the ULP branch for every enrolled op
    # except the four in ONLY_EVER_TOLERANCE -- the coarse-LUT pair and the two binary
    # ops carrying their own per-format rtol/atol -- not silently resolve all of them to
    # tolerance. Named rather than written as a bare "- 4", so a fifth op quietly
    # slipping off the ULP branch fails instead of fitting the slack.
    assert set(enrolled_ops()) - saw_ulp == ONLY_EVER_TOLERANCE, sorted(
        op.name for op in set(enrolled_ops()) - saw_ulp
    )
    # And specifically the input-keyed ones: the loop that left `input_format` unset
    # sent exactly these to TOLERANCE_CONTRACT, so they are the regression's witnesses.
    # Less the documented tolerance-only ops, which are input-keyed as well now that
    # every input format is swept -- they are excused above, by name.
    assert (
        _TRANSCENDENTALS_ENROLLED_WITH_AN_INPUT_FORMAT - ONLY_EVER_TOLERANCE <= saw_ulp
    )


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

#: The input axis the guards sweep. Wider than the output axis, and derived from the
#: table rather than from ``ULP_CAPABLE_FORMATS``: an input format only has to be
#: *unpackable*, not gateable, so a `{in: Bfp4_b, out: Bfp8_b}` row was invisible to
#: every guard here while resolving perfectly well for a driver.
_QUERYABLE_INPUT_FORMATS = sorted(
    {
        key.input_format
        for table in _SFPU_ACCURACY_BUDGET.values()
        for key in table
        if key.input_format is not None
    }
    | set(ULP_CAPABLE_FORMATS),
    key=lambda f: f.name,
) + [None]

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


#: Ops whose correct result is the *only* result: a predicate writing 1.0/0.0, a
#: selection that passes an operand through or zeroes it, a constant fill, a clamp, an
#: integer-valued result, or a single IEEE add. Unlike EXACT_BY_CONSTRUCTION above, one
#: step of slack is not the pack path here -- it is the contract breaking. Listed by what
#: the op computes, deliberately not derived from the table: a set read back out of the
#: rows would agree with them by construction and pin nothing.
EXACT_ZERO_BY_CONSTRUCTION = (
    MathOperation.Floor,
    MathOperation.Ceil,
    MathOperation.Trunc,
    MathOperation.Fill,
    MathOperation.Threshold,
    MathOperation.Isfinite,
    MathOperation.Isinf,
    MathOperation.Isnan,
    MathOperation.Isneginf,
    MathOperation.Isposinf,
    MathOperation.LogicalNot,
    MathOperation.Signbit,
    MathOperation.UnaryEq,
    MathOperation.UnaryNe,
    MathOperation.SfpuElwEq,
    MathOperation.SfpuElwNe,
    MathOperation.SfpuElwGt,
    MathOperation.SfpuElwGe,
    MathOperation.SfpuElwLt,
    MathOperation.SfpuElwLe,
    MathOperation.SfpuIsclose,
    MathOperation.SfpuMask,
    MathOperation.SfpuAddTopRow,
)


#: What an exactly-rounded op may still cost on a cell that *converts*. The op is exact;
#: the output conversion is not, and the exhaustive sweep measures 2 steps where a
#: sampled domain measured 0 -- it reaches the magnitudes where bf16->fp16 rounds. Same
#: reasoning as ``test_an_exact_op_never_carries_a_wide_budget``'s one step of slack,
#: one wider because this sweep leaves nothing out. Past this it is not the pack path.
_PACK_PATH_STEPS = 2


def _mantissa_bits(fmt):
    """From the metric's own table, so a format added there is covered here.

    ``0`` for a format with no per-element ULP: the coarser block floats appear on the
    *input* axis only, where all that matters is that nothing downstream is narrower
    than they are.
    """
    from helpers.ulp import _ULP_DTYPES, has_ulp_gate

    if not has_ulp_gate(fmt):
        return 0
    return _ULP_DTYPES[ulp_dtype(fmt)].mantissa_bits


def _exact_allowance(op, input_format, output_format):
    """How much slack an exactly-rounded *op* may carry on one cell, and why.

    Per cell, not per op: the allowance is the cost of the output *pack*, and granting
    it unconditionally let a row that cannot pay it widen from 0 to 2 with this guard
    still green.

    Two rules, because the two lists differ in what the pack can move:

    * a value-passing op -- ``Abs``, ``Neg``, ``Identity`` -- pays it on every cell,
      same-format included. Dest holds fp32 and the pack back to a 16-bit output rounds,
      which is exactly the single step the table records for all three on bf16->bf16.
    * an op in ``EXACT_ZERO_BY_CONSTRUCTION`` writes 1.0/0.0, a constant, or an integer,
      and those survive a pack the output format can represent. It pays only where the
      output has *fewer* mantissa bits than the input: bf16 cannot hold every integer
      fp16 can, which is the 1 step Floor/Ceil/Trunc/Threshold measure on
      ``Float16 -> Float16_b`` and nowhere else.

    An unset *input_format* resolves the row that wildcards ``in:``, which covers the
    narrowing cells too, so it gets the allowance.
    """
    if output_format in _ULP_PROXY_DTYPES:
        # Not skipped, as it used to be. A block float's spacing is the block's rather
        # than the op's, which is why the exhaustive rows that measured 2-3 steps into
        # Bfp8_b are parked on `metric: tolerance` -- so any Bfp8_b row that *is* on the
        # ULP metric here is the 0-step enrolment, and the only other magnitude bound
        # left for it is the 25.6-step usable ceiling. A regenerated 20 would have
        # passed every host guard in this file.
        return 0, "a Bfp8_b ULP row here is the 0-step enrolment or nothing"
    if op not in EXACT_ZERO_BY_CONSTRUCTION:
        return (
            _PACK_PATH_STEPS,
            "the value passes through an fp32 Dest and is packed back",
        )
    if input_format is None:
        return (
            _PACK_PATH_STEPS,
            "the row wildcards `in:`, so it covers a narrowing cell",
        )
    if _mantissa_bits(output_format) < _mantissa_bits(input_format):
        return _PACK_PATH_STEPS, "the output has fewer mantissa bits than the input"
    return 0, "the output can represent every value this op produces from that input"


@pytest.mark.parametrize("op", EXACT_ZERO_BY_CONSTRUCTION, ids=lambda op: op.name)
def test_an_exactly_rounded_op_carries_a_zero_budget(op):
    """ "Any drift at all is a regression" is this stack's claim for these ops, and a
    budget of 1 would retire it silently -- the provenance guard permits a measured 0 to
    be written as 1, so nothing else here would notice.

    The pack-path allowance is per *cell*, not per op: a same-format row performs no
    output conversion, and a predicate's 1.0/0.0 is exact everywhere, so both are held
    at 0 while a converting cell may carry ``_PACK_PATH_STEPS``.
    """
    seen = False
    for in_fmt, fmt, contract in _every_variant(op):
        if contract.metric == Metric.ULP:
            seen = True
            allowance, why = _exact_allowance(op, in_fmt, fmt)
            assert contract.max_ulp <= allowance, (
                f"{op.name} on {in_fmt and in_fmt.name}->{fmt.name} carries "
                f"max_ulp={contract.max_ulp}, past the {allowance} it may have because "
                f"{why}. This op is exactly rounded by construction; anything more is "
                "the contract going away. Re-measure before widening it."
            )
    assert seen, f"{op.name} resolves to no ULP contract at all; the row was dropped"


def _every_variant(op):
    """Every contract an op can resolve to, across the whole keyed variant space.

    Yields ``(input_format, output_format, contract)``: the input is part of the answer,
    not only part of the query, because whether a cell *converts* is what decides how
    much slack an exactly-rounded op may carry.

    Passing only ``output_format`` is not enough: by the ``matches()`` rule an unset
    caller dimension cannot match a key that sets one, so any ``BudgetKey(arch=...)``,
    ``BudgetKey(dest_acc=...)`` or ``BudgetKey(input_format=...)`` entry is invisible to
    such a query. The last one is not hypothetical — every one of the enrolled
    transcendental entries pins ``input_format``, so a query without it dropped all 19 of
    those ops out of the guards below and left
    ``test_no_budget_exceeds_its_formats_meaningful_ceiling`` covering 9 ops instead of 28.
    """
    for fmt in ULP_CAPABLE_FORMATS:
        for input_format in _QUERYABLE_INPUT_FORMATS:
            for approx_mode in list(ApproximationMode) + [None]:
                for dest_acc in list(DestAccumulation) + [None]:
                    # arch is required; None is unrepresentable.
                    for arch in ChipArchitecture:
                        yield input_format, fmt, accuracy_contract(
                            op,
                            output_format=fmt,
                            input_format=input_format,
                            approx_mode=approx_mode,
                            dest_acc=dest_acc,
                            arch=arch,
                        )


@pytest.mark.parametrize("op", EXACT_BY_CONSTRUCTION, ids=lambda op: op.name)
def test_an_exact_op_never_carries_a_wide_budget(op):
    """These ops clear a sign bit, copy, or land on an integer. The pack path is the only
    slack they may carry; more than that is not the op, and a budget hiding it defeats
    the point of having these enrolled as the canaries. The allowance is two steps rather
    than one because the exhaustive sweep reaches the magnitudes where a cross-format
    output actually rounds -- a sampled domain measured those cells at 0. Per cell, not
    per op: a same-format cell converts nothing, so it gets no allowance at all."""
    for in_fmt, fmt, contract in _every_variant(op):
        if contract.metric == Metric.ULP:
            allowance, why = _exact_allowance(op, in_fmt, fmt)
            assert contract.max_ulp <= allowance, (
                f"{op.name} on {in_fmt and in_fmt.name}->{fmt.name} carries "
                f"max_ulp={contract.max_ulp}, past the {allowance} it may have because "
                f"{why}. These ops are exact by construction; a budget this wide means "
                "the number was fitted to a failure. Investigate the datapath or the "
                "golden instead."
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
        for _, fmt, contract in _every_variant(op):
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

    The exhaustive sweep does not enrol a block float at all, whatever it measures: it
    enumerates a format in value order, so sixteen adjacent values share a block and the
    exponent fits all of them -- the best case for quantization, not a representative
    one. `Abs` and `Neg` read 393 steps that way and their rows are parked on
    `metric: tolerance` because of it, not because of the number. Every op still
    enrolled here is enrolled from a row measured the other way.
    """
    enrolled_on_bfp8 = {
        op
        for op in enrolled_ops()
        for _, fmt, contract in _every_variant(op)
        if fmt is DataFormat.Bfp8_b and contract.metric == Metric.ULP
    }
    # The ceiling is necessary but not sufficient, and it is not the guard: every
    # Bfp8_b cell already passes through `test_no_budget_exceeds_its_formats_usable_
    # ceiling`, because `ULP_CAPABLE_FORMATS` includes Bfp8_b via `_ULP_PROXY_DTYPES`.
    # A regeneration enrolling `Abs` at the 3 steps a sorted sweep reads would clear
    # 25.6, clear provenance, and be exactly the failure this test is named for. So the
    # bound is on *membership*, and it is an equality.
    ceiling = usable_budget_ceiling(DataFormat.Bfp8_b)
    for op in enrolled_ops():
        for _, fmt, contract in _every_variant(op):
            if fmt is DataFormat.Bfp8_b and contract.metric == Metric.ULP:
                assert contract.max_ulp <= ceiling, (
                    f"{op.name} carries a {contract.max_ulp}-step Bfp8_b budget against "
                    f"a {ceiling:.0f}-step ceiling; past it the number gates nothing."
                )
    assert enrolled_on_bfp8 == {
        MathOperation.Floor,
        MathOperation.Ceil,
        MathOperation.Trunc,
        # Fill is block-friendly by a different mechanism from the three above, and a
        # stronger one: its output is a single constant, so every block is uniform
        # whatever the input held and the shared exponent is exact by construction.
        #
        # Threshold is deliberately absent, and so is every op enrolled only through a
        # sampled `{in: Bfp4_b, out: Bfp8_b}` or `{in: Float32, out: Bfp8_b}` row. Those
        # 0s and 13-to-25s are the block exponent fitting a degenerate or narrow
        # stimulus, not the op: Threshold's came from uniform(-5, 5) against
        # THRESHOLD_T=5.0, where the pass-through branch never fires, while the
        # exhaustive sweep reads 16545. They are recorded as tolerance with their
        # measurements.
        MathOperation.Fill,
    }, sorted(op.name for op in enrolled_on_bfp8)


#: The input formats the unary driver pairs with a Bfp8_b *output* for the three
#: 0-step-enrolled ops: the BROAD_FORMATS 4x4 matrix plus the Bfp4_b-input row. The
#: budget spans all five pipelines, and `for_op_pipeline` resolves range against the
#: input format -- Bfp8_b is absent from `_FORMAT_MAX_MAGNITUDE`, so it inherits the
#: maximal bf16 fallback, never wins `narrowest_range_format`, and the output narrows
#: nothing. Asserting only the Bfp8_b->Bfp8_b diagonal would pin one of the five.
_BFP8_B_OUTPUT_INPUT_FORMATS = (
    DataFormat.Float32,
    DataFormat.Float16,
    DataFormat.Float16_b,
    DataFormat.Bfp8_b,
    DataFormat.Bfp4_b,
)


@pytest.mark.parametrize(
    "op",
    [MathOperation.Floor, MathOperation.Ceil, MathOperation.Trunc],
    ids=lambda op: op.name,
)
@pytest.mark.parametrize(
    "input_format", _BFP8_B_OUTPUT_INPUT_FORMATS, ids=lambda f: f.name
)
def test_the_bfp8_b_enrolment_depends_on_the_swept_domain_not_on_the_format(
    op, input_format
):
    """A shared exponent does not represent integers exactly in general: the in-block step
    scales with the block maximum, so it is exact only while every maximum stays under
    ``2**7``. Floor/Ceil/Trunc qualify because ``_OP_DOMAIN_REGISTRY`` bounds them to
    ``uniform(-10, 10)`` -- a property of the *stimulus*, not of the format, and the
    same mechanism takes Abs and Neg to 15616 steps. So it is asserted, not assumed.

    Over every pipeline the budget covers, resolved the way the driver resolves it: all
    three specs are static ``uniform(-10, 10)`` today, but `_OP_DOMAIN_REGISTRY` entries
    may be callables of `data_format` -- `_reciprocal_spec` already is -- so a
    format-sensitive spec here could push one input pipeline past the bound.

    Floor/Ceil/Trunc only, though five ops are enrolled on Bfp8_b. This bounds the
    *input* spec, and the fifth -- ``Fill`` -- produces a constant outside its input
    range, so including it would pass without testing anything. Its enrolment rests on
    the block being uniform rather than on the domain, which
    ``test_the_integer_valued_ops_are_the_only_ones_enrolled_on_bfp8_b`` states instead.
    """
    spec = exclude_undefined(
        op, for_op_pipeline(op, input_format, DataFormat.Bfp8_b).spec_A
    )
    assert spec.low is not None and spec.high is not None, op.name
    # ceil of the bound, because the invariant is about *result* block maxima and
    # floor/ceil/trunc round outward by up to one integer. An input domain inside
    # (127, 128) -- uniform(-127.5, 127.5), say -- produces block maxima of exactly
    # 128, where the in-block step becomes 2**(7-6) = 2, odd integers stop being
    # representable and the 0-step budget is no longer a valid criterion.
    reachable = math.ceil(max(abs(spec.low), abs(spec.high)))
    assert reachable < BFP8_B_EXACT_INTEGER_DOMAIN, (
        f"{op.name} from {input_format.name} is swept over [{spec.low}, {spec.high}], "
        f"whose results reach {reachable} and so whose block maxima can reach "
        f"{BFP8_B_EXACT_INTEGER_DOMAIN}. Its 0-step Bfp8_b budget relied on every "
        "block maximum staying below that; re-measure before widening."
    )


# ── Integers never reach the ULP metric through the registry ──────────────────


def test_the_registry_never_carries_a_budget_that_gates_nothing():
    """The two ways a step budget can be inert, and the ops that must never carry one.

    The *downgrade* itself is pinned elsewhere and not repeated here:
    ``test_an_enrolled_op_keeps_todays_gate_on_a_format_without_a_per_element_ulp``
    drives ``accuracy_contract`` over every block format that lacks a per-element ULP,
    and ``test_ulp.py`` pins ``has_ulp_gate`` False for every integer format. What is
    left is the table.

    Both halves are needed, because neither sees what the other does. A row *keyed* on
    a non-gateable format is caught below by its key -- and that is wider than the
    integers: Bfp4_b, Bfp2_b, the MX formats and Tf32 are downgraded the same way.
    ``validate_registry`` does reach those formats now that its sweep is every
    ``DataFormat`` member, but it only asks whether the resolution is *unambiguous*,
    never whether the winner stayed on the ULP metric -- so a
    ``BudgetKey(output_format=Bfp4_b)`` carrying a measured budget resolves cleanly,
    gets downgraded, and passes it. Not subsumed by the widened sweep. An op enrolled
    under ``DEFAULT`` has ``key.output_format is None``, so the key check cannot see it
    at all; only the enrolment check can.
    """
    for op, table in _SFPU_ACCURACY_BUDGET.items():
        for key, contract in table.items():
            if contract.metric != Metric.ULP or key.output_format is None:
                continue
            assert has_ulp_gate(key.output_format), (
                f"{op.name} carries a step budget keyed on {key.output_format.name}, "
                "which has no per-element ULP, so accuracy_contract downgrades it to "
                "the tolerance metric and the number gates nothing."
            )

    # Enrolling an integer-only op would be meaningless rather than merely loose, and
    # the driver would raise at the call. Derived from the canonical classification and
    # driver sets, not from name tokens: a token list over Int32/Int16/Int8/Shift/
    # Bitwise misses UnaryMaxUint32 and UnaryMinUint32 ("Uint32" does not contain
    # "Int32"), every SFPU_BINARY_INT member such as SfpuGtInt, and SfpuGcd.
    integer_ops = _integer_only_ops()
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
    assert not (set(enrolled_ops()) & integer_ops), sorted(
        op.name for op in set(enrolled_ops()) & integer_ops
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
        ("input_format", "Float32"),
        ("output_format", "Float32"),
        ("dest_acc", True),
        ("dest_acc", False),
        ("arch", "wormhole"),
    ],
    ids=lambda v: str(v),
)
def test_a_budget_key_dimension_that_is_not_an_enum_member_is_refused(field, bogus):
    """The same rule as ``AccuracyContract.metric``, on the five dimensions that had no
    check. All of them are bare ``Enum``s, so ``DestAccumulation.No.value is False`` and
    ``ChipArchitecture.WORMHOLE.value == "wormhole"`` never compare equal to their
    members -- and the failure mode is silence, not an exception -- and the query is a ``BudgetKey`` too,
    so one check covers both sides: such a key is counted
    as set by ``specificity``, matched by nothing in ``matches()``, seen as no duplicate
    as no duplicate row by the loader and as no tie by ``validate_registry()``, and rendered
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


@pytest.mark.parametrize("alias", ["1", "0", "1.0"])
def test_a_numeric_alias_for_a_boolean_enum_is_refused(tmp_path, alias):
    """``True == 1`` in Python, so a by-value lookup alone would load ``approx: 1`` as
    ``Yes``: a typo in a key dimension would select a contract instead of failing."""
    with _refuses("is not a ApproximationMode"):
        _table(tmp_path, f"Abs:\n  - {{approx: {alias}, max_ulp: 1}}\n")


def test_a_duplicate_field_inside_a_row_is_refused(tmp_path):
    """The same silent last-wins rule applies inside a row, where it would swap the key
    a budget is filed under."""
    with _refuses("duplicate entry for 'out'"):
        _table(tmp_path, "Abs:\n  - {out: Float16_b, out: Float32, max_ulp: 1}\n")


def test_anchors_and_merge_keys_load_and_may_override(tmp_path):
    """The table shares rows through anchors, and a ``<<`` merge that overrides a field
    is not a duplicate."""
    table = _table(
        tmp_path,
        """\
        Abs:
          - &base {out: Float16_b, max_ulp: 2}
        Neg:
          - *base
          - {<<: *base, out: Float32, max_ulp: 5}
        """,
    )
    neg = table[MathOperation.Neg]
    assert neg[BudgetKey(output_format=DataFormat.Float16_b)].max_ulp == 2
    assert neg[BudgetKey(output_format=DataFormat.Float32)].max_ulp == 5


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


#: Ops whose budget was measured on the hand-built sweep that drives them, rather than on
#: the standard random sweep. Each was run under ``--ulp-report`` on that sweep's own
#: stimulus on Wormhole, 2026-09-18; the per-format counts are in the YAML row comments.
#: Enrolling a further op from one of these sweeps means measuring it there first.
MEASURED_ON_SWEEP = {
    "signbit": {MathOperation.Signbit},
    "isinf_isnan": {
        MathOperation.Isinf,
        MathOperation.Isposinf,
        MathOperation.Isneginf,
        MathOperation.Isnan,
        MathOperation.Isfinite,
    },
    "threshold": {
        MathOperation.LogicalNot,
        MathOperation.UnaryEq,
        MathOperation.UnaryNe,
        MathOperation.ReluMin,
        MathOperation.ReluMax,
    },
}


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

    hand_built = {
        # The signbit sweep is not parametrised over a list; it drives this one op.
        "signbit": {MathOperation.Signbit},
        "isinf_isnan": set(ISINF_ISNAN_MATHOPS),
        "threshold": set(_THRESHOLD_OPS),
    }
    assert all(hand_built.values()), "a sweep set went empty; the derivation has broken"

    enrolled = set(enrolled_ops())
    for sweep, ops in sorted(hand_built.items()):
        unrecorded = sorted(
            op.name for op in (enrolled & ops) - MEASURED_ON_SWEEP[sweep]
        )
        assert not unrecorded, (
            f"{', '.join(unrecorded)} carries a budget but is driven by the {sweep} "
            "sweep, whose hand-built stimulus no recorded measurement covers. Measure it "
            "there and add it to MEASURED_ON_SWEEP, or key the budget away from the "
            "formats it reaches."
        )


# ─────────────────────────────────────────────────────────────────────────────
# A budget must be backed by the measurement it records
#
# The table used to be shadowed by a second file holding what every op *should*
# resolve to. That file pinned nothing: resolving the registry reproduced it exactly,
# so it agreed with the table by construction and only ever caught "you changed a
# number and did not change the copy". Regenerate both and it caught nothing.
#
# The provenance is the real invariant. Every row carries the measurement its budget
# came from, and the emitter's rule is that a budget sits at or above it with bounded
# headroom. Widening a budget without re-measuring therefore has to falsify the comment
# next to it -- a sentence someone has to write -- rather than re-run a generator.
# ─────────────────────────────────────────────────────────────────────────────

#: How far a budget may sit above the measurement it records. The emitter's own headroom
#: is 1.25x on most rows and 2x at its widest; nothing in the table exceeds that.
MEASUREMENT_HEADROOM = 2

#: ``max 65536 ULP`` in the emitted rows, ``0 ULP`` in the hand-measured ones.
_MEASUREMENT = re.compile(r"(?:max )?(\d+) ULP")


def _measured_budget_rows(path=_TABLE_PATH):
    """Every ``max_ulp`` row in the table, with the measurement it records.

    The measurement is the row's own trailing comment, or its op's header comment where
    one sweep covered the whole op. Read from the text, not the loaded table: the comment
    *is* the provenance, YAML discards it, and a budget whose comment no longer supports
    it is precisely the drift this guards against.
    """
    rows, op, op_measured = [], None, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):  # `OpName:`, optionally with a header comment
            head, _, comment = line.partition("#")
            op = head.split(":")[0].strip()
            found = _MEASUREMENT.search(comment)
            op_measured = int(found.group(1)) if found else None
            continue
        body, _, comment = line.strip().partition("#")
        declared = re.search(r"max_ulp:\s*(\d+)", body)
        if not declared:
            continue  # a tolerance row has no step budget to back
        found = _MEASUREMENT.search(comment)
        measured = int(found.group(1)) if found else op_measured
        rows.append((op, body.strip(), int(declared.group(1)), measured))
    return rows


def test_the_provenance_parser_sees_every_budget_the_registry_enforces():
    """The two audits below recognise a budget row by a regex over the file's text, and a
    row it misses is silently dropped from both -- it keeps its enforced ``max_ulp`` but
    loses its measurement requirement and its ``MEASUREMENT_HEADROOM`` bound.

    ``- {max_ulp : 5}`` and ``- {max_ulp: +5}`` both load as 5 and both miss; ``0x10``
    is worse, loading as 16 while the regex captures ``0``, so the audit would validate
    a number the registry does not enforce. Tie the parse back to the loaded table.
    """
    parsed = sorted((op, budget) for op, _, budget, _ in _measured_budget_rows())
    loaded = sorted(
        (op.name, contract.max_ulp)
        for op, table in _SFPU_ACCURACY_BUDGET.items()
        for contract in table.values()
        if contract.metric == Metric.ULP
    )
    assert parsed == loaded


def test_every_step_budget_names_the_measurement_it_came_from():
    """A budget with no measurement behind it is a guess, and the table's whole claim is
    that it holds none. The number may sit on the row or on the op, whichever the sweep
    covered."""
    rows = _measured_budget_rows()
    assert rows, "no max_ulp rows found -- the parser has drifted from the table"
    unbacked = [(op, body) for op, body, _, measured in rows if measured is None]
    assert not unbacked, "budgets with no recorded measurement:\n" + "\n".join(
        f"  {op}: {body}" for op, body in unbacked
    )


def test_no_step_budget_exceeds_the_measurement_it_records():
    """The guard that replaced the expected-budget file.

    Raising ``max_ulp`` until a failure goes away now has to move the measurement beside
    it past what was actually measured. A budget *below* its measurement is the other
    error -- it cannot pass, so it was never measured on the sweep it claims.
    """
    for op, body, budget, measured in _measured_budget_rows():
        if measured is None:
            continue  # test_every_step_budget_names_the_measurement_it_came_from owns this
        where = f"{op}: {body} (records {measured} ULP)"
        if measured == 0:
            # Floored to 1 where a finite sample cannot assert exactness; 0 only where
            # the op is exact by construction and the sweep was exhaustive.
            assert budget <= 1, f"{where}: a 0-ULP measurement cannot justify {budget}"
        else:
            assert budget >= measured, f"{where}: budget {budget} is below it"
            assert budget <= MEASUREMENT_HEADROOM * measured, (
                f"{where}: budget {budget} is more than "
                f"{MEASUREMENT_HEADROOM}x the measurement"
            )
