# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the SFPU accuracy budget registry (no kernel, no device).

``helpers/sfpu_accuracy_budget.py`` loads ``sfpu_accuracy_budget.yaml`` and resolves, per
op variant, the gate the device drivers hand to ``passed_test``. A budget resolved from
the wrong row is a silently wrong gate rather than an error, so the file guards:

* the building blocks: ``AccuracyContract`` and ``BudgetKey`` refuse anything they
  cannot represent faithfully (the table file itself is test_ulp_table.py's);
* the resolution rule, on small purpose-built tables so enrolling an op cannot break it;
* ``accuracy_contract``: the fallbacks and the downgrades that make enrolment incremental;
* the live table: invariants a reader cannot see by looking at the numbers, and the
  measurement every step budget must record.
"""

import functools
import math
from dataclasses import fields
from itertools import product

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    MathOpType,
)
from helpers.sfpu_accuracy_budget import (
    _SFPU_ACCURACY_BUDGET,
    DEFAULT,
    TOLERANCE_CONTRACT,
    AccuracyContract,
    BudgetKey,
    Metric,
    accuracy_contract,
    assert_against_contract,
    domains_of,
    enrolled_ops,
    resolve_contract,
    usable_budget_ceiling,
    validate_registry,
)
from helpers.sfpu_domains import _UNARY_OPS_NOT_SWEPT, sfpu_unary_ops
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from helpers.ulp import (
    _ULP_DTYPES,
    MAX_MEANINGFUL_ULP,
    ULP_FORMATS,
    has_ulp_gate,
    ulp_dtype,
)
from helpers.ulp_domains import Domain, Problem
from helpers.ulp_sweep import _verdict, sweep_cells
from helpers.ulp_table import MEASURED_ARCH, TABLE_PATH, Row, load_table
from helpers.ulp_testing import TANH_DOMAINS, refused
from helpers.utils import passed_test

UNSWEPT_ARCHS = [a for a in ChipArchitecture if a != MEASURED_ARCH]

#: The float formats the SFPU does not compute in: the block floats, Tf32 and the MX
#: formats (MxInt* included -- ``is_integer()`` does not cover them).
NOT_COMPUTE_FORMATS = sorted(
    (f for f in DataFormat if f not in ULP_FORMATS and not f.is_integer()),
    key=lambda f: f.name,
)

#: The input axis for the live-table sweeps: the formats a row may name, and ``None``,
#: which resolves the rows that wildcard ``input``.
QUERYABLE_INPUT_FORMATS = [*ULP_FORMATS, None]


def _contracts(*rows: Row):
    """One op's rows as the registry holds them."""
    return {row.key: row.contract for row in rows}


def _every_variant(op):
    """Yield ``(input_format, output_format, contract)`` for every variant of *op*.

    Every dimension is swept, ``None`` included where a caller may leave it unset: an
    unset query dimension only matches a wildcard, so a query that omitted
    ``input_format`` would miss every row that pins it -- which is most of the table.
    """
    for fmt in ULP_FORMATS:
        for input_format in QUERYABLE_INPUT_FORMATS:
            for approx_mode in [*ApproximationMode, None]:
                for dest_acc in [*DestAccumulation, None]:
                    for arch in ChipArchitecture:  # required; None is not accepted
                        yield input_format, fmt, accuracy_contract(
                            op,
                            output_format=fmt,
                            input_format=input_format,
                            approx_mode=approx_mode,
                            dest_acc=dest_acc,
                            arch=arch,
                        )


@functools.lru_cache(maxsize=None)
def _live_step_budgets():
    """``(op, input_format, output_format, contract)`` for every ULP contract the live
    table resolves to. Computed once: more than one test reads it, and the sweep is the
    slow part.
    """
    return tuple(
        (op, in_fmt, fmt, contract)
        for op in enrolled_ops()
        for in_fmt, fmt, contract in _every_variant(op)
        if contract.metric is Metric.ULP
    )


def _integer_only_ops() -> set:
    """Every SFPU op whose operands and result are integers.

    No single classification covers them, so this is the union of the typed integer
    binaries, the unary and binary drivers' integer sets, and ``SfpuGcd`` (in none of
    those). ``ReluMin`` is in the unary set but has a float path too.
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


# ── AccuracyContract ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs, problem",
    [
        ({"metric": Metric.ULP}, Problem.MISSING_FIGURE),
        # Both at once reads as deliberate but is a half-finished conversion.
        ({"max_ulp": 1, "atol": 0.13}, Problem.MIXED_METRIC),
        ({"metric": Metric.TOLERANCE, "max_ulp": 1}, Problem.MIXED_METRIC),
        ({"max_ulp": -1}, Problem.BAD_NUMBER),
        # YAML reads `true` as a bool, which is an int and would enforce a 1-step budget.
        ({"max_ulp": True}, Problem.BAD_NUMBER),
        ({"max_ulp": 1.0}, Problem.BAD_NUMBER),
        # `atol: yes` applies as 1.0, NaN is ignored by passed_test, inf gates nothing.
        ({"metric": Metric.TOLERANCE, "atol": True}, Problem.BAD_NUMBER),
        ({"metric": Metric.TOLERANCE, "atol": math.nan}, Problem.BAD_NUMBER),
        ({"metric": Metric.TOLERANCE, "atol": math.inf}, Problem.BAD_NUMBER),
        # passed_test ignores a negative override and silently uses the default.
        ({"metric": Metric.TOLERANCE, "atol": -0.001}, Problem.BAD_NUMBER),
        ({"metric": Metric.TOLERANCE, "rtol": -0.001}, Problem.BAD_NUMBER),
        ({"max_ulp": 1, "near_zero_atol": -0.001}, Problem.BAD_NUMBER),
        ({"metric": None, "max_ulp": 1}, Problem.NOT_A_MEMBER),
    ],
    ids=repr,
)
def test_a_malformed_contract_is_refused(kwargs, problem):
    with refused(problem):
        AccuracyContract(**kwargs)


def test_the_metric_is_a_closed_set():
    """Two members, so the ``__post_init__`` membership check above is complete."""
    assert set(Metric) == {Metric.ULP, Metric.TOLERANCE}
    assert AccuracyContract(max_ulp=1).metric is Metric.ULP
    assert TOLERANCE_CONTRACT.metric is Metric.TOLERANCE


#: ``(contract, passed_test_kwargs(), tolerance_kwargs())``. ``tolerance_kwargs`` declines
#: to gate on ULP; if it regressed to the other shape, passed_test would accept it and
#: silently switch every enrolled op to a ULP budget.
_TRANSLATIONS = [
    (AccuracyContract(max_ulp=3), {"max_ulp": 3, "near_zero_atol": None}, {}),
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
    # The flush request reaches only the ULP arm: passed_test refuses it without a budget.
    # Either value is passed on as passed_test reads it -- False is "unflushed", not the
    # per-dtype default, which only None keeps.
    for flush in (True, False):
        flushed = contract.passed_test_kwargs(flush_subnormals=flush)
        if contract.metric is Metric.ULP:
            assert flushed == {**by_ulp, "flush_subnormals": flush}
        else:
            assert flushed == by_ulp


@pytest.mark.parametrize(
    "arch, gated",
    [(ChipArchitecture.WORMHOLE, True), (ChipArchitecture.BLACKHOLE, False)],
    ids=lambda v: getattr(v, "name", str(v)),
)
def test_the_binary_gate_enforces_the_whole_contract(arch, gated, monkeypatch):
    """`assert_against_contract` is the binary and ternary drivers' gate, and they run
    only on hardware, so this is the one host check that it takes the step budget and
    not just the tolerance: a one-step drift on SfpuElwEq, exact at 0 steps on Wormhole,
    fails there and passes the tolerance it falls back to off Wormhole. It also ranks
    with subnormal outputs flushed: an fp16-subnormal golden that the pack writes as 0
    is no step of SfpuElwsub's error."""
    import helpers.chip_architecture as chip

    monkeypatch.setattr(chip, "get_chip_architecture", lambda: arch)
    lanes = DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM
    bf16 = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    golden = torch.ones(lanes, dtype=torch.bfloat16)
    drifted = golden.clone()
    drifted[0] = 1.0 + 2.0**-7  # one bf16 step above 1.0

    def eq_gate():
        assert_against_contract(
            MathOperation.SfpuElwEq, bf16, DestAccumulation.No, golden, drifted
        )

    if gated:
        with pytest.raises(AssertionError):  # allow-pytest.raises: host-only test
            eq_gate()
    else:
        eq_gate()

    fp16 = InputOutputFormat(DataFormat.Float16, DataFormat.Float16)
    golden = torch.ones(lanes, dtype=torch.float16)
    # Half the smallest fp16 normal: an fp16 subnormal, 512 steps from 0 unflushed.
    golden[0] = float(torch.finfo(torch.float16).smallest_normal) / 2
    flushed = golden.clone()
    flushed[0] = 0.0
    assert_against_contract(
        MathOperation.SfpuElwsub, fp16, DestAccumulation.No, golden, flushed
    )


@pytest.mark.parametrize("method", ["passed_test_kwargs", "tolerance_kwargs"])
def test_every_contract_is_accepted_by_passed_test(method):
    """Callable, not just shaped right: a renamed keyword would otherwise surface only
    on hardware."""
    golden = torch.ones(DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM, dtype=torch.bfloat16)
    for contract, _, _ in _TRANSLATIONS:
        assert passed_test(
            golden, golden.clone(), DataFormat.Float16_b, **getattr(contract, method)()
        )


# ── BudgetKey ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "field, bogus",
    [
        ("approx_mode", True),
        ("input_format", ApproximationMode.No),
        ("output_format", DestAccumulation.No),
        ("dest_acc", True),
        ("dest_acc", False),
        ("arch", DataFormat.Float32),
    ],
    ids=str,
)
def test_a_budget_key_dimension_that_is_not_an_enum_member_is_refused(field, bogus):
    """The enums are bare, so ``True`` never equals ``DestAccumulation.Yes``. Such a key
    would count as set, match nothing, and its budget would gate nothing. Queries are
    BudgetKeys too, so this also covers the caller side."""
    with refused(Problem.NOT_A_MEMBER):
        BudgetKey(**{field: bogus})


def test_every_budget_key_field_counts_toward_specificity():
    assert BudgetKey(
        approx_mode=ApproximationMode.No,
        input_format=DataFormat.Float16_b,
        output_format=DataFormat.Float32,
        dest_acc=DestAccumulation.Yes,
        arch=ChipArchitecture.WORMHOLE,
    ).specificity == len(fields(BudgetKey))


def test_a_key_describes_itself_for_an_error_message():
    assert DEFAULT.describe() == "DEFAULT"
    described = BudgetKey(
        output_format=DataFormat.Float32, dest_acc=DestAccumulation.No
    ).describe()
    assert "output_format" in described and "dest_acc" in described


# ── Resolution rule, on purpose-built tables ──────────────────────────────────


def test_the_default_key_matches_every_variant():
    assert DEFAULT.specificity == 0
    assert DEFAULT.matches(
        BudgetKey(
            approx_mode=ApproximationMode.Yes,
            input_format=DataFormat.Float32,
            output_format=DataFormat.Float32,
            dest_acc=DestAccumulation.No,
            arch=ChipArchitecture.WORMHOLE,
        )
    )
    assert DEFAULT.matches(BudgetKey())


@pytest.mark.parametrize(
    "broad, narrow",
    [
        (AccuracyContract(max_ulp=64), AccuracyContract(max_ulp=4)),
        # A narrower tolerance for one variant needs a key, not a driver override.
        (
            AccuracyContract(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05),
            AccuracyContract(metric=Metric.TOLERANCE, atol=0.001, rtol=0.001),
        ),
    ],
    ids=["ulp", "tolerance"],
)
def test_a_more_specific_key_wins_over_the_default(broad, narrow):
    table = {DEFAULT: broad, BudgetKey(output_format=DataFormat.Float32): narrow}
    for fmt, expected in ((DataFormat.Float32, narrow), (DataFormat.Float16_b, broad)):
        query = BudgetKey(output_format=fmt, arch=MEASURED_ARCH)
        assert resolve_contract(table, query) == expected, fmt.name


def test_specificity_counts_every_set_dimension():
    """The only comparison of two non-DEFAULT keys. The Fill rows stack 1-, 2- and
    4-field keys on one op, and only an equal-specificity tie raises, so a miscount
    would silently repoint budgets."""
    table = {
        BudgetKey(output_format=DataFormat.Float32): AccuracyContract(max_ulp=4),
        BudgetKey(
            output_format=DataFormat.Float32, dest_acc=DestAccumulation.No
        ): AccuracyContract(max_ulp=8),
    }
    for dest_acc, expected in ((DestAccumulation.No, 8), (DestAccumulation.Yes, 4)):
        query = BudgetKey(output_format=DataFormat.Float32, dest_acc=dest_acc)
        assert resolve_contract(table, query).max_ulp == expected


def test_a_per_arch_override_beats_the_shared_entry():
    """One shared value plus per-arch overrides, rather than keying every row on arch."""
    table = {
        DEFAULT: AccuracyContract(max_ulp=4),
        BudgetKey(arch=ChipArchitecture.BLACKHOLE): AccuracyContract(max_ulp=2),
    }
    for arch, expected in (
        (ChipArchitecture.BLACKHOLE, 2),
        (ChipArchitecture.WORMHOLE, 4),
        (ChipArchitecture.QUASAR, 4),
    ):
        query = BudgetKey(output_format=DataFormat.Float32, arch=arch)
        assert resolve_contract(table, query).max_ulp == expected, arch


def test_an_unset_query_dimension_only_matches_a_wildcard():
    """A caller that does not know ``dest_acc`` must not get a budget measured for one
    setting of it."""
    table = {BudgetKey(dest_acc=DestAccumulation.Yes): AccuracyContract(max_ulp=1)}
    query = BudgetKey(output_format=DataFormat.Float32)
    assert resolve_contract(table, query) is TOLERANCE_CONTRACT


def test_equally_specific_keys_are_an_error_not_a_tie_break():
    """The table's iteration order must not decide a budget."""
    table = {
        BudgetKey(approx_mode=ApproximationMode.No): AccuracyContract(max_ulp=4),
        BudgetKey(output_format=DataFormat.Float32): AccuracyContract(max_ulp=64),
    }
    query = BudgetKey(
        output_format=DataFormat.Float32, approx_mode=ApproximationMode.No
    )
    with refused(Problem.AMBIGUOUS):
        resolve_contract(table, query)


def test_an_empty_table_falls_back_to_the_tolerance_metric():
    query = BudgetKey(output_format=DataFormat.Float32)
    assert resolve_contract({}, query) is TOLERANCE_CONTRACT


# ── accuracy_contract: fallbacks and downgrades ───────────────────────────────


def test_an_unenrolled_op_keeps_todays_gate():
    """Picked from whatever is still unenrolled, so enrolling more ops cannot break it."""
    unenrolled = sorted(set(MathOperation) - set(enrolled_ops()), key=lambda o: o.name)
    assert unenrolled, "every op is enrolled; this test has nothing left to check"
    for op in unenrolled[:5]:
        for fmt in ULP_FORMATS:
            contract = accuracy_contract(op, output_format=fmt, arch=MEASURED_ARCH)
            assert contract is TOLERANCE_CONTRACT, op.name


@pytest.mark.parametrize("fmt", NOT_COMPUTE_FORMATS, ids=lambda f: f.name)
def test_an_enrolled_op_keeps_todays_gate_on_a_format_the_sfpu_does_not_compute_in(fmt):
    """Abs has a step budget, but these formats keep their block-aware compares -- by
    falling back, not by raising."""
    # The same query on a gated format is a step budget, so the fall-through below is
    # the downgrade under test and not a query that matched nothing.
    query = dict(op=MathOperation.Abs, arch=MEASURED_ARCH)
    assert (
        accuracy_contract(output_format=DataFormat.Float16_b, **query).metric
        is Metric.ULP
    )
    assert accuracy_contract(output_format=fmt, **query) is TOLERANCE_CONTRACT


@pytest.mark.parametrize("fmt", NOT_COMPUTE_FORMATS, ids=lambda f: f.name)
def test_an_enrolled_op_keeps_todays_gate_when_the_input_is_not_a_compute_format(fmt):
    """The input-axis twin: Abs's rows wildcard ``input``, so only ``steps_bind``
    reading the input format keeps a block-float input off the step budget."""
    query = dict(
        op=MathOperation.Abs,
        output_format=DataFormat.Float16_b,
        arch=MEASURED_ARCH,
    )
    assert (
        accuracy_contract(input_format=DataFormat.Float16_b, **query).metric
        is Metric.ULP
    )
    assert accuracy_contract(input_format=fmt, **query) is TOLERANCE_CONTRACT


@pytest.mark.parametrize("arch", UNSWEPT_ARCHS, ids=lambda a: a.name)
def test_a_step_budget_does_not_bind_on_an_unswept_architecture(arch):
    """Unkeyed rows are Wormhole measurements with no headroom for another kernel."""
    on_wh = accuracy_contract(
        MathOperation.Abs, output_format=DataFormat.Float32, arch=MEASURED_ARCH
    )
    assert on_wh.metric is Metric.ULP and on_wh.max_ulp == 0
    elsewhere = accuracy_contract(
        MathOperation.Abs, output_format=DataFormat.Float32, arch=arch
    )
    assert elsewhere is TOLERANCE_CONTRACT


@pytest.mark.parametrize("arch", UNSWEPT_ARCHS, ids=lambda a: a.name)
@pytest.mark.parametrize(
    "op", [MathOperation.SigmoidAppx, MathOperation.GeluAppx], ids=lambda o: o.name
)
def test_a_declared_tolerance_survives_an_unswept_architecture(op, arch):
    """The arch downgrade applies to step budgets only; these two keep their declared
    ``atol=0.13`` on every architecture."""
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
    "output_format", [DataFormat.Float16_b, DataFormat.Bfp4_b], ids=lambda f: f.name
)
def test_a_downgrade_lands_on_the_ops_own_tolerance_row(
    arch, output_format, monkeypatch
):
    """A downgraded ULP row falls back to the op's own tolerance row, not the global
    default. No live op has both shapes yet, so the table is patched."""
    monkeypatch.setitem(
        _SFPU_ACCURACY_BUDGET,
        MathOperation.Abs,
        {
            DEFAULT: AccuracyContract(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05),
            BudgetKey(output_format=DataFormat.Float16_b): AccuracyContract(max_ulp=3),
        },
    )
    contract = accuracy_contract(
        MathOperation.Abs, output_format=output_format, arch=arch
    )
    if output_format is DataFormat.Float16_b and arch is MEASURED_ARCH:
        assert contract.metric is Metric.ULP and contract.max_ulp == 3
    else:  # Bfp4_b is a block float; off Wormhole the unkeyed row does not bind
        assert contract.metric is Metric.TOLERANCE
        assert contract.atol == 0.13 and contract.rtol == 0.05


def test_a_bare_tolerance_row_opts_out_of_ulp_without_retracting_a_declared_atol(
    monkeypatch,
):
    """The sweep writes a numberless ``metric: tolerance`` row for every cell past the
    ceiling. It beats a shared ``atol`` row on specificity, and resolving it as-is gave
    SigmoidAppx the per-format default: eight device failures no host test saw."""
    op = MathOperation.Abs
    monkeypatch.setitem(
        _SFPU_ACCURACY_BUDGET,
        op,
        {
            DEFAULT: AccuracyContract(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05),
            BudgetKey(
                input_format=DataFormat.Float16_b, output_format=DataFormat.Float16_b
            ): AccuracyContract(metric=Metric.TOLERANCE),
            BudgetKey(output_format=DataFormat.Float32): AccuracyContract(
                metric=Metric.TOLERANCE, atol=0.5, rtol=0.5
            ),
        },
    )
    bare_cell = accuracy_contract(
        op,
        input_format=DataFormat.Float16_b,
        output_format=DataFormat.Float16_b,
        arch=MEASURED_ARCH,
    )
    assert bare_cell.metric is Metric.TOLERANCE
    assert bare_cell.atol == 0.13 and bare_cell.rtol == 0.05
    # The most specific numbered row still wins where there is one.
    numbered = accuracy_contract(
        op, output_format=DataFormat.Float32, arch=MEASURED_ARCH
    )
    assert numbered.atol == 0.5
    # With only bare rows there is nothing to fall through to.
    monkeypatch.setitem(
        _SFPU_ACCURACY_BUDGET,
        op,
        {BudgetKey(output_format=DataFormat.Float16_b): TOLERANCE_CONTRACT},
    )
    only_bare = accuracy_contract(
        op, output_format=DataFormat.Float16_b, arch=MEASURED_ARCH
    )
    assert only_bare == TOLERANCE_CONTRACT


@pytest.mark.parametrize("arch", UNSWEPT_ARCHS, ids=lambda a: a.name)
def test_a_ulp_row_that_names_its_arch_binds_there(arch, monkeypatch):
    """A row keyed on an arch was measured there, so the arch downgrade exempts it."""
    monkeypatch.setitem(
        _SFPU_ACCURACY_BUDGET,
        MathOperation.Abs,
        {
            BudgetKey(output_format=DataFormat.Float16_b): AccuracyContract(max_ulp=3),
            BudgetKey(output_format=DataFormat.Float16_b, arch=arch): AccuracyContract(
                max_ulp=5
            ),
        },
    )

    def resolved(on):
        return accuracy_contract(
            MathOperation.Abs,
            output_format=DataFormat.Float16_b,
            approx_mode=ApproximationMode.No,
            dest_acc=DestAccumulation.No,
            arch=on,
        )

    contract = resolved(arch)
    assert contract.metric is Metric.ULP and contract.max_ulp == 5
    for other in UNSWEPT_ARCHS:
        if other is not arch:
            assert resolved(other) is TOLERANCE_CONTRACT


@pytest.mark.parametrize("enrolled", [False, True], ids=["unenrolled", "enrolled"])
def test_a_bad_query_is_refused_whether_or_not_the_op_is_enrolled(enrolled):
    """Otherwise a miswired driver passes until the day its op is enrolled."""
    op = next(
        op
        for op in sorted(MathOperation, key=lambda o: o.name)
        if (op in _SFPU_ACCURACY_BUDGET) == enrolled
    )
    with refused(Problem.NOT_A_MEMBER):
        accuracy_contract(op, output_format=DataFormat.Float32, arch=True)


def test_a_query_left_over_from_another_test_is_replaced_not_flagged(monkeypatch):
    """``--ulp-measure`` files a reading under the last variant looked up, and refuses
    two lookups racing one comparison -- but only within one test. The exhaustive sweep
    resolves a contract and then skips a tolerance cell; flagging that dropped every
    reading that followed a skip (40 of 130 tests, measured)."""
    import helpers.sfpu_accuracy_budget as budget

    def resolve(test_id):
        monkeypatch.setenv("PYTEST_CURRENT_TEST", f"{test_id} (call)")
        accuracy_contract(
            MathOperation.Abs, output_format=DataFormat.Float16_b, arch=MEASURED_ARCH
        )

    monkeypatch.setattr(budget, "LAST_QUERY", None)
    monkeypatch.setattr(budget, "PENDING_AMBIGUOUS", False)

    # An ambiguous test that exits before comparing must not hand its flag on.
    resolve("t_zero")
    resolve("t_zero")
    assert budget.PENDING_AMBIGUOUS
    resolve("t_one")
    assert not budget.PENDING_AMBIGUOUS
    resolve("t_two")
    assert not budget.PENDING_AMBIGUOUS
    assert budget.LAST_QUERY.test == "t_two"

    resolve("t_three")
    resolve("t_three")
    assert budget.PENDING_AMBIGUOUS
    resolve("t_three")  # a third lookup in the same test is still ambiguous
    assert budget.PENDING_AMBIGUOUS


def test_the_measure_recorder_files_one_row_under_the_variant_just_resolved(
    tmp_path, monkeypatch
):
    """``--ulp-measure`` tags a comparison with ``accuracy_contract``'s last query and
    consumes it: a second comparison in the same test that never went through the
    registry must not inherit the variant, and the row names the variant asked for
    rather than the tensors' format."""
    read = _measure_rows(tmp_path, monkeypatch)
    golden = torch.full((32,), 1.5, dtype=torch.bfloat16)

    accuracy_contract(
        MathOperation.Abs,
        output_format=DataFormat.Float16_b,
        input_format=DataFormat.Float16,
        dest_acc=DestAccumulation.Yes,
        arch=MEASURED_ARCH,
    )
    assert passed_test(golden, golden.clone(), DataFormat.Float16_b)
    assert passed_test(golden, golden.clone(), DataFormat.Float16_b)  # no lookup

    (row,) = read()
    assert row.op is MathOperation.Abs
    assert row.key == BudgetKey(
        input_format=DataFormat.Float16,
        output_format=DataFormat.Float16_b,
        dest_acc=DestAccumulation.Yes,
        arch=MEASURED_ARCH,
    )
    # A worst lane of 0 means something only over lanes that were measured.
    assert (row.worst, row.lanes, row.unmeasurable, row.nonfinite) == (0, 32, 0, 0)


def test_the_measure_recorder_writes_the_nonfinite_count_the_headroom_report_reads(
    tmp_path, monkeypatch
):
    """The count is the sweep's only link to the headroom report's non-finite check:
    written through the recorder and read through the report's own reader."""
    from helpers.ulp import ulp_distance
    from helpers.ulp_budget_diff import read_cells
    from helpers.utils import _record_ulp_measurement

    read = _measure_rows(tmp_path, monkeypatch)
    golden = torch.full((32,), 1.5, dtype=torch.bfloat16)
    accuracy_contract(
        MathOperation.Abs,
        output_format=DataFormat.Float16_b,
        input_format=DataFormat.Float16_b,
        arch=MEASURED_ARCH,
    )
    _record_ulp_measurement(
        ulp_distance(golden, golden.clone()),
        mask=torch.ones(32, dtype=torch.bool),
        nonfinite=7,
    )
    assert [cell.nonfinite for cell in read_cells(read()).values()] == [7]


def test_a_record_the_sweep_requires_fails_rather_than_being_dropped(
    tmp_path, monkeypatch
):
    """The headroom report judges only the cells it reads, so a sweep record that would
    be dropped fails its cell; a functional driver's is only lost."""
    from helpers.ulp import ulp_distance
    from helpers.utils import RecordDropped, _record_ulp_measurement

    read = _measure_rows(tmp_path, monkeypatch)
    golden = torch.full((32,), 1.5, dtype=torch.bfloat16)
    reading = dict(
        distance=ulp_distance(golden, golden.clone()),
        mask=torch.ones(32, dtype=torch.bool),
    )
    _record_ulp_measurement(**reading)  # no query pending: dropped
    with pytest.raises(RecordDropped):  # allow-pytest.raises: host-only test
        _record_ulp_measurement(**reading, required=True)
    assert read() == []
    accuracy_contract(
        MathOperation.Abs, output_format=DataFormat.Float16_b, arch=MEASURED_ARCH
    )
    _record_ulp_measurement(**reading, required=True)
    assert len(read()) == 1


def _measure_rows(tmp_path, monkeypatch):
    """The recorder pointed at a file, and the reader of what it wrote."""
    import helpers.utils as utils
    from helpers.ulp_table import read_measurements

    path = tmp_path / "measure.jsonl"
    monkeypatch.setattr(utils, "_ULP_MEASURE_PATH", str(path))
    return lambda: read_measurements(path)


def test_the_measure_recorder_ranks_the_lanes_the_verdict_ranks(tmp_path, monkeypatch):
    """The ULP arm hands the recorder the lanes it ranks, without the ones a
    ``near_zero_atol`` floor rescued: those carry the largest step counts by
    construction, and filing them would fold a floor-carried pass back as a budget."""
    rows = _measure_rows(tmp_path, monkeypatch)
    fmt = DataFormat.Float16_b
    golden = torch.full((32,), 1.5, dtype=torch.bfloat16)
    golden[0] = 1e-6
    result = golden.clone()
    result[0] = 3e-6  # about 200 steps away, but 2e-6 in absolute terms
    result[1] = 1.5078125  # one real bf16 step above 1.5

    accuracy_contract(MathOperation.Abs, output_format=fmt, arch=MEASURED_ARCH)
    assert passed_test(golden, result, fmt, max_ulp=1, near_zero_atol=1e-5)
    (row,) = rows()
    assert row.worst == 1, row


def test_the_measure_recorder_drops_an_ambiguous_or_promoted_variant(
    tmp_path, monkeypatch
):
    """Two lookups and then one comparison cannot say which variant it was, and a
    variant TestConfig promoted to a 32-bit Dest ran another kernel than it names --
    against a golden built for the one it names. Neither is filed."""
    import helpers.chip_architecture as chip

    monkeypatch.setenv("CHIP_ARCH", "wormhole")
    monkeypatch.setattr(chip, "_cached_chip_architecture", None)
    rows = _measure_rows(tmp_path, monkeypatch)
    golden = torch.full((32,), 1.5, dtype=torch.bfloat16)

    for _ in range(2):
        accuracy_contract(
            MathOperation.Abs, output_format=DataFormat.Float16_b, arch=MEASURED_ARCH
        )
    assert passed_test(golden, golden.clone(), DataFormat.Float16_b)
    assert rows() == []

    half = golden.to(torch.float16)
    accuracy_contract(
        MathOperation.Abs,
        output_format=DataFormat.Float16,
        input_format=DataFormat.Float16_b,
        dest_acc=DestAccumulation.No,
        arch=MEASURED_ARCH,
    )
    assert passed_test(half, half.clone(), DataFormat.Float16)
    assert rows() == []

    # The promotion is the resolved architecture's, not the process's: Quasar runs the
    # 16-bit Dest it was asked for, so the same cell resolved for it is filed, under it.
    accuracy_contract(
        MathOperation.Abs,
        output_format=DataFormat.Float16,
        input_format=DataFormat.Float16_b,
        dest_acc=DestAccumulation.No,
        arch=ChipArchitecture.QUASAR,
    )
    assert passed_test(half, half.clone(), DataFormat.Float16)
    (row,) = rows()
    assert (row.key.arch, row.key.dest_acc) == (
        ChipArchitecture.QUASAR,
        DestAccumulation.No,
    )


def test_every_comparison_spends_the_query_it_was_resolved_for(tmp_path, monkeypatch):
    """A path that writes no row -- a shape mismatch, an output with no per-element
    ULP -- still spends the lookup, or the next comparison in the test would inherit the
    variant and the next lookup read as ambiguous."""
    import helpers.sfpu_accuracy_budget as budget

    _measure_rows(tmp_path, monkeypatch)
    golden = torch.full((32,), 1.5, dtype=torch.bfloat16)
    for result, fmt in (
        (golden.clone().reshape(1, 32), DataFormat.Float16_b),  # broadcast compare
        (golden.clone(), DataFormat.Bfp4_b),  # no per-element ULP
    ):
        accuracy_contract(MathOperation.Abs, output_format=fmt, arch=MEASURED_ARCH)
        passed_test(golden, result, fmt)
        assert (budget.LAST_QUERY, budget.PENDING_AMBIGUOUS) == (None, False), fmt


def test_a_comparison_does_not_load_the_budget_table():
    """``passed_test`` spends the registry's pending query on every call, but must not
    import the registry to do it: every caller -- matmul, pack and Quasar suites run on
    their own -- would then load and validate the whole table, and a bad row would error
    each comparison after its verdict. In a fresh interpreter, since this one has it."""
    import subprocess
    import sys

    import helpers.sfpu_accuracy_budget as budget
    import helpers.utils as utils

    code = (
        f"import sys, torch; from {utils.__name__} import passed_test; "
        "from helpers.format_config import DataFormat; "
        "golden = torch.ones(32, dtype=torch.bfloat16); "
        "assert passed_test(golden, golden.clone(), DataFormat.Float16_b); "
        f"sys.exit({budget.__name__!r} in sys.modules)"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=TABLE_PATH.parent.parent)
    assert done.returncode == 0


def test_a_tolerance_reading_ranks_fp16_subnormals_flushed(tmp_path, monkeypatch):
    """Every step-budget gate on Float16 ranks the subnormal band flushed, so a reading
    from a tolerance cell must too, or one file mixes the two and a folded-back figure
    is looser than its gate: a golden 6e-08 (the smallest fp16 subnormal) against a
    flushed 0 is one step unflushed and none flushed."""
    rows = _measure_rows(tmp_path, monkeypatch)
    golden = torch.full((32,), 1.5, dtype=torch.float16)
    golden[0] = 6e-08
    result = golden.clone()
    result[0] = 0.0
    accuracy_contract(
        MathOperation.Abs, output_format=DataFormat.Float16, arch=MEASURED_ARCH
    )
    assert passed_test(golden, result, DataFormat.Float16)
    (row,) = rows()
    assert row.worst == 0, row


def test_an_unpreparable_measure_path_is_refused_before_anything_runs(
    tmp_path, monkeypatch
):
    """A path that is a directory raises, and the plugin makes that a usage error: the
    sweep's records are required, and under `-n` the workers set the path themselves,
    so a controller that only switched itself off left every worker failing every
    sweep cell while a serial run passed and recorded nothing."""
    import helpers.utils as utils

    monkeypatch.setattr(utils, "_ULP_MEASURE_PATH", str(tmp_path))  # a directory
    with pytest.raises(OSError):  # allow-pytest.raises: host-only test
        utils.prepare_ulp_measure_file()

    target = tmp_path / "nested" / "measure.jsonl"
    monkeypatch.setattr(utils, "_ULP_MEASURE_PATH", str(target))
    target.parent.mkdir()
    target.write_text("stale\n", encoding="utf-8")
    utils.prepare_ulp_measure_file()
    assert target.read_text() == ""  # truncated, so runs do not fold together

    # `--ulp-measure out/dir/x.jsonl` with no `out/dir` yet: the parent is created, not
    # warned about.
    fresh = tmp_path / "out" / "dir" / "measure.jsonl"
    monkeypatch.setattr(utils, "_ULP_MEASURE_PATH", str(fresh))
    utils.prepare_ulp_measure_file()
    assert utils._ULP_MEASURE_PATH == str(fresh) and fresh.read_text() == ""


def test_a_measure_recorder_write_failure_does_not_fail_the_comparison(
    tmp_path, monkeypatch
):
    """Reporting only: an unwritable path warns and is otherwise ignored, so it can
    neither fail a passing test nor hide a failing comparison's summary."""
    import helpers.utils as utils

    monkeypatch.setattr(utils, "_ULP_MEASURE_PATH", str(tmp_path))  # a directory
    monkeypatch.setattr(utils, "_ULP_MEASURE_WARNED", False)
    golden = torch.full((32,), 1.5, dtype=torch.bfloat16)
    for _ in range(2):
        accuracy_contract(
            MathOperation.Abs, output_format=DataFormat.Float16_b, arch=MEASURED_ARCH
        )
        assert passed_test(golden, golden.clone(), DataFormat.Float16_b)
    assert utils._ULP_MEASURE_WARNED


def test_arch_must_be_passed_explicitly():
    """The one dimension whose numbers do not transfer cannot default to Wormhole."""
    with pytest.raises(TypeError):  # allow-pytest.raises: host-only test
        accuracy_contract(MathOperation.Abs, output_format=DataFormat.Float32)


# ── The live table ────────────────────────────────────────────────────────────


def test_the_registry_resolves_unambiguously_for_every_variant():
    validate_registry()


def test_enrolled_ops_is_sorted_and_stable():
    ops = enrolled_ops()
    assert list(ops) == sorted(ops, key=lambda op: op.name)
    assert len(set(ops)) == len(ops)


#: Enrolled ops with no step budget anywhere: the 3-segment LUT pair, two binaries
#: whose per-format tolerances moved into the table, and three transcendentals whose
#: *best* cell is still past its output's usable ceiling (6 bf16, 51 fp16)
#: once the near-zero floor the emitter grants (up to 1e-3) is applied -- the
#: measurements are on their rows, not repeated here to drift. Softplus answers 0 below
#: x = -5, a 0.0065 cut-off a 1e-3 floor cannot cover; Digamma and Lgamma are wrong for
#: negative and small positive x. Recorded, not fixed; tracked: Digamma #51128, Softplus
#: #51866 (input clamps, under #52178) and Lgamma #55356.
#:
#: Erfc left this set with the claim applied to the ranking, the Dest-capacity rule and
#: the emitted floor: it has cells inside the ceiling now and the rest fall through to
#: tolerance, like Sign, Heaviside, GeluTanh, Tanhshrink, Xielu, I1 and SfpuElwmul.
#: Polygamma came back: its only cells inside the ceiling were Bfp4_b inputs, whose
#: quantization rather than the op made them look good. Expm1Cw stays: its only gateable cells were Float16 at
#: dest_acc=No, where it answers the golden's overflow stand-in negated (-130560 for
#: 130560, -131008 for 131008: #54770), the op's sign flip and not the Dest's overflow.
ONLY_EVER_TOLERANCE = frozenset(
    {
        MathOperation.SigmoidAppx,
        MathOperation.GeluAppx,
        MathOperation.SfpuElwpow,
        MathOperation.SfpuXlogy,
        MathOperation.Softplus,
        MathOperation.Lgamma,
        MathOperation.Digamma,
        MathOperation.Expm1Cw,
        MathOperation.Polygamma,
    }
)


def test_every_enrolled_op_reaches_its_step_budget():
    """The sweep must reach the ULP branch for every enrolled op but ONLY_EVER_TOLERANCE.

    A sweep that left ``input_format`` unset sent every input-keyed op -- nearly the
    whole table -- to ``TOLERANCE_CONTRACT``; they would show up in ``missing`` here.
    """
    with_budget = {op for op, _, _, _ in _live_step_budgets()}
    missing = set(enrolled_ops()) - with_budget
    assert missing == ONLY_EVER_TOLERANCE, sorted(op.name for op in missing)


#: Ops exact by construction: a sign-bit change, a copy, or an integer-valued result.
EXACT_BY_CONSTRUCTION = (
    MathOperation.Abs,
    MathOperation.Neg,
    MathOperation.Identity,
    MathOperation.Floor,
    MathOperation.Ceil,
    MathOperation.Trunc,
    MathOperation.Round,
)

#: The subset writing an integer, which survives any pack that can represent it.
INTEGER_VALUED = (
    MathOperation.Floor,
    MathOperation.Ceil,
    MathOperation.Trunc,
    MathOperation.Round,
)

#: Ops whose correct result is the *only* result: an integer, a predicate's 1.0/0.0, a
#: sign's -1/0/1, a pass-through-or-zero selection, an operand selection, a constant, a
#: clamp, or a single IEEE add. One step here is the contract breaking, not the pack path.
#: Listed by what the op computes, not read back from the table, which would agree with
#: it by construction. ReluMax, ReluMin and Frac stay out: each carries a sampled
#: `measured: 1` wildcard row, which would have to be split by input first --
#: on a wildcard the guard below cannot tell a narrowing cell's pack step from drift.
#: SfpuBinaryMax/Min had the same wildcards and are keyed by input now.
EXACT_ZERO_BY_CONSTRUCTION = (
    *INTEGER_VALUED,
    MathOperation.Fill,
    MathOperation.Threshold,
    MathOperation.Clamp,
    MathOperation.Hardtanh,
    MathOperation.Isfinite,
    MathOperation.Isinf,
    MathOperation.Isnan,
    MathOperation.Isneginf,
    MathOperation.Isposinf,
    MathOperation.LogicalNot,
    MathOperation.Signbit,
    MathOperation.Sign,
    MathOperation.Heaviside,
    MathOperation.EqualZero,
    MathOperation.NotEqualZero,
    MathOperation.LessThanZero,
    MathOperation.GreaterThanZero,
    MathOperation.LessThanEqualZero,
    MathOperation.GreaterThanEqualZero,
    MathOperation.UnaryEq,
    MathOperation.UnaryNe,
    MathOperation.UnaryGt,
    MathOperation.UnaryGe,
    MathOperation.UnaryLt,
    MathOperation.UnaryLe,
    MathOperation.SfpuElwEq,
    MathOperation.SfpuElwNe,
    MathOperation.SfpuElwGt,
    MathOperation.SfpuElwGe,
    MathOperation.SfpuElwLt,
    MathOperation.SfpuElwLe,
    MathOperation.SfpuIsclose,
    MathOperation.SfpuMask,
    MathOperation.SfpuAddTopRow,
    MathOperation.SfpuBinaryMax,
    MathOperation.SfpuBinaryMin,
)

#: The subset whose every result is exact in every format -- a predicate's 1.0/0.0 or a
#: constant -- so even a narrowing cell has nothing for the pack to round.
EXACT_IN_EVERY_FORMAT = tuple(
    op
    for op in EXACT_ZERO_BY_CONSTRUCTION
    if op
    not in (
        *INTEGER_VALUED,
        MathOperation.Threshold,
        MathOperation.Clamp,
        MathOperation.Hardtanh,
        MathOperation.SfpuMask,
        MathOperation.SfpuAddTopRow,
        MathOperation.SfpuBinaryMax,
        MathOperation.SfpuBinaryMin,
    )
)

#: What the output pack may cost an exact op on a cell that converts: one step. The
#: figure is the measurement itself, so one is also the most such a cell may read.
_PACK_PATH_STEPS = 1


def _mantissa_bits(fmt):
    """0 for a format without a per-element ULP (these appear on the input axis only)."""
    return _ULP_DTYPES[ulp_dtype(fmt)].mantissa_bits if has_ulp_gate(fmt) else 0


def _exact_allowance(op, input_format, output_format):
    """``(steps, reason)``: the slack an exact op may carry on one cell -- the cost of
    the output pack, and only where there is one."""
    if op in EXACT_IN_EVERY_FORMAT:
        return 0, "a 1.0/0.0 or a constant is exact in every format, so no pack rounds"
    if op not in EXACT_ZERO_BY_CONSTRUCTION:
        return (
            _PACK_PATH_STEPS,
            "the value passes through an fp32 Dest and is packed back",
        )
    if input_format is None:
        return (
            _PACK_PATH_STEPS,
            "the row wildcards `input`, so it covers a narrowing cell",
        )
    if _mantissa_bits(output_format) < _mantissa_bits(input_format):
        # e.g. Float16 -> Float16_b: bf16 cannot hold every integer fp16 can.
        return _PACK_PATH_STEPS, "the output has fewer mantissa bits than the input"
    return 0, "the output can represent every value this op produces from that input"


@pytest.mark.parametrize(
    "op",
    sorted(
        set(EXACT_BY_CONSTRUCTION) | set(EXACT_ZERO_BY_CONSTRUCTION),
        key=lambda op: op.name,
    ),
    ids=lambda op: op.name,
)
def test_an_exact_op_never_carries_a_wide_budget(op):
    """These are the canaries: a budget past the pack path means the number was fitted
    to a failure. For the exactly rounded ops "any drift is a regression" is the whole
    claim, so each must also *have* a step budget somewhere: a dropped row would read
    as a passing op."""
    seen = False
    for budget_op, in_fmt, fmt, contract in _live_step_budgets():
        if budget_op is not op:
            continue
        seen = True
        allowance, why = _exact_allowance(op, in_fmt, fmt)
        assert contract.max_ulp <= allowance, (
            f"{op.name} on {in_fmt and in_fmt.name}->{fmt.name} carries "
            f"max_ulp={contract.max_ulp}, past the {allowance} it may have because "
            f"{why}. The op is exact by construction; investigate the datapath or the "
            "golden rather than widening the budget."
        )
    if op in EXACT_ZERO_BY_CONSTRUCTION:
        assert (
            seen
        ), f"{op.name} resolves to no ULP contract at all; the row was dropped"


def _driver_variants(test_function):
    """Every parameter combination pytest collects for *test_function*, as dicts: the
    cross product of its ``parametrize`` marks, read from the function itself so a
    format or op added to a driver is covered without editing this file."""
    axes = []
    for mark in getattr(test_function, "pytestmark", []):
        if mark.name != "parametrize":
            continue
        names = [n.strip() for n in mark.args[0].split(",")]
        rows = []
        for value in mark.args[1]:
            value = getattr(value, "values", value)  # a pytest.param
            rows.append(dict(zip(names, value if len(names) > 1 else (value,))))
        axes.append(rows)
    for combo in product(*axes):
        yield {k: v for part in combo for k, v in part.items()}


def _unary_step_budget_drivers():
    """The tests in test_eltwise_unary_sfpu.py that call the driver with
    ``gate_on_step_budget=True``, read from the source: a list kept here instead would
    stay green after the keyword was dropped, and that sweep would quietly gate on
    tolerance alone."""
    import ast
    import inspect

    import test_eltwise_unary_sfpu as unary

    tree = ast.parse(inspect.getsource(unary))
    return sorted(
        fn.name
        for fn in tree.body
        if isinstance(fn, ast.FunctionDef)
        and fn.name.startswith("test_")
        and any(
            isinstance(node, ast.Call)
            and any(
                kw.arg == "gate_on_step_budget"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in node.keywords
            )
            for node in ast.walk(fn)
        )
    )


#: The unary step-budget drivers that run one op rather than a ``mathop`` axis.
_SINGLE_OP_DRIVERS = {"test_eltwise_unary_sfpu_signbit": MathOperation.Signbit}


def test_the_unary_step_budget_drivers_are_the_measured_sweeps():
    """Exactly the sweeps MEASURED_ON_SWEEP records gate on the step budget, read from
    the call sites rather than trusted."""
    import test_eltwise_unary_sfpu as unary

    drivers = _unary_step_budget_drivers()
    assert drivers == sorted(
        f"test_eltwise_unary_sfpu_{sweep}" for sweep in MEASURED_ON_SWEEP
    )
    for name in drivers:
        has_op_axis = all(
            "mathop" in variant for variant in _driver_variants(getattr(unary, name))
        )
        assert (
            has_op_axis or name in _SINGLE_OP_DRIVERS
        ), f"{name} has no mathop axis; name its op in _SINGLE_OP_DRIVERS"


def _exact_op_driver_variants():
    """``(driver, op, formats, approx, dest_acc)`` for every variant a hand-built driver
    runs an exact op in and gates on the op's *whole* contract, step budget included.

    Every binary driver does (``sfpu_binary`` -> ``assert_against_contract``), with the
    Dest promotion applied first and ``_APPROX_MODE``; of the unary drivers only the
    ones passing ``gate_on_step_budget`` (:func:`_unary_step_budget_drivers`) -- the
    rest gate on tolerance. *driver* is the test function."""
    import inspect

    import test_eltwise_binary_sfpu as binary
    import test_eltwise_unary_sfpu as unary

    drivers = [
        (fn, None, binary._APPROX_MODE)
        for name, fn in inspect.getmembers(binary, inspect.isfunction)
        if name.startswith("test_")
    ] + [
        (getattr(unary, name), _SINGLE_OP_DRIVERS.get(name), None)
        for name in _unary_step_budget_drivers()
    ]
    exact = set(EXACT_ZERO_BY_CONSTRUCTION)
    for fn, fixed_op, approx in drivers:
        for variant in _driver_variants(fn):
            op = fixed_op or variant.get("mathop")
            if op in exact:
                yield (
                    fn,
                    op,
                    variant["formats"],
                    variant.get("approx_mode", approx),
                    variant["dest_acc"],
                )


def _exact_driver_skip(driver, op, formats, dest_acc):
    """Why *driver* skips this variant, or ``None``: the drivers' own skip predicates,
    asked rather than re-typed, so an exclusion here can neither outlive nor outgrow the
    skip. A binary driver applies the Float32 one exactly when it calls
    ``skip_if_32bit_input_without_dest_acc``."""
    import inspect

    import test_eltwise_unary_sfpu as unary
    from helpers.constraints import skip_reason_32bit_input_without_dest_acc

    if driver is unary.test_eltwise_unary_sfpu_isinf_isnan:
        return unary.isinf_isnan_skip_reason(formats, op, dest_acc)
    if "skip_if_32bit_input_without_dest_acc(" in inspect.getsource(driver):
        return skip_reason_32bit_input_without_dest_acc(formats, dest_acc)
    return None


def _step_gateable(*formats) -> bool:
    """A variant a step budget can gate: every format named one the SFPU computes in --
    not an integer, which wants bit equality, nor a block float, whose error is the
    unpacker's and packer's. The exact-op guards and the not-measurable audit filter on
    it, so they agree on what "gateable" means."""
    return all(f is None or f in ULP_FORMATS for f in formats)


def test_every_driven_variant_of_an_exact_op_is_gated():
    """`test_an_exact_op_never_carries_a_wide_budget` asks only that each exact op keep
    *some* step budget: drop every `SfpuMask` Float16_b row and keep its Float32 one, and
    the Float16_b variants fall back to tolerance while it still passes. This walks every
    variant the hand-built drivers run an exact op in between formats the SFPU computes
    in (`_step_gateable`), and requires each to resolve to a step budget -- at the Dest
    the variant asks for and at the one it is promoted to. A variant the driver skips
    never runs, so it must resolve
    to *no* step budget: one there claims a measurement nobody took -- which is how a
    re-sort that dropped Isinf/Isnan/Isneginf's `not measured` rows showed up. The swept
    cells of the unary exact ops are held by
    `test_every_swept_cell_of_an_exact_op_is_gated_or_waived`."""
    from helpers.data_format_inference import effective_dest_acc

    ungated, claimed = [], []
    for driver, op, formats, approx, dest in _exact_op_driver_variants():
        out_fmt = formats.output_format
        if not _step_gateable(formats.input_format, out_fmt):
            continue
        promoted = effective_dest_acc(
            formats.input_format, out_fmt, dest, MEASURED_ARCH
        )
        for d in {dest, promoted}:
            contract = accuracy_contract(
                op,
                output_format=out_fmt,
                input_format=formats.input_format,
                approx_mode=approx,
                dest_acc=d,
                arch=MEASURED_ARCH,
            )
            cell = (
                f"{driver.__name__}: {op.name} "
                f"{formats.input_format.name}->{out_fmt.name}"
            )
            if _exact_driver_skip(driver, op, formats, d):
                if contract.metric is Metric.ULP:
                    claimed.append(f"{cell} dest={d.name}")
                continue
            if contract.metric is Metric.ULP:
                continue
            ungated.append(f"{cell} dest={d.name}")
    assert not ungated, (
        "exact-op variant(s) a driver runs that resolve to no step budget, so a "
        "regression there passes on tolerance:\n  " + "\n  ".join(sorted(set(ungated)))
    )
    assert not claimed, (
        "variant(s) the driver skips that resolve to a step budget, a measurement the "
        "table does not have; key a `not measured` tolerance row on them:\n  "
        + "\n  ".join(sorted(set(claimed)))
    )


def test_every_exact_op_is_driven_by_a_gate():
    """`test_every_driven_variant_of_an_exact_op_is_gated` and
    `test_every_swept_cell_of_an_exact_op_is_gated_or_waived` hold only the ops something
    drives: an exact op neither swept nor run by a step-budget driver would be checked
    by nothing but the existence of a row. Driven means on a variant a step budget can
    gate and the driver does not skip, the same filter the first of those applies: an
    op left with only Int32 outputs, or only skipped variants, is driven by nothing."""
    driven = {
        op
        for driver, op, formats, _, dest in _exact_op_driver_variants()
        if _step_gateable(formats.input_format, formats.output_format)
        and not _exact_driver_skip(driver, op, formats, dest)
    }
    unreached = set(EXACT_ZERO_BY_CONSTRUCTION) - driven - set(_swept_exact_ops())
    assert not unreached, sorted(op.name for op in unreached)


#: Swept cells of an exact-by-construction op that the table holds on the tolerance
#: metric, with what was measured there. Each would be a real deviation on an op that
#: should be exact, with no cause established yet;
#: `test_every_swept_cell_of_an_exact_op_is_gated_or_waived` keeps the list from growing
#: unnoticed, and fails when an entry is no longer needed. Empty today. Every
#: class it used to hold was the sweep's, not the ops': Abs/Neg/Identity's 512-step
#: Float16 cells were the metric keeping fp16 subnormals the pack does not reproduce;
#: Floor/Ceil's Float32 -> Float16 dest_acc=No cells were the fp16 Dest's flush of the
#: strided lanes below 2**-14. All of them measure 0.
_EXACT_OP_DEMOTIONS: dict = {}


def _swept_exact_ops():
    """The exact ops the exhaustive sweep drives: every sign-bit/copy/integer op, and
    the unary members of EXACT_ZERO_BY_CONSTRUCTION it has a domain for. The predicates
    it has none for (Isinf, UnaryEq, ...) are gated on their hand-built sweeps instead.
    """
    from helpers.sfpu_domains import _UNARY_OPS_NOT_SWEPT, sfpu_unary_ops

    swept = set(sfpu_unary_ops()) - set(_UNARY_OPS_NOT_SWEPT)
    return sorted(
        set(EXACT_BY_CONSTRUCTION)
        | {op for op in EXACT_ZERO_BY_CONSTRUCTION if op in swept},
        key=lambda op: op.name,
    )


@pytest.mark.parametrize("op", _swept_exact_ops(), ids=lambda op: op.name)
def test_every_swept_cell_of_an_exact_op_is_gated_or_waived(op):
    """`test_an_exact_op_never_carries_a_wide_budget` reads only ULP rows, so a cell the
    emitter demoted to tolerance is invisible to it -- and the sweep does not gate
    tolerance cells. Every swept, non-block cell of these ops must resolve to a step
    budget, or be listed above with its measurement."""

    demoted = set()
    # The Wormhole cells, whatever CHIP_ARCH this host sets: the contracts below are
    # resolved at MEASURED_ARCH, and Quasar promotes nothing.
    for in_fmt, out_fmt, approx, dest in sweep_cells(MEASURED_ARCH):
        contract = accuracy_contract(
            op,
            output_format=out_fmt,
            input_format=in_fmt,
            approx_mode=approx,
            dest_acc=dest,
            arch=MEASURED_ARCH,
        )
        if contract.metric is not Metric.ULP:
            demoted.add((op, in_fmt, out_fmt, dest))
    waived = {cell for cell in _EXACT_OP_DEMOTIONS if cell[0] is op}
    assert demoted <= waived, sorted(
        f"{i.name}->{o.name} dest={d.name}" for _, i, o, d in demoted - waived
    )
    assert waived <= demoted, "stale waiver(s): " + ", ".join(
        sorted(f"{i.name}->{o.name} dest={d.name}" for _, i, o, d in waived - demoted)
    )


def test_no_budget_exceeds_its_formats_usable_ceiling():
    """Past ``usable_budget_ceiling`` a step budget is looser than the tolerance it
    replaces -- and it is the whole gate, since the ULP arm of passed_test skips
    isclose and PCC. Such an op belongs on the tolerance metric."""
    for op, _, fmt, contract in _live_step_budgets():
        ceiling = usable_budget_ceiling(fmt)
        assert contract.max_ulp <= ceiling, (
            f"{op.name} on {fmt.name} has max_ulp={contract.max_ulp}, past the "
            f"{ceiling:.0f}-step point where a budget stops being tighter than the "
            "tolerance it replaces. Put the op on the tolerance metric and record "
            "the measurement instead."
        )


def test_the_usable_ceiling_is_tighter_than_the_meaningful_one():
    for fmt in ULP_FORMATS:
        assert usable_budget_ceiling(fmt) < MAX_MEANINGFUL_ULP[ulp_dtype(fmt)], fmt.name
    # 6.4, rounded down: a 7-step budget accepts bf16 128 -> 135, which the 0.05 +
    # 0.05 * 128 = 6.45 tolerance it replaces refuses.
    assert usable_budget_ceiling(DataFormat.Float16_b) == 6


def test_no_declared_tolerance_is_shadowed_by_a_numberless_row():
    """The live-table form of the bare-row test: every variant an op's numbered
    tolerance row covers resolves to that tolerance or to a step budget, never to the
    per-format default."""
    shadowed = []
    for op, table in _SFPU_ACCURACY_BUDGET.items():
        numbered = [
            key for key, contract in table.items() if contract.declares_tolerance
        ]
        if not numbered:
            continue
        input_formats = sorted(
            {key.input_format for key in table} - {None}, key=lambda f: f.name
        ) + [None]
        for dims in product(
            [*ApproximationMode, None],
            input_formats,
            ULP_FORMATS,
            [*DestAccumulation, None],
            ChipArchitecture,
        ):
            query = BudgetKey(*dims)
            if not any(key.matches(query) for key in numbered):
                continue
            contract = accuracy_contract(
                op,
                approx_mode=query.approx_mode,
                input_format=query.input_format,
                output_format=query.output_format,
                dest_acc=query.dest_acc,
                arch=query.arch,
            )
            # A list cell's contract carries `domains`, so it is never `==` the default;
            # ask what the fallback is, not what it equals.
            if contract.metric is Metric.TOLERANCE and not contract.declares_tolerance:
                shadowed.append(f"{op.name} {query.describe()}")
    assert not shadowed, "\n".join(shadowed[:20])


def test_no_integer_only_op_is_enrolled():
    """A step budget on an integer op is meaningless, and the driver would raise. The
    listed ops pin the derivation: each was missed by an earlier name-based one."""
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
    enrolled_integer = set(enrolled_ops()) & integer_ops
    assert not enrolled_integer, sorted(op.name for op in enrolled_integer)


#: Ops measured on the hand-built sweep that drives them, under ``--ulp-report`` on
#: Wormhole; the rows' notes carry the variant and lane counts. Those sweeps gate on
#: the whole contract (``gate_on_step_budget``), so these rows are what they enforce.
#: For the eight predicates without a registered domain (the isinf/isnan five,
#: LogicalNot, UnaryEq, UnaryNe) these sweeps are the only gate; Signbit, ReluMin and
#: ReluMax are also swept exhaustively, and on their 16-bit cells it is those more
#: specific exhaustive rows that the sweeps here enforce.
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
    """The signbit, isinf/isnan and threshold sweeps use hand-built stimuli and gate on
    the step budget, so an op they drive must have been measured there."""
    from test_eltwise_unary_sfpu import _THRESHOLD_OPS, ISINF_ISNAN_MATHOPS

    hand_built = {
        "signbit": {MathOperation.Signbit},  # not parametrised; drives this one op
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


# ── Every step budget is the measurement it came from ─────────────────────────
#
# The gate is the row's own `measured` figure; the loader refuses a ULP row without one.
# There is no second number to widen by hand.


def test_a_demoted_row_measured_past_the_ceiling():
    """A tolerance row the emitter wrote -- it pins ``input`` and ``output``, as every row it
    writes does -- with a measurement is a demotion only if its ``measured`` is past
    ``usable_budget_ceiling``: ``_verdict``'s call for that figure. One inside the
    ceiling should have been enrolled. A broader row is a sampled hand judgement.

    Empty today: the emitter now writes a demotion as a ``domains`` list with no
    ``metric`` or ``measured``, and the only tolerance rows with a ``measured`` are the
    ``nonfinite`` ones, which are not measurable. A tripwire for a hand-edited row."""
    wrong = [
        (op, row)
        for op, rows in load_table().items()
        for row in rows
        if row.metric is Metric.TOLERANCE
        and row.measured is not None
        and not row.unmeasurable
        and row.input is not None
        and row.output is not None
        and _verdict(row.measured, row.output) is not Metric.TOLERANCE
    ]
    assert not wrong, wrong


# ── A gated cell is not quietly parked ────────────────────────────────────────
#
# The emitter writes `not measurable` for a cell in which one lane disagrees with the
# golden about being finite, and the sweep gates no tolerance row -- so an overflow a
# later emit introduces into a gated cell would take the whole cell off the gate with
# nothing to notice. Exact ops are held by _EXACT_OP_DEMOTIONS; this holds the rest.

#: Swept cells on a gateable output that the table holds as ``not measurable``, keyed
#: ``(op, in, out, approx, dest)`` with ``None`` for an axis the cause does not depend
#: on, each with that cause. A cell whose disagreeing lanes are a tracked defect on a
#: handful of inputs does not belong here: name the inputs in
#: ``ulp_sweep._KNOWN_NONFINITE_LANES`` instead, and the rest of the cell stays gated.
_UNMEASURABLE_CELLS_ACKNOWLEDGED = {
    # -- the approximation's own shortfall at the fp16 and fp32 overflow edges -----
    (
        MathOperation.Exp,
        DataFormat.Float16,
        DataFormat.Float16,
        ApproximationMode.Yes,
        None,
    ): (
        "approximate exp answers 64256..65408 for x in 11.09..11.12, where exp(x) is "
        "past 65504: the approximation's own shortfall at the overflow edge, which no "
        "store or golden fix removes"
    ),
    (
        MathOperation.Exp,
        DataFormat.Float32,
        DataFormat.Float16,
        ApproximationMode.Yes,
        None,
    ): ("the same shortfall from a strided Float32 input, one lane"),
    (
        MathOperation.Exp,
        DataFormat.Float16,
        DataFormat.Float32,
        ApproximationMode.Yes,
        DestAccumulation.Yes,
    ): (
        "the same shortfall at fp32's overflow edge: exp(88.75) is 3.497e38, past "
        "FLT_MAX, and the approximation answers 3.396e38, one lane"
    ),
    (
        MathOperation.ExpWithBase,
        DataFormat.Float16,
        DataFormat.Float16,
        ApproximationMode.Yes,
        None,
    ): (
        "the same shortfall through the 0.5 scale: approximate exp(0.5 x) answers "
        "64256..64640 for x in 22.19..22.22, where the answer is past 65504"
    ),
    (
        MathOperation.ExpWithBase,
        DataFormat.Float32,
        DataFormat.Float16,
        ApproximationMode.Yes,
        None,
    ): (
        "the same shortfall through the 0.5 scale from a strided Float32 input, one lane"
    ),
    (
        MathOperation.ExpWithBase,
        DataFormat.Float16,
        DataFormat.Float32,
        ApproximationMode.Yes,
        DestAccumulation.Yes,
    ): (
        "the same shortfall at fp32's overflow edge through the 0.5 scale: exp(0.5 * "
        "177.5) is past FLT_MAX, and the approximation answers 3.396e38, one lane"
    ),
    # -- kernel behaviour over a wide band of the format -----------------------------
    # Each is what the sweep found and the row records; none is a golden or store
    # artefact, and none is a handful of lanes an issue could name. Each names the issue
    # that tracks it, or says why it is no defect, so retiring the entry is tied to
    # something: the stale check below fails once no row needs it.
    (MathOperation.Digamma, None, None, None, None): (
        "wrong for negative x, which the LUT has no branch for (#58682): -inf of the "
        "wrong sign for large negative x (the saturation below the fit range is #51128)"
    ),
    (MathOperation.Expm1Cw, None, None, None, None): (
        "past the overflow point (x ~ 89) the kernel returns -1, the x -> -inf limit, "
        "instead of inf: only the low side is clamped, and SFPSETEXP wraps the "
        "exponent above it (#54770, closed because ELU/SELU/CELU feed it only x < 0; "
        "the test-only calculate_expm1_cw wrapper is the one caller that reaches it)"
    ),
    (MathOperation.I0, None, None, None, None): (
        "a finite answer to an infinite golden from |x| ~ 92 up: the kernel is a bare "
        "Taylor series with no large-|x| branch (#50465, closed without a fix on main; "
        "the fix is #52126). The Float16 cells' one lane in the gap below, x = +-13.3, "
        "is named in _KNOWN_NONFINITE_LANES, so those cells stay gated"
    ),
    (MathOperation.I1, None, None, None, None): (
        "saturates at +-1.16e37 where i1 overflows fp32, half the format: the input is "
        "clamped at 88.5 (#52174, under #52178)"
    ),
    (MathOperation.Lgamma, None, None, None, None): (
        "no defect: the driver runs calculate_lgamma_stirling alone, which answers "
        "lgamma(1 - x) below 0.5 (the reflection correction is the next stage, "
        "lgamma.h), so at the negative-integer poles it answers finite until lgamma(1 - x) "
        "itself overflows near |x| = 4e36. Its in-domain error is #55356. Separately, "
        "a 16-bit Float16 Dest answers inf at x = 8172 and 8176, where lgamma is a top "
        "fp16 value: #58607, named lane by lane in _KNOWN_NONFINITE_LANES"
    ),
    (MathOperation.Polygamma, None, None, None, None): (
        "wrong for x < 0, where the Euler-Maclaurin tail is evaluated outside its "
        "domain (#52278): 0 for large negative x where the golden is inf, and a finite "
        "or NaN answer beside the poles at the negative integers"
    ),
    (MathOperation.Rpow, None, None, None, None): (
        "2**x for |x| above ~8e31: 1 where the answer is inf on a 16-bit Dest, whose "
        "power body has no overflow guard (#52675), and inf where it is 0 on a 32-bit "
        "Dest, whose Veltkamp split overflows (#55129); NaN on a Float16 output is the "
        "store's inf (#57215)"
    ),
}


#: How many not-measurable cells each *wildcard* acknowledgement above covers today. An
#: acknowledgement that leaves a dimension open -- an op-wide one most of all -- would
#: otherwise approve every cell of the op that is ever demoted, for any cause, and the
#: stale check below could not retire it while one such cell remained. Pinning the count
#: makes a newly demoted cell a change someone has to read: the table moved, the cause
#: must be looked at, and the number here bumped on purpose. Exact-cell keys need no pin.
_UNMEASURABLE_CELLS_ACKNOWLEDGED_COUNTS = {
    (
        MathOperation.Exp,
        DataFormat.Float16,
        DataFormat.Float16,
        ApproximationMode.Yes,
        None,
    ): 2,
    (
        MathOperation.Exp,
        DataFormat.Float32,
        DataFormat.Float16,
        ApproximationMode.Yes,
        None,
    ): 2,
    (MathOperation.Digamma, None, None, None, None): 4,
    (
        MathOperation.ExpWithBase,
        DataFormat.Float16,
        DataFormat.Float16,
        ApproximationMode.Yes,
        None,
    ): 2,
    (
        MathOperation.ExpWithBase,
        DataFormat.Float32,
        DataFormat.Float16,
        ApproximationMode.Yes,
        None,
    ): 2,
    (MathOperation.Expm1Cw, None, None, None, None): 15,
    (MathOperation.I0, None, None, None, None): 10,
    (MathOperation.I1, None, None, None, None): 10,
    (MathOperation.Lgamma, None, None, None, None): 1,
    (MathOperation.Polygamma, None, None, None, None): 11,
    (MathOperation.Rpow, None, None, None, None): 10,
}


def _not_measurable_cells():
    """``(op, in, out, approx_or_None, dest_or_None)`` for every ``not measurable`` row
    on an output a step budget could gate."""
    return [
        (op, row.input, row.output, row.approx, row.dest)
        for op, rows in load_table().items()
        for row in rows
        if row.unmeasurable and _step_gateable(row.output)
    ]


#: The repo's `check-large-files` pre-commit hook (scripts/check_file_size.py): a file of
#: more than this many KB, rounded up, fails it, and the table has no exclude there.
_REPO_FILE_SIZE_LIMIT_KB = 500


def test_the_table_fits_the_repos_file_size_limit():
    """The hook only runs in pre-commit, so a re-emit that grows the table past it is
    otherwise found by CI. It grows with every op enrolled, every entry a list gains and
    every not-measurable verdict (``ulp_sweep.NAMED_LANES`` lanes each): trim a
    recurring note before raising anything, or give the hook a justified exclude."""
    kb = math.ceil(TABLE_PATH.stat().st_size / 1024)
    assert kb <= _REPO_FILE_SIZE_LIMIT_KB, (
        f"{TABLE_PATH.name} is {kb} KB, past the {_REPO_FILE_SIZE_LIMIT_KB} KB "
        "check-large-files limit"
    )


#: Rows whose every claimed lane disagreed with the golden about being finite: there
#: is no rest of the cell to measure, so `nonfinite_verdict` records no `measured` (a 0
#: would claim it bit-exact). The table cannot tell such a row from one that lost its
#: figure, so each is listed by hand after reading the emit. Empty today.
_EVERY_CLAIMED_LANE_NONFINITE: frozenset = frozenset()


def test_a_row_parked_by_disagreeing_lanes_records_the_rest_of_its_cell():
    """`nonfinite_verdict` keeps the step count over the measurable lanes beside the
    lanes that park a cell, so a cell parked by four lanes still says what the other
    tens of thousands measure, and the headroom report (from #57527) has a figure to
    hold it to. A row written before the emitter kept that figure records nothing for
    them; re-emit its op. A cell with no measurable lane left is listed in
    `_EVERY_CLAIMED_LANE_NONFINITE` instead."""
    bare = [
        (op, row)
        for op, rows in load_table().items()
        for row in rows
        if row.unmeasurable
        and row.measured is None
        and (op, row.key) not in _EVERY_CLAIMED_LANE_NONFINITE
    ]
    assert not bare, (
        f"{len(bare)} row(s) name the lanes that park their cell but not the maximum "
        "over the rest of it: re-emit their ops, or, where the emit still records no "
        "figure (no lane of the cell is finite on both sides), list the row in "
        f"_EVERY_CLAIMED_LANE_NONFINITE: {bare}"
    )


def _acknowledges(key, cell) -> bool:
    return all(k is None or k == c for k, c in zip(key, cell))


#: Swept cells of an enrolled unary op that resolve to no row of the op's own, with why.
#: Empty: every enrolled op covers every cell the sweep drives. The emitter writes whole
#: grids, so a hole can only open where it keeps a block verbatim (an op-wide
#: `atol`/`rtol` anchor) and the sweep's format set then grows -- the cell is measured,
#: resolves to the global default, and nothing holds it. A cell listed here is a
#: decision, with its reason, not a gap.
_UNRESOLVED_SWEPT_CELLS: dict = {}


def test_every_swept_cell_of_an_enrolled_op_resolves_to_a_row_of_its_own():
    """The sweep gates a ULP row and the headroom report (from #57527) judges a
    tolerance row; a swept cell that resolves to ``TOLERANCE_CONTRACT`` -- no row of the
    op covers it -- is measured and judged by nothing, and reads in the table exactly
    like a cell nobody ever enrolled. ``test_every_swept_cell_of_an_exact_op_is_gated_or_waived`` asks this
    of the exact ops; this asks it of every enrolled op."""
    from helpers.sfpu_domains import sfpu_unary_ops

    unary = sfpu_unary_ops()
    unresolved, resolved = [], []
    for op in enrolled_ops():
        if op not in unary:
            continue
        table = _SFPU_ACCURACY_BUDGET[op]
        for in_fmt, out_fmt, approx, dest in sweep_cells(MEASURED_ARCH):
            query = BudgetKey(
                input_format=in_fmt,
                output_format=out_fmt,
                approx_mode=approx,
                dest_acc=dest,
                arch=MEASURED_ARCH,
            )
            cell = (op, in_fmt, out_fmt, approx, dest)
            if resolve_contract(table, query) is TOLERANCE_CONTRACT:
                if cell not in _UNRESOLVED_SWEPT_CELLS:
                    unresolved.append(cell)
            elif cell in _UNRESOLVED_SWEPT_CELLS:
                resolved.append(cell)

    def _name(cell):
        op, i, o, a, d = cell
        return f"{op.name} {i.name}->{o.name} approx={a.name} dest={d.name}"

    assert not unresolved, (
        "swept cells no row of the op covers (measured by the sweep, judged by "
        "nothing): give them a row, or list them in _UNRESOLVED_SWEPT_CELLS with why:\n"
        + "\n".join(f"  {_name(c)}" for c in unresolved)
    )
    assert not resolved, "listed as unresolved but a row covers them now: " + ", ".join(
        _name(c) for c in resolved
    )


def test_a_not_measurable_verdict_on_a_gateable_cell_is_acknowledged():
    cells = _not_measurable_cells()
    unacknowledged = [
        cell
        for cell in cells
        if not any(_acknowledges(key, cell) for key in _UNMEASURABLE_CELLS_ACKNOWLEDGED)
    ]
    assert not unacknowledged, (
        "not-measurable cells with no acknowledged cause (a tracked defect on a few "
        "inputs belongs in ulp_sweep._KNOWN_NONFINITE_LANES; anything else, here):\n"
        + "\n".join(
            f"  {op.name} {i.name}->{o.name} approx={a and a.name} dest={d and d.name}"
            for op, i, o, a, d in unacknowledged
        )
    )
    stale = [
        key
        for key in _UNMEASURABLE_CELLS_ACKNOWLEDGED
        if not any(_acknowledges(key, cell) for cell in cells)
    ]
    assert not stale, "acknowledgements no row needs any more: " + ", ".join(
        f"{op.name} {i and i.name}->{o and o.name} approx={a and a.name} "
        f"dest={d and d.name}"
        for op, i, o, a, d in stale
    )
    wildcard = [
        k for k in _UNMEASURABLE_CELLS_ACKNOWLEDGED if any(v is None for v in k[1:])
    ]
    unpinned = [k for k in wildcard if k not in _UNMEASURABLE_CELLS_ACKNOWLEDGED_COUNTS]
    assert (
        not unpinned
    ), "wildcard acknowledgements with no cell count pinned: " + ", ".join(
        f"{k[0].name} {tuple(getattr(v, 'name', None) for v in k[1:])}"
        for k in unpinned
    )
    covered = {
        key: sum(1 for cell in cells if _acknowledges(key, cell))
        for key in _UNMEASURABLE_CELLS_ACKNOWLEDGED_COUNTS
    }
    moved = {
        key: (now, pinned)
        for key, pinned in _UNMEASURABLE_CELLS_ACKNOWLEDGED_COUNTS.items()
        if (now := covered[key]) != pinned
    }
    assert not moved, (
        "the cells a wildcard acknowledgement covers changed; read the new rows, then "
        "bump the pin: "
        + ", ".join(
            f"{k[0].name} {tuple(getattr(v, 'name', None) for v in k[1:])}: {now} "
            f"cells, pinned {pinned}"
            for k, (now, pinned) in moved.items()
        )
    )
    orphaned = set(_UNMEASURABLE_CELLS_ACKNOWLEDGED_COUNTS) - set(wildcard)
    assert not orphaned, f"count pins with no acknowledgement: {sorted(orphaned)}"


def test_every_unary_op_is_enrolled_or_excused():
    """ "Never measured" and "deliberately left on tolerance" both read as absent from
    the table, and the sweep collects only the ops the table names. So every unary op
    either has a block or ``_UNARY_OPS_NOT_SWEPT`` says why not (Erfc, Lgamma and Xielu
    sat here).
    """
    unaccounted = sorted(
        set(sfpu_unary_ops()) - set(enrolled_ops()) - set(_UNARY_OPS_NOT_SWEPT),
        key=lambda op: op.name,
    )
    assert not unaccounted, (
        "no accuracy contract, and no reason given, for: "
        + ", ".join(op.name for op in unaccounted)
        + ". Measure it with --ulp-emit (add the op's key line to the table first), or "
        "add it to _UNARY_OPS_NOT_SWEPT with why it cannot be swept."
    )


# ── Rows with a `domains` list ───────────────────────────────────────────────

_TANH = Row(
    input=DataFormat.Float16_b,
    output=DataFormat.Float16_b,
    dest=DestAccumulation.No,
    near_zero_atol=1e-7,
    domains=TANH_DOMAINS,
)


def test_a_floor_rides_with_a_list_on_either_metric_but_not_alone():
    entries = (Domain(-math.inf, math.inf, 100),)
    AccuracyContract(metric=Metric.TOLERANCE, near_zero_atol=1e-7, domains=entries)
    with refused(Problem.MIXED_METRIC):
        AccuracyContract(metric=Metric.TOLERANCE, near_zero_atol=1e-7)


def test_domains_of_hands_the_sweep_driver_the_list_and_the_floor(monkeypatch):
    """Pure: beside the driver's one `accuracy_contract` lookup it must not publish a
    query of its own, or the recorder would drop that test's reading. And a Wormhole
    list binds nowhere else, as a Wormhole figure does not."""
    import helpers.sfpu_accuracy_budget as budget

    monkeypatch.setattr(
        budget, "_SFPU_ACCURACY_BUDGET", {MathOperation.Tanh: _contracts(_TANH)}
    )
    monkeypatch.setattr(budget, "LAST_QUERY", None)
    query = dict(
        output_format=DataFormat.Float16_b,
        input_format=DataFormat.Float16_b,
        dest_acc=DestAccumulation.No,
        approx_mode=ApproximationMode.No,
    )
    assert domains_of(MathOperation.Tanh, arch=MEASURED_ARCH, **query) == (
        _TANH.domains,
        1e-7,
    )
    assert budget.LAST_QUERY is None
    for arch in UNSWEPT_ARCHS:
        assert domains_of(MathOperation.Tanh, arch=arch, **query) == ((), None)
    assert domains_of(MathOperation.Abs, arch=MEASURED_ARCH, **query) == ((), None)
