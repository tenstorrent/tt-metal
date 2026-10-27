# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only guards for the SFPU accuracy budget registry: the contract and key
types, the resolution rule on purpose-built tables, ``accuracy_contract``'s fallbacks
and downgrades, and the invariants of the live table a reader cannot see in its
numbers. The table file itself is test_ulp_table.py's."""

import ast
import functools
import inspect
import math
from itertools import product

import helpers.chip_architecture as chip
import helpers.sfpu_accuracy_budget as budget
import pytest
import test_eltwise_binary_sfpu as binary
import test_eltwise_unary_sfpu as unary
import torch
from conftest import TANH_DOMAINS, refused
from helpers.chip_architecture import ChipArchitecture
from helpers.constraints import skip_reason_32bit_input_without_dest_acc
from helpers.data_format_inference import effective_dest_acc
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    MathOpType,
)
from helpers.sfpu_accuracy_budget import (
    _SFPU_ACCURACY_BUDGET,
    TOLERANCE_CONTRACT,
    AccuracyContract,
    BudgetKey,
    accuracy_contract,
    assert_against_contract,
    domains_of,
    enrolled_ops,
    resolve_contract,
    validate_registry,
)
from helpers.sfpu_domains import sfpu_unary_ops, swept_unary_ops
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from helpers.ulp import ULP_FORMATS, has_ulp_gate, mantissa_bits, ulp_dtype
from helpers.ulp_domains import Domain, Problem, usable_budget_ceiling
from helpers.ulp_sweep import sweep_cells
from helpers.ulp_table import MEASURED_ARCH, Metric, Row, load_table
from helpers.utils import passed_test

UNSWEPT_ARCHS = [a for a in ChipArchitecture if a != MEASURED_ARCH]

# The input axis for the live-table sweeps: the formats a row may name, and None, which
# resolves the rows that wildcard `input`.
QUERYABLE_INPUT_FORMATS = [*ULP_FORMATS, None]

_LANES = DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM


def _contracts(*rows: Row):
    """One op's rows as the registry holds them."""
    return {row.key: row.contract for row in rows}


def _every_variant(op):
    """``(input_format, output_format, contract)`` for every variant of *op*, every
    dimension swept, ``None`` included where a caller may leave it unset."""
    for fmt in ULP_FORMATS:
        for input_format in QUERYABLE_INPUT_FORMATS:
            for approx_mode in [*ApproximationMode, None]:
                for dest_acc in [*DestAccumulation, None]:
                    for arch in ChipArchitecture:
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
    table resolves to; computed once, as the walk is the slow part."""
    return tuple(
        (op, in_fmt, fmt, contract)
        for op in enrolled_ops()
        for in_fmt, fmt, contract in _every_variant(op)
        if contract.metric is Metric.ULP
    )


def _integer_only_ops() -> set:
    """Every SFPU op whose operands and result are integers: the typed integer
    binaries, the drivers' integer sets and SfpuGcd (in none of those), less ReluMin,
    which has a float path too."""
    typed_binaries = {
        op
        for op in MathOperation
        if op.value.operation_type is MathOpType.SFPU_BINARY_INT
    }
    return (
        typed_binaries
        | set(unary._INT_UNARY_OPS)
        | set(binary._INT_DRIVEN_BINARY_OPS)
        | {MathOperation.SfpuGcd}
    ) - {MathOperation.ReluMin}


# ── AccuracyContract ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs, problem",
    [
        ({"metric": Metric.ULP}, Problem.MISSING_FIGURE),
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
    assert set(Metric) == {Metric.ULP, Metric.TOLERANCE}
    assert AccuracyContract(max_ulp=1).metric is Metric.ULP
    assert TOLERANCE_CONTRACT.metric is Metric.TOLERANCE


# One contract of each shape the table can hold.
#: ``(contract, passed_test_kwargs(), tolerance_kwargs())``, exact: on identical tensors
#: passed_test passes for any bounds, so only these catch `custom_atol` and `custom_rtol`
#: swapped or `near_zero_atol` dropped.
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


@pytest.mark.parametrize(
    "arch, gated",
    [(ChipArchitecture.WORMHOLE, True), (ChipArchitecture.BLACKHOLE, False)],
    ids=lambda v: getattr(v, "name", str(v)),
)
def test_the_binary_gate_enforces_the_whole_contract(arch, gated, monkeypatch):
    """`assert_against_contract` is the binary and ternary drivers' gate, and they run
    only on hardware, so this is the one host check that it takes the step budget: a
    one-step drift on SfpuElwEq, exact on Wormhole, fails there and passes the tolerance
    it falls back to elsewhere. It also ranks subnormal outputs flushed: an
    fp16-subnormal golden the pack writes as 0 is no step of SfpuElwsub's error."""
    monkeypatch.setattr(chip, "get_chip_architecture", lambda: arch)
    bf16 = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    golden = torch.ones(_LANES, dtype=torch.bfloat16)
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
    golden = torch.ones(_LANES, dtype=torch.float16)
    golden[0] = float(torch.finfo(torch.float16).smallest_normal) / 2
    flushed = golden.clone()
    flushed[0] = 0.0
    assert_against_contract(
        MathOperation.SfpuElwsub, fp16, DestAccumulation.No, golden, flushed
    )


def test_every_contract_is_accepted_by_passed_test():
    """Callable, not just shaped right: a renamed keyword would otherwise surface only
    on hardware. The flush request reaches only the ULP arm (passed_test refuses it
    without a budget), and `tolerance_kwargs` never gates on ULP: if it did, passed_test
    would accept it and silently switch every enrolled op to a step budget."""
    golden = torch.ones(_LANES, dtype=torch.bfloat16)
    for contract, expected_ulp, expected_tolerance in _TRANSLATIONS:
        by_tolerance = contract.tolerance_kwargs()
        assert by_tolerance == expected_tolerance
        assert passed_test(golden, golden.clone(), DataFormat.Float16_b, **by_tolerance)
        for flush in (None, True, False):
            by_ulp = contract.passed_test_kwargs(flush_subnormals=flush)
            flushes = flush is not None and contract.metric is Metric.ULP
            assert by_ulp == (
                {**expected_ulp, "flush_subnormals": flush} if flushes else expected_ulp
            )
            assert passed_test(golden, golden.clone(), DataFormat.Float16_b, **by_ulp)


# ── BudgetKey and the resolution rule ─────────────────────────────────────────


@pytest.mark.parametrize(
    "field, bogus",
    [("approx_mode", True), ("arch", DataFormat.Float32)],
    ids=str,
)
def test_a_budget_key_dimension_that_is_not_an_enum_member_is_refused(field, bogus):
    """`True` never equals `DestAccumulation.Yes`: such a key would count as set, match
    nothing, and gate nothing. Queries are BudgetKeys too."""
    with refused(Problem.NOT_A_MEMBER):
        BudgetKey(**{field: bogus})


def test_a_bad_query_is_refused_before_the_unenrolled_early_return():
    """`accuracy_contract` builds the key before it looks the op up, so a miswired
    driver for an op with no row is refused rather than handed the default."""
    unenrolled = next(
        op for op in MathOperation if op not in budget._SFPU_ACCURACY_BUDGET
    )
    with refused(Problem.NOT_A_MEMBER):
        accuracy_contract(unenrolled, output_format=DataFormat.Float16_b, arch=True)


@pytest.mark.parametrize(
    "broad, narrow",
    [
        (AccuracyContract(max_ulp=64), AccuracyContract(max_ulp=4)),
        (
            AccuracyContract(metric=Metric.TOLERANCE, atol=0.13, rtol=0.05),
            AccuracyContract(metric=Metric.TOLERANCE, atol=0.001, rtol=0.001),
        ),
    ],
    ids=["ulp", "tolerance"],
)
def test_a_more_specific_key_wins_over_the_default(broad, narrow):
    table = {BudgetKey(): broad, BudgetKey(output_format=DataFormat.Float32): narrow}
    for fmt, expected in ((DataFormat.Float32, narrow), (DataFormat.Float16_b, broad)):
        query = BudgetKey(output_format=fmt, arch=MEASURED_ARCH)
        assert resolve_contract(table, query) == expected, fmt.name


def test_specificity_counts_every_set_dimension():
    """Only an equal-specificity tie raises, so a miscount would silently repoint
    budgets."""
    table = {
        BudgetKey(output_format=DataFormat.Float32): AccuracyContract(max_ulp=4),
        BudgetKey(
            output_format=DataFormat.Float32, dest_acc=DestAccumulation.No
        ): AccuracyContract(max_ulp=8),
    }
    for dest_acc, expected in ((DestAccumulation.No, 8), (DestAccumulation.Yes, 4)):
        query = BudgetKey(output_format=DataFormat.Float32, dest_acc=dest_acc)
        assert resolve_contract(table, query).max_ulp == expected


def test_an_unset_query_dimension_only_matches_a_wildcard():
    """A caller that does not know `dest_acc` must not get a budget measured for one
    setting of it."""
    table = {BudgetKey(dest_acc=DestAccumulation.Yes): AccuracyContract(max_ulp=1)}
    query = BudgetKey(output_format=DataFormat.Float32)
    assert resolve_contract(table, query) is TOLERANCE_CONTRACT


def test_equally_specific_keys_are_an_error_not_a_tie_break():
    table = {
        BudgetKey(approx_mode=ApproximationMode.No): AccuracyContract(max_ulp=4),
        BudgetKey(output_format=DataFormat.Float32): AccuracyContract(max_ulp=64),
    }
    query = BudgetKey(
        output_format=DataFormat.Float32, approx_mode=ApproximationMode.No
    )
    with refused(Problem.AMBIGUOUS):
        resolve_contract(table, query)


# ── accuracy_contract: fallbacks and downgrades ───────────────────────────────


def test_an_unenrolled_op_keeps_todays_gate():
    unenrolled = sorted(set(MathOperation) - set(enrolled_ops()), key=lambda o: o.name)
    assert unenrolled, "every op is enrolled; this test has nothing left to check"
    for op in unenrolled[:5]:
        for fmt in ULP_FORMATS:
            contract = accuracy_contract(op, output_format=fmt, arch=MEASURED_ARCH)
            assert contract is TOLERANCE_CONTRACT, op.name


def test_an_enrolled_op_keeps_todays_gate_on_a_format_the_sfpu_does_not_compute_in():
    """Abs has a step budget, but a block float keeps its block-aware compare."""
    query = dict(op=MathOperation.Abs, arch=MEASURED_ARCH)
    assert (
        accuracy_contract(output_format=DataFormat.Float16_b, **query).metric
        is Metric.ULP
    )
    assert (
        accuracy_contract(output_format=DataFormat.Bfp8_b, **query)
        is TOLERANCE_CONTRACT
    )


def test_an_enrolled_op_keeps_todays_gate_when_the_input_is_not_a_compute_format():
    """The input-axis twin: Abs's rows wildcard `input`, so only `steps_bind` reading
    the input format keeps a block-float input off the step budget."""
    query = dict(
        op=MathOperation.Abs, output_format=DataFormat.Float16_b, arch=MEASURED_ARCH
    )
    assert (
        accuracy_contract(input_format=DataFormat.Float16_b, **query).metric
        is Metric.ULP
    )
    assert (
        accuracy_contract(input_format=DataFormat.Bfp8_b, **query) is TOLERANCE_CONTRACT
    )


def test_a_step_budget_does_not_bind_on_an_unswept_architecture():
    """Unkeyed rows are Wormhole measurements with no headroom for another kernel."""
    query = dict(output_format=DataFormat.Float32)
    on_wh = accuracy_contract(MathOperation.Abs, arch=MEASURED_ARCH, **query)
    assert on_wh.metric is Metric.ULP and on_wh.max_ulp == 0
    for arch in UNSWEPT_ARCHS:
        elsewhere = accuracy_contract(MathOperation.Abs, arch=arch, **query)
        assert elsewhere is TOLERANCE_CONTRACT, arch


def test_a_declared_tolerance_survives_an_unswept_architecture():
    """The arch downgrade applies to step budgets only."""
    contract = accuracy_contract(
        MathOperation.SigmoidAppx,
        output_format=DataFormat.Float16_b,
        approx_mode=ApproximationMode.Yes,
        dest_acc=DestAccumulation.No,
        arch=ChipArchitecture.BLACKHOLE,
    )
    assert contract.metric is Metric.TOLERANCE
    assert contract.atol == 0.13 and contract.rtol == 0.05


@pytest.mark.parametrize(
    "arch, output_format",
    [
        (MEASURED_ARCH, DataFormat.Bfp4_b),
        (ChipArchitecture.BLACKHOLE, DataFormat.Float16_b),
    ],
    ids=["block-float", "unswept-arch"],
)
def test_a_downgrade_lands_on_the_ops_own_tolerance_row(
    arch, output_format, monkeypatch
):
    """A downgraded ULP row falls back to the op's own tolerance row, not the global
    default."""
    monkeypatch.setitem(
        _SFPU_ACCURACY_BUDGET,
        MathOperation.Abs,
        {
            BudgetKey(): AccuracyContract(
                metric=Metric.TOLERANCE, atol=0.13, rtol=0.05
            ),
            BudgetKey(output_format=DataFormat.Float16_b): AccuracyContract(max_ulp=3),
        },
    )
    gated = accuracy_contract(
        MathOperation.Abs, output_format=DataFormat.Float16_b, arch=MEASURED_ARCH
    )
    assert gated.metric is Metric.ULP and gated.max_ulp == 3
    contract = accuracy_contract(
        MathOperation.Abs, output_format=output_format, arch=arch
    )
    assert contract.metric is Metric.TOLERANCE
    assert contract.atol == 0.13 and contract.rtol == 0.05


def test_a_bare_tolerance_row_opts_out_of_ulp_without_retracting_a_declared_atol(
    monkeypatch,
):
    """The sweep writes a numberless `metric: tolerance` row for every cell past the
    ceiling; it beats a shared `atol` row on specificity but must not replace it with
    the per-format default."""
    op = MathOperation.Abs
    monkeypatch.setitem(
        _SFPU_ACCURACY_BUDGET,
        op,
        {
            BudgetKey(): AccuracyContract(
                metric=Metric.TOLERANCE, atol=0.13, rtol=0.05
            ),
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
    numbered = accuracy_contract(
        op, output_format=DataFormat.Float32, arch=MEASURED_ARCH
    )
    assert numbered.atol == 0.5
    monkeypatch.setitem(
        _SFPU_ACCURACY_BUDGET,
        op,
        {BudgetKey(output_format=DataFormat.Float16_b): TOLERANCE_CONTRACT},
    )
    only_bare = accuracy_contract(
        op, output_format=DataFormat.Float16_b, arch=MEASURED_ARCH
    )
    assert only_bare == TOLERANCE_CONTRACT


def test_a_ulp_row_that_names_its_arch_binds_there(monkeypatch):
    """A row keyed on an arch was measured there, so the arch downgrade exempts it."""
    arch = ChipArchitecture.BLACKHOLE
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


def test_a_query_left_over_from_another_test_is_replaced_not_flagged(monkeypatch):
    """`--ulp-measure` files a reading under the last variant looked up and refuses
    two lookups racing one comparison, but only within one test: the sweep resolves a
    contract and then skips a tolerance cell, and flagging that dropped every reading
    that followed a skip."""

    def resolve(test_id):
        monkeypatch.setenv("PYTEST_CURRENT_TEST", f"{test_id} (call)")
        accuracy_contract(
            MathOperation.Abs, output_format=DataFormat.Float16_b, arch=MEASURED_ARCH
        )

    monkeypatch.setattr(budget, "LAST_QUERY", None)
    monkeypatch.setattr(budget, "PENDING_AMBIGUOUS", False)
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
    resolve("t_three")
    assert budget.PENDING_AMBIGUOUS


# ── The live table ────────────────────────────────────────────────────────────


def test_the_registry_resolves_unambiguously_for_every_variant():
    validate_registry()


# Enrolled ops with no step budget anywhere: the LUT pair and two binaries hold a
# declared tolerance; the rest are past their output's usable ceiling on every cell,
# tracked by Digamma #51128, Softplus #51866 (under #52178), Lgamma #55356, Expm1Cw
# #54770 and Polygamma #52278.
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
    with_budget = {op for op, _, _, _ in _live_step_budgets()}
    missing = set(enrolled_ops()) - with_budget
    assert missing == ONLY_EVER_TOLERANCE, sorted(op.name for op in missing)


# Ops exact by construction: a sign-bit change, a copy, or an integer-valued result.
EXACT_BY_CONSTRUCTION = (
    MathOperation.Abs,
    MathOperation.Neg,
    MathOperation.Identity,
    MathOperation.Floor,
    MathOperation.Ceil,
    MathOperation.Trunc,
    MathOperation.Round,
)

# The subset writing an integer, which survives any pack that can represent it.
INTEGER_VALUED = (
    MathOperation.Floor,
    MathOperation.Ceil,
    MathOperation.Trunc,
    MathOperation.Round,
)

# Ops whose correct result is the *only* result: an integer, a predicate's 1.0/0.0, a
# sign's -1/0/1, a pass-through-or-zero selection, an operand selection, a constant, a
# clamp, or a single IEEE add. Listed by what the op computes, not read back from the
# table. ReluMax, ReluMin and Frac stay out: each carries a `measured: 1` wildcard row
# that would have to be split by input first.
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

# The subset whose every result is exact in every format (a predicate's 1.0/0.0 or a
# constant), so even a narrowing cell has nothing for the pack to round.
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

# What the output pack may cost an exact op on a cell that converts: one step. The
# figure is the measurement itself, so one is also the most such a cell may read.
_PACK_PATH_STEPS = 1


def _mantissa_bits(fmt):
    """0 for a format without a per-element ULP (these appear on the input axis only)."""
    return mantissa_bits(ulp_dtype(fmt)) if has_ulp_gate(fmt) else 0


def _exact_allowance(op, input_format, output_format):
    """``(steps, reason)``: the slack an exact op may carry on one cell, the cost of the
    output pack and only where there is one."""
    if op in EXACT_IN_EVERY_FORMAT:
        return 0, "a 1.0/0.0 or a constant is exact in every format, so no pack rounds"
    if op not in EXACT_ZERO_BY_CONSTRUCTION:
        return _PACK_PATH_STEPS, "the value passes through an fp32 Dest and is packed"
    if input_format is None:
        return (
            _PACK_PATH_STEPS,
            "the row wildcards `input`, so it covers a narrowing cell",
        )
    if _mantissa_bits(output_format) < _mantissa_bits(input_format):
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
    """The canaries: a budget past the pack path means the number was fitted to a
    failure. Each exactly rounded op must also *have* a step budget somewhere, or a
    dropped row would read as a passing op."""
    seen = False
    for budget_op, in_fmt, fmt, contract in _live_step_budgets():
        if budget_op is not op:
            continue
        seen = True
        allowance, why = _exact_allowance(op, in_fmt, fmt)
        assert contract.max_ulp <= allowance, (
            f"{op.name} on {in_fmt and in_fmt.name}->{fmt.name} carries "
            f"max_ulp={contract.max_ulp}, past the {allowance} it may have because {why}"
        )
    if op in EXACT_ZERO_BY_CONSTRUCTION:
        assert seen, f"{op.name} resolves to no ULP contract at all"


def _driver_variants(test_function):
    """Every parameter combination pytest collects for *test_function*, as dicts: the
    cross product of its ``parametrize`` marks, read from the function itself."""
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
    ``gate_on_step_budget=True``, read from the source so a list here cannot outlive
    the keyword."""
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


# The unary step-budget drivers that run one op rather than a `mathop` axis.
_SINGLE_OP_DRIVERS = {"test_eltwise_unary_sfpu_signbit": MathOperation.Signbit}


def test_the_unary_step_budget_drivers_are_the_measured_sweeps():
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
    runs an exact op in and gates on the whole contract: every binary driver, with the
    Dest promotion applied first and ``_APPROX_MODE``, and the unary drivers passing
    ``gate_on_step_budget``."""
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
    asked rather than re-typed."""
    if driver is unary.test_eltwise_unary_sfpu_isinf_isnan:
        return unary.isinf_isnan_skip_reason(formats, op, dest_acc)
    if "skip_if_32bit_input_without_dest_acc(" in inspect.getsource(driver):
        return skip_reason_32bit_input_without_dest_acc(formats, dest_acc)
    return None


def _step_gateable(*formats) -> bool:
    """A variant a step budget can gate: every format named one the SFPU computes in."""
    return all(f is None or f in ULP_FORMATS for f in formats)


def test_every_driven_variant_of_an_exact_op_is_gated():
    """Every variant a hand-built driver runs an exact op in, on a gateable output,
    resolves to a step budget at the Dest it asks for and at the one it is promoted to.
    A variant the driver skips never runs, so it must resolve to *no* step budget: one
    there claims a measurement nobody took."""
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
                f"{formats.input_format.name}->{out_fmt.name} dest={d.name}"
            )
            if _exact_driver_skip(driver, op, formats, d):
                if contract.metric is Metric.ULP:
                    claimed.append(cell)
            elif contract.metric is not Metric.ULP:
                ungated.append(cell)
    assert not ungated, "exact-op variants with no step budget: " + ", ".join(
        sorted(set(ungated))
    )
    assert not claimed, "skipped variants with a step budget: " + ", ".join(
        sorted(set(claimed))
    )


def _swept_exact_ops():
    """The exact ops the exhaustive sweep drives: every sign-bit/copy/integer op, and
    the unary members of EXACT_ZERO_BY_CONSTRUCTION it has a domain for."""
    swept = swept_unary_ops()
    return sorted(
        set(EXACT_BY_CONSTRUCTION)
        | {op for op in EXACT_ZERO_BY_CONSTRUCTION if op in swept},
        key=lambda op: op.name,
    )


def test_every_exact_op_is_driven_by_a_gate():
    """An exact op neither swept nor run by a step-budget driver, on a gateable and
    unskipped variant, would be checked by nothing but the existence of a row."""
    driven = {
        op
        for driver, op, formats, _, dest in _exact_op_driver_variants()
        if _step_gateable(formats.input_format, formats.output_format)
        and not _exact_driver_skip(driver, op, formats, dest)
    }
    unreached = set(EXACT_ZERO_BY_CONSTRUCTION) - driven - set(_swept_exact_ops())
    assert not unreached, sorted(op.name for op in unreached)


def test_every_swept_cell_of_an_enrolled_op_resolves_to_a_row_of_its_own():
    """The sweep gates a ULP row and the headroom report judges a tolerance row; a
    swept cell resolving to `TOLERANCE_CONTRACT` is measured and judged by nothing. A
    swept cell of an exact op must resolve to a step budget outright: the emitter
    demoting one is a real deviation on an op that should be exact."""
    exact = set(_swept_exact_ops())
    unresolved, demoted = [], []
    for op in enrolled_ops():
        if op not in sfpu_unary_ops():
            continue
        for in_fmt, out_fmt, approx, dest in sweep_cells(MEASURED_ARCH):
            contract = accuracy_contract(
                op,
                output_format=out_fmt,
                input_format=in_fmt,
                approx_mode=approx,
                dest_acc=dest,
                arch=MEASURED_ARCH,
            )
            cell = f"{op.name} {in_fmt.name}->{out_fmt.name} {approx.name} {dest.name}"
            if contract is TOLERANCE_CONTRACT:
                unresolved.append(cell)
            elif op in exact and contract.metric is not Metric.ULP:
                demoted.append(cell)
    assert not unresolved, "swept cells no row of the op covers: " + ", ".join(
        unresolved
    )
    assert not demoted, "exact-op cells off the step budget: " + ", ".join(demoted)


def test_no_budget_exceeds_its_formats_usable_ceiling():
    """Past `usable_budget_ceiling` a step budget is looser than the tolerance it
    replaces, and it is the whole gate."""
    for op, _, fmt, contract in _live_step_budgets():
        assert contract.max_ulp <= usable_budget_ceiling(
            fmt
        ), f"{op.name} on {fmt.name}: max_ulp={contract.max_ulp}"


def test_no_integer_only_op_is_enrolled():
    """A step budget on an integer op is meaningless. The listed ops pin the
    derivation; each was missed by an earlier name-based one."""
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


# Ops measured on the hand-built sweep that drives them, which gates on the whole
# contract (`gate_on_step_budget`). For the eight predicates without a registered
# domain these sweeps are the only gate.
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
    hand_built = {
        "signbit": {MathOperation.Signbit},
        "isinf_isnan": set(unary.ISINF_ISNAN_MATHOPS),
        "threshold": set(unary._THRESHOLD_OPS),
    }
    assert all(hand_built.values()), "a sweep set went empty"
    enrolled = set(enrolled_ops())
    for sweep, ops in sorted(hand_built.items()):
        unrecorded = sorted(
            op.name for op in (enrolled & ops) - MEASURED_ON_SWEEP[sweep]
        )
        assert not unrecorded, (
            f"{', '.join(unrecorded)} carries a budget but is driven by the {sweep} "
            "sweep, whose hand-built stimulus no recorded measurement covers"
        )


def test_a_demoted_row_measured_past_the_ceiling():
    """A tolerance row the emitter wrote (it pins `in` and `out`) with a measurement
    inside the ceiling should have been enrolled.

    Empty today: the emitter writes a demotion as a `domains` list with no `metric` or
    `measured`, and the only tolerance rows with a `measured` are the `nonfinite`
    ones, which are not measurable. A tripwire for a hand-edited row."""
    wrong = [
        (op, row)
        for op, rows in load_table().items()
        for row in rows
        if row.metric is Metric.TOLERANCE
        and row.measured is not None
        and not row.unmeasurable
        and row.input is not None
        and row.output is not None
        and row.measured <= usable_budget_ceiling(row.output)
    ]
    assert not wrong, wrong


# ── A gated cell is not quietly parked ────────────────────────────────────────

# Swept cells on a gateable output that the table holds as `not measurable`, keyed
# `(op, in, out, approx, dest)` with `None` for an axis the cause does not depend on,
# each with that cause. A cell whose disagreeing lanes are a tracked defect on a
# handful of inputs belongs in `ulp_sweep._KNOWN_NONFINITE_LANES` instead.
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
    # Each names the issue that tracks it, or says why it is no defect, so retiring the
    # entry is tied to something: the stale check below fails once no row needs it.
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


# How many not-measurable cells each *wildcard* acknowledgement above covers today, so
# a newly demoted cell is a change someone has to read rather than one an op-wide
# acknowledgement approves for any cause. Exact-cell keys need no pin.
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


#: Rows whose every claimed lane disagreed with the golden about being finite: there
#: is no rest of the cell to measure, so `nonfinite_verdict` records no `measured` (a 0
#: would claim it bit-exact) and a re-emit writes the same row. The table cannot tell
#: such a row from one that lost its figure, so each is listed by hand after reading
#: the emit. Empty today.
_EVERY_CLAIMED_LANE_NONFINITE: frozenset = frozenset()


def test_a_row_parked_by_disagreeing_lanes_records_the_rest_of_its_cell():
    """`nonfinite_verdict` keeps the step count over the measurable lanes beside the
    lanes that park a cell, so the headroom report has a figure to hold it to; a row
    without one was written before the emitter kept it, so its op needs a re-emit. A
    cell with no measurable lane left is listed in `_EVERY_CLAIMED_LANE_NONFINITE`."""
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


def _key_name(key):
    op, i, o, a, d = key
    return f"{op.name} {i and i.name}->{o and o.name} approx={a and a.name} dest={d and d.name}"


def test_a_not_measurable_verdict_on_a_gateable_cell_is_acknowledged():
    cells = _not_measurable_cells()
    unacknowledged = [
        cell
        for cell in cells
        if not any(_acknowledges(key, cell) for key in _UNMEASURABLE_CELLS_ACKNOWLEDGED)
    ]
    assert (
        not unacknowledged
    ), "not-measurable cells with no acknowledged cause: " + ", ".join(
        _key_name(cell) for cell in unacknowledged
    )
    stale = [
        key
        for key in _UNMEASURABLE_CELLS_ACKNOWLEDGED
        if not any(_acknowledges(key, cell) for cell in cells)
    ]
    assert not stale, "acknowledgements no row needs any more: " + ", ".join(
        _key_name(key) for key in stale
    )
    wildcard = [
        k for k in _UNMEASURABLE_CELLS_ACKNOWLEDGED if any(v is None for v in k[1:])
    ]
    unpinned = [k for k in wildcard if k not in _UNMEASURABLE_CELLS_ACKNOWLEDGED_COUNTS]
    assert (
        not unpinned
    ), "wildcard acknowledgements with no cell count pinned: " + ", ".join(
        _key_name(k) for k in unpinned
    )
    moved = {
        key: (sum(1 for cell in cells if _acknowledges(key, cell)), pinned)
        for key, pinned in _UNMEASURABLE_CELLS_ACKNOWLEDGED_COUNTS.items()
    }
    moved = {key: counts for key, counts in moved.items() if counts[0] != counts[1]}
    assert (
        not moved
    ), "the cells a wildcard acknowledgement covers changed: " + ", ".join(
        f"{_key_name(k)}: {now} cells, pinned {pinned}"
        for k, (now, pinned) in moved.items()
    )
    orphaned = set(_UNMEASURABLE_CELLS_ACKNOWLEDGED_COUNTS) - set(wildcard)
    assert not orphaned, f"count pins with no acknowledgement: {sorted(orphaned)}"


def test_every_unary_op_is_enrolled_or_excused():
    """ "Never measured" and "deliberately left on tolerance" both read as absent from
    the table, so every unary op either has a row or is listed in
    `_UNARY_OPS_NOT_SWEPT`, whose comment gives the reason."""
    unaccounted = sorted(
        set(swept_unary_ops()) - set(enrolled_ops()),
        key=lambda op: op.name,
    )
    assert (
        not unaccounted
    ), "no accuracy contract and not listed in _UNARY_OPS_NOT_SWEPT: " + ", ".join(
        op.name for op in unaccounted
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
    query of its own, or the recorder would drop that test's reading. A Wormhole list
    binds nowhere else, as a Wormhole figure does not."""
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
