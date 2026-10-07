# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the SFPU accuracy budget registry (no kernel, no device).

``helpers/sfpu_accuracy_budget.py`` loads ``sfpu_accuracy_budget.yaml`` and resolves, per
op variant, the gate the device drivers hand to ``passed_test``. A budget resolved from
the wrong row is a silently wrong gate rather than an error, so the file guards:

* the building blocks: ``AccuracyContract``, ``BudgetKey`` and the YAML loader refuse
  anything they cannot represent faithfully;
* the resolution rule, on small purpose-built tables so enrolling an op cannot break it;
* ``accuracy_contract``: the fallbacks and the downgrades that make enrolment incremental;
* the live table: invariants a reader cannot see by looking at the numbers, and the
  measurement every step budget must record.
"""

import functools
import math
import re
import textwrap
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
    _BUDGET_KEY_TYPES,
    _SFPU_ACCURACY_BUDGET,
    _TABLE_PATH,
    DEFAULT,
    MEASURED_ARCH,
    TOLERANCE_CONTRACT,
    AccuracyContract,
    BudgetKey,
    Metric,
    _load_table,
    _winner,
    accuracy_contract,
    assert_against_contract,
    enrolled_ops,
    resolve_contract,
    usable_budget_ceiling,
    validate_registry,
)
from helpers.sfpu_domains import _UNARY_OPS_NOT_SWEPT, sfpu_unary_ops
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from helpers.ulp import (
    _ULP_DTYPES,
    _ULP_PROXY_DTYPES,
    MAX_MEANINGFUL_ULP,
    ULP_FORMATS,
    has_ulp_gate,
    ulp_dtype,
)
from helpers.ulp_sweep import _DATED, _EMITTED_NOTE, _row_fields, _split_key_line
from helpers.utils import passed_test

UNSWEPT_ARCHS = [a for a in ChipArchitecture if a != MEASURED_ARCH]

#: Formats with neither a per-element ULP nor an integer type: Bfp4_b, Bfp2_b, Tf32 and
#: the MX formats (MxInt* included -- ``is_integer()`` does not cover them).
BLOCK_FORMATS_WITHOUT_ULP = sorted(
    (f for f in DataFormat if not has_ulp_gate(f) and not f.is_integer()),
    key=lambda f: f.name,
)

#: Every output format a step budget can gate. Derived, so a new proxy dtype is covered.
ULP_CAPABLE_FORMATS = list(ULP_FORMATS) + list(_ULP_PROXY_DTYPES)

#: The input axis for the live-table sweeps: every gateable format plus every input the
#: table pins. An input only has to be unpackable, so the table may name one (Bfp4_b)
#: that is not in ULP_CAPABLE_FORMATS. ``None`` resolves the rows that wildcard ``in:``.
QUERYABLE_INPUT_FORMATS = sorted(
    {
        key.input_format
        for table in _SFPU_ACCURACY_BUDGET.values()
        for key in table
        if key.input_format is not None
    }
    | set(ULP_CAPABLE_FORMATS),
    key=lambda f: f.name,
) + [None]


def _refuses(match, kind=ValueError):
    """The suite's ``expect_error`` fixture needs a device; these are host-only tests."""
    return pytest.raises(kind, match=match)  # allow-pytest.raises: host-only test


def _table(tmp_path, text):
    """*text* as a budget table on disk, loaded the way the real one is."""
    path = tmp_path / "budget.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return _load_table(path)


def _every_variant(op):
    """Yield ``(input_format, output_format, contract)`` for every variant of *op*.

    Every dimension is swept, ``None`` included where a caller may leave it unset: an
    unset query dimension only matches a wildcard, so a query that omitted
    ``input_format`` would miss every row that pins it -- which is most of the table.
    """
    for fmt in ULP_CAPABLE_FORMATS:
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
    "kwargs, match",
    [
        ({"metric": Metric.ULP}, "needs max_ulp"),
        # Both at once reads as deliberate but is a half-finished conversion.
        ({"max_ulp": 1, "atol": 0.13}, "silently ignored"),
        ({"metric": Metric.TOLERANCE, "max_ulp": 1}, "belong to the ulp metric"),
        ({"max_ulp": -1}, "must not be negative"),
        # YAML reads `true` as a bool, which is an int and would enforce a 1-step budget.
        ({"max_ulp": True}, "must be an int step count"),
        ({"max_ulp": 1.0}, "must be an int step count"),
        ({"max_ulp": "3"}, "must be an int step count"),
        # `atol: yes` applies as 1.0, NaN is ignored by passed_test, inf gates nothing.
        ({"metric": Metric.TOLERANCE, "atol": True}, "must be (a number|finite)"),
        ({"metric": Metric.TOLERANCE, "atol": math.nan}, "must be (a number|finite)"),
        ({"metric": Metric.TOLERANCE, "atol": math.inf}, "must be (a number|finite)"),
        # passed_test ignores a negative override and silently uses the default.
        ({"metric": Metric.TOLERANCE, "atol": -0.001}, "must not be negative"),
        ({"metric": Metric.TOLERANCE, "rtol": -0.001}, "must not be negative"),
        ({"max_ulp": 1, "near_zero_atol": -0.001}, "must not be negative"),
        # `metric="ulp"` is not Metric.ULP; it used to fall through to the tolerance arm.
        ({"metric": "ulp", "max_ulp": 1}, "must be a Metric member"),
        ({"metric": "tolerance", "max_ulp": 1}, "must be a Metric member"),
        ({"metric": "pcc", "max_ulp": 1}, "must be a Metric member"),
        ({"metric": 0, "max_ulp": 1}, "must be a Metric member"),
        ({"metric": None, "max_ulp": 1}, "must be a Metric member"),
    ],
    ids=repr,
)
def test_a_malformed_contract_is_refused(kwargs, match):
    with _refuses(match):
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
        with _refuses("Assert against golden failed", AssertionError):
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
        ("input_format", "Float32"),
        ("output_format", "Float32"),
        ("dest_acc", True),
        ("dest_acc", False),
        ("arch", "wormhole"),
    ],
    ids=str,
)
def test_a_budget_key_dimension_that_is_not_an_enum_member_is_refused(field, bogus):
    """The enums are bare, so ``True`` never equals ``DestAccumulation.Yes``. Such a key
    would count as set, match nothing, and render like the correct one: its budget would
    gate nothing. Queries are BudgetKeys too, so this also covers the caller side."""
    with _refuses(f"BudgetKey.{field} must be a"):
        BudgetKey(**{field: bogus})


def test_every_budget_key_field_is_guarded():
    """``_BUDGET_KEY_TYPES`` is maintained by hand beside the dataclass."""
    assert {f.name for f in fields(BudgetKey)} == set(_BUDGET_KEY_TYPES)
    assert BudgetKey(
        approx_mode=ApproximationMode.No,
        input_format=DataFormat.Float16_b,
        output_format=DataFormat.Float32,
        dest_acc=DestAccumulation.Yes,
        arch=ChipArchitecture.WORMHOLE,
    ).specificity == len(_BUDGET_KEY_TYPES)


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
        assert resolve_contract(table, query, label="op") == expected, fmt.name


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
        assert resolve_contract(table, query, label="op").max_ulp == expected


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
        assert resolve_contract(table, query, label="op").max_ulp == expected, arch


def test_an_unset_query_dimension_only_matches_a_wildcard():
    """A caller that does not know ``dest_acc`` must not get a budget measured for one
    setting of it."""
    table = {BudgetKey(dest_acc=DestAccumulation.Yes): AccuracyContract(max_ulp=1)}
    query = BudgetKey(output_format=DataFormat.Float32)
    assert resolve_contract(table, query, label="op") is TOLERANCE_CONTRACT


def test_equally_specific_keys_are_an_error_not_a_tie_break():
    """The table's iteration order must not decide a budget."""
    table = {
        BudgetKey(approx_mode=ApproximationMode.No): AccuracyContract(max_ulp=4),
        BudgetKey(output_format=DataFormat.Float32): AccuracyContract(max_ulp=64),
    }
    query = BudgetKey(
        output_format=DataFormat.Float32, approx_mode=ApproximationMode.No
    )
    with _refuses("equally specific"):
        resolve_contract(table, query, label="Ambiguous")


def test_an_empty_table_falls_back_to_the_tolerance_metric():
    query = BudgetKey(output_format=DataFormat.Float32)
    assert resolve_contract({}, query, label="op") is TOLERANCE_CONTRACT


# ── accuracy_contract: fallbacks and downgrades ───────────────────────────────


def test_an_unenrolled_op_keeps_todays_gate():
    """Picked from whatever is still unenrolled, so enrolling more ops cannot break it."""
    unenrolled = sorted(set(MathOperation) - set(enrolled_ops()), key=lambda o: o.name)
    assert unenrolled, "every op is enrolled; this test has nothing left to check"
    for op in unenrolled[:5]:
        for fmt in ULP_FORMATS:
            contract = accuracy_contract(op, output_format=fmt, arch=MEASURED_ARCH)
            assert contract is TOLERANCE_CONTRACT, op.name


@pytest.mark.parametrize("fmt", BLOCK_FORMATS_WITHOUT_ULP, ids=lambda f: f.name)
def test_an_enrolled_op_keeps_todays_gate_on_a_format_without_a_per_element_ulp(fmt):
    """Abs has a step budget, but these formats keep their block-aware compares -- by
    falling back, not by raising."""
    assert not has_ulp_gate(fmt)
    # The same query on a gated format is a step budget, so the fall-through below is
    # the downgrade under test and not a query that matched nothing.
    query = dict(op=MathOperation.Abs, arch=MEASURED_ARCH)
    assert (
        accuracy_contract(output_format=DataFormat.Float16_b, **query).metric
        is Metric.ULP
    )
    assert accuracy_contract(output_format=fmt, **query) is TOLERANCE_CONTRACT


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
    else:  # Bfp4_b has no per-element ULP; off Wormhole the unkeyed row does not bind
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
    with _refuses("BudgetKey.arch must be a"):
        accuracy_contract(op, output_format=DataFormat.Float32, arch="wormhole")


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
    assert budget.LAST_QUERY[0] == "t_two"

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

    rows = read()
    assert len(rows) == 1, rows
    assert rows[0]["op"] == "Abs" and rows[0]["in"] == "Float16"
    assert rows[0]["out"] == "Float16_b" and rows[0]["dest"] == "Yes"
    assert rows[0]["approx"] is None and rows[0]["max"] == 0
    # The architecture whose contract it is, spelled as the table's `arch:` rows are.
    assert rows[0]["arch"] == MEASURED_ARCH.name
    # A max of 0 means something only over lanes that were measured.
    assert (rows[0]["lanes"], rows[0]["unmeasurable"]) == (32, 0)


def _measure_rows(tmp_path, monkeypatch):
    import json

    import helpers.utils as utils

    path = tmp_path / "measure.jsonl"
    monkeypatch.setattr(utils, "_ULP_MEASURE_PATH", str(path))

    def rows():
        lines = path.read_text().splitlines() if path.exists() else []
        return [json.loads(line) for line in lines]

    return rows


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
    assert row["max"] == 1, row


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
    assert (row["arch"], row["dest"]) == ("QUASAR", "No")


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

    code = textwrap.dedent(
        """
        import sys, torch
        from helpers.format_config import DataFormat
        from helpers.utils import passed_test
        golden = torch.ones(32, dtype=torch.bfloat16)
        assert passed_test(golden, golden.clone(), DataFormat.Float16_b)
        print("helpers.sfpu_accuracy_budget" in sys.modules)
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_TABLE_PATH.parent.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.strip().splitlines()[-1] == "False", done.stdout


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
    assert row["max"] == 0, row


def test_an_unpreparable_measure_path_warns_and_records_nothing(tmp_path, monkeypatch):
    """The session-start truncation is under the same rule as every write: a path that
    is a directory warns and switches the recorder off rather than aborting pytest
    before anything is compared."""
    import helpers.utils as utils

    monkeypatch.setattr(utils, "_ULP_MEASURE_PATH", str(tmp_path))  # a directory
    monkeypatch.setattr(utils, "_ULP_MEASURE_WARNED", False)
    utils.prepare_ulp_measure_file()
    assert utils._ULP_MEASURE_PATH is None and utils._ULP_MEASURE_WARNED

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
    with _refuses("arch", TypeError):
        accuracy_contract(MathOperation.Abs, output_format=DataFormat.Float32)


# ── The YAML loader ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, match",
    [
        # YAML keeps only the last of two equal keys, so a whole op would vanish.
        (
            "Abs:\n  - {max_ulp: 0}\nNeg:\n  - {max_ulp: 1}\nAbs:\n  - {max_ulp: 99}\n",
            "duplicate entry for 'Abs'",
        ),
        # Same last-wins rule inside a row, where it would swap the key.
        (
            "Abs:\n  - {out: Float16_b, out: Float32, max_ulp: 1}\n",
            "duplicate entry for 'out'",
        ),
        # Two list items with one key: nothing collapses them, the later would win.
        (
            "Abs:\n  - {out: Float16_b, max_ulp: 1}\n  - {out: Float16_b, max_ulp: 4}\n",
            "repeats BudgetKey",
        ),
        ("Nope:\n  - {max_ulp: 1}\n", "'Nope' is not a MathOperation"),
        # Each failure names the op it came from.
        ("Abs:\n  - {max_ulp: 1, budget: 2}\n", "Abs: unknown field"),
        (
            "Abs:\n  - {out: Float17, max_ulp: 1}\n",
            "Abs: 'Float17' is not a DataFormat",
        ),
        ("Abs:\n  - {metric: pcc, max_ulp: 1}\n", "Abs: 'pcc' is not a Metric"),
        ("Abs:\n", "Abs has no rows"),
        # The contract invariants still come from AccuracyContract.
        (
            "Abs:\n  - {max_ulp: 1, atol: 0.5}\n",
            "a ulp contract replaces the tolerance",
        ),
        # True == 1, so a by-value lookup would load `approx: 1` as Yes.
        ("Abs:\n  - {approx: 1, max_ulp: 1}\n", "is not a ApproximationMode"),
        ("Abs:\n  - {approx: 0, max_ulp: 1}\n", "is not a ApproximationMode"),
        ("Abs:\n  - {approx: 1.0, max_ulp: 1}\n", "is not a ApproximationMode"),
    ],
    ids=[
        "repeated-op",
        "repeated-field",
        "repeated-row",
        "unknown-op",
        "unknown-field",
        "unknown-format",
        "unknown-metric",
        "no-rows",
        "bad-contract",
        "approx-1",
        "approx-0",
        "approx-1.0",
    ],
)
def test_the_loader_refuses_what_it_cannot_represent(tmp_path, text, match):
    with _refuses(match):
        _table(tmp_path, text)


def test_rows_with_different_keys_are_kept(tmp_path):
    table = _table(
        tmp_path, "Abs:\n  - {max_ulp: 4}\n  - {out: Float16_b, max_ulp: 1}\n"
    )
    assert len(table[MathOperation.Abs]) == 2


def test_anchors_and_merge_keys_load_and_may_override(tmp_path):
    """A ``<<`` merge that overrides a field is not a duplicate."""
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
    """YAML 1.1 reads bare ``No`` as ``False``; both spellings must load the same."""
    quoted = _table(tmp_path, 'Abs:\n  - {approx: "No", dest: "Yes", max_ulp: 1}\n')
    bare = _table(tmp_path, "Abs:\n  - {approx: No, dest: Yes, max_ulp: 1}\n")
    assert quoted == bare
    key = next(iter(quoted[MathOperation.Abs]))
    assert key.approx_mode is ApproximationMode.No
    assert key.dest_acc is DestAccumulation.Yes


# ── The live table ────────────────────────────────────────────────────────────


def test_the_registry_resolves_unambiguously_for_every_variant():
    validate_registry()


def test_enrolled_ops_is_sorted_and_stable():
    ops = enrolled_ops()
    assert list(ops) == sorted(ops, key=lambda op: op.name)
    assert len(set(ops)) == len(ops)


#: Enrolled ops with no step budget anywhere: the 3-segment LUT pair, two binaries
#: whose per-format tolerances moved into the table, five transcendentals whose *best*
#: cell is already past its output's usable ceiling (6 bf16, 51 fp16, 25 Bfp8_b) -- the
#: measurements are on their rows, not repeated here to drift -- and Expm1Cw, which
#: returns -1 where expm1 overflows (x past ~88.7) on every cell, so no cell has a lane
#: count a step budget can describe. Recorded, not fixed; tracked: Erfc #51137, Digamma
#: #51128, Softplus #51866 (input clamps, under #52178) and Lgamma #55356.
#:
#: Sign, Heaviside, GeluTanh, Tanhshrink, Xielu, I1 and SfpuElwmul are not here: per
#: variant, some of their cells are inside the ceiling, and the rest fall through to
#: tolerance.
ONLY_EVER_TOLERANCE = frozenset(
    {
        MathOperation.SigmoidAppx,
        MathOperation.GeluAppx,
        MathOperation.SfpuElwpow,
        MathOperation.SfpuXlogy,
        MathOperation.Erfc,
        MathOperation.Polygamma,
        MathOperation.Softplus,
        MathOperation.Lgamma,
        MathOperation.Digamma,
        MathOperation.Expm1Cw,
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
#: it by construction. ReluMax, ReluMin and Frac stay out: each carries a `max_ulp: 1`
#: wildcard row from the 2026-09-18 sample, which would have to be split by input first --
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

#: What the output pack may cost an exact op on a cell that converts: a measured 1 step,
#: written as 2 by the emitter's headroom.
_PACK_PATH_STEPS = 2


def _mantissa_bits(fmt):
    """0 for a format without a per-element ULP (these appear on the input axis only)."""
    return _ULP_DTYPES[ulp_dtype(fmt)].mantissa_bits if has_ulp_gate(fmt) else 0


def _exact_allowance(op, input_format, output_format):
    """``(steps, reason)``: the slack an exact op may carry on one cell -- the cost of
    the output pack, and only where there is one."""
    if output_format in _ULP_PROXY_DTYPES:
        # Rows measured at 2-3 steps into Bfp8_b are parked on the tolerance metric, so
        # a Bfp8_b ULP row is the 0-step enrolment; otherwise only the 25-step usable
        # ceiling would bound it.
        return 0, "a Bfp8_b ULP row here is the 0-step enrolment or nothing"
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
            "the row wildcards `in:`, so it covers a narrowing cell",
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
    claim, so each must also *have* a step budget somewhere: the provenance guard lets a
    measured 0 be written as 1, and a dropped row would read as a passing op."""
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
    ``_skip_fp32_no_dest_acc``."""
    import inspect

    import test_eltwise_binary_sfpu as binary
    import test_eltwise_unary_sfpu as unary

    if driver is unary.test_eltwise_unary_sfpu_isinf_isnan:
        return unary.isinf_isnan_skip_reason(formats, op, dest_acc)
    if "_skip_fp32_no_dest_acc(" in inspect.getsource(driver):
        return binary.fp32_no_dest_acc_skip_reason(formats, dest_acc)
    return None


def _step_gateable_output(fmt) -> bool:
    """An output a step budget can gate: not an integer, which wants bit equality, nor a
    block float, which keeps its lattice compare. The exact-op guards and the
    not-measurable audit filter on it, so they agree on what "gateable" means."""
    return has_ulp_gate(fmt) and fmt not in _ULP_PROXY_DTYPES


def test_every_driven_variant_of_an_exact_op_is_gated():
    """`test_an_exact_op_never_carries_a_wide_budget` asks only that each exact op keep
    *some* step budget: drop every `SfpuMask` Float16_b row and keep its Float32 one, and
    the Float16_b variants fall back to tolerance while it still passes. This walks every
    variant the hand-built drivers run an exact op in, on a float non-block output, and
    requires each to resolve to a step budget -- at the Dest the variant asks for and at
    the one it is promoted to. A variant the driver skips never runs, so it must resolve
    to *no* step budget: one there claims a measurement nobody took -- which is how a
    re-sort that dropped Isinf/Isnan/Isneginf's `not measured` rows showed up. The swept
    cells of the unary exact ops are held by
    `test_every_swept_cell_of_an_exact_op_is_gated_or_waived`."""
    from helpers.data_format_inference import effective_dest_acc

    ungated, claimed = [], []
    for driver, op, formats, approx, dest in _exact_op_driver_variants():
        out_fmt = formats.output_format
        if not _step_gateable_output(out_fmt):
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
        if _step_gateable_output(formats.output_format)
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
#: strided lanes below 2**-14; and the one-flip Bfp8_b/Bfp4_b cells of Floor and the
#: predicates were the sweep's -0.0 lane, which the block quantizer turns into
#: -2**-127 for the golden (``ulp_sweep._normal_input``). All of them measure 0.
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
    from helpers.ulp_sweep import sweep_cells

    demoted = set()
    # The Wormhole cells, whatever CHIP_ARCH this host sets: the contracts below are
    # resolved at MEASURED_ARCH, and Quasar promotes nothing.
    for in_fmt, out_fmt, approx, dest in sweep_cells(MEASURED_ARCH):
        if out_fmt in _ULP_PROXY_DTYPES:
            continue  # a block output is never enrolled from this sweep
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


def test_a_block_float_output_is_never_enrolled_only_incidentally_covered():
    """In a shared-exponent block a small element next to a large one quantizes to zero,
    so even Abs reaches 15616 bf16 steps on a Bfp8_b output, and the exhaustive sweep
    never enrols a block-float cell: it walks a format in value order, so adjacent
    values share a block -- the best case, not a representative one. So no row may pin
    a step budget to a block-float output.

    Floor, Ceil and Trunc still *resolve* to a budget there, through their op-wide
    ``{max_ulp: 0}`` row, wherever no more specific row shadows it. That is incidental
    cover and not a gate: the sweep skips block-float outputs and the unary functional
    driver takes only the tolerance arm. It is allowed only in that shape -- the default
    row of an integer-valued op -- so a deliberate Bfp8_b enrolment fails here."""
    for op, table in _SFPU_ACCURACY_BUDGET.items():
        for key, contract in table.items():
            assert not (
                contract.metric is Metric.ULP and key.output_format in _ULP_PROXY_DTYPES
            ), f"{op.name} pins a step budget to {key.output_format.name}: {key.describe()}"
        for input_format, approx_mode, dest_acc in product(
            [f for f in QUERYABLE_INPUT_FORMATS if f is not None],
            [*ApproximationMode, None],
            DestAccumulation,
        ):
            query = BudgetKey(
                approx_mode=approx_mode,
                input_format=input_format,
                output_format=DataFormat.Bfp8_b,
                dest_acc=dest_acc,
                arch=MEASURED_ARCH,
            )
            found = _winner(table, query, op.name)
            if found is None or found[1].metric is not Metric.ULP:
                continue
            key = found[0]
            assert key == DEFAULT and op in INTEGER_VALUED, (
                f"{op.name} {input_format.name}->Bfp8_b approx={approx_mode} "
                f"dest={dest_acc.name} is step-gated by {key.describe()}; a block "
                "float's lattice compare is the stronger criterion"
            )


def test_no_step_budget_is_keyed_on_a_format_without_a_per_element_ulp():
    """accuracy_contract downgrades such a row, so its number gates nothing.
    validate_registry cannot see this: it only checks the resolution is unambiguous."""
    for op, table in _SFPU_ACCURACY_BUDGET.items():
        for key, contract in table.items():
            if contract.metric is Metric.ULP and key.output_format is not None:
                assert has_ulp_gate(key.output_format), (
                    f"{op.name} carries a step budget keyed on "
                    f"{key.output_format.name}, which has no per-element ULP, so the "
                    "number gates nothing."
                )


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
            ULP_CAPABLE_FORMATS,
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
            if contract == TOLERANCE_CONTRACT:
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
#: Wormhole, 2026-09-18; the counts are in the YAML row comments. Those sweeps gate on
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


# ── Every step budget records the measurement it came from ────────────────────
#
# The measurement lives in the YAML comment beside the row (or on the op's header line),
# so it is read from the text: YAML discards comments. Widening a budget without
# re-measuring then has to falsify that comment.

#: How far a *sampled* row's budget may sit above its measurement. Its comment records
#: a sample, and the hand-set headroom over the tail the sample did not see varies; an
#: exhaustive row saw every value and carries exactly the emitter's budget instead.
MEASUREMENT_HEADROOM = 2

#: ``max 65536 ULP`` in the emitted rows, ``0 ULP`` in the hand-measured ones.
_MEASUREMENT = re.compile(r"(?:max )?(\d+) ULP")

#: What marks a row's comment as the exhaustive sweep's: ``write_table``'s suffix.
_EXHAUSTIVE = "exhaustive"


def _run_of(note, key_run):
    """The run a row's measurement came from, read as the emitter writes it.

    A dated note names its own run (``_DATED``, the emitter's own test). An undated one
    is credited to the ``measured by:`` run on the key line only when it is a note the
    emitter writes (``_EMITTED_NOTE``), the rule ``_stamp_kept`` applies before a
    re-emit replaces that clause. Any other undated note, or none, names no run: a
    hand-written ``max 0 ULP over 2048 pts`` under an exhaustive key line is a sample,
    and is held to the sampled rules."""
    note = note.strip()
    if _DATED.search(note):
        return note
    return key_run if _EMITTED_NOTE.fullmatch(note) else ""


def _measured_budget_rows(path=_TABLE_PATH):
    """``(op_name, row_text, max_ulp, measured_or_None, exhaustive)`` for every
    ``max_ulp`` row. *exhaustive* is whether :func:`_run_of` the row is the sweep.

    Deciding that by "the row has a number" instead read every emitted row -- which has
    a number and no label -- as a sampled one, and the exhaustive audit below then
    covered only the handful of rows that carry their own exhaustive label. Acosh's
    ``max_ulp: 7  # max 6 ULP`` raised to 12 with its comment untouched passed."""
    rows, op, op_measured, op_run = [], None, None, ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):  # `OpName:`, optionally with a header comment
            head, header, op_run = _split_key_line(line)
            op = head.split(":")[0].strip()
            found = _MEASUREMENT.search(header)
            op_measured = int(found.group(1)) if found else None
            continue
        body, _, comment = line.strip().partition("#")
        declared = re.search(r"max_ulp:\s*(\d+)", body)
        if not declared:
            continue  # a tolerance row has no step budget to back
        found = _MEASUREMENT.search(comment)
        measured = int(found.group(1)) if found else op_measured
        exhaustive = _EXHAUSTIVE in _run_of(comment, op_run)
        rows.append((op, body.strip(), int(declared.group(1)), measured, exhaustive))
    return rows


def test_the_provenance_parser_sees_every_budget_the_registry_enforces():
    """A row the regex misses (``max_ulp : 5``, ``+5``, ``0x10``) would silently escape
    the two audits below, so tie the parse back to the loaded table."""
    parsed = sorted((op, budget) for op, _, budget, _, _ in _measured_budget_rows())
    loaded = sorted(
        (op.name, contract.max_ulp)
        for op, table in _SFPU_ACCURACY_BUDGET.items()
        for contract in table.values()
        if contract.metric is Metric.ULP
    )
    assert parsed == loaded


def test_every_step_budget_names_the_measurement_it_came_from():
    rows = _measured_budget_rows()
    assert rows, "no max_ulp rows found -- the parser has drifted from the table"
    unbacked = [(op, body) for op, body, _, measured, _ in rows if measured is None]
    assert not unbacked, "budgets with no recorded measurement:\n" + "\n".join(
        f"  {op}: {body}" for op, body in unbacked
    )


#: Exact by construction without being canaries: a clamp returns one of its operands,
#: and x - trunc(x) is exact, so a sampled 0 on these states what the op guarantees
#: rather than what the sample happened to miss -- where the output can hold that
#: operand (:func:`_rounds_at_pack`). SfpuBinaryMax/Min are selections too, and already
#: canaries in ``EXACT_ZERO_BY_CONSTRUCTION``.
EXACT_SELECTIONS = (
    MathOperation.ReluMax,
    MathOperation.ReluMin,
    MathOperation.UnaryMax,
    MathOperation.UnaryMin,
    MathOperation.Frac,
)


def test_the_emitter_and_the_guards_agree_on_which_ops_are_exact():
    """The emitter keeps a strided 0 only on EXACT_BY_CONSTRUCTION_OPS, and the guards
    here judge by their own finer lists; an op in one and not the other would be
    floored by one and held to 0 by the other."""
    from helpers.sfpu_accuracy_budget import EXACT_BY_CONSTRUCTION_OPS

    listed = {*EXACT_BY_CONSTRUCTION, *EXACT_ZERO_BY_CONSTRUCTION, *EXACT_SELECTIONS}
    assert listed == EXACT_BY_CONSTRUCTION_OPS, sorted(
        op.name for op in listed ^ EXACT_BY_CONSTRUCTION_OPS
    )


def _sampled_zero_budgets(path=_TABLE_PATH):
    """``(op_name, row_text)`` for every ``max_ulp: 0`` row whose measurement was a
    sample: :func:`_run_of` it is not the exhaustive sweep."""
    found, op, key_run = [], None, ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            head, _, key_run = _split_key_line(line)
            op = head.split(":")[0].strip()
            continue
        body, _, note = line.strip().partition("#")
        if not re.search(r"max_ulp:\s*0\b", body):
            continue
        if _EXHAUSTIVE not in _run_of(note, key_run):
            found.append((op, body.strip()))
    return found


def _rounds_at_pack(body) -> bool:
    """Whether *body* pins a cell whose output holds fewer mantissa bits than its input
    still has in Dest: there the operand an exact op returns -- a selected value, an
    integer -- is rounded at pack, as Abs, Neg and Identity's 1-step Float32 -> Float16_b
    cells record. Dest keeps the input's precision except where a 32-bit input meets a
    16-bit Dest (``dest: "No"``), which the unpack has narrowed already. A 16-bit input
    keeps it in a 16-bit Dest too: Float16 -> Float16_b at ``dest: "No"`` reads 1 step on
    Abs and ReluMin. A row that leaves `in` or `out` unpinned is an op-wide canary; the
    narrowing cells it would cover are pinned by rows of their own.
    """
    fields = _row_fields(body)
    if "in" not in fields or "out" not in fields:
        return False
    in_fmt = DataFormat[fields["in"]]
    if fields.get("dest") == "No" and in_fmt.is_32_bit():
        return False
    return _mantissa_bits(DataFormat[fields["out"]]) < _mantissa_bits(in_fmt)


def test_only_a_32_bit_input_is_narrowed_before_a_16_bit_dest():
    """A Float16 operand stays fp16 in a 16-bit Dest and is rounded to bf16 at pack, so
    a sampled 0 there cannot be an exact op's guarantee; a Float32 one was narrowed by
    the unpack, leaving the pack nothing to round."""
    row = "- {{in: {}, out: Float16_b, dest: {}, max_ulp: 0}}"
    assert _rounds_at_pack(row.format("Float16", '"No"'))
    assert _rounds_at_pack(row.format("Float32", '"Yes"'))
    assert _rounds_at_pack(row.format("Float32", "Any").replace(", dest: Any", ""))
    assert not _rounds_at_pack(row.format("Float32", '"No"'))
    assert not _rounds_at_pack(row.format("Float16_b", '"No"'))


def test_a_sampled_zero_on_an_inexact_op_is_floored_to_one():
    """A finite sample cannot assert exactness, so a sampled 0 is written as 1 unless
    the op is exact by construction on that cell; only the exhaustive sweep, which saw
    every value, may keep a 0 on an op that rounds. A sampled 0 gated bit-exact fails on
    any golden, domain or rounding change -- the Elwdiv note records that happening.

    Exact by construction is per cell: an op returning an operand, or an integer, is
    exact only where the output can hold it. Only a 1.0/0.0 or a constant
    (``EXACT_IN_EVERY_FORMAT``) is exact on every cell."""
    exact = {
        *EXACT_BY_CONSTRUCTION,
        *EXACT_ZERO_BY_CONSTRUCTION,
        *EXACT_SELECTIONS,
    }
    unfloored = [
        f"{op}: {body}"
        for op, body in _sampled_zero_budgets()
        if MathOperation[op] not in EXACT_IN_EVERY_FORMAT
        and (MathOperation[op] not in exact or _rounds_at_pack(body))
    ]
    assert not unfloored, "\n".join(unfloored)


def _budgets_past_their_measurement(path=_TABLE_PATH):
    """Every ``max_ulp`` row in *path* whose budget is not the one its measurement
    allows, as messages. An exhaustive row carries exactly the budget the emitter
    derives from its measurement -- ``_verdict``'s rule, 0 for 0 and otherwise
    ``EMIT_HEADROOM`` rounded up -- so a budget widened by hand has to falsify the
    comment beside it, which is the table header's rule for raising one. A sampled row
    may sit anywhere in ``[measured, MEASUREMENT_HEADROOM * measured]``, and at 1 over
    a measured 0: a finite sample cannot assert exactness."""
    from helpers.ulp_sweep import _is_exact, _row_fields, _verdict

    problems = []
    for op, body, budget, measured, exhaustive in _measured_budget_rows(path):
        if measured is None:
            continue  # owned by test_every_step_budget_names_the_measurement_it_came_from
        where = f"{op}: {body} (records {measured} ULP)"
        if exhaustive:
            # The emitter's own call: the input format decides whether a 0 was seen on
            # every value (kept) or on a stride of Float32 (written as 1, unless the op
            # is exact by construction).
            fields = _row_fields(body)
            verdict = _verdict(measured, fields["out"], fields.get("in"), _is_exact(op))
            if ("ulp", budget) != verdict:
                problems.append(
                    f"{where}: the emitter writes {verdict[1]} for that measurement, "
                    f"not {budget}. Re-measure rather than edit the number."
                )
        elif measured == 0:
            if budget > 1:
                problems.append(f"{where}: a 0-ULP measurement cannot justify {budget}")
        elif budget < measured:
            problems.append(f"{where}: budget {budget} is below it")
        elif budget > MEASUREMENT_HEADROOM * measured:
            problems.append(
                f"{where}: budget {budget} is more than {MEASUREMENT_HEADROOM}x the "
                "measurement"
            )
    return problems


def test_a_demotion_note_names_the_budget_the_emitter_computes():
    """A tolerance row the emitter demoted says ``budget B > ceiling C``, and both
    numbers are checkable: B is ``_verdict``'s budget for the measurement beside it,
    and it does cross C. A block kept verbatim is never re-rendered, so a
    stale figure there -- Gelu's 31406 from the old float ceil, where 28550 x 1.1 is
    exactly 31405 -- survives every re-emit unless something reads it."""
    from helpers.ulp_sweep import _is_exact, _verdict

    note = re.compile(r"max (\d+) ULP, budget (\d+) > ceiling (\d+)")
    wrong, op = [], None
    for line in _TABLE_PATH.read_text(encoding="utf-8").splitlines():
        if line and not line[0].isspace() and not line.startswith("#"):
            op = line.split(":")[0].strip()
        found = note.search(line)
        if not found:
            continue
        measured, budget, ceiling = (int(g) for g in found.groups())
        fields = _row_fields(line)
        expected = _verdict(measured, fields["out"], fields.get("in"), _is_exact(op))
        if expected != ("tolerance", budget) or budget <= ceiling:
            wrong.append(f"{op}: {line.split('#', 1)[0].strip()} -> {expected}")
    assert not wrong, "\n".join(wrong)


def test_no_step_budget_exceeds_the_measurement_it_records():
    problems = _budgets_past_their_measurement()
    assert not problems, "\n".join(problems)


#: One op as the emitter writes it: the run label once on the key line, each row with
#: its number alone, and a row from another run naming that run itself.
_LABELLED_OP = """\
Acosh:  # measured by: exhaustive Float16_b/Float16/Bfp8_b sweep, wormhole, 2026-09-30, except where a row says otherwise
  - {{in: Float16, out: Float16, dest: "No", max_ulp: {exhaustive}}}  # max 6 ULP
  - {{in: Float32, out: Float16, dest: "No", max_ulp: {sampled}}}  # max 6 ULP, wormhole, 2026-09-21
"""


def test_an_emitted_row_is_held_to_the_run_on_its_key_line(tmp_path):
    """The raise the audit exists to reject: a row the sweep wrote, its budget edited
    and its comment untouched. The row carries no label of its own, so the run it is
    held to is the key line's, and the key line says exhaustive: only the emitter's
    own number passes. The dated row beside it names a sampled run and keeps the 2x
    envelope, so a raise within it is not this audit's to reject."""
    path = tmp_path / "budget.yaml"
    path.write_text(_LABELLED_OP.format(exhaustive=7, sampled=12), encoding="utf-8")
    assert _budgets_past_their_measurement(path) == []

    path.write_text(_LABELLED_OP.format(exhaustive=12, sampled=12), encoding="utf-8")
    problems = _budgets_past_their_measurement(path)
    assert len(problems) == 1 and "the emitter writes 7" in problems[0], problems
    assert _measured_budget_rows(path)[0][4] is True, "the emitted row read as sampled"


def test_only_a_note_the_emitter_writes_is_credited_to_the_key_lines_run(tmp_path):
    """The other half of that rule, and the one ``_stamp_kept`` applies before a re-emit
    replaces the clause: an undated *hand-written* note under an exhaustive key line was
    not measured by that run. Credited to it, a sampled bit-exact gate would read as
    exhaustive, escape the sampled-zero floor, and pass the exact-budget audit 0 for 0.
    """
    path = tmp_path / "budget.yaml"
    path.write_text(
        "Acosh:  # measured by: exhaustive Float16_b/Float16/Bfp8_b sweep, wormhole, "
        "2026-09-30, except where a row says otherwise\n"
        '  - {in: Float32, out: Float32, dest: "Yes", max_ulp: 0}  # max 0 ULP over 2048 pts\n'
        '  - {in: Float16, out: Float16, dest: "No", max_ulp: 0}  # max 0 ULP\n',
        encoding="utf-8",
    )
    assert [row[4] for row in _measured_budget_rows(path)] == [False, True]
    assert _sampled_zero_budgets(path) == [
        ("Acosh", '- {in: Float32, out: Float32, dest: "Yes", max_ulp: 0}')
    ]


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
_GOLDEN_IN_INPUT_FORMAT = (
    "the golden is tilized and untilized in the input format, so an fp16 input's "
    "reference overflows at 65504 where the kernel's 32-bit Dest holds the answer "
    "(exp2(16) reads inf against an exact 65536); #58590 fixes the golden"
)
_NO_INFINITY_IN_A_16BIT_DEST = (
    "a 16-bit Dest has no infinity: where the answer overflows, the kernel's result "
    "reads as the Dest's largest magnitude (-130560 for sinh(-65504)), a finite answer "
    "to an infinite golden"
)
_UNMEASURABLE_CELLS_ACKNOWLEDGED = {
    **{
        (MathOperation.Tan, DataFormat.Float16, out, None, None): (
            _GOLDEN_IN_INPUT_FORMAT
            + "; tan(177.5) is -66347, which fp16 has no room for"
        )
        for out in (DataFormat.Float16_b, DataFormat.Float32)
    },
    # -- the golden, not the kernel -----------------------------------------------
    **{
        (
            op,
            DataFormat.Float16,
            out,
            None,
            DestAccumulation.Yes,
        ): _GOLDEN_IN_INPUT_FORMAT
        for op in (
            MathOperation.Cosh,
            MathOperation.Exp,
            MathOperation.Exp2,
            MathOperation.Expm1,
            MathOperation.Selu,
            MathOperation.Sinh,
            MathOperation.Square,
            MathOperation.UnaryPower,
            MathOperation.Xielu,
        )
        for out in (DataFormat.Float16_b, DataFormat.Float32)
    },
    (
        MathOperation.Cbrt,
        DataFormat.Float16_b,
        DataFormat.Float16,
        None,
        DestAccumulation.Yes,
    ): (
        "the golden is rounded to the bfloat16 input format, so cbrt(2.8e14) = 65439 "
        "reads 65536 -> inf against the kernel's correct 65440; #58590 fixes the golden"
    ),
    # -- the store or the Dest, not the op ------------------------------------------
    (MathOperation.Sinh, DataFormat.Float16, None, None, DestAccumulation.No): (
        _NO_INFINITY_IN_A_16BIT_DEST
    ),
    # -- the approximation's own shortfall at the fp16 overflow edge ----------------
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
    # -- kernel behaviour over a wide band of the format, not yet triaged -----------
    # Each is what the sweep found and the row records; none is a golden or store
    # artefact, and none is a handful of lanes an issue could name. They hold their
    # cells on tolerance until the kernel is looked at.
    (MathOperation.Digamma, None, None, None, None): (
        "non-finite of the wrong sign for |x| above ~1e36, and a finite -61312 where a "
        "block-quantized input lands on the pole at 0"
    ),
    (MathOperation.ExpWithBase, None, None, ApproximationMode.Yes, None): (
        "past the overflow point the approximate kernel returns x itself instead of "
        "inf, and NaN for large negative x where the answer is 0"
    ),
    (
        MathOperation.ExpWithBase,
        DataFormat.Float16,
        None,
        ApproximationMode.No,
        DestAccumulation.Yes,
    ): (_GOLDEN_IN_INPUT_FORMAT),
    (MathOperation.Expm1Cw, None, None, None, None): (
        "past the overflow point (x >= 90) the kernel returns -1, the x -> -inf limit, "
        "instead of inf"
    ),
    (MathOperation.I0, None, None, None, None): (
        "saturates at 6.05e37 where i0 overflows fp32: a finite answer to an infinite "
        "golden from |x| ~ 90 up"
    ),
    (MathOperation.I1, None, None, None, None): (
        "saturates at -1.16e37 where i1 overflows fp32, half the format"
    ),
    (MathOperation.Lgamma, None, None, None, None): (
        "saturates at 3.32e38 where lgamma overflows fp32"
    ),
    (MathOperation.Polygamma, None, None, None, None): (
        "0 for |x| above ~1e36 where the golden is inf, and inf near x = -7 where the "
        "golden is 1.8e31"
    ),
    (MathOperation.Rpow, None, None, None, None): (
        "NaN where the answer is 0 and 1 where it is inf, for |x| above ~8e31"
    ),
}


#: How many not-measurable cells each *wildcard* acknowledgement above covers today. An
#: acknowledgement that leaves a dimension open -- an op-wide one most of all -- would
#: otherwise approve every cell of the op that is ever demoted, for any cause, and the
#: stale check below could not retire it while one such cell remained. Pinning the count
#: makes a newly demoted cell a change someone has to read: the table moved, the cause
#: must be looked at, and the number here bumped on purpose. Exact-cell keys need no pin.
_UNMEASURABLE_CELLS_ACKNOWLEDGED_COUNTS = {
    (MathOperation.Tan, DataFormat.Float16, DataFormat.Float16_b, None, None): 2,
    (MathOperation.Tan, DataFormat.Float16, DataFormat.Float32, None, None): 2,
    (
        MathOperation.Cosh,
        DataFormat.Float16,
        DataFormat.Float16_b,
        None,
        DestAccumulation.Yes,
    ): 1,
    (
        MathOperation.Cosh,
        DataFormat.Float16,
        DataFormat.Float32,
        None,
        DestAccumulation.Yes,
    ): 1,
    (
        MathOperation.Exp,
        DataFormat.Float16,
        DataFormat.Float16_b,
        None,
        DestAccumulation.Yes,
    ): 2,
    (
        MathOperation.Exp,
        DataFormat.Float16,
        DataFormat.Float32,
        None,
        DestAccumulation.Yes,
    ): 2,
    (
        MathOperation.Exp2,
        DataFormat.Float16,
        DataFormat.Float16_b,
        None,
        DestAccumulation.Yes,
    ): 1,
    (
        MathOperation.Exp2,
        DataFormat.Float16,
        DataFormat.Float32,
        None,
        DestAccumulation.Yes,
    ): 1,
    (
        MathOperation.Expm1,
        DataFormat.Float16,
        DataFormat.Float16_b,
        None,
        DestAccumulation.Yes,
    ): 1,
    (
        MathOperation.Expm1,
        DataFormat.Float16,
        DataFormat.Float32,
        None,
        DestAccumulation.Yes,
    ): 1,
    (
        MathOperation.Selu,
        DataFormat.Float16,
        DataFormat.Float16_b,
        None,
        DestAccumulation.Yes,
    ): 1,
    (
        MathOperation.Selu,
        DataFormat.Float16,
        DataFormat.Float32,
        None,
        DestAccumulation.Yes,
    ): 1,
    (
        MathOperation.Sinh,
        DataFormat.Float16,
        DataFormat.Float16_b,
        None,
        DestAccumulation.Yes,
    ): 1,
    (
        MathOperation.Sinh,
        DataFormat.Float16,
        DataFormat.Float32,
        None,
        DestAccumulation.Yes,
    ): 1,
    (
        MathOperation.Square,
        DataFormat.Float16,
        DataFormat.Float16_b,
        None,
        DestAccumulation.Yes,
    ): 1,
    (
        MathOperation.Square,
        DataFormat.Float16,
        DataFormat.Float32,
        None,
        DestAccumulation.Yes,
    ): 1,
    (
        MathOperation.UnaryPower,
        DataFormat.Float16,
        DataFormat.Float16_b,
        None,
        DestAccumulation.Yes,
    ): 1,
    (
        MathOperation.UnaryPower,
        DataFormat.Float16,
        DataFormat.Float32,
        None,
        DestAccumulation.Yes,
    ): 1,
    (
        MathOperation.Xielu,
        DataFormat.Float16,
        DataFormat.Float16_b,
        None,
        DestAccumulation.Yes,
    ): 1,
    (
        MathOperation.Xielu,
        DataFormat.Float16,
        DataFormat.Float32,
        None,
        DestAccumulation.Yes,
    ): 1,
    (
        MathOperation.Cbrt,
        DataFormat.Float16_b,
        DataFormat.Float16,
        None,
        DestAccumulation.Yes,
    ): 1,
    (MathOperation.Sinh, DataFormat.Float16, None, None, DestAccumulation.No): 2,
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
    (MathOperation.Digamma, None, None, None, None): 22,
    (MathOperation.ExpWithBase, None, None, ApproximationMode.Yes, None): 25,
    (
        MathOperation.ExpWithBase,
        DataFormat.Float16,
        None,
        ApproximationMode.No,
        DestAccumulation.Yes,
    ): 2,
    (MathOperation.Expm1Cw, None, None, None, None): 25,
    (MathOperation.I0, None, None, None, None): 22,
    (MathOperation.I1, None, None, None, None): 20,
    (MathOperation.Lgamma, None, None, None, None): 20,
    (MathOperation.Polygamma, None, None, None, None): 46,
    (MathOperation.Rpow, None, None, None, None): 22,
}


def _not_measurable_cells(path=_TABLE_PATH):
    """``(op, in, out, approx_or_None, dest_or_None)`` for every ``not measurable`` row
    on an output a step budget could gate."""
    from helpers.ulp_sweep import _row_fields

    cells, op = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line[0].isspace() and not line.startswith("#"):
            op = line.split(":")[0].strip()
        if "not measurable" not in line or not line.lstrip().startswith("- "):
            continue
        fields = _row_fields(line)
        out_fmt = DataFormat[fields["out"]]
        if not _step_gateable_output(out_fmt):
            continue
        cells.append(
            (
                MathOperation[op],
                DataFormat[fields["in"]],
                out_fmt,
                ApproximationMode[fields["approx"]] if "approx" in fields else None,
                DestAccumulation[fields["dest"]] if "dest" in fields else None,
            )
        )
    return cells


def test_a_row_parked_by_disagreeing_lanes_records_the_rest_of_its_cell():
    """`nonfinite_reason` writes the step count over the measurable lanes beside the
    lanes that park a cell, so a cell parked by four lanes still says what the other
    tens of thousands measure, and the headroom report (from #57527) has a figure to
    hold it to. A row written before the emitter kept that figure records nothing for
    them; re-emit its op."""
    shape = re.compile(r"; max \d+ ULP over the \d+ measurable lanes")
    bare, op = [], None
    for line in _TABLE_PATH.read_text(encoding="utf-8").splitlines():
        if line and not line[0].isspace() and not line.startswith("#"):
            op = line.split(":")[0].strip()
        if "lane(s) disagreeing" in line and not shape.search(line):
            bare.append(f"{op}: {line.split('#', 1)[0].strip()}")
    assert not bare, (
        f"{len(bare)} row(s) name the lanes that park their cell but not the maximum "
        "over the rest of it; re-emit their ops:\n  " + "\n  ".join(bare)
    )


def _acknowledges(key, cell) -> bool:
    return all(k is None or k == c for k, c in zip(key, cell))


#: Swept cells of an enrolled unary op that resolve to no row of the op's own, with why.
#: Empty: every enrolled op covers every cell the sweep drives. The emitter writes whole
#: grids, so a hole can only open where it keeps a block verbatim (a floor row) and the
#: sweep's format set then grows -- the cell is measured, resolves to the global default,
#: and nothing holds it. A cell listed here is a decision, with its reason, not a gap.
_UNRESOLVED_SWEPT_CELLS: dict = {}


def test_every_swept_cell_of_an_enrolled_op_resolves_to_a_row_of_its_own():
    """The sweep gates a ULP row and the headroom report (from #57527) judges a
    tolerance row; a swept cell that resolves to ``TOLERANCE_CONTRACT`` -- no row of the
    op covers it -- is measured and judged by nothing, and reads in the table exactly
    like a cell nobody ever enrolled. ``test_every_swept_cell_of_an_exact_op_is_gated_or_waived`` asks this
    of the exact ops; this asks it of every enrolled op."""
    from helpers.sfpu_domains import sfpu_unary_ops
    from helpers.ulp_sweep import sweep_cells

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
            if resolve_contract(table, query, label=op.name) is TOLERANCE_CONTRACT:
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
        f"{op.name} {i.name}->{o.name}" for op, i, o, _, _ in stale
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
    the table, and the emitter cannot create an op's block. So every unary op either
    has one or ``_UNARY_OPS_NOT_SWEPT`` says why not (Erfc, Lgamma and Xielu sat here).
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
