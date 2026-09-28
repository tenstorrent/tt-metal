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
    DEFAULT,
    MEASURED_ARCH,
    TOLERANCE_CONTRACT,
    AccuracyContract,
    BudgetKey,
    Metric,
    _load_table,
    _winner,
    accuracy_contract,
    enrolled_ops,
    resolve_contract,
    usable_budget_ceiling,
    validate_registry,
)
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from helpers.ulp import (
    _ULP_DTYPES,
    _ULP_PROXY_DTYPES,
    MAX_MEANINGFUL_ULP,
    ULP_FORMATS,
    has_ulp_gate,
    ulp_dtype,
)
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
#: whose per-format tolerances moved into the table, and three ops past the usable
#: ceiling on every float column (measurements on their YAML rows). Sign and Heaviside
#: are not here: WH reads -0.0 as negative, so they carry budgets only on the cells
#: where that lane is not in play.
ONLY_EVER_TOLERANCE = frozenset(
    {
        MathOperation.SigmoidAppx,
        MathOperation.GeluAppx,
        MathOperation.SfpuElwpow,
        MathOperation.SfpuXlogy,
        MathOperation.GeluTanh,
        MathOperation.Tanhshrink,
        MathOperation.SfpuElwmul,
    }
)


def test_every_enrolled_op_reaches_its_step_budget():
    """The sweep must reach the ULP branch for every enrolled op but the ones above.

    The input-keyed ops are pinned separately: a sweep that left ``input_format`` unset
    sent every one of them to ``TOLERANCE_CONTRACT`` while this still passed for the rest.
    """
    input_keyed = {
        op
        for op, table in _SFPU_ACCURACY_BUDGET.items()
        if any(key.input_format is not None for key in table)
    }
    assert len(input_keyed) == 54, sorted(op.name for op in input_keyed)
    with_budget = {op for op, _, _, _ in _live_step_budgets()}
    missing = set(enrolled_ops()) - with_budget
    assert missing == ONLY_EVER_TOLERANCE, sorted(op.name for op in missing)
    assert input_keyed - ONLY_EVER_TOLERANCE <= with_budget


#: Ops exact by construction: a sign-bit change, a copy, or an integer-valued result.
EXACT_BY_CONSTRUCTION = (
    MathOperation.Abs,
    MathOperation.Neg,
    MathOperation.Identity,
    MathOperation.Floor,
    MathOperation.Ceil,
    MathOperation.Trunc,
)

#: The subset writing an integer, which survives any pack that can represent it.
INTEGER_VALUED = (MathOperation.Floor, MathOperation.Ceil, MathOperation.Trunc)

#: Ops whose correct result is the *only* result: an integer, a predicate's 1.0/0.0, a
#: pass-through-or-zero selection, a constant, a clamp, or a single IEEE add. One step
#: here is the contract breaking, not the pack path. Listed by what the op computes, not
#: read back from the table, which would agree with it by construction.
EXACT_ZERO_BY_CONSTRUCTION = (
    *INTEGER_VALUED,
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

#: What the output pack may cost an exact op on a cell that converts: the exhaustive
#: sweep reaches the magnitudes where a cross-format output rounds and measures 2.
_PACK_PATH_STEPS = 2


def _mantissa_bits(fmt):
    """0 for a format without a per-element ULP (these appear on the input axis only)."""
    return _ULP_DTYPES[ulp_dtype(fmt)].mantissa_bits if has_ulp_gate(fmt) else 0


def _exact_allowance(op, input_format, output_format):
    """``(steps, reason)``: the slack an exact op may carry on one cell -- the cost of
    the output pack, and only where there is one."""
    if output_format in _ULP_PROXY_DTYPES:
        # Rows measured at 2-3 steps into Bfp8_b are parked on the tolerance metric, so
        # a Bfp8_b ULP row is the 0-step enrolment; otherwise only the 25.6-step usable
        # ceiling would bound it.
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
        # e.g. Float16 -> Float16_b: bf16 cannot hold every integer fp16 can.
        return _PACK_PATH_STEPS, "the output has fewer mantissa bits than the input"
    return 0, "the output can represent every value this op produces from that input"


@pytest.mark.parametrize("op", EXACT_BY_CONSTRUCTION, ids=lambda op: op.name)
def test_an_exact_op_never_carries_a_wide_budget(op):
    """These are the canaries: a budget past the pack path means the number was fitted
    to a failure."""
    for budget_op, in_fmt, fmt, contract in _live_step_budgets():
        if budget_op is not op:
            continue
        allowance, why = _exact_allowance(op, in_fmt, fmt)
        assert contract.max_ulp <= allowance, (
            f"{op.name} on {in_fmt and in_fmt.name}->{fmt.name} carries "
            f"max_ulp={contract.max_ulp}, past the {allowance} it may have because "
            f"{why}. The op is exact by construction; investigate the datapath or the "
            "golden rather than widening the budget."
        )


#: Swept cells of an exact-by-construction op that the table holds on the tolerance
#: metric, with what was measured there. Each is a real deviation on an op that should
#: be exact, and none has a cause established yet; the test below keeps the list from
#: growing unnoticed, and fails when an entry is no longer needed.
_EXACT_OP_DEMOTIONS = {
    **{
        (op, in_fmt, DataFormat.Float16, DestAccumulation.Yes): "512 ULP measured"
        for op in (MathOperation.Abs, MathOperation.Neg, MathOperation.Identity)
        for in_fmt in (DataFormat.Float16_b, DataFormat.Bfp8_b)
    },
    (
        MathOperation.Floor,
        DataFormat.Bfp8_b,
        DataFormat.Float16,
        DestAccumulation.Yes,
    ): "15360 ULP measured",
    **{
        (MathOperation.Floor, DataFormat.Bfp8_b, DataFormat.Float16_b, dest): (
            "16129 ULP measured"
        )
        for dest in DestAccumulation
    },
}


@pytest.mark.parametrize("op", EXACT_BY_CONSTRUCTION, ids=lambda op: op.name)
def test_every_swept_cell_of_an_exact_op_is_gated_or_waived(op):
    """`test_an_exact_op_never_carries_a_wide_budget` reads only ULP rows, so a cell the
    emitter demoted to tolerance is invisible to it -- and the sweep does not gate
    tolerance cells. Every swept, non-block cell of these ops must resolve to a step
    budget, or be listed above with its measurement."""
    from helpers.ulp_sweep import sweep_cells

    demoted = set()
    for in_fmt, out_fmt, approx, dest in sweep_cells():
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


@pytest.mark.parametrize("op", EXACT_ZERO_BY_CONSTRUCTION, ids=lambda op: op.name)
def test_an_exactly_rounded_op_carries_a_zero_budget(op):
    """ "Any drift is a regression" is the claim for these ops. The provenance guard lets
    a measured 0 be written as 1, so only this test notices that 1."""
    seen = False
    for budget_op, in_fmt, fmt, contract in _live_step_budgets():
        if budget_op is not op:
            continue
        seen = True
        allowance, why = _exact_allowance(op, in_fmt, fmt)
        assert contract.max_ulp <= allowance, (
            f"{op.name} on {in_fmt and in_fmt.name}->{fmt.name} carries "
            f"max_ulp={contract.max_ulp}, past the {allowance} it may have because "
            f"{why}. This op is exactly rounded by construction; anything more is "
            "the contract going away. Re-measure before widening it."
        )
    assert seen, f"{op.name} resolves to no ULP contract at all; the row was dropped"


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
    assert usable_budget_ceiling(DataFormat.Float16_b) == 6.4


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
#: Wormhole, 2026-09-18; the counts are in the YAML row comments.
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
    """The signbit, isinf/isnan and threshold sweeps use hand-built stimuli, and a budget
    binds on them too, so an op they drive must have been measured there."""
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


def _measured_budget_rows(path=_TABLE_PATH):
    """``(op_name, row_text, max_ulp, measured_or_None, exhaustive)`` for every
    ``max_ulp`` row. *exhaustive* is whether the comment the measurement was read from
    is the sweep's."""
    rows, op, op_measured, op_exhaustive = [], None, None, False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):  # `OpName:`, optionally with a header comment
            head, _, comment = line.partition("#")
            op = head.split(":")[0].strip()
            found = _MEASUREMENT.search(comment)
            op_measured = int(found.group(1)) if found else None
            op_exhaustive = _EXHAUSTIVE in comment
            continue
        body, _, comment = line.strip().partition("#")
        declared = re.search(r"max_ulp:\s*(\d+)", body)
        if not declared:
            continue  # a tolerance row has no step budget to back
        found = _MEASUREMENT.search(comment)
        measured = int(found.group(1)) if found else op_measured
        exhaustive = _EXHAUSTIVE in comment if found else op_exhaustive
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


def test_no_step_budget_exceeds_the_measurement_it_records():
    """An exhaustive row carries exactly the budget the emitter derives from its
    measurement -- ``_verdict``'s rule, 0 for 0 and otherwise ``EMIT_HEADROOM`` rounded
    up -- so a budget widened by hand has to falsify the comment beside it, which is
    the table header's rule for raising one. A 2x envelope would have admitted Acosh's
    ``max_ulp: 7  # max 6 ULP`` raised to 12 with its comment untouched.

    A sampled row may sit anywhere in ``[measured, MEASUREMENT_HEADROOM * measured]``,
    and at 1 over a measured 0: a finite sample cannot assert exactness."""
    from helpers.ulp_sweep import _row_fields, _verdict

    for op, body, budget, measured, exhaustive in _measured_budget_rows():
        if measured is None:
            continue  # owned by test_every_step_budget_names_the_measurement_it_came_from
        where = f"{op}: {body} (records {measured} ULP)"
        if exhaustive:
            out_fmt = _row_fields(body)["out"]
            assert ("ulp", budget) == _verdict(measured, out_fmt), (
                f"{where}: the emitter writes {_verdict(measured, out_fmt)[1]} for that "
                f"measurement, not {budget}. Re-measure rather than edit the number."
            )
        elif measured == 0:
            assert budget <= 1, f"{where}: a 0-ULP measurement cannot justify {budget}"
        else:
            assert budget >= measured, f"{where}: budget {budget} is below it"
            assert budget <= MEASUREMENT_HEADROOM * measured, (
                f"{where}: budget {budget} is more than "
                f"{MEASUREMENT_HEADROOM}x the measurement"
            )


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
    **{
        (op, DataFormat.Float16, DataFormat.Float16_b, None, DestAccumulation.Yes): (
            "the golden is tilized and untilized in the input format, so an fp16 input's "
            "reference overflows at 65504 where the kernel's 32-bit Dest holds the "
            "answer (exp2(16) reads inf against an exact 65536); #58590 fixes the golden"
        )
        for op in (MathOperation.Exp, MathOperation.Exp2, MathOperation.Square)
    },
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
        if not has_ulp_gate(out_fmt) or out_fmt in _ULP_PROXY_DTYPES:
            continue  # a block output is never gated from this sweep
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
